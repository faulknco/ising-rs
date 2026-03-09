# GPU Phase 2: Performance & Correctness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the per-device efficiency gap between the Phase 1 GPU baseline and HPC-level Monte Carlo by eliminating host↔device transfers, batching replica operations, adding autocorrelation-aware sampling, and validating correctness against CPU results.

**Architecture:** GPU-resident observable computation eliminates the 26 GB/run transfer bottleneck. Batched replica kernels amortize launch overhead. Online blocking analysis ensures statistical independence. Mixed-precision policy (f32 kernels + f64 reductions) preserves accuracy at N=128+. CPU/GPU parity benchmarks gate every change.

**Tech Stack:** Rust 1.94+, cudarc 0.12, CUDA C (sm_75), Python 3 (numpy, scipy).

**Prerequisite:** Phase 1 plan (`2026-03-08-gpu-acceleration-impl.md`) fully implemented.

---

## Phase 1 Bottleneck Analysis

The Phase 1 implementation has a critical performance bottleneck:

```
Per measurement sweep (Ising, N=64, 20 replicas):
  GPU kernel:           ~0.1 ms
  dtoh_sync_copy:       ~0.5 ms  ← stalls pipeline
  Host energy O(N³):    ~0.8 ms  ← redundant, data was on device
  ─────────────────────────────
  Total per sweep:      ~1.4 ms
  × 100,000 sweeps:     ~140 sec
  Transfer volume:      20 × 262 KB × 100,000 = 524 GB (!)

After Phase 2 (GPU-resident observables):
  GPU kernel:           ~0.1 ms
  GPU reduction:        ~0.02 ms
  dtoh partial sums:    ~0.001 ms (24 bytes, not 262 KB)
  ─────────────────────────────
  Total per sweep:      ~0.12 ms
  × 100,000 sweeps:     ~12 sec  (12x speedup)
  Transfer volume:      20 × 24 bytes × 100,000 = 48 MB (10,000x reduction)
```

---

## Task 1: GPU-resident energy reduction kernel for Ising

**Files:**
- Modify: `src/cuda/reduce_kernel.cu`
- Modify: `src/cuda/reduce_gpu.rs`

**Context:** `reduce_energy_ising` already exists but isn't used in the gpu_fss measurement loop. The gpu_fss Ising path calls `get_spins()` + `ising_e_m_host()` instead. We need to wire the existing kernel into the measurement path and verify correctness.

**Step 1: Add a combined Ising reduction function to reduce_gpu.rs**

Add a function that calls both `reduce_mag_ising` and `reduce_energy_ising` in sequence, returning `(energy_per_spin, mag_per_spin)` as f64. This replaces the host-side `ising_e_m_host()`.

```rust
/// GPU-side measurement: returns (energy_per_spin, abs_mag_per_spin) for Ising cubic.
/// No host↔device spin transfer — only partial sums come back (~n_blocks floats).
pub fn measure_ising_gpu(
    device: &Arc<CudaDevice>,
    spins: &CudaSlice<i8>,
    n: usize,
    j: f32,
) -> anyhow::Result<(f64, f64)> {
    let (total_energy, total_mag) = reduce_ising(device, spins, n, j)?;
    let n3 = (n * n * n) as f64;
    Ok((total_energy / n3, total_mag.abs() / n3))
}
```

**Step 2: Verify correctness against host computation**

Write a test (non-CUDA gated, uses `#[ignore]`) that:
1. Creates a `LatticeGpu` with known seed
2. Runs 100 sweeps
3. Computes observables via `get_spins()` + host function
4. Computes observables via `measure_ising_gpu()`
5. Asserts both match within f32 precision (~1e-5 relative error)

```rust
#[test]
#[ignore] // Requires --features cuda
fn gpu_reduction_matches_host() {
    // ... create lattice, sweep, compare GPU vs host E and M
}
```

**Step 3: Commit**

```bash
git add src/cuda/reduce_gpu.rs
git commit -m "feat(cuda): add measure_ising_gpu — GPU-resident observable computation"
```

---

## Task 2: Wire GPU-resident measurement into gpu_fss Ising path

**Files:**
- Modify: `src/bin/gpu_fss.rs`
- Modify: `src/cuda/lattice_gpu.rs`

**Context:** Replace the `get_spins()` + `ising_e_m_host()` pattern in the gpu_fss sampling loop with `measure_ising_gpu()`. This eliminates the 26 GB transfer bottleneck.

**Step 1: Add `measure_gpu()` method to LatticeGpu**

```rust
/// Measure E and |M| per spin using GPU reduction. No host↔device spin transfer.
pub fn measure_gpu(&self, j: f32) -> anyhow::Result<(f64, f64)> {
    use crate::cuda::reduce_gpu;
    reduce_gpu::load_reduce_kernels(&self.device)?; // idempotent if already loaded
    reduce_gpu::measure_ising_gpu(&self.device, &self.spins, self.n, j)
}
```

**Step 2: Update gpu_fss.rs Ising sampling loop**

Replace:
```rust
let spins = lat.get_spins().expect("get_spins failed");
let (e, m) = ising_e_m_host(&spins, n);
let e_per = e / n3;
let m_per = (m / n3).abs();
```

With:
```rust
let (e_per, m_per) = lat.measure_gpu(1.0).expect("GPU measure failed");
```

Keep the `ising_e_m_host` function for the parity benchmark (Task 6).

**Step 3: Verify output matches Phase 1**

Run gpu_fss with small parameters (N=4, 4 replicas, 100 samples) and compare CSV output between old (host) and new (GPU) paths. Values should match within f32 precision.

**Step 4: Commit**

```bash
git add src/cuda/lattice_gpu.rs src/bin/gpu_fss.rs
git commit -m "perf(cuda): eliminate host transfer in Ising measurement — GPU-resident E,M"
```

---

## Task 3: GPU-resident energy reduction for continuous spins

**Files:**
- Modify: `src/cuda/reduce_kernel.cu`
- Modify: `src/cuda/reduce_gpu.rs`

**Context:** `measure_raw()` in `ContinuousGpuLattice` uses GPU reduction for magnetisation but falls back to host for energy (`energy_continuous_host`). We need a GPU energy kernel for continuous spins.

**Step 1: Write `reduce_energy_continuous` kernel**

```c
// Continuous-spin energy: E = -J * sum_{<ij>} S_i · S_j (forward neighbours only)
extern "C" __global__ void reduce_energy_continuous(
    const float* spins,
    float* partial_energy,
    int    N,
    int    n_comp,
    float  J
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_sites = N * N * N;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float val = 0.0f;
    if (gid < n_sites) {
        int z = gid / (N * N);
        int r = gid % (N * N);
        int y = r / N;
        int x = r % N;

        int fwd[3];
        fwd[0] = z*N*N + y*N + (x+1)%N;
        fwd[1] = z*N*N + ((y+1)%N)*N + x;
        fwd[2] = ((z+1)%N)*N*N + y*N + x;

        for (int f = 0; f < 3; f++) {
            float dot = 0.0f;
            for (int c = 0; c < n_comp; c++) {
                dot += spins[gid * n_comp + c] * spins[fwd[f] * n_comp + c];
            }
            val -= J * dot;
        }
    }

    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_energy[blockIdx.x] = sdata[0];
}
```

**Step 2: Add Rust wrapper and register kernel**

Add `reduce_continuous_energy()` to `reduce_gpu.rs` and register `"reduce_energy_continuous"` in `load_reduce_kernels()`.

**Step 3: Update `measure_raw()` in `gpu_lattice_continuous.rs`**

Replace host energy computation with GPU kernel call. Remove `device.dtoh_sync_copy(&self.spins)`.

**Step 4: Correctness test**

Same pattern as Task 1: compare GPU vs host energy on a known configuration.

**Step 5: Commit**

```bash
git add src/cuda/reduce_kernel.cu src/cuda/reduce_gpu.rs src/cuda/gpu_lattice_continuous.rs
git commit -m "perf(cuda): GPU-resident energy for continuous spins — eliminates N³ transfer"
```

---

## Task 4: Pre-allocated reduction buffers

**Files:**
- Modify: `src/cuda/reduce_gpu.rs`
- Modify: `src/cuda/lattice_gpu.rs`
- Modify: `src/cuda/gpu_lattice_continuous.rs`

**Context:** Currently `reduce_ising()` allocates `partial_mag` and `partial_e` arrays inside each call via `device.alloc_zeros()`. Over 100,000 samples, that's 200,000 allocations. Pre-allocate these buffers once and reuse.

**Step 1: Add `ReductionBuffers` struct**

```rust
pub struct ReductionBuffers {
    pub partial_f1: CudaSlice<f32>,
    pub partial_f2: CudaSlice<f32>,
    pub partial_f3: CudaSlice<f32>,
}

impl ReductionBuffers {
    pub fn new(device: &Arc<CudaDevice>, n_blocks: usize) -> anyhow::Result<Self> {
        Ok(Self {
            partial_f1: device.alloc_zeros::<f32>(n_blocks)?,
            partial_f2: device.alloc_zeros::<f32>(n_blocks)?,
            partial_f3: device.alloc_zeros::<f32>(n_blocks)?,
        })
    }
}
```

**Step 2: Store buffers in LatticeGpu and ContinuousGpuLattice**

Add `reduce_bufs: ReductionBuffers` field. Allocate once in `new()`.

**Step 3: Update reduction functions to take `&mut ReductionBuffers`**

Change `reduce_ising()` and `reduce_continuous_mag()` signatures.

**Step 4: Commit**

```bash
git add src/cuda/reduce_gpu.rs src/cuda/lattice_gpu.rs src/cuda/gpu_lattice_continuous.rs
git commit -m "perf(cuda): pre-allocate reduction buffers — eliminate per-sample alloc overhead"
```

---

## Task 5: Online blocking analysis for autocorrelation

**Files:**
- Create: `src/blocking.rs`
- Modify: `src/lib.rs`

**Context:** Phase 1 has no integrated autocorrelation time (τ_int) estimation. Samples may be correlated, making error bars too small. We need an online blocking analysis that runs during sampling and reports effective independent samples.

**Step 1: Implement Flyvbjerg-Petersen blocking**

```rust
/// Online blocking analysis for autocorrelation-aware error estimation.
/// Implements Flyvbjerg & Petersen (1989): "Error estimates on averages
/// of correlated data."

pub struct BlockingAnalyzer {
    data: Vec<f64>,
}

impl BlockingAnalyzer {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn push(&mut self, value: f64) {
        self.data.push(value);
    }

    /// Compute blocking analysis: returns (mean, error, tau_int_estimate).
    /// Error accounts for autocorrelation via systematic blocking.
    pub fn analyze(&self) -> (f64, f64, f64) {
        let n = self.data.len();
        if n < 4 { return (0.0, 0.0, 0.0); }

        let mean: f64 = self.data.iter().sum::<f64>() / n as f64;
        let naive_var = self.data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (n - 1) as f64;

        // Block the data: repeatedly average pairs
        let mut block_data = self.data.clone();
        let mut prev_var_mean = naive_var / n as f64;
        let mut tau_int = 0.5; // initial estimate

        for _level in 0..20 {
            let m = block_data.len();
            if m < 4 { break; }

            let m_new = m / 2;
            let mut new_data = Vec::with_capacity(m_new);
            for i in 0..m_new {
                new_data.push((block_data[2*i] + block_data[2*i+1]) / 2.0);
            }
            block_data = new_data;

            let bm = block_data.len();
            let block_mean: f64 = block_data.iter().sum::<f64>() / bm as f64;
            let block_var = block_data.iter()
                .map(|&x| (x - block_mean).powi(2))
                .sum::<f64>() / (bm - 1) as f64;
            let var_mean = block_var / bm as f64;

            // When var_mean plateaus, we've reached the decorrelation scale
            if var_mean > prev_var_mean * 0.95 && var_mean < prev_var_mean * 1.05 {
                tau_int = var_mean / (naive_var / n as f64) / 2.0;
                return (mean, var_mean.sqrt(), tau_int);
            }
            prev_var_mean = var_mean;
        }

        (mean, prev_var_mean.sqrt(), tau_int)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}
```

**Step 2: Add unit tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uncorrelated_data_gives_tau_near_half() {
        let mut ba = BlockingAnalyzer::new();
        let mut rng = rand::thread_rng();
        use rand::Rng;
        for _ in 0..10000 {
            ba.push(rng.gen::<f64>());
        }
        let (mean, err, tau) = ba.analyze();
        assert!((mean - 0.5).abs() < 0.02);
        assert!(tau < 2.0, "uncorrelated data should have tau ~0.5, got {tau}");
    }

    #[test]
    fn correlated_data_gives_larger_tau() {
        let mut ba = BlockingAnalyzer::new();
        let mut x = 0.0_f64;
        let mut rng = rand::thread_rng();
        use rand::Rng;
        for _ in 0..10000 {
            x = 0.95 * x + 0.05 * rng.gen::<f64>(); // AR(1), tau ~10
            ba.push(x);
        }
        let (_mean, _err, tau) = ba.analyze();
        assert!(tau > 2.0, "correlated AR(1) data should have tau > 2, got {tau}");
    }
}
```

**Step 3: Export from lib.rs**

```rust
pub mod blocking;
```

**Step 4: Commit**

```bash
git add src/blocking.rs src/lib.rs
git commit -m "feat: add Flyvbjerg-Petersen blocking analysis for autocorrelation"
```

---

## Task 6: CPU/GPU parity benchmark pack

**Files:**
- Create: `tests/gpu_parity.rs`

**Context:** Before any further optimization, establish automated correctness checking: run the same (model, N, seed, T) on CPU and GPU, compare observables within tolerance. This is a regression gate for all subsequent changes.

**Step 1: Write parity benchmark**

```rust
//! CPU/GPU parity tests.
//! Run with: cargo test --features cuda -- --ignored gpu_parity

#[test]
#[ignore] // Requires --features cuda and GPU
fn gpu_parity_ising_n8() {
    // 1. CPU: run 500 Metropolis sweeps at beta=0.22, seed=42, N=8
    //    Compute <E>, <|M|>, <M²>, <M⁴>
    // 2. GPU: same parameters using LatticeGpu
    //    Compute same observables via GPU reduction
    // 3. Assert: |E_cpu - E_gpu| / |E_cpu| < 1e-3
    //    Assert: |M_cpu - M_gpu| / |M_cpu| < 1e-3
    //    (f32 vs f64 gives ~1e-4 relative error on individual sums)
}

#[test]
#[ignore]
fn gpu_parity_heisenberg_n8() { /* Same pattern for continuous spins */ }

#[test]
#[ignore]
fn gpu_parity_xy_n8() { /* Same pattern */ }
```

**Step 2: Define acceptance thresholds**

```rust
const E_TOLERANCE: f64 = 1e-3;  // relative error on energy per spin
const M_TOLERANCE: f64 = 5e-3;  // relative error on |M| per spin (noisier)
```

**Step 3: Commit**

```bash
git add tests/gpu_parity.rs
git commit -m "test: add CPU/GPU parity benchmarks with acceptance thresholds"
```

---

## Task 7: Mixed-precision policy — f64 accumulation in reduction kernels

**Files:**
- Modify: `src/cuda/reduce_kernel.cu`
- Modify: `src/cuda/reduce_gpu.rs`

**Context:** At N=128 (2 million sites), summing f32 values loses ~3 decimal digits due to catastrophic cancellation. GPU kernels should keep f32 for spin storage and per-site arithmetic, but promote partial sums to f64 during block-wide reduction.

**Step 1: Create mixed-precision reduction kernels**

Duplicate the existing reduction kernels with `_f64` suffix. In shared memory, use `double` for the accumulation:

```c
extern "C" __global__ void reduce_mag_ising_f64(
    const signed char* spins,
    double* partial_mag,   // f64 output
    int    n_sites
) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < n_sites) ? (double)spins[gid] : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_mag[blockIdx.x] = sdata[0];
}
```

**Step 2: Add Rust wrappers for f64 reductions**

```rust
pub fn reduce_ising_f64(
    device: &Arc<CudaDevice>,
    spins: &CudaSlice<i8>,
    n: usize,
    j: f32,
) -> anyhow::Result<(f64, f64)> { /* ... */ }
```

**Step 3: Use f64 reductions when N >= 64**

Add a threshold: for N < 64, f32 reductions are fine. For N >= 64, use f64. This is a policy decision that can be exposed as a CLI flag later.

**Step 4: Commit**

```bash
git add src/cuda/reduce_kernel.cu src/cuda/reduce_gpu.rs
git commit -m "feat(cuda): mixed-precision reduction — f64 accumulation for N≥64"
```

---

## Task 8: GPU-resident parallel tempering — batched replica sweeps

**Files:**
- Create: `src/cuda/batched_sweep.cu`
- Create: `src/cuda/batched_sweep.rs`
- Modify: `src/cuda/mod.rs`
- Modify: `build.rs`

**Context:** Phase 1 launches one kernel per replica per parity per sweep. For 20 replicas that's 40 kernel launches per sweep with synchronization after each pair. We can batch all replicas into a single kernel launch by offsetting into a concatenated spin array.

**Step 1: Write batched Ising sweep kernel**

```c
// Batched checkerboard Metropolis: update all replicas in one launch.
// spins_all: concatenated spin arrays [replica_0 | replica_1 | ... | replica_R-1]
// betas: per-replica inverse temperatures
extern "C" __global__ void batched_metropolis_kernel(
    signed char* spins_all,
    curandState*  rng_states,
    const float*  betas,        // [n_replicas]
    int           N,
    int           n_replicas,
    float         J,
    float         h,
    int           parity
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int sites_per_replica = (N * N * N) / 2;
    int total = sites_per_replica * n_replicas;
    if (tid >= total) return;

    int replica = tid / sites_per_replica;
    int local_tid = tid % sites_per_replica;
    int offset = replica * N * N * N;
    float beta = betas[replica];

    // Same checkerboard logic as before, but index into spins_all[offset + idx]
    // ...
}
```

**Step 2: Write Rust wrapper**

`BatchedIsingSweep` struct that allocates a single concatenated spin buffer and a single RNG buffer for all replicas.

**Step 3: Update gpu_fss.rs to use batched sweeps**

Replace the per-replica `lat.step()` loop with a single `batched.sweep()` call.

**Step 4: Verify correctness via parity test**

Same output as per-replica path.

**Step 5: Commit**

```bash
git add src/cuda/batched_sweep.cu src/cuda/batched_sweep.rs src/cuda/mod.rs build.rs src/bin/gpu_fss.rs
git commit -m "perf(cuda): batched replica sweeps — single kernel launch for all replicas"
```

---

## Task 9: Integrate blocking analysis into gpu_fss output

**Files:**
- Modify: `src/bin/gpu_fss.rs`

**Context:** Wire the blocking analyzer from Task 5 into the gpu_fss measurement loop. Report τ_int alongside observables, and add autocorrelation-corrected error bars to the summary CSV.

**Step 1: Add BlockingAnalyzer per temperature index per observable**

```rust
use ising::blocking::BlockingAnalyzer;

let mut ba_e: Vec<BlockingAnalyzer> = (0..n_replicas).map(|_| BlockingAnalyzer::new()).collect();
let mut ba_m: Vec<BlockingAnalyzer> = (0..n_replicas).map(|_| BlockingAnalyzer::new()).collect();
```

**Step 2: Feed samples into analyzers during measurement loop**

```rust
ba_e[t_idx].push(e_per);
ba_m[t_idx].push(m_per);
```

**Step 3: Update summary CSV with error bars and tau_int**

Add columns: `E_err,M_err,tau_E,tau_M` computed from blocking analysis.

**Step 4: Commit**

```bash
git add src/bin/gpu_fss.rs
git commit -m "feat: autocorrelation-corrected error bars in gpu_fss via blocking analysis"
```

---

## Task 10: Autotuning layer — adaptive delta and exchange rate

**Files:**
- Create: `src/autotune.rs`
- Modify: `src/lib.rs`

**Context:** The Metropolis proposal width (delta) for continuous spins and the replica exchange interval (exchange_every) strongly affect efficiency. Auto-tune during warmup: adjust delta to target ~40% acceptance rate, adjust exchange_every based on acceptance ratio.

**Step 1: Write acceptance rate tracker**

```rust
pub struct AcceptanceTracker {
    accepted: usize,
    total: usize,
}

impl AcceptanceTracker {
    pub fn new() -> Self { Self { accepted: 0, total: 0 } }
    pub fn record(&mut self, accepted: bool) { self.total += 1; if accepted { self.accepted += 1; } }
    pub fn rate(&self) -> f64 { if self.total == 0 { 0.5 } else { self.accepted as f64 / self.total as f64 } }
    pub fn reset(&mut self) { self.accepted = 0; self.total = 0; }
}
```

**Step 2: Write delta autotuner**

```rust
/// Adjust proposal width to target acceptance rate.
/// Call every ~100 sweeps during warmup.
pub fn tune_delta(current_delta: f64, acceptance_rate: f64, target: f64) -> f64 {
    let ratio = acceptance_rate / target;
    let new_delta = current_delta * ratio.clamp(0.5, 2.0);
    new_delta.clamp(0.01, 3.0) // safety bounds
}
```

**Step 3: Wire into gpu_fss continuous path warmup**

During warmup, every 100 sweeps, compute acceptance rate and adjust delta.

**Step 4: Unit tests**

```rust
#[test]
fn tune_delta_increases_when_acceptance_too_high() {
    let d = tune_delta(0.5, 0.8, 0.4); // 80% acceptance, target 40%
    assert!(d > 0.5); // should increase delta to reduce acceptance
}
```

**Step 5: Commit**

```bash
git add src/autotune.rs src/lib.rs src/bin/gpu_fss.rs
git commit -m "feat: adaptive delta autotuning during warmup"
```

---

## Task 11: Multi-spin Ising backend (MSC)

**Files:**
- Create: `src/cuda/msc_kernel.cu`
- Create: `src/cuda/msc_lattice.rs`
- Modify: `src/cuda/mod.rs`
- Modify: `build.rs`

**Context:** Multi-spin coding packs 32 Ising replicas into one 32-bit word. Bit k at site i holds the spin of replica k. All 32 replicas update simultaneously with a single set of neighbor reads. This gives 32x throughput for Ising specifically.

**Step 1: Write MSC checkerboard kernel**

```c
// Multi-spin coded Ising: 32 replicas per uint32_t word.
// spins_msc[site] = 32-bit word, bit k = spin of replica k (1=up, 0=down)
// boltz_masks[5][32]: pre-computed acceptance masks per delta_E per replica
extern "C" __global__ void msc_metropolis_kernel(
    unsigned int* spins_msc,    // [N³] words, 32 replicas each
    curandState*  rng_states,
    const unsigned int* boltz_masks, // [5][32] pre-computed
    int           N,
    int           parity
) {
    // For each site with correct parity:
    //   1. Read 6 neighbour words
    //   2. For each bit position: count aligned neighbours (popcount tricks)
    //   3. Compute delta_E index (0..4 for z=6 cubic)
    //   4. Accept/reject using pre-computed Boltzmann mask
    //   5. XOR flip accepted bits
}
```

**Step 2: Pre-compute Boltzmann masks on host**

For each replica temperature and each possible delta_E value, compute whether to accept. Pack into 32-bit masks.

**Step 3: Write Rust wrapper `MscLattice`**

```rust
pub struct MscLattice {
    n: usize,
    n_replicas: usize, // max 32
    device: Arc<CudaDevice>,
    spins_msc: CudaSlice<u32>,  // [N³] words
    rng_states: CudaSlice<u8>,
    boltz_masks: CudaSlice<u32>, // [5 * 32] pre-computed
}
```

**Step 4: Commit**

```bash
git add src/cuda/msc_kernel.cu src/cuda/msc_lattice.rs src/cuda/mod.rs build.rs
git commit -m "feat(cuda): multi-spin coded Ising — 32 replicas per word"
```

---

## Task 12: Nsight performance benchmark script

**Files:**
- Create: `scripts/benchmark_gpu.sh`

**Context:** Profile kernel arithmetic intensity, memory bandwidth utilisation, and occupancy using `nsys` and `ncu`. Produces a summary that identifies the next bottleneck after Phase 2 optimisations.

**Step 1: Write benchmark script**

```bash
#!/usr/bin/env bash
# GPU performance benchmarks using NVIDIA Nsight tools.
#
# Prerequisites: nsys and ncu on PATH (from CUDA Toolkit)
# Usage: bash scripts/benchmark_gpu.sh

set -e
OUTDIR="benchmarks"
mkdir -p "$OUTDIR"

echo "=== Building gpu_fss with profiling ==="
cargo build --release --features cuda --bin gpu_fss

echo "=== Nsight Systems timeline (Ising N=32) ==="
nsys profile -o "$OUTDIR/ising_n32" \
  target/release/gpu_fss \
  --model ising --sizes 32 \
  --tmin 4.4 --tmax 4.6 --replicas 4 \
  --warmup 500 --samples 1000 \
  --exchange-every 10

echo "=== Nsight Compute kernel analysis ==="
ncu --set full -o "$OUTDIR/ising_n32_kernels" \
  target/release/gpu_fss \
  --model ising --sizes 32 \
  --tmin 4.5 --tmax 4.5 --replicas 1 \
  --warmup 100 --samples 200 \
  --exchange-every 10

echo "=== Benchmark complete ==="
echo "Open $OUTDIR/ising_n32.nsys-rep in Nsight Systems"
echo "Open $OUTDIR/ising_n32_kernels.ncu-rep in Nsight Compute"
```

**Step 2: Commit**

```bash
chmod +x scripts/benchmark_gpu.sh
git add scripts/benchmark_gpu.sh
git commit -m "feat: add Nsight GPU performance benchmark script"
```

---

## Summary

| Task | Component | Key Deliverable | Speedup |
|---|---|---|---|
| 1 | GPU-resident Ising E/M | `measure_ising_gpu()` | Enables Task 2 |
| 2 | Wire into gpu_fss | Eliminate 26 GB transfer | ~12x |
| 3 | GPU-resident continuous E | `reduce_energy_continuous` kernel | ~5x for XY/Heisenberg |
| 4 | Pre-allocated buffers | `ReductionBuffers` struct | ~1.2x (allocation overhead) |
| 5 | Blocking analysis | `BlockingAnalyzer` | Correct error bars |
| 6 | CPU/GPU parity | `tests/gpu_parity.rs` | Correctness gate |
| 7 | Mixed-precision f64 | f64 reduction kernels | Accuracy at N≥64 |
| 8 | Batched replica sweeps | Single-launch kernel | ~3x (launch overhead) |
| 9 | Integrate blocking | Error bars + τ_int in CSV | Statistical reliability |
| 10 | Autotuning | Adaptive delta | ~1.5x (optimal acceptance) |
| 11 | Multi-spin Ising | 32 replicas per word | ~10-30x for Ising |
| 12 | Nsight benchmarks | Profiling scripts | Guides next phase |

**Execution order:** Tasks 1-3 are the critical path (eliminate transfers). Task 6 should gate Tasks 7-8 (correctness before optimisation). Tasks 5, 9, 10 are independent. Task 11 is the highest-effort/highest-reward item. Task 12 is lowest priority.

**Combined estimated speedup over Phase 1:** 30-100x for Ising (with MSC), 10-20x for continuous spins. N=128 Ising becomes feasible in minutes instead of hours.
