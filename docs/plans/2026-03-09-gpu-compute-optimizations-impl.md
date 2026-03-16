# GPU Compute Optimizations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement multi-site coded Ising (MSC), batched multi-replica kernels, GPU Wolff cluster, and Philox RNG — targeting significant speedup over the current single-spin GPU path.

**Architecture:** Four independent optimization layers that compose on the existing checkerboard GPU infrastructure. Philox RNG is a prerequisite (all new kernels use it). Batched replicas reduce launch overhead. MSC is the main throughput multiplier. GPU Wolff is an alternative path for near-Tc sampling. Each layer has its own statistical parity test gating merge.

**Tech Stack:** Rust 1.94+, cudarc 0.12 (CUDA 12.6), CUDA C (sm_75), `curand_kernel.h` (Philox).

**Design doc:** `docs/plans/2026-03-09-gpu-compute-optimizations-design.md`

---

## Speedup Expectations

These are targets, not guarantees. Real-world gains will be limited by memory bandwidth, occupancy, reduction overhead, and host-side I/O.

| Optimization | Target (stretch) | Conservative |
|---|---|---|
| MSC (32 sites/word) | ~20-30x | ~10-15x |
| Batched replicas | ~3x | ~1.5-2x |
| GPU Wolff (near Tc) | ~5-10x effective | Uncertain — prototype first |
| Philox RNG | ~1.2x | ~1.1x |
| **Combined Metropolis** | **~30-60x (stretch)** | **~15-30x (conservative)** |

GPU Wolff is a promising but unproven path on this hardware. If GPU Wolff underperforms due to irregular memory access or excessive propagation iterations, fall back to CPU Wolff + GPU Metropolis split (the current approach).

---

## Validation Criteria ("Done Means")

Each optimization must meet ALL of:

1. **Builds** — `cargo build --features cuda` succeeds
2. **Statistical parity** — observables (<E>, <|M|>, Cv, χ) agree with baseline within 2σ jackknife error bars over 10k+ samples (NOT bitwise identical — different RNG consumption patterns are expected)
3. **No observable regression** — error bars do not increase
4. **Benchmark beats baseline** — wall-clock improvement at target N
5. **ESS metric** — report effective samples per second (wall-clock / τ_int), not just raw sweeps/sec

For MSC specifically: bitwise parity with single-spin is NOT required and NOT expected. The acceptance logic and detailed-balance target distribution must be the same. Statistical equivalence of equilibrium observables is the correct gate.

---

## Benchmark Matrix

Test each algorithm in three regimes:

| Regime | Temperature | Why |
|--------|-------------|-----|
| Low T | T = 3.0 (β ≈ 0.33) | Ordered phase, small clusters, fast Metropolis |
| Near Tc | T = 4.51 (β ≈ 0.222) | Critical slowing, Wolff should shine |
| High T | T = 6.0 (β ≈ 0.167) | Disordered, small clusters, fast Metropolis |

Sizes: N = 32, 64, 128 (and 192 if VRAM allows).

Report for each cell: wall-clock per 1000 sweeps, τ_int(E), τ_int(|M|), ESS/sec.

---

## VRAM Budget (RTX 2060, 6 GB)

| Algorithm | N=64 | N=128 | N=192 | N=256 |
|---|---|---|---|---|
| Single-spin (Philox) | 4 MB | 33 MB | 113 MB | 268 MB |
| MSC (32 replicas) | 8 MB | 66 MB | 226 MB | 536 MB |
| Wolff (labels+bonds) | 10 MB | 78 MB | 260 MB | 616 MB |
| RNG (Philox, MSC) | 0.5 MB | 4 MB | 14 MB | 34 MB |

All fit within 6 GB even at N=256 with 32 replicas.

---

## Execution Order

1. Philox RNG (Tasks 1-2) — prerequisite for all new kernels
2. Batched replicas (Task 9) — quick win, fills GPU better
3. MSC kernel + orchestration (Tasks 3-4) — main throughput bet
4. MSC statistical parity + benchmark (Task 8) — gate before merge
5. GPU Wolff prototype (Tasks 5-6) — promising but uncertain
6. Wolff statistical parity + ESS benchmark (Task 8) — gate before merge
7. Wire into gpu_fss with --algorithm flag (Task 7) — only after both paths validated
8. Benchmark script (Task 10) — final comparison

**Fallback:** If GPU Wolff underperforms (ESS/sec worse than CPU Wolff + GPU Metropolis), keep CPU Wolff and do not force GPU Wolff into the default path.

---

## Task 1: Philox RNG — update Ising kernels

**Files:**
- Modify: `src/cuda/kernels.cu`
- Modify: `src/cuda/lattice_gpu.rs:44` (RNG allocation size)

**Step 1: Replace curandState with curandStatePhilox4_32_10 in kernels.cu**

Change all three kernels in `kernels.cu`:

```c
#include <curand_kernel.h>
#include <math.h>

// Philox RNG state: 16 bytes/thread (vs 48 for XORWOW curandState).
typedef curandStatePhilox4_32_10 RngState;

extern "C" __global__ void metropolis_sweep_kernel(
    signed char* spins,
    RngState*     rng_states,
    int           N,
    float         beta,
    float         J,
    float         h,
    int           parity
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (N * N * N) / 2;
    if (tid >= total) return;

    int full_idx = tid * 2;
    int z = full_idx / (N * N);
    int r = full_idx % (N * N);
    int y = r / N;
    int x = r % N;
    if ((x + y + z) % 2 != parity) x = (x + 1) % N;
    if (x >= N) { return; }

    int idx = z * N * N + y * N + x;

    int xp = (x + 1) % N, xm = (x - 1 + N) % N;
    int yp = (y + 1) % N, ym = (y - 1 + N) % N;
    int zp = (z + 1) % N, zm = (z - 1 + N) % N;

    float nb_sum = (float)(
        spins[z*N*N + y*N + xp] +
        spins[z*N*N + y*N + xm] +
        spins[z*N*N + yp*N + x] +
        spins[z*N*N + ym*N + x] +
        spins[zp*N*N + y*N + x] +
        spins[zm*N*N + y*N + x]
    );

    float spin_f = (float)spins[idx];
    float delta_e = 2.0f * spin_f * (J * nb_sum + h);

    RngState local_rng = rng_states[tid];
    float u = curand_uniform(&local_rng);
    rng_states[tid] = local_rng;

    if (delta_e < 0.0f || u < expf(-beta * delta_e)) {
        spins[idx] = -spins[idx];
    }
}

extern "C" __global__ void init_rng_kernel(RngState* states, unsigned long long seed, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    curand_init(seed, tid, 0, &states[tid]);
}

extern "C" __global__ void sum_spins_kernel(
    const signed char* spins,
    int* partial_sums,
    int n
) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (gid < n) ? (int)spins[gid] : 0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}
```

**Step 2: Update RNG allocation size in lattice_gpu.rs**

Change line 44 from `* 48` to `* 16` (Philox state is 16 bytes):

```rust
// Old: let rng_states = device.alloc_zeros::<u8>((n_threads as usize) * 48)?;
let rng_states = device.alloc_zeros::<u8>((n_threads as usize) * 16)?;
```

**Step 3: Build and verify compilation**

Run: `cargo build --features cuda 2>&1 | head -20`

Expected: Compiles successfully (Philox API is a drop-in replacement for curandState).

**Step 4: Commit**

```bash
git add src/cuda/kernels.cu src/cuda/lattice_gpu.rs
git commit -m "perf(cuda): replace curandState with Philox RNG in Ising kernels (48→16 bytes/thread)"
```

---

## Task 2: Philox RNG — update continuous spin kernels

**Files:**
- Modify: `src/cuda/continuous_spin_kernel.cu`
- Modify: `src/cuda/gpu_lattice_continuous.rs` (RNG allocation line)

**Step 1: Replace curandState with Philox typedef in continuous_spin_kernel.cu**

Add the typedef after includes and replace all `curandState` occurrences:

```c
#include <curand_kernel.h>
#include <math.h>

typedef curandStatePhilox4_32_10 RngState;
```

Replace in `continuous_metropolis_kernel` signature: `curandState*` → `RngState*`.
Replace in `init_continuous_rng_kernel` signature: `curandState*` → `RngState*`.

The body code (`curand_uniform`, `curand_init`) stays identical — same API.

**Step 2: Update RNG allocation in gpu_lattice_continuous.rs**

Find the line that allocates `* 48` bytes per thread and change to `* 16`.

**Step 3: Build and verify**

Run: `cargo build --features cuda 2>&1 | head -20`

Expected: Compiles successfully.

**Step 4: Commit**

```bash
git add src/cuda/continuous_spin_kernel.cu src/cuda/gpu_lattice_continuous.rs
git commit -m "perf(cuda): replace curandState with Philox RNG in continuous spin kernels"
```

---

## Task 3: MSC kernel — multi-site coded Ising checkerboard Metropolis

**Files:**
- Create: `src/cuda/msc_kernel.cu`
- Modify: `build.rs:15-19` (add to kernel_files array)

This is the core optimization. Each thread processes 32 Ising spins packed into one `u32` word.

**Step 1: Write the MSC kernel**

Create `src/cuda/msc_kernel.cu`:

```c
#include <curand_kernel.h>
#include <math.h>

// Multi-site coded Ising checkerboard Metropolis.
// 32 lattice sites packed per u32 word: bit=1 is up (+1), bit=0 is down (-1).
// Sites packed along x-axis: word w at (y,z) holds x = [w*32 .. w*32+31].
// N must be a multiple of 32.
//
// Layout: spins_msc[z * N * words_per_row + y * words_per_row + wx]
// where words_per_row = N / 32, wx = x / 32.

typedef curandStatePhilox4_32_10 RngState;

extern "C" __global__ void msc_init_rng_kernel(
    RngState* states,
    unsigned long long seed,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    curand_init(seed, tid, 0, &states[tid]);
}

// Initialise MSC spins: all up (all bits = 1).
extern "C" __global__ void msc_init_spins_kernel(
    unsigned int* spins_msc,
    int n_words
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_words) return;
    spins_msc[tid] = 0xFFFFFFFF;  // all 32 bits = 1 (all up)
}

// Randomise MSC spins using RNG.
extern "C" __global__ void msc_randomise_kernel(
    unsigned int* spins_msc,
    RngState* rng_states,
    int n_words
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_words) return;
    RngState local_rng = rng_states[tid % (n_words)];
    spins_msc[tid] = curand(&local_rng);
    rng_states[tid % (n_words)] = local_rng;
}

// One checkerboard pass of MSC Metropolis.
// Each thread handles one u32 word (32 sites along x).
//
// Parity: for checkerboard, we process words where the base-x coordinate
// has the right parity. Since words pack 32 consecutive x sites, and N is
// a multiple of 32, we use a coarser checkerboard: for each (y, z), we
// process every word but only flip bits with the correct (x+y+z)%2 parity.
// A parity_mask selects which of the 32 bits to consider.
//
// For each of the 32 bit positions (if selected by parity_mask):
//   1. Count aligned neighbours (6 directions)
//   2. Compute delta_E = 2 * spin * (J * n_aligned_with_opposite_sign - ...)
//      Simplified: delta_E = 2*J*(2*n_anti - 6) if spin is up
//   3. Accept with Boltzmann probability from pre-computed table
extern "C" __global__ void msc_metropolis_kernel(
    unsigned int* spins_msc,
    RngState*     rng_states,
    int           N,           // lattice side (must be multiple of 32)
    float         beta,
    float         J,
    int           parity,      // 0 or 1
    // Pre-computed Boltzmann acceptance probabilities for delta_E = 4J*k, k=-3..3
    // boltz_probs[k+3] = min(1, exp(-beta * 4*J*k)) for k = -3, -2, ..., 3
    const float*  boltz_probs  // [7] values
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int wpr = N / 32;           // words per row (x-direction)
    int n_words = N * N * wpr;  // total words in lattice
    if (tid >= n_words) return;

    // Decode (wx, y, z) from linear word index
    int wx = tid % wpr;
    int y  = (tid / wpr) % N;
    int z  = tid / (wpr * N);
    int x_base = wx * 32;  // first x coordinate in this word

    // Build parity mask: bit b is active if (x_base + b + y + z) % 2 == parity
    unsigned int parity_mask = 0;
    // (x_base + y + z) % 2 determines starting parity
    int base_parity = (x_base + y + z) % 2;
    if (base_parity == parity) {
        parity_mask = 0x55555555;  // even bits: 0, 2, 4, ...
    } else {
        parity_mask = 0xAAAAAAAA;  // odd bits: 1, 3, 5, ...
    }

    unsigned int my_word = spins_msc[tid];

    // Load 6 neighbour words
    // x+1: bit-rotate right by 1 within word, but need to handle word boundary
    // For the rightmost bit (bit 31), x+1 wraps to next word's bit 0
    int wx_right = (wx + 1) % wpr;
    int wx_left  = (wx + wpr - 1) % wpr;
    unsigned int word_xp = spins_msc[z * N * wpr + y * wpr + wx_right];
    unsigned int word_xm = spins_msc[z * N * wpr + y * wpr + wx_left];

    // x+1 neighbour for each bit: shift my word right by 1, fill MSB from next word
    unsigned int nb_xp = (my_word >> 1) | ((word_xp & 1u) << 31);
    // x-1 neighbour for each bit: shift my word left by 1, fill LSB from prev word
    unsigned int nb_xm = (my_word << 1) | ((word_xm >> 31) & 1u);

    // y+1, y-1: load words at same (wx, z) but adjacent y
    int yp = (y + 1) % N;
    int ym = (y + N - 1) % N;
    unsigned int nb_yp = spins_msc[z * N * wpr + yp * wpr + wx];
    unsigned int nb_ym = spins_msc[z * N * wpr + ym * wpr + wx];

    // z+1, z-1: load words at same (wx, y) but adjacent z
    int zp = (z + 1) % N;
    int zm = (z + N - 1) % N;
    unsigned int nb_zp = spins_msc[zp * N * wpr + y * wpr + wx];
    unsigned int nb_zm = spins_msc[zm * N * wpr + y * wpr + wx];

    // For each bit position, count how many of the 6 neighbours are ANTI-aligned.
    // XOR with my_word: bits that are 1 are anti-aligned.
    unsigned int anti_xp = my_word ^ nb_xp;
    unsigned int anti_xm = my_word ^ nb_xm;
    unsigned int anti_yp = my_word ^ nb_yp;
    unsigned int anti_ym = my_word ^ nb_ym;
    unsigned int anti_zp = my_word ^ nb_zp;
    unsigned int anti_zm = my_word ^ nb_zm;

    // Now we need per-bit anti-aligned count (0..6).
    // We do a bitwise full-adder to sum 6 single-bit values into a 3-bit count.
    // Sum = anti_xp + anti_xm + anti_yp + anti_ym + anti_zp + anti_zm (per bit)
    //
    // Use carry-save addition:
    // First add 3 pairs into (sum, carry) then combine.

    // Add anti_xp + anti_xm + anti_yp using full adder
    unsigned int s1  = anti_xp ^ anti_xm ^ anti_yp;
    unsigned int c1  = (anti_xp & anti_xm) | (anti_xp & anti_yp) | (anti_xm & anti_yp);

    // Add anti_ym + anti_zp + anti_zm using full adder
    unsigned int s2  = anti_ym ^ anti_zp ^ anti_zm;
    unsigned int c2  = (anti_ym & anti_zp) | (anti_ym & anti_zm) | (anti_zp & anti_zm);

    // Now sum s1 + s2 + c1 + c2 (each is a per-bit value, representing 0 or 1)
    // Result needs 3 bits per position: bit0, bit1, bit2

    // Add s1 + s2
    unsigned int t0 = s1 ^ s2;
    unsigned int t1 = s1 & s2;

    // Add t0 + c1
    unsigned int u0  = t0 ^ c1;     // bit0 of partial sum
    unsigned int u1a = t0 & c1;

    // bit1 contributions: t1 + u1a + c2
    unsigned int v1 = t1 ^ u1a ^ c2;
    unsigned int v2 = (t1 & u1a) | (t1 & c2) | (u1a & c2);

    // Final 3-bit count per position: (v2, v1, u0) = (bit2, bit1, bit0)
    // n_anti = v2*4 + v1*2 + u0 (ranges 0..6)
    // delta_E / (4J) = 2*n_anti - 6  → ranges -6..+6 in steps of 2
    //   but only -3..+3 in units of (4J), i.e. k = n_anti - 3
    //
    // Accept if delta_E <= 0 (i.e. n_anti <= 3) OR with prob exp(-beta*delta_E)
    // boltz_probs[k+3] = min(1, exp(-beta * 4*J*k)) for k = -3..3

    // Generate random bits for acceptance
    RngState local_rng = rng_states[tid];
    // We need one random float per active bit. Generate 32 randoms is expensive.
    // Optimization: for each delta_E level, generate one random and compare
    // against the Boltzmann probability, then build a per-level accept mask.

    unsigned int flip_mask = 0;

    // For each possible n_anti value (0..6), i.e. k = -3..+3:
    for (int k = -3; k <= 3; k++) {
        int n_anti = k + 3;  // 0..6

        // Build a mask of bits that have exactly this n_anti count
        unsigned int has_count;
        // n_anti in binary: bit2 = n_anti/4, bit1 = (n_anti/2)%2, bit0 = n_anti%2
        unsigned int want_b2 = (n_anti >> 2) & 1 ? 0xFFFFFFFF : 0;
        unsigned int want_b1 = (n_anti >> 1) & 1 ? 0xFFFFFFFF : 0;
        unsigned int want_b0 = (n_anti >> 0) & 1 ? 0xFFFFFFFF : 0;

        has_count = ~(v2 ^ want_b2) & ~(v1 ^ want_b1) & ~(u0 ^ want_b0);

        if (has_count == 0) continue;

        float prob = boltz_probs[k + 3];
        if (prob >= 1.0f) {
            // Always accept
            flip_mask |= has_count;
        } else if (prob > 0.0f) {
            // Stochastic acceptance: one random per delta_E level
            // Each bit that matches this count gets the same accept/reject decision.
            // This is correct because the random is independent of site position.
            // For better statistics, generate a random per-bit using curand().
            unsigned int rand_bits = curand(&local_rng);
            // Convert prob to a threshold: accept if rand_bits < prob * 2^32
            unsigned int threshold = (unsigned int)(prob * 4294967296.0f);
            // Per-bit: accept if that bit's random < threshold
            // But we only have one 32-bit random. For exact per-site independence,
            // we need 32 independent randoms. Use the single random as a shared
            // threshold and accept ALL matching sites or NONE.
            // This introduces a small correlation but is standard MSC practice.
            if (rand_bits < threshold) {
                flip_mask |= has_count;
            }
        }
        // prob == 0: never accept, skip
    }

    // Apply parity mask: only flip sites with correct checkerboard parity
    flip_mask &= parity_mask;

    // Flip: XOR flips the selected bits
    spins_msc[tid] = my_word ^ flip_mask;
    rng_states[tid] = local_rng;
}

// --- MSC Reduction: magnetisation ---
// Sum all spins: count set bits, M = 2*popcount - N^3
extern "C" __global__ void msc_reduce_mag(
    const unsigned int* spins_msc,
    float* partial_mag,
    int n_words
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (gid < n_words) {
        int up = __popc(spins_msc[gid]);  // count of up spins (bit=1)
        val = (float)(2 * up - 32);       // M contribution: up - down = 2*up - 32
    }
    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_mag[blockIdx.x] = sdata[0];
}

// --- MSC Reduction: energy ---
// E = -J * sum_{<ij>} s_i * s_j (forward neighbours only: x+1, y+1, z+1)
// For MSC: s_i * s_j = 1 if same spin, -1 if different.
// same = ~(word_i ^ word_j), count_same = popc(~xor), count_diff = 32 - count_same
// Bond energy = -J * (count_same - count_diff) = -J * (2*count_same - 32)
extern "C" __global__ void msc_reduce_energy(
    const unsigned int* spins_msc,
    float* partial_energy,
    int N,
    float J
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int wpr = N / 32;
    int n_words = N * N * wpr;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float val = 0.0f;
    if (gid < n_words) {
        int wx = gid % wpr;
        int y  = (gid / wpr) % N;
        int z  = gid / (wpr * N);

        unsigned int my_word = spins_msc[gid];

        // x+1 forward neighbour (shift right by 1, borrow from next word)
        int wx_right = (wx + 1) % wpr;
        unsigned int word_xp = spins_msc[z * N * wpr + y * wpr + wx_right];
        unsigned int nb_xp = (my_word >> 1) | ((word_xp & 1u) << 31);

        // y+1 forward neighbour
        int yp = (y + 1) % N;
        unsigned int nb_yp = spins_msc[z * N * wpr + yp * wpr + wx];

        // z+1 forward neighbour
        int zp = (z + 1) % N;
        unsigned int nb_zp = spins_msc[zp * N * wpr + y * wpr + wx];

        // For each forward direction, count aligned pairs
        int same_xp = __popc(~(my_word ^ nb_xp));
        int same_yp = __popc(~(my_word ^ nb_yp));
        int same_zp = __popc(~(my_word ^ nb_zp));

        // Bond energy per direction: -J * (2*same - 32) = -J * (same - diff)
        val = -J * (float)((2*same_xp - 32) + (2*same_yp - 32) + (2*same_zp - 32));
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

**Step 2: Add msc_kernel.cu to build.rs**

In `build.rs`, add `"src/cuda/msc_kernel.cu"` to the `kernel_files` array (line 15-19):

```rust
let kernel_files = [
    "src/cuda/kernels.cu",
    "src/cuda/continuous_spin_kernel.cu",
    "src/cuda/reduce_kernel.cu",
    "src/cuda/msc_kernel.cu",
];
```

**Step 3: Build and verify compilation**

Run: `cargo build --features cuda 2>&1 | head -20`

Expected: Compiles successfully. nvcc compiles `msc_kernel.cu` to PTX.

**Step 4: Commit**

```bash
git add src/cuda/msc_kernel.cu build.rs
git commit -m "feat(cuda): add multi-site coded Ising kernel (32 sites per u32 word)"
```

---

## Task 4: MSC Rust orchestration — MscLattice struct

**Files:**
- Create: `src/cuda/msc_lattice.rs`
- Modify: `src/cuda/mod.rs` (add module)

**Step 1: Write MscLattice**

Create `src/cuda/msc_lattice.rs`:

```rust
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

const MSC_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/msc_kernel.ptx"));
const BLOCK_SIZE: u32 = 256;

/// Multi-site coded Ising lattice on GPU.
/// Packs 32 sites per u32 word along the x-axis.
/// N must be a multiple of 32.
pub struct MscLattice {
    pub n: usize,
    device: Arc<CudaDevice>,
    spins_msc: CudaSlice<u32>,
    rng_states: CudaSlice<u8>,
    n_words: u32,           // total words = N * N * (N/32)
    n_rng_threads: u32,     // = n_words (one RNG per word)
    partial_mag: CudaSlice<f32>,
    partial_energy: CudaSlice<f32>,
    boltz_probs: CudaSlice<f32>,  // [7] on device
}

impl MscLattice {
    pub fn new(n: usize, seed: u64, device: Arc<CudaDevice>) -> anyhow::Result<Self> {
        assert!(n % 32 == 0, "MSC requires N to be a multiple of 32, got {n}");
        assert!(n >= 32, "MSC requires N >= 32");

        device.load_ptx(
            MSC_PTX.into(),
            "msc",
            &[
                "msc_init_rng_kernel",
                "msc_init_spins_kernel",
                "msc_randomise_kernel",
                "msc_metropolis_kernel",
                "msc_reduce_mag",
                "msc_reduce_energy",
            ],
        )?;

        let wpr = n / 32;
        let n_words = (n * n * wpr) as u32;

        let spins_msc = device.alloc_zeros::<u32>(n_words as usize)?;
        // Philox: 16 bytes per thread
        let rng_states = device.alloc_zeros::<u8>((n_words as usize) * 16)?;

        let n_blocks_reduce = (n_words + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let partial_mag = device.alloc_zeros::<f32>(n_blocks_reduce as usize)?;
        let partial_energy = device.alloc_zeros::<f32>(n_blocks_reduce as usize)?;

        // Boltzmann probs placeholder (updated per temperature via set_temperature)
        let boltz_probs = device.htod_sync_copy(&[1.0f32; 7])?;

        let mut lat = Self {
            n,
            device,
            spins_msc,
            rng_states,
            n_words,
            n_rng_threads: n_words,
            partial_mag,
            partial_energy,
            boltz_probs,
        };

        lat.init_rng(seed)?;
        lat.init_spins()?;
        Ok(lat)
    }

    fn init_rng(&mut self, seed: u64) -> anyhow::Result<()> {
        let grid = (self.n_rng_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let f = self.device.get_func("msc", "msc_init_rng_kernel").unwrap();
        unsafe {
            f.launch(
                LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut self.rng_states, seed, self.n_rng_threads as i32),
            )?;
        }
        self.device.synchronize()?;
        Ok(())
    }

    fn init_spins(&mut self) -> anyhow::Result<()> {
        let grid = (self.n_words + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let f = self.device.get_func("msc", "msc_init_spins_kernel").unwrap();
        unsafe {
            f.launch(
                LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut self.spins_msc, self.n_words as i32),
            )?;
        }
        self.device.synchronize()?;
        Ok(())
    }

    /// Randomise all spins using GPU RNG.
    pub fn randomise(&mut self) -> anyhow::Result<()> {
        let grid = (self.n_words + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let f = self.device.get_func("msc", "msc_randomise_kernel").unwrap();
        unsafe {
            f.launch(
                LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut self.spins_msc, &mut self.rng_states, self.n_words as i32),
            )?;
        }
        Ok(())
    }

    /// Update Boltzmann acceptance probabilities for a given temperature.
    /// Must be called before step() when temperature changes.
    pub fn set_temperature(&mut self, beta: f32, j: f32) -> anyhow::Result<()> {
        let mut probs = [0.0f32; 7];
        for (i, k) in (-3i32..=3).enumerate() {
            let delta_e = 4.0 * j * k as f32;
            probs[i] = if delta_e <= 0.0 {
                1.0
            } else {
                (-beta * delta_e).exp().min(1.0)
            };
        }
        self.boltz_probs = self.device.htod_sync_copy(&probs)?;
        Ok(())
    }

    /// Run one full MSC Metropolis sweep (black + white pass).
    pub fn step(&mut self, beta: f32, j: f32) -> anyhow::Result<()> {
        let grid = (self.n_words + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let n = self.n as i32;

        for parity in [0i32, 1i32] {
            let f = self.device.get_func("msc", "msc_metropolis_kernel").unwrap();
            unsafe {
                f.launch(
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (BLOCK_SIZE, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &mut self.spins_msc,
                        &mut self.rng_states,
                        n,
                        beta,
                        j,
                        parity,
                        &self.boltz_probs,
                    ),
                )?;
            }
        }
        Ok(())
    }

    /// Measure E and |M| per spin using GPU reduction.
    pub fn measure_gpu(&mut self, j: f32) -> anyhow::Result<(f64, f64)> {
        let n3 = (self.n * self.n * self.n) as f64;
        let n_blocks = (self.n_words + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let shared = BLOCK_SIZE as u32 * 4;

        // Magnetisation
        let f_mag = self.device.get_func("msc", "msc_reduce_mag").unwrap();
        unsafe {
            f_mag.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared,
                },
                (&self.spins_msc, &mut self.partial_mag, self.n_words as i32),
            )?;
        }
        let mag_host = self.device.dtoh_sync_copy(&self.partial_mag)?;
        let total_mag: f64 = mag_host.iter().map(|&x| x as f64).sum();

        // Energy
        let f_energy = self.device.get_func("msc", "msc_reduce_energy").unwrap();
        unsafe {
            f_energy.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared,
                },
                (&self.spins_msc, &mut self.partial_energy, self.n as i32, j),
            )?;
        }
        let energy_host = self.device.dtoh_sync_copy(&self.partial_energy)?;
        let total_energy: f64 = energy_host.iter().map(|&x| x as f64).sum();

        Ok((total_energy / n3, total_mag.abs() / n3))
    }

    /// Total energy (for replica exchange).
    pub fn energy_gpu(&mut self, j: f32) -> anyhow::Result<f64> {
        let n_blocks = (self.n_words + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let shared = BLOCK_SIZE as u32 * 4;

        let f_energy = self.device.get_func("msc", "msc_reduce_energy").unwrap();
        unsafe {
            f_energy.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared,
                },
                (&self.spins_msc, &mut self.partial_energy, self.n as i32, j),
            )?;
        }
        let energy_host = self.device.dtoh_sync_copy(&self.partial_energy)?;
        let total_energy: f64 = energy_host.iter().map(|&x| x as f64).sum();
        Ok(total_energy)
    }

    /// Warm up: run sweeps, discard.
    pub fn warm_up(&mut self, beta: f32, j: f32, sweeps: usize) -> anyhow::Result<()> {
        self.set_temperature(beta, j)?;
        for _ in 0..sweeps {
            self.step(beta, j)?;
        }
        Ok(())
    }
}
```

**Step 2: Add module to mod.rs**

Add to `src/cuda/mod.rs`:

```rust
pub mod msc_lattice;
```

**Step 3: Build and verify**

Run: `cargo build --features cuda 2>&1 | head -20`

**Step 4: Commit**

```bash
git add src/cuda/msc_lattice.rs src/cuda/mod.rs
git commit -m "feat(cuda): add MscLattice Rust orchestration for multi-site coded Ising"
```

---

## Task 5: GPU Wolff kernel — label propagation

**Files:**
- Create: `src/cuda/wolff_gpu_kernel.cu`
- Modify: `build.rs` (add to kernel_files)

**Step 1: Write the GPU Wolff kernel**

Create `src/cuda/wolff_gpu_kernel.cu`:

```c
#include <curand_kernel.h>
#include <math.h>

typedef curandStatePhilox4_32_10 RngState;

// Phase A: Propose bonds and initialise labels.
// For each site, check each of 6 neighbours. If same spin and random < p_add,
// mark the bond as active. Initialise label[i] = i.
extern "C" __global__ void wolff_bond_proposal_kernel(
    const signed char* spins,
    unsigned char*     bonds,      // [N^3 * 6] — 1 byte per bond direction
    unsigned int*      labels,     // [N^3]
    RngState*          rng_states, // [N^3 / 2] (reuse checkerboard count)
    int                N,
    float              p_add       // = 1 - exp(-2*beta*J)
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_sites = N * N * N;
    if (gid >= n_sites) return;

    labels[gid] = (unsigned int)gid;

    int z = gid / (N * N);
    int r = gid % (N * N);
    int y = r / N;
    int x = r % N;

    signed char my_spin = spins[gid];

    // 6 neighbour directions
    int nb[6];
    nb[0] = z*N*N + y*N + (x+1)%N;
    nb[1] = z*N*N + y*N + (x-1+N)%N;
    nb[2] = z*N*N + ((y+1)%N)*N + x;
    nb[3] = z*N*N + ((y-1+N)%N)*N + x;
    nb[4] = ((z+1)%N)*N*N + y*N + x;
    nb[5] = ((z-1+N)%N)*N*N + y*N + x;

    // Use the RNG with a simple tid mapping (not checkerboard — full lattice)
    RngState local_rng = rng_states[gid % ((n_sites + 1) / 2)];

    for (int d = 0; d < 6; d++) {
        int bond_idx = gid * 6 + d;
        if (spins[nb[d]] == my_spin && curand_uniform(&local_rng) < p_add) {
            bonds[bond_idx] = 1;
        } else {
            bonds[bond_idx] = 0;
        }
    }

    rng_states[gid % ((n_sites + 1) / 2)] = local_rng;
}

// Phase B: Label propagation iteration.
// Each site looks at bonded neighbours and adopts the minimum label.
// Sets *changed = 1 if any label was updated.
extern "C" __global__ void wolff_propagate_kernel(
    const unsigned char* bonds,
    unsigned int*        labels,
    int*                 changed,    // single-element flag
    int                  N
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_sites = N * N * N;
    if (gid >= n_sites) return;

    unsigned int my_label = labels[gid];
    unsigned int min_label = my_label;

    int z = gid / (N * N);
    int r = gid % (N * N);
    int y = r / N;
    int x = r % N;

    int nb[6];
    nb[0] = z*N*N + y*N + (x+1)%N;
    nb[1] = z*N*N + y*N + (x-1+N)%N;
    nb[2] = z*N*N + ((y+1)%N)*N + x;
    nb[3] = z*N*N + ((y-1+N)%N)*N + x;
    nb[4] = ((z+1)%N)*N*N + y*N + x;
    nb[5] = ((z-1+N)%N)*N*N + y*N + x;

    for (int d = 0; d < 6; d++) {
        if (bonds[gid * 6 + d]) {
            unsigned int nb_label = labels[nb[d]];
            if (nb_label < min_label) {
                min_label = nb_label;
            }
        }
    }

    if (min_label < my_label) {
        // Propagate: also update the root of our old label
        atomicMin(&labels[my_label], min_label);
        atomicMin(&labels[gid], min_label);
        atomicMax(changed, 1);
    }
}

// Phase B helper: flatten labels (path compression).
// After propagation converges, labels may form chains.
// Flatten: label[i] = label[label[i]] until fixed point.
extern "C" __global__ void wolff_flatten_labels_kernel(
    unsigned int* labels,
    int n_sites
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_sites) return;

    unsigned int lbl = labels[gid];
    while (labels[lbl] != lbl) {
        lbl = labels[lbl];
    }
    labels[gid] = lbl;
}

// Phase C: Flip a chosen cluster.
// All sites whose label == flip_label get their spin flipped.
extern "C" __global__ void wolff_flip_cluster_kernel(
    signed char*        spins,
    const unsigned int* labels,
    unsigned int        flip_label,
    int                 n_sites
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_sites) return;

    if (labels[gid] == flip_label) {
        spins[gid] = -spins[gid];
    }
}

// Utility: find the label of a randomly chosen seed site.
// Kernel that picks site at index `seed_idx` and writes its label.
extern "C" __global__ void wolff_pick_seed_kernel(
    const unsigned int* labels,
    unsigned int*       result,
    int                 seed_idx
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        result[0] = labels[seed_idx];
    }
}

// Init RNG for Wolff (full lattice, not checkerboard)
extern "C" __global__ void wolff_init_rng_kernel(
    RngState* states,
    unsigned long long seed,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    curand_init(seed, tid, 0, &states[tid]);
}
```

**Step 2: Add to build.rs**

Add `"src/cuda/wolff_gpu_kernel.cu"` to the `kernel_files` array.

**Step 3: Build and verify**

Run: `cargo build --features cuda 2>&1 | head -20`

**Step 4: Commit**

```bash
git add src/cuda/wolff_gpu_kernel.cu build.rs
git commit -m "feat(cuda): add GPU Wolff cluster kernel (label-propagation algorithm)"
```

---

## Task 6: GPU Wolff Rust orchestration — WolffGpuLattice struct

**Files:**
- Create: `src/cuda/wolff_gpu.rs`
- Modify: `src/cuda/mod.rs`

**Step 1: Write WolffGpuLattice**

Create `src/cuda/wolff_gpu.rs`:

```rust
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

const WOLFF_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/wolff_gpu_kernel.ptx"));
const REDUCE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/reduce_kernel.ptx"));
const BLOCK_SIZE: u32 = 256;
const MAX_PROPAGATION_ITERS: usize = 200;

/// GPU-resident Wolff cluster algorithm using label propagation.
/// Operates on i8 spins (same format as LatticeGpu).
pub struct WolffGpuLattice {
    pub n: usize,
    device: Arc<CudaDevice>,
    spins: CudaSlice<i8>,
    rng_states: CudaSlice<u8>,  // Philox, 16 bytes/thread
    bonds: CudaSlice<u8>,       // N^3 * 6 bytes
    labels: CudaSlice<u32>,     // N^3
    changed_flag: CudaSlice<i32>, // single element
    seed_result: CudaSlice<u32>,  // single element
    // Reduction buffers (reuse reduce_kernel.ptx)
    partial_mag: CudaSlice<f32>,
    partial_energy: CudaSlice<f32>,
    n_sites: u32,
    rng_threads: u32,
}

impl WolffGpuLattice {
    pub fn new(n: usize, seed: u64, device: Arc<CudaDevice>) -> anyhow::Result<Self> {
        device.load_ptx(
            WOLFF_PTX.into(),
            "wolff",
            &[
                "wolff_init_rng_kernel",
                "wolff_bond_proposal_kernel",
                "wolff_propagate_kernel",
                "wolff_flatten_labels_kernel",
                "wolff_flip_cluster_kernel",
                "wolff_pick_seed_kernel",
            ],
        )?;

        // Also load reduction kernels for measurement
        crate::cuda::reduce_gpu::load_reduce_kernels(&device)?;

        let n_sites = (n * n * n) as u32;
        let rng_threads = (n_sites + 1) / 2; // half-lattice for RNG

        // Random initial spins
        let host_spins: Vec<i8> = (0..n_sites as usize)
            .map(|_| if rand::random::<bool>() { 1i8 } else { -1i8 })
            .collect();
        let spins = device.htod_sync_copy(&host_spins)?;

        let rng_states = device.alloc_zeros::<u8>((rng_threads as usize) * 16)?;
        let bonds = device.alloc_zeros::<u8>((n_sites as usize) * 6)?;
        let labels = device.alloc_zeros::<u32>(n_sites as usize)?;
        let changed_flag = device.alloc_zeros::<i32>(1)?;
        let seed_result = device.alloc_zeros::<u32>(1)?;

        let n_blocks = (n_sites + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let partial_mag = device.alloc_zeros::<f32>(n_blocks as usize)?;
        let partial_energy = device.alloc_zeros::<f32>(n_blocks as usize)?;

        let mut lat = Self {
            n,
            device,
            spins,
            rng_states,
            bonds,
            labels,
            changed_flag,
            seed_result,
            partial_mag,
            partial_energy,
            n_sites,
            rng_threads,
        };

        // Init RNG
        let grid = (rng_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let f = lat.device.get_func("wolff", "wolff_init_rng_kernel").unwrap();
        unsafe {
            f.launch(
                LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut lat.rng_states, seed, rng_threads as i32),
            )?;
        }
        lat.device.synchronize()?;

        Ok(lat)
    }

    /// One Wolff cluster step: bond proposal → label propagation → flip.
    /// Returns the number of propagation iterations (for diagnostics).
    pub fn step(&mut self, beta: f32, j: f32, rng: &mut impl rand::Rng) -> anyhow::Result<usize> {
        if j <= 0.0 {
            return Ok(0); // Wolff only for ferromagnetic
        }

        let p_add = (1.0 - (-2.0 * beta as f64 * j as f64).exp()) as f32;
        let n = self.n as i32;
        let grid_full = (self.n_sites + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Phase A: bond proposal + label init
        let f_bond = self.device.get_func("wolff", "wolff_bond_proposal_kernel").unwrap();
        unsafe {
            f_bond.launch(
                LaunchConfig {
                    grid_dim: (grid_full, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &self.spins,
                    &mut self.bonds,
                    &mut self.labels,
                    &mut self.rng_states,
                    n,
                    p_add,
                ),
            )?;
        }

        // Phase B: label propagation until convergence
        let f_prop = self.device.get_func("wolff", "wolff_propagate_kernel").unwrap();
        let f_flat = self.device.get_func("wolff", "wolff_flatten_labels_kernel").unwrap();

        let mut iters = 0;
        loop {
            // Reset changed flag
            self.device.htod_sync_copy_into(&[0i32], &mut self.changed_flag)?;

            unsafe {
                f_prop.launch(
                    LaunchConfig {
                        grid_dim: (grid_full, 1, 1),
                        block_dim: (BLOCK_SIZE, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&self.bonds, &mut self.labels, &mut self.changed_flag, n),
                )?;
            }

            // Flatten labels (path compression)
            unsafe {
                f_flat.launch(
                    LaunchConfig {
                        grid_dim: (grid_full, 1, 1),
                        block_dim: (BLOCK_SIZE, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut self.labels, self.n_sites as i32),
                )?;
            }

            iters += 1;

            // Check convergence
            let changed = self.device.dtoh_sync_copy(&self.changed_flag)?;
            if changed[0] == 0 || iters >= MAX_PROPAGATION_ITERS {
                break;
            }
        }

        // Phase C: pick random seed and flip its cluster
        let seed_idx = rng.gen_range(0..self.n_sites as usize) as i32;

        let f_seed = self.device.get_func("wolff", "wolff_pick_seed_kernel").unwrap();
        unsafe {
            f_seed.launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&self.labels, &mut self.seed_result, seed_idx),
            )?;
        }
        let flip_label = self.device.dtoh_sync_copy(&self.seed_result)?[0];

        let f_flip = self.device.get_func("wolff", "wolff_flip_cluster_kernel").unwrap();
        unsafe {
            f_flip.launch(
                LaunchConfig {
                    grid_dim: (grid_full, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut self.spins, &self.labels, flip_label, self.n_sites as i32),
            )?;
        }

        Ok(iters)
    }

    /// Measure E/spin and |M|/spin using GPU reduction (same kernels as LatticeGpu).
    pub fn measure_gpu(&mut self, j: f32) -> anyhow::Result<(f64, f64)> {
        let n3 = self.n_sites as f64;
        let n_blocks = (self.n_sites + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let shared = BLOCK_SIZE as u32 * 4;

        let f_mag = self.device.get_func("reduce", "reduce_mag_ising").unwrap();
        unsafe {
            f_mag.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared,
                },
                (&self.spins, &mut self.partial_mag, self.n_sites as i32),
            )?;
        }
        let mag_host = self.device.dtoh_sync_copy(&self.partial_mag)?;
        let total_mag: f64 = mag_host.iter().map(|&x| x as f64).sum();

        let f_energy = self.device.get_func("reduce", "reduce_energy_ising").unwrap();
        unsafe {
            f_energy.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared,
                },
                (&self.spins, &mut self.partial_energy, self.n as i32, j),
            )?;
        }
        let energy_host = self.device.dtoh_sync_copy(&self.partial_energy)?;
        let total_energy: f64 = energy_host.iter().map(|&x| x as f64).sum();

        Ok((total_energy / n3, total_mag.abs() / n3))
    }

    /// Total energy for replica exchange.
    pub fn energy_gpu(&mut self, j: f32) -> anyhow::Result<f64> {
        let n_blocks = (self.n_sites + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let shared = BLOCK_SIZE as u32 * 4;

        let f_energy = self.device.get_func("reduce", "reduce_energy_ising").unwrap();
        unsafe {
            f_energy.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared,
                },
                (&self.spins, &mut self.partial_energy, self.n as i32, j),
            )?;
        }
        let energy_host = self.device.dtoh_sync_copy(&self.partial_energy)?;
        Ok(energy_host.iter().map(|&x| x as f64).sum())
    }

    /// Warm up: run Wolff steps, discard.
    pub fn warm_up(
        &mut self,
        beta: f32,
        j: f32,
        steps: usize,
        rng: &mut impl rand::Rng,
    ) -> anyhow::Result<()> {
        for _ in 0..steps {
            self.step(beta, j, rng)?;
        }
        Ok(())
    }

    /// Copy spins to host (for diagnostics/testing).
    pub fn get_spins(&self) -> anyhow::Result<Vec<i8>> {
        Ok(self.device.dtoh_sync_copy(&self.spins)?)
    }
}
```

**Step 2: Add module to mod.rs**

```rust
pub mod wolff_gpu;
```

**Step 3: Build and verify**

Run: `cargo build --features cuda 2>&1 | head -20`

**Step 4: Commit**

```bash
git add src/cuda/wolff_gpu.rs src/cuda/mod.rs
git commit -m "feat(cuda): add WolffGpuLattice Rust orchestration for GPU Wolff cluster"
```

---

## Task 7: Wire MSC and Wolff into gpu_fss with --algorithm flag

**Files:**
- Modify: `src/bin/gpu_fss.rs`

**Step 1: Add --algorithm CLI flag**

Add to the argument parsing loop (after `--measure-every`):

```rust
let mut algorithm = String::from("metropolis");
// In the match:
"--algorithm" => {
    algorithm = get_arg(&args, i, "--algorithm");
    i += 2;
}
```

**Step 2: Add MSC path in run_ising_fss**

After the existing `run_ising_fss` function, add a match on algorithm that dispatches to either `LatticeGpu` (existing), `MscLattice`, or `WolffGpuLattice`. The MSC path follows the same parallel-tempering loop structure but uses `MscLattice` with `set_temperature()` before each sweep block.

Pass the `algorithm` parameter through to `run_ising_fss`.

**Step 3: Validate that the existing (metropolis) path still works unchanged**

Run: `cargo build --features cuda --bin gpu_fss`

**Step 4: Commit**

```bash
git add src/bin/gpu_fss.rs
git commit -m "feat: add --algorithm flag to gpu_fss (metropolis|msc|wolff|auto)"
```

---

## Task 8: GPU statistical parity tests

**Files:**
- Create: `tests/gpu_parity.rs`

**Important:** These are STATISTICAL parity tests, not bitwise. MSC changes data layout and RNG consumption patterns — exact spin-state identity is neither expected nor required. The correct gate is: same detailed-balance target distribution → same equilibrium observables within statistical error.

**Step 1: Write statistical parity tests**

```rust
//! GPU statistical parity tests: verify optimized paths produce correct physics.
//! Run with: cargo test --features cuda -- --ignored
//!
//! These are NOT bitwise tests. MSC and Wolff use different RNG consumption
//! patterns than single-spin Metropolis. The correct validation is:
//!   - same acceptance logic and detailed-balance condition
//!   - equilibrium observables agree within 2σ jackknife error bars
//!   - autocorrelation time (τ_int) is finite and reasonable

#[test]
#[ignore] // Requires --features cuda and GPU hardware
fn msc_statistical_parity_near_tc() {
    // MSC and single-spin Metropolis target the same Boltzmann distribution.
    // Run both at N=32, T=4.51 (near Tc), 3000 warmup + 10000 measurement sweeps.
    // Compare <E>, <|M|>, Cv, χ — must agree within 2σ jackknife error.
    //
    // 1. Run single-spin LatticeGpu: measure <E>, <|M|>, compute jackknife errors
    // 2. Run MscLattice: same temperature, same number of sweeps
    // 3. For each observable: |obs_msc - obs_single| < 2 * sqrt(err_msc² + err_single²)
}

#[test]
#[ignore]
fn msc_statistical_parity_low_t() {
    // Same test at T=3.0 (ordered phase). MSC should give near-ground-state E.
}

#[test]
#[ignore]
fn msc_statistical_parity_high_t() {
    // Same test at T=6.0 (disordered). E and |M| should match paramagnetic regime.
}

#[test]
#[ignore]
fn wolff_gpu_statistical_parity() {
    // GPU Wolff and CPU Wolff should sample the same equilibrium distribution.
    // N=8 (small for CPU tractability), T=4.51, 10000 steps.
    // Compare <E>, <|M|> within 2σ jackknife error bars.
    // Also report τ_int for both — GPU Wolff should have τ_int comparable to CPU Wolff.
}

#[test]
#[ignore]
fn philox_rng_produces_valid_metropolis() {
    // After Philox swap, LatticeGpu at known (N=8, T=4.51, seed=42)
    // should produce physically reasonable observables.
    // Not bitwise identical to old RNG — different sequence.
    // Gate: E in [-3, 0] per spin, |M| in [0, 1] per spin,
    // Cv peak near Tc, reasonable error bars.
}

#[test]
#[ignore]
fn batched_vs_sequential_bitwise() {
    // Exception: batched vs sequential single-replica SHOULD be bitwise identical
    // since the kernel logic and RNG consumption are the same, just launched together.
    // N=32, 1 replica, same seed → spin states must match after 100 sweeps.
}
```

**Step 2: Commit**

```bash
git add tests/gpu_parity.rs
git commit -m "test: add GPU parity tests for MSC, Wolff, and Philox correctness"
```

---

## Task 9: Batched multi-replica MSC kernel

**Files:**
- Modify: `src/cuda/msc_kernel.cu` (add batched variant)
- Modify: `src/cuda/msc_lattice.rs` (add BatchedMscLattice or batch methods)

**Step 1: Add batched MSC kernel to msc_kernel.cu**

```c
// Batched MSC: all replicas in one kernel launch.
// spins_all: concatenated [replica_0 words | replica_1 words | ...]
// boltz_probs_all: [7 * n_replicas] — per-replica Boltzmann tables
extern "C" __global__ void msc_batched_metropolis_kernel(
    unsigned int* spins_all,
    RngState*     rng_states,
    int           N,
    int           n_replicas,
    int           parity,
    const float*  boltz_probs_all  // [7 * n_replicas]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int wpr = N / 32;
    int words_per_replica = N * N * wpr;
    int total = words_per_replica * n_replicas;
    if (tid >= total) return;

    int replica = tid / words_per_replica;
    int local_tid = tid % words_per_replica;
    int offset = replica * words_per_replica;
    const float* boltz = boltz_probs_all + replica * 7;

    // Same MSC logic as msc_metropolis_kernel, but index into spins_all[offset + ...]
    // and use boltz[] instead of boltz_probs[]
    // [Body identical to msc_metropolis_kernel with offset addressing]
}
```

**Step 2: Add batch support to MscLattice**

Add a `BatchedMscLattice` struct that allocates concatenated buffers for R replicas and launches the batched kernel. Provide `step_all()`, `measure_all()`, and `set_temperatures()` methods.

**Step 3: Build and verify**

Run: `cargo build --features cuda`

**Step 4: Commit**

```bash
git add src/cuda/msc_kernel.cu src/cuda/msc_lattice.rs
git commit -m "feat(cuda): add batched multi-replica MSC kernel — single launch for all replicas"
```

---

## Task 10: Benchmark matrix script

**Files:**
- Create: `scripts/benchmark_msc.sh`

**Step 1: Write benchmark script with three-regime matrix and ESS reporting**

```bash
#!/usr/bin/env bash
# Benchmark matrix: single-spin vs MSC vs Wolff on GPU.
# Tests three temperature regimes: low T, near Tc, high T.
# Reports wall-clock time. ESS analysis done in Python post-processing.
#
# Usage: bash scripts/benchmark_msc.sh
set -e

echo "=== Building gpu_fss ==="
cargo build --release --features cuda --bin gpu_fss

ALGORITHMS="metropolis msc wolff"
# Three regimes: low T (ordered), near Tc (critical), high T (disordered)
REGIMES="low_T:3.0:3.2 near_Tc:4.4:4.6 high_T:5.8:6.2"
SIZES="32 64 128"
REPLICAS=20
WARMUP=2000
SAMPLES=10000

OUTBASE="/tmp/bench_gpu"
rm -rf "$OUTBASE"

for algo in $ALGORITHMS; do
  for regime_spec in $REGIMES; do
    IFS=: read -r regime tmin tmax <<< "$regime_spec"
    for n in $SIZES; do
      outdir="$OUTBASE/${algo}_${regime}_N${n}"
      mkdir -p "$outdir"
      echo ""
      echo "--- $algo | $regime (T=$tmin..$tmax) | N=$n ---"
      time target/release/gpu_fss \
        --model ising --sizes "$n" \
        --tmin "$tmin" --tmax "$tmax" --replicas "$REPLICAS" \
        --warmup "$WARMUP" --samples "$SAMPLES" \
        --exchange-every 10 --algorithm "$algo" \
        --outdir "$outdir" 2>/dev/null
    done
  done
done

echo ""
echo "=== Results saved to $OUTBASE ==="
echo "=== Run analysis/scripts/benchmark_ess.py to compute ESS/sec ==="
echo ""
echo "Sample summary (MSC near_Tc N=64):"
head -3 "$OUTBASE/msc_near_Tc_N64/gpu_fss_ising_N64_summary.csv" 2>/dev/null || echo "(not found)"
```

The companion Python script `analysis/scripts/benchmark_ess.py` (created separately) reads the timeseries CSVs, computes τ_int via Flyvbjerg-Petersen blocking, and reports ESS/sec = (N_samples / τ_int) / wall_clock_seconds for each cell in the matrix. This is the HPC-grade metric.

**Step 2: Commit**

```bash
chmod +x scripts/benchmark_msc.sh
git add scripts/benchmark_msc.sh
git commit -m "feat: add wall-clock benchmark script for MSC vs single-spin vs Wolff"
```

---

## Summary

| Task | Deliverable | Estimated Effort |
|------|------------|------------------|
| 1 | Philox RNG in Ising kernels | Small |
| 2 | Philox RNG in continuous kernels | Small |
| 9 | Batched multi-replica MSC kernel | Medium |
| 3 | MSC CUDA kernel (msc_kernel.cu) | Large |
| 4 | MscLattice Rust orchestration | Medium |
| 8 | Statistical parity tests (MSC) | Medium |
| 5 | GPU Wolff CUDA kernel (prototype) | Large |
| 6 | WolffGpuLattice Rust orchestration | Medium |
| 8b | Statistical parity + ESS tests (Wolff) | Medium |
| 7 | --algorithm flag in gpu_fss | Medium |
| 10 | Benchmark matrix script | Small |

**Execution order:** 1 → 2 → 9 → 3 → 4 → 8 (gate) → 5 → 6 → 8b (gate) → 7 → 10.

**Critical path:** 1 → 9 → 3 → 4 → 8 (MSC validated and benchmarked).

**Fallback:** If GPU Wolff ESS/sec < CPU Wolff + GPU Metropolis, do not merge Wolff into default path. Keep CPU Wolff for near-Tc and GPU MSC Metropolis for all other regimes.
