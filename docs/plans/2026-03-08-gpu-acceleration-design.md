# GPU Acceleration Design — Checkerboard Metropolis + Parallel Tempering + Histogram Reweighting

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bridge consumer GPU (RTX 2060, 6 GB VRAM) to HPC-level accuracy for all three spin models (Ising, XY, Heisenberg) at N=128–256, using the full modern Monte Carlo toolchain.

**Architecture:** Two CUDA kernels (multi-spin-coded Ising + generic continuous-spin Metropolis/overrelax), parallel tempering with replica exchange, and Ferrenberg-Swendsen histogram reweighting in Python. Fully additive — opt-in via `--features cuda`, zero changes to existing CPU code.

**Tech Stack:** Rust + cudarc 0.12 (FFI to CUDA), CUDA C kernels compiled to PTX via build.rs, Python (numpy/scipy) for histogram reweighting.

**Hardware target:** Windows RTX 2060 (1920 CUDA cores, 6 GB VRAM, sm_75 Turing).

---

## 1. CUDA Kernel Architecture

### Kernel 1: `ising_msc_checkerboard_kernel`

Multi-spin coded (MSC) Ising with 32 temperature replicas packed per 32-bit word.

- Asynchronous MSC: bit position k holds the spin of replica k at the same lattice site
- Checkerboard update: all black sites (`(x+y+z) % 2 == 0`) in parallel, sync, then white sites
- Neighbours computed from index arithmetic — no adjacency list on GPU
- Geometry support: cubic (2 colours), BCC (2 colours, bipartite), FCC (4 colours)
- One kernel launch = one half-sweep across all 32 replicas simultaneously
- Accept/reject: pre-compute Boltzmann lookup tables for each replica temperature, pass as constant memory

### Kernel 2: `continuous_spin_checkerboard_kernel`

Generic kernel parameterised by `n_components` (2 = XY, 3 = Heisenberg).

- Spins stored as `float * n_components` per site on device
- Metropolis update: propose random perturbation within cone angle delta, accept/reject via delta_E
- Overrelaxation sub-kernel: deterministic reflection through local field direction (zero-cost acceptance, no RNG). Interleave `n_overrelax` overrelax sweeps per Metropolis sweep
- Same checkerboard colouring as Ising kernel
- `n_components` is a compile-time or launch-time parameter

### Shared Infrastructure

- `init_rng_kernel` — existing, reuse for per-thread curandState initialisation
- `reduce_kernel` — parallel reduction for E, |M|, M^2, M^4 on device. Avoids CPU-GPU round-trip per sample. Uses shared memory + warp shuffle for block-wide reduction.
- `GeometryInfo` struct passed to kernels: `{ n, n_colors, color_offsets[] }` — cubic/BCC use 2 colours, FCC uses 4

## 2. Parallel Tempering

### Sweep Phase (GPU, parallel)

Allocate R replicas at temperatures T_1 < T_2 < ... < T_R spanning the critical region. Each replica is a full lattice in GPU memory.

- One kernel launch sweeps all R replicas simultaneously
- For Ising MSC: 32 replicas packed per word — one launch handles 32 temperatures for free
- For continuous spins: each replica is a separate block (or group of blocks)

### Exchange Phase (CPU, serial)

After every `N_exchange` sweeps, attempt to swap adjacent replicas (i, i+1):

```
p_swap = min(1, exp((beta_i - beta_{i+1}) * (E_i - E_{i+1})))
```

Only energies need to be read back from GPU — one float per replica, negligible transfer cost.

### Temperature Selection

- Space R = 20–40 temperatures across the critical window
- Optimal spacing: tune so acceptance rate is 20–30% between adjacent replicas
- For Ising MSC at N=128: critical window ~0.03 in beta, 32 replicas gives delta_beta ~ 0.001 per replica

### Memory Budget (RTX 2060, ~5.5 GB usable)

| Model | N=128, 20 replicas | N=256, 20 replicas |
|---|---|---|
| Ising MSC (32 replicas/word) | 256 KB | 2 MB |
| XY (2 x f32) | 32 MB | 256 MB |
| Heisenberg (3 x f32) | 48 MB | 384 MB |

All fit comfortably. RNG state adds ~50 MB at N=128 (48 bytes per thread, N^3/2 threads).

## 3. Histogram Reweighting

### Implementation: Pure Python (no Rust/CUDA changes)

**Single-histogram (Ferrenberg-Swendsen 1988):**

During each simulation, collect energy time series. Reweight to nearby temperature:

```
<O>_beta = sum_E O(E) H(E) exp(-(beta - beta_0) * E) / sum_E H(E) exp(-(beta - beta_0) * E)
```

Valid within a window delta_beta ~ L^(-1/nu) around beta_0.

**Multiple-histogram (WHAM):**

Combine histograms from all R parallel-tempering replicas. Self-consistent iteration solves for free energies, then extracts any observable at any beta in the range. Produces continuous curves of Cv(T), chi(T), U(T) from the discrete simulation points.

### Data Format

- Current: one CSV row per temperature with pre-averaged observables (41 rows)
- New: raw energy/magnetisation time series per replica (binary format for compactness)
- GPU binary also outputs backward-compatible summary CSV (pre-averaged) so existing notebooks still work
- Reweighting happens in Python using the raw time series

### New Analysis Code

- `analysis/scripts/reweighting.py` — single-histogram + WHAM implementation
- Updated FSS notebooks: load histogram data, reweight to smooth curves, extract Binder crossings from continuous U(T)

## 4. CLI Design

### `src/bin/gpu_fss.rs` — Unified GPU FSS binary

```bash
cargo run --release --features cuda --bin gpu_fss -- \
  --model ising|xy|heisenberg \
  --sizes 8,16,32,64,128 \
  --tmin 4.4 --tmax 4.6 --replicas 32 \
  --warmup 5000 --samples 100000 \
  --exchange-every 10 \
  --seed 42 --outdir analysis/data
```

### `src/bin/gpu_jfit.rs` — Unified GPU J-fitting binary

```bash
cargo run --release --features cuda --bin gpu_jfit -- \
  --model ising|xy|heisenberg \
  --graph analysis/graphs/bcc_N12.json \
  --tmin 6.0 --tmax 6.7 --replicas 20 \
  --warmup 5000 --samples 100000 \
  --exchange-every 10 \
  --outdir analysis/data
```

### Output Files (per model/size)

1. `gpu_fss_{model}_N{n}_timeseries.bin` — raw E and |M| per sweep per replica (binary, compact)
2. `gpu_fss_{model}_N{n}_summary.csv` — pre-averaged observables per replica temperature (backward-compatible)

## 5. File Changes

### New Files

| File | Purpose |
|---|---|
| `src/cuda/ising_msc_kernel.cu` | Multi-spin coded Ising checkerboard kernel |
| `src/cuda/continuous_spin_kernel.cu` | Generic XY/Heisenberg checkerboard Metropolis + overrelax |
| `src/cuda/reduce_kernel.cu` | Parallel reduction for observables |
| `src/cuda/parallel_tempering.rs` | Host-side replica exchange logic |
| `src/cuda/gpu_lattice_continuous.rs` | GPU lattice wrapper for continuous spins (XY/Heisenberg) |
| `src/bin/gpu_fss.rs` | Unified GPU FSS binary |
| `src/bin/gpu_jfit.rs` | Unified GPU J-fitting binary |
| `analysis/scripts/reweighting.py` | Single-histogram + WHAM implementation |
| `scripts/run_gpu_publication.sh` | Production run script for all three models |

### Modified Files

| File | Change |
|---|---|
| `src/cuda/mod.rs` | Export new modules |
| `Cargo.toml` | Add `[[bin]]` entries for gpu_fss, gpu_jfit |
| `build.rs` | Compile new .cu files to PTX |

### Untouched

- All CPU code (`src/xy/`, `src/heisenberg/`, `src/wolff.rs`, `src/metropolis.rs`)
- All existing analysis notebooks
- All existing CLI binaries
- All existing tests

Fully additive — `--features cuda` opt-in. Without the flag, everything compiles as before.

## 6. Expected Performance

| Model | N | Current (CPU Wolff) | GPU Metropolis + PT | Speedup |
|---|---|---|---|---|
| Ising | 32 | ~2 min | ~1 sec | ~100x |
| Ising | 128 | infeasible | ~5 min | N/A |
| Ising (MSC) | 128 | infeasible | ~10 sec (32 replicas) | N/A |
| Heisenberg | 32 | ~4 min | ~2 sec | ~100x |
| Heisenberg | 128 | infeasible | ~20 min | N/A |
| XY | 128 | infeasible | ~15 min | N/A |

Estimates assume 2000 warmup + 100,000 sample sweeps with parallel tempering. Wall times are order-of-magnitude.

## 7. References

- Preis et al. (2009) — GPU accelerated MC of 2D and 3D Ising model
- Weigel (2012) — Performance potential for simulating spin models on GPU
- Block et al. (2010) — Multi-GPU accelerated multi-spin Monte Carlo
- Komura & Okabe (2012, 2014) — GPU Swendsen-Wang for Ising, Potts, XY
- Lulli et al. (2013) — Compact asynchronous MSC parallel tempering on GPU
- Ferrenberg & Swendsen (1988, 1989) — Histogram reweighting
