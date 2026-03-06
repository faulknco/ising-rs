# Design: FSS Analysis + Domain Wall Dynamics + CUDA GPU Backend

Date: 2026-03-06

## Overview

Extend the existing `ising-rs` Rust project with three capabilities:

1. **Finite-Size Scaling (FSS)** — sweep multiple lattice sizes, extract critical exponents via Binder cumulant crossing and scaling collapse
2. **Domain Wall Dynamics** — quench experiment measuring coarsening exponent vs Allen-Cahn theory
3. **CUDA GPU Backend** — `cudarc`-based checkerboard Metropolis kernel targeting NVIDIA RTX 2060 (Windows 10)

Analysis is delivered as two Jupyter notebooks consuming CSV data produced by Rust CLI binaries.

---

## Architecture

```
ising-rs/
├── src/
│   ├── fss.rs              # multi-N sweep runner, FSS observables (M2, M4 moments)
│   ├── coarsening.rs       # quench + domain wall density tracker
│   └── cuda/
│       ├── mod.rs
│       ├── kernels.cu      # checkerboard Metropolis CUDA kernels
│       └── lattice_gpu.rs  # CudaSlice<i8> GPU lattice, device RNG
├── src/bin/
│   ├── fss.rs              # CLI: --sizes, --wolff, --outdir
│   └── coarsening.rs       # CLI: --n, --t-quench, --steps, --outdir
├── analysis/
│   ├── data/               # CSV output from CLI binaries (gitignored)
│   ├── fss.ipynb
│   └── coarsening.ipynb
├── build.rs                # nvcc .cu → .ptx compilation
└── Cargo.toml
```

---

## Part 1: Finite-Size Scaling

### Rust

**`src/fss.rs`**
- `FssConfig`: sizes `Vec<usize>`, all other fields from existing `SweepConfig`
- `run_fss(config)` iterates over each N, calls existing `sweep::run()`, returns `Vec<(usize, Vec<Observables>)>`
- `observables.rs` gains `m2: f64` (⟨M²⟩) and `m4: f64` (⟨M⁴⟩) fields needed for Binder cumulant
- Binder cumulant: `U = 1 - m4 / (3 * m2^2)` computed in notebook from exported moments

**`src/bin/fss.rs`**
```
cargo run --release --bin fss -- \
  --sizes 8,12,16,20,24,28 \
  --wolff \
  --warmup 500 --samples 200 \
  --outdir analysis/data
```
Writes `analysis/data/fss_N8.csv`, `fss_N12.csv`, ... each with columns: `T,E,M,M2,M4,Cv,chi`

### Notebook: `analysis/fss.ipynb`

Sections:
1. Load all CSVs, overlay ⟨E⟩, |⟨M⟩|, Cv, χ for all N
2. Binder cumulant U(T) for all N — crossing point = precise Tc
3. Peak scaling: `log(Cv_max)` vs `log(N)` → slope = α/ν; `log(χ_max)` vs `log(N)` → slope = γ/ν
4. Scaling collapse: plot `N^{-γ/ν} χ` vs `(T−Tc)N^{1/ν}`, optimise ν for best collapse
5. Summary table: measured vs 3D Ising theory (ν=0.6301, β=0.3265, α=0.1096, γ=1.2372)

---

## Part 2: Domain Wall Dynamics

### Rust

**`src/coarsening.rs`**
- `domain_wall_density(lattice)`: fraction of NN pairs with opposite spins — O(N³) scan
- `CoarseningConfig`: n, geometry, t_high (disorder start), t_quench (target T), total_steps, sample_every
- Protocol:
  1. Randomise at `t_high`, run 200 warmup sweeps
  2. Set T = `t_quench`, run `total_steps` Metropolis sweeps
  3. Every `sample_every` sweeps, record `(step, domain_wall_density)`

**`src/bin/coarsening.rs`**
```
cargo run --release --bin coarsening -- \
  --n 30 --t-quench 0.5 --steps 50000 --sample-every 100
```
Writes `analysis/data/coarsening_N30.csv` with columns: `t,rho`

### Notebook: `analysis/coarsening.ipynb`

Sections:
1. Load CSV, plot ρ(t) on linear and log-log axes
2. OLS fit in log-log space after transient (skip first ~10% of steps)
3. Extract coarsening exponent z: `ρ ~ t^{-z}`
4. Compare to theory: z=1/2 (2D), z=1/3 (3D Allen-Cahn)
5. Visualise snapshots: domain wall pixels at t=100, 1000, 10000 steps

---

## Part 3: CUDA GPU Backend

### Target hardware
- NVIDIA RTX 2060, 6GB VRAM, 1920 CUDA cores
- Windows 10, CUDA Toolkit 12.x

### Rust implementation

**`src/cuda/kernels.cu`**
- Checkerboard decomposition: colour sites by `(x+y+z) % 2`
- `metropolis_black_kernel<<<grid, block>>>`: each thread handles one black site
- `metropolis_white_kernel<<<grid, block>>>`: each thread handles one white site
- Per-thread `curandState` RNG seeded from `global_seed + thread_id`
- ΔE computed from 6 neighbours in device memory (L1 cache friendly)
- One full sweep = black kernel + white kernel

**`src/cuda/lattice_gpu.rs`**
- `LatticeGpu`: `CudaSlice<i8>` spins on device, `CudaSlice<curandState>` RNG states
- `step_gpu(beta, j, h)`: launches both kernels
- `get_spins()`: transfers device → host only when needed
- `magnetisation_gpu()`: reduction kernel (sum of spins / N³)

**`build.rs`**
- Detects `CUDA_PATH` env var
- Compiles `src/cuda/kernels.cu` with `nvcc -ptx` → `target/kernels.ptx`
- Embedded in binary via `include_str!`
- Falls back gracefully if CUDA not present (CPU path unaffected)

**`Cargo.toml` additions**
```toml
[dependencies]
cudarc = { version = "0.12", features = ["cuda-12050"] }

[features]
cuda = ["cudarc"]
```

**CLI integration**: `--gpu` flag on both `fss` and `coarsening` binaries routes to GPU backend.

### Expected performance (RTX 2060, N=28 cubic)
- 21,952 spins, 10,976 threads per kernel launch
- Estimated 50-200× speedup over single-threaded CPU Metropolis
- Full FSS run (6 sizes, Wolff) estimated ~2-3 min GPU vs ~25 min CPU

---

## Windows 10 Setup

```powershell
# 1. Install CUDA Toolkit 12.x from developer.nvidia.com
# 2. Set environment variable
[Environment]::SetEnvironmentVariable("CUDA_PATH", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x", "User")
# 3. Install Rust stable (rustup.rs)
# 4. Build
cargo build --release --features cuda
```

---

## Implementation Order

1. Add M2/M4 moments to `observables.rs`
2. `src/fss.rs` + `src/bin/fss.rs`
3. `src/coarsening.rs` + `src/bin/coarsening.rs`
4. `analysis/fss.ipynb`
5. `analysis/coarsening.ipynb`
6. `src/cuda/kernels.cu` + `src/cuda/lattice_gpu.rs` + `build.rs`
7. Wire `--gpu` flag into CLI binaries

## Non-goals

- No changes to existing WASM/browser simulation
- No GPU Wolff algorithm (checkerboard decomposition doesn't apply directly)
- No Windows-specific CI (local build only)
