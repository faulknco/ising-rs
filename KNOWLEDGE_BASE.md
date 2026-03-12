# ising-rs Knowledge Base

A comprehensive reference for the ising-rs project: physics, implementation, analysis pipeline, infrastructure, and lessons learned across multiple development sessions.

---

## Table of Contents

1. [Project Goals](#project-goals)
2. [The Physics](#the-physics)
3. [Algorithms](#algorithms)
4. [Code Architecture](#code-architecture)
5. [Data Generation Pipeline](#data-generation-pipeline)
6. [Analysis Pipeline](#analysis-pipeline)
7. [Key Results](#key-results)
8. [Infrastructure and CI](#infrastructure-and-ci)
9. [Debugging War Stories](#debugging-war-stories)
10. [Configuration Reference](#configuration-reference)
11. [Glossary](#glossary)

---

## Project Goals

Build a publication-quality Monte Carlo simulator for the Ising model that:

1. **Validates against known theory** — reproduce 3D Ising universality class exponents to within a few percent
2. **Works on arbitrary graph topologies** — cubic, BCC, FCC, triangular, bond-diluted, and custom edge-list graphs
3. **Includes GPU acceleration** — CUDA checkerboard decomposition for Metropolis sweeps
4. **Studies nonequilibrium physics** — Kibble-Zurek defect scaling and Allen-Cahn domain coarsening
5. **Connects to real materials** — exchange coupling (J) fitting to match experimental Curie temperatures for BCC Fe and FCC Ni
6. **Ships as a complete package** — paper draft, analysis notebooks, WASM demo, CI, and documentation

---

## The Physics

### The Ising Model

The Hamiltonian:

```
H = -J * sum_{<i,j>} sigma_i * sigma_j - h * sum_i sigma_i
```

- `sigma_i = +/- 1` — spin at site i
- `J` — exchange coupling (J > 0: ferromagnet, J < 0: antiferromagnet)
- `h` — external magnetic field
- `<i,j>` — sum over nearest-neighbour pairs

The model exhibits a continuous phase transition at the **Curie temperature** T_c, separating the ordered (ferromagnetic) phase from the disordered (paramagnetic) phase.

### Critical Phenomena (3D Ising Universality Class)

Near T_c, observables follow power laws characterised by **critical exponents**:

| Exponent | Relation | 3D Ising Value | Physical Meaning |
|----------|----------|----------------|------------------|
| beta | M ~ (T_c - T)^beta | 0.3265 | Order parameter vanishing |
| gamma | chi ~ \|T - T_c\|^{-gamma} | 1.2372 | Susceptibility divergence |
| alpha | C_v ~ \|T - T_c\|^{-alpha} | 0.1096 | Heat capacity singularity |
| nu | xi ~ \|T - T_c\|^{-nu} | 0.6301 | Correlation length divergence |

These are connected by **scaling relations**:
- Rushbrooke: `alpha + 2*beta + gamma = 2`
- Hyperscaling: `2*beta/nu + gamma/nu = d` (d = dimension = 3)
- Fisher: `gamma = nu * (2 - eta)`

### Finite-Size Scaling (FSS)

On a finite lattice of size L, the true divergence is rounded. FSS theory predicts:
- `chi_max ~ L^{gamma/nu}` — susceptibility peak scales with system size
- `M(T_c) ~ L^{-beta/nu}` — magnetisation at T_c vanishes with system size
- `|dU/dT|_max ~ L^{1/nu}` — Binder cumulant slope diverges with system size

The **Binder cumulant** is the key quantity for locating T_c:

```
U = 1 - <m^4> / (3 * <m^2>^2)
```

At T_c, U is scale-invariant — curves for different L cross at the same point. This crossing gives T_c independent of any exponent assumptions.

### Known Exact Values

- **2D square lattice** (Onsager): T_c = 2 / ln(1 + sqrt(2)) = 2.2692
- **3D cubic lattice** (numerical): T_c/J = 4.5115232(16)
- **3D cubic lattice**: coordination number z = 6, ground state energy E_0 = -3J per spin

### Observables Computed

For each temperature point, we measure:

- **Energy per spin**: `<E>/N` — from the Hamiltonian
- **Magnetisation per spin**: `<|M|>/N` — absolute value to avoid cancellation
- **Heat capacity**: `C_v = beta^2 * (<E^2> - <E>^2) * N` — from energy fluctuations
- **Susceptibility**: `chi = beta * (<M^2> - <|M|>^2) * N` — connected susceptibility using signed magnetisation
- **Binder cumulant**: `U = 1 - <M^4> / (3 * <M^2>^2)` — from 2nd and 4th moments of |m|

The connected susceptibility uses `<m_signed^2> - <m_signed>^2` (not `<|m|>`), which is important for avoiding systematic bias.

### Anisotropy Component Observables (GPU)

For the Heisenberg model with uniaxial anisotropy `H = -J S_i.S_j - D (S_i^z)^2`, the GPU pipeline tracks:

- **Mz**: `|sum_i S_i^z| / N` — easy-axis order parameter (D > 0)
- **Mxy**: `sqrt(sum_i S_i^x)^2 + (sum_i S_i^y)^2) / N` — easy-plane order parameter (D < 0)
- **chi_z**: `beta * (<Mz^2> - <Mz>^2) * N` — z-component susceptibility
- **chi_xy**: `beta * (<Mxy^2> - <Mxy>^2) * N` — in-plane susceptibility

The analysis pipeline (`analyze_heisenberg_anisotropy.py`) auto-selects the correct observable:
- D > 0: uses Mz, chi_z (easy-axis regime)
- D < 0: uses Mxy, chi_xy (easy-plane regime)
- D = 0: uses M, chi (isotropic Heisenberg)

The `--init-state` flag controls spin initialization:
- `cold`: all spins along z — best for D > 0 (easy-axis)
- `planar`: all spins along x — best for D < 0 (easy-plane)
- `random`: uniformly random on S2/S1 — default for D = 0

### Kibble-Zurek Mechanism

When cooling through T_c at a finite rate, the system cannot keep up with the diverging relaxation time. The defect (domain wall) density scales as:

```
rho ~ tau_Q^{-kappa}    where kappa = nu / (1 + nu * z)
```

- `tau_Q` — quench time (sweeps)
- `z` — dynamic critical exponent (z ~ 2.02 for Metropolis, z ~ 0.33 for Wolff)
- Theoretical: kappa = 0.279 for 3D Ising with Metropolis dynamics

We use a **snap-freeze** (linear temperature ramp) protocol: cool from T_start > T_c to T_end < T_c over tau_Q Metropolis sweeps, then measure the domain wall density.

### Domain Coarsening (Allen-Cahn)

After a quench below T_c, domains grow as:

```
rho(t) ~ t^{-1/2}
```

This is the Allen-Cahn law for non-conserved order parameter dynamics. The domain wall density `rho` counts the fraction of nearest-neighbour pairs with opposite spins.

---

## Algorithms

### Metropolis Algorithm

Single-spin flip dynamics:
1. Pick a random spin i
2. Compute energy change: `delta_E = 2 * sigma_i * (J * sum_neighbours + h)`
3. Accept flip if `delta_E < 0` or with probability `exp(-beta * delta_E)`

One **sweep** = N attempted flips (N = number of spins). Each spin is visited once on average.

**Critical slowing down**: near T_c, the autocorrelation time diverges as `tau ~ xi^z` with z ~ 2.17 for 3D Ising. This makes Metropolis very slow near the phase transition.

### Wolff Cluster Algorithm

Eliminates critical slowing down by flipping entire correlated clusters:
1. Pick a random seed spin
2. Grow cluster: for each same-spin neighbour, add with probability `p_add = 1 - exp(-2*beta*J)`
3. Flip the entire cluster

Key properties:
- **Dynamic exponent** z ~ 0.33 (vs 2.17 for Metropolis) — roughly 7x faster decorrelation near T_c
- Only works for **ferromagnetic** J > 0 (p_add = 0 for J < 0, cluster never grows)
- Only exact for **h = 0**; with external field, follow each Wolff step with a Metropolis sweep
- At low T: p_add ~ 1, clusters are large (nearly all spins)
- At high T: p_add ~ 0, clusters are small (just the seed)

### GPU Checkerboard Decomposition

For Metropolis on regular lattices, colour the lattice like a checkerboard (red/black). All red spins can be updated simultaneously because they only depend on black neighbours (and vice versa). This gives massive GPU parallelism.

---

## Code Architecture

### Module Overview

```
src/
  lib.rs              # Module declarations
  lattice.rs          # Lattice struct: spins, precomputed neighbours, PBC
  metropolis.rs       # sweep() and warm_up() functions
  wolff.rs            # step() and warm_up() functions
  observables.rs      # measure(), measure_wolff(), measure_wolff_raw(), energy_magnetisation()
  fitting.rs          # CriticalExponents::fit() — log-log OLS for beta, alpha, gamma
  fss.rs              # run_fss() — FSS driver over multiple lattice sizes
  sweep.rs            # run() and run_raw() — temperature sweep driver
  kibble_zurek.rs     # run_kz() and run_kz_sweep()
  coarsening.rs       # run_coarsening() and domain_wall_density()
  graph.rs            # parse_edge_csv(), parse_json_graph() — arbitrary topology loading
  cli.rs              # Shared CLI parsing: get_arg(), parse_arg(), parse_geometry()
  wasm.rs             # WebAssembly bindings for browser demo
  cuda/               # GPU kernels (optional feature)
    mod.rs
    lattice_gpu.rs
    fss_gpu.rs
    coarsening_gpu.rs
    kz_gpu.rs
```

### Key Design Decisions

- **Flat arrays with precomputed neighbours**: `Lattice.spins` is a `Vec<i8>`, `Lattice.neighbours` is a `Vec<Vec<usize>>`. Neighbours are computed once at construction — O(1) lookup per flip attempt. This is the same approach whether the topology is cubic, triangular, or an arbitrary graph.
- **Spins as i8**: +1 or -1. Compact, fast arithmetic.
- **Periodic boundary conditions (PBC)**: implemented via modular arithmetic in neighbour construction. For 3D cubic: `((i-1+N) % N, j, k)` etc.
- **`measure_wolff_raw()`**: returns per-sample E and M vectors (not just averages). Essential for histogram reweighting — you need the raw time series to reweight to nearby temperatures.
- **RNG**: `rand_xoshiro::Xoshiro256PlusPlus` — fast, high-quality, seedable for reproducibility.

### Binary CLIs

All 5 binaries share `cli.rs` for argument parsing. Key flags:
- `--n` — lattice size (linear dimension)
- `--geometry` — square, triangular, cubic
- `--wolff` — use Wolff algorithm instead of Metropolis
- `--raw` — output per-sample data (for histogram reweighting)
- `--seed` — RNG seed for reproducibility
- `--warmup`, `--samples` — equilibration and measurement sweeps

---

## Data Generation Pipeline

### Two-Tier Strategy

The FSS analysis uses two separate data runs:

**Tier 1: Full-range sweep**
```bash
cargo run --release --bin fss -- --wolff --sizes 8,12,16,20,24,32,40,48 \
    --tmin 3.5 --tmax 5.5 --steps 41 --warmup 2000 --samples 2000 \
    --outdir analysis/data
```
- Coarse temperature grid (dT = 0.05)
- 2000 samples per point
- Shows overall phase transition shape
- Output: `analysis/data/fss_N{L}.csv` (aggregated observables)

**Tier 2: High-resolution near T_c**
```bash
cargo run --release --bin fss -- --wolff --raw --sizes 16,20,24,32,40,48 \
    --tmin 4.30 --tmax 4.70 --steps 41 --warmup 5000 --samples 10000 \
    --outdir analysis/data/hires
```
- Fine temperature grid (dT = 0.01) centred on T_c
- 10000 samples per point with `--raw` flag
- Outputs per-sample data for histogram reweighting
- Output: `analysis/data/hires/fss_raw_N{L}.csv`

### Kibble-Zurek Data

```bash
cargo run --release --bin kz -- --n 20 --geometry cubic \
    --tau-min 100 --tau-max 100000 --tau-steps 20 \
    --trials 50 --seed 42 --outdir analysis/data
```

Multiple trials per tau_Q value, averaged to reduce noise. Logarithmically spaced tau_Q values.

### Coarsening Data

```bash
cargo run --release --bin coarsening -- --n 30 --geometry cubic \
    --t-quench 2.5 --steps 200000 --sample-every 100 \
    --warmup 200 --seed 42 --outdir analysis/data
```

---

## Analysis Pipeline

### Jupyter Notebooks

**`analysis/fss.ipynb`** — Main FSS analysis:
1. Load full-range CSV data for each lattice size
2. Load hires raw CSV data for histogram reweighting
3. Compute observables with `compute_observables_from_raw()` including jackknife errors
4. Plot E, M, Cv, chi, Binder cumulant vs T for all sizes
5. Extract T_c from Binder cumulant crossings (using HIRES data, not coarse data)
6. Ferrenberg-Swendsen histogram reweighting on 200-point fine grid
7. Peak scaling fits: chi_max vs L, M(T_c) vs L, |dU/dT|_max vs L
8. Global scaling collapse with adjacent-point cost optimisation

**`analysis/validation.ipynb`** — Cross-checks:
- 2D Onsager exact solution comparison
- Exact enumeration on small lattices
- Autocorrelation analysis (Metropolis vs Wolff)
- Fluctuation-dissipation theorem verification

### Jackknife Error Estimation

20-block jackknife resampling:
1. Divide N samples into 20 blocks
2. For each block, compute the observable leaving that block out
3. Error = sqrt((n-1)/n * sum((x_i - x_bar)^2))

Applied to all observables: E, M, Cv, chi, Binder cumulant.

### Ferrenberg-Swendsen Histogram Reweighting

Given raw samples {E_i, M_i} at simulation temperature T_0 (beta_0), estimate observables at a nearby temperature T (beta):

```
<O>(beta) = sum_i O_i * exp(-(beta - beta_0) * E_i * N) / sum_i exp(-(beta - beta_0) * E_i * N)
```

We use a **patchwork** scheme: for each target temperature, reweight from the nearest simulated temperature. This gives a smooth 200-point grid from 41 simulation temperatures.

Critical requirement: need **per-sample** E and M data (not just averages), hence the `--raw` flag.

### Scaling Collapse

The scaling collapse tests whether data for different L values collapse onto a single curve when plotted as:

```
chi * L^{-gamma/nu}  vs  (T - T_c) * L^{1/nu}
```

We optimise T_c, gamma/nu, and nu using the **adjacent-point cost function** (Kawashima & Ito 1993):
- Sort all rescaled data points by x = (T - T_c) * L^{1/nu}
- Sum the squared differences between adjacent points in x
- Minimise over (T_c, nu, gamma/nu)

This is superior to binned variance cost functions, which can have degenerate minima. Bounds: T_c in [4.45, 4.55], nu in [0.5, 0.9], gamma/nu in [1.5, 2.5].

---

## Key Results

### 3D Cubic Lattice FSS

| Quantity | Measured | Theory | Error |
|----------|----------|--------|-------|
| T_c (Binder crossing) | 4.512(4) | 4.5115 | 0.01% |
| gamma/nu (chi peak scaling) | 1.933(30) | 1.964 | 1.6% |
| beta/nu (M at T_c scaling) | 0.492(26) | 0.518 | 5.0% |
| nu (collapse optimisation) | 0.667(37) | 0.630 | 5.9% |
| Hyperscaling check: 2*beta/nu + gamma/nu | 2.918 | 3.0 | 2.7% |

### Kibble-Zurek

- kappa = 0.258(15) at N=80
- Theoretical: kappa = nu/(1 + nu*z) = 0.279
- Error: 7.4%

### Exchange Coupling Fitting

- BCC Fe: J_fit = 14.3 meV (literature: 16.3 meV)
- Method: simulate on BCC graph, scan J values, match T_c to experimental Curie temperature

---

## Infrastructure and CI

### CI Pipeline (`.github/workflows/ci.yml`)

5 jobs:
1. **fmt** — `cargo fmt -- --check` (runs first, gates the rest)
2. **build-and-test (ubuntu)** — build, 38 unit tests, clippy
3. **build-and-test (windows)** — same
4. **build-and-test (macos)** — same
5. **cargo-deny** — license and vulnerability audit (runs in parallel with build)

### Branch Protection (master)

- Required status checks: all 5 CI jobs must pass
- Strict mode: branch must be up-to-date before merging
- Enforce admins: even repo owner can't bypass
- No force pushes, no branch deletion

### Tooling

- `rust-toolchain.toml` — pins Rust 1.94 with rustfmt and clippy components
- `Cargo.toml` — `rust-version = "1.94"` (MSRV)
- `deny.toml` — cargo-deny config for license allowlist (MIT, Apache-2.0, BSD, ISC, Unicode)
- `.github/dependabot.yml` — weekly updates for cargo and GitHub Actions, major versions ignored
- `CONTRIBUTING.md` — pre-PR checklist: fmt, clippy, test
- Issue and PR templates

### Key Dependencies

- `rand 0.8` + `rand_xoshiro 0.6` — RNG
- `wasm-bindgen 0.2` — WASM support
- `anyhow 1` — error handling in binaries
- `cudarc 0.12` (optional) — CUDA bindings

---

## Debugging War Stories

### The Binder Crossing Disaster

**Problem**: Tc extracted from Binder cumulant crossings was 4.459 — way off from the known value of 4.5115.

**Root cause**: Using coarse-grid data (dT = 0.05, 2000 samples) for Binder crossings. The curves were too noisy and the crossings were unreliable — N=32/40 crossing at 4.31, N=40/48 at 4.37.

**Fix**: Switch to hires data (dT = 0.01, 10000 samples) for Binder crossing extraction. Result: T_c = 4.5121 +/- 0.0039.

**Lesson**: Binder crossings are extremely sensitive to statistical noise. You need high-stats, fine-temperature-grid data near T_c.

### The Scaling Collapse Failure

**Problem**: Global scaling collapse optimiser stuck at parameter bounds (T_c = 4.30, nu = 0.40) — completely unphysical.

**Root cause**: Binned variance cost function had a degenerate minimum. With wide bounds [(4.0, 5.0), (0.3, 1.5), (1.0, 3.0)], the optimiser found a trivial solution where all points collapsed to the same x-value.

**Fix**:
1. Replaced binned variance cost with **adjacent-point cost** (Kawashima & Ito 1993)
2. Tightened bounds to [(4.45, 4.55), (0.5, 0.9), (1.5, 2.5)]

**Lesson**: Cost function choice matters enormously for collapse optimisation. Adjacent-point cost is more robust than binned methods.

### The Connected Susceptibility Subtlety

**Problem**: Early susceptibility values were systematically wrong.

**Root cause**: Using `<|m|>^2` instead of `<m_signed>^2` in the fluctuation formula. The correct connected susceptibility is:

```
chi = beta * N * (<m_signed^2> - <m_signed>^2)
```

Not `<|m|^2> - <|m|>^2`. The absolute value introduces a positive bias.

**Lesson**: Always track both signed and unsigned magnetisation separately.

### Cargo-Deny Config Evolution

**Problem**: `deny.toml` failed in CI with "unexpected value" and "deprecated key" errors.

**Root cause**: The cargo-deny-action@v2 ships cargo-deny 0.19.0, which has a completely different config format than older versions. `vulnerability = "deny"` and `unlicensed = "deny"` are both removed in v0.19.

**Fix**: Simplified config to only use current v0.19 keys: `[advisories] ignore = []`, `[licenses] allow = [...]`.

### Rust-Toolchain vs CI

**Problem**: `cargo fmt -- --check` failed in CI even though it passed locally.

**Root cause**: `rust-toolchain.toml` pins to channel `1.94`, but without specifying `components = ["rustfmt", "clippy"]`. The `dtolnay/rust-toolchain@stable` action installed the `stable` toolchain with rustfmt, but `rust-toolchain.toml` overrode it to use the `1.94` toolchain which didn't have rustfmt installed.

**Fix**: Added `components = ["rustfmt", "clippy"]` to `rust-toolchain.toml`.

### Tests in tests/ Not Formatted

**Problem**: `cargo fmt` passed locally but CI showed diffs in `tests/cli.rs`.

**Root cause**: The initial `cargo fmt` only formatted `src/` files. The `tests/` directory was missed because we staged files manually with `git add src/`.

**Fix**: Run `cargo fmt` (which formats everything) and stage `tests/cli.rs` too.

---

## Configuration Reference

### Simulation Parameters (What Values to Use)

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| Lattice size N | 8-48 | Larger = better statistics but slower. Use multiple sizes for FSS. |
| Warmup sweeps | 2000-5000 | Must be >> autocorrelation time. More at low T. |
| Measurement samples | 2000 (coarse), 10000 (hires) | More = smaller error bars. |
| T range (full) | 3.5 - 5.5 | Covers both phases well for 3D cubic. |
| T range (hires) | 4.30 - 4.70 | Centred on T_c = 4.5115, fine enough for reweighting. |
| Temperature steps | 41 | Gives dT = 0.05 (full) or dT = 0.01 (hires). |
| Jackknife blocks | 20 | Standard choice. Too few = noisy errors, too many = correlated blocks. |
| KZ trials | 10-50 | More trials = smoother rho(tau_Q) curve. |
| KZ tau_Q range | 100 - 100000 | Logarithmically spaced. Need 2+ decades for power law fit. |

### 3D Cubic Reference Values

- T_c = 4.5115232(16) J/k_B
- Coordination number z = 6
- Ground state energy: E_0 = -3J per spin (each bond counted once)
- Ground state magnetisation: M = 1 per spin (all aligned)
- Binder cumulant at T=0: U = 2/3 (all samples identical)
- Binder cumulant at T=infinity: U = 0 (Gaussian distribution)

### 2D Square Reference Values

- T_c = 2/ln(1 + sqrt(2)) = 2.2692 J/k_B (Onsager exact)
- Coordination number z = 4
- Ground state energy: E_0 = -2J per spin

---

## Glossary

| Term | Definition |
|------|-----------|
| **Allen-Cahn** | Domain wall density decays as t^{-1/2} after quench below T_c |
| **Binder cumulant** | U = 1 - <m^4>/(3<m^2>^2). Scale-invariant at T_c. |
| **Critical slowing down** | Autocorrelation time diverges at T_c as xi^z |
| **Domain wall density** | Fraction of nearest-neighbour pairs with opposite spins |
| **Ferrenberg-Swendsen** | Histogram reweighting to interpolate observables between simulated temperatures |
| **FSS** | Finite-size scaling: extracting infinite-volume exponents from finite lattices |
| **Harris criterion** | Disorder is relevant if alpha > 0 (suppresses T_c) |
| **Hyperscaling** | 2*beta/nu + gamma/nu = d (spatial dimension) |
| **Jackknife** | Resampling method for error estimation; less biased than bootstrap for correlated data |
| **Kawashima-Ito** | Adjacent-point cost function for scaling collapse optimisation |
| **Kibble-Zurek** | Defect density after finite-rate quench: rho ~ tau_Q^{-kappa} |
| **Metropolis** | Single-spin flip Monte Carlo with acceptance exp(-beta*delta_E) |
| **PBC** | Periodic boundary conditions: lattice wraps around in all directions |
| **Scaling collapse** | Plotting rescaled data to verify universal scaling functions |
| **Sweep** | One pass through all N spins (N flip attempts) |
| **Universality class** | Systems with same symmetry and dimension share critical exponents |
| **Wolff** | Cluster flip algorithm: builds and flips correlated clusters, z ~ 0.33 |

---

---

## GPU Pipeline (added 2026-03-12)

### Overview

The GPU pipeline (`src/bin/gpu_fss.rs`) runs parallel tempering FSS simulations on NVIDIA GPUs via CUDA. It supports all three universality classes with model-specific optimizations.

### Architecture

- **Ising (Z2):** Multi-site coded (MSC) lattice — 32 spins packed per u32 word. Batched kernel launches all replicas simultaneously. ~30x speedup over single-spin GPU.
- **Heisenberg (O(3)):** 3-component f32 Cartesian spins. Checkerboard Metropolis + microcanonical over-relaxation (5 sweeps per Metropolis). CPU-side Wolff embedding every 10 sweeps.
- **XY (O(2)):** Same as Heisenberg with n_comp=2.
- **Quantized variants:** FP16 Heisenberg (48 bits/spin vs 96) and angle-only XY (16 bits/spin). Compute in f32, store in f16.

### Wolff Embedding for O(n) Models

CPU-side cluster algorithm that dramatically reduces critical slowing down (z ≈ 0.1 vs z ≈ 2 for Metropolis):

1. Download all spins from GPU to host
2. For each replica (parallel via `std::thread::scope`):
   - Choose random unit vector **r**
   - Project all spins onto r: σ_i = S_i · r
   - Build Wolff cluster on projected ±1 variables (DFS + u64-packed bitset)
   - Reflect cluster spins: S_i → S_i − 2(S_i · r)r
3. Upload modified spins back to GPU

Implementation: `src/cuda/wolff.rs` (shared module for Cartesian + angle representations).

### Key GPU Gotchas

- **curandStatePhilox4_32_10** is 64 bytes per state (not 16 or 48)
- RNG states dominate VRAM: 226 MB/replica at N=192 vs 85 MB for spins
- RTX 2060 (6GB): max 16 replicas at N=192 (f32), ~20 with FP16
- Windows WDDM silently pages GPU memory when VRAM exceeded — causes massive slowdown with no error
- Windows working set trimming pages host RAM to disk when process is idle — degraded Wolff performance

### Current Results (2026-03-12)

| Model | Tc Error | γ/ν | ν | β/ν | Sizes | Status |
|-------|----------|-----|---|-----|-------|--------|
| Ising | 0.015% | 2.6% | 0.1% | 0.3% | 8-192 | Publication quality |
| XY | 0.001% | 0.8% | 2.3% | 1.0% | 8-128 | Publication quality |
| Heisenberg | 0.3% | 1.1% | 11.6% | 2.0%† | 16-128 | Usable |

† β/ν via hyperscaling. Direct fit needs narrower T grid for N=192.

### Analysis

Use `analysis/scripts/analyze_gpu_fss.py` with `--method single` (single-histogram reweighting). WHAM fails for N≥64.

Key flags: `--fit-sizes` to exclude problematic sizes, `--collapse-sizes` for scaling collapse subset.

---

## Anisotropy Crossover (Phase 2, started 2026-03-12)

### Physics

3D Heisenberg model with uniaxial anisotropy: H = −J Σ S_i·S_j − D Σ (S_i^z)²

- D > 0: easy-axis → Ising universality class at large D
- D = 0: isotropic → Heisenberg universality class
- D < 0: easy-plane → XY universality class at large |D|

The crossover between universality classes as D varies is the target of Phase 2 research.

### Implementation Status (branch: `feature/gpu-anisotropy-port`)

- **CPU complete:** Metropolis with D, component-resolved observables (M_z, M_xy, χ_z, χ_xy), overrelaxation disabled for D≠0
- **GPU not started:** CUDA kernels need D parameter threaded through
- **Campaign scripts ready:** 7 D values × 5 sizes × 49 temps

---

*Last updated: 2026-03-12*
