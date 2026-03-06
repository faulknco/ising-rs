# Ising Model Research Notes
Date: 2026-03-06

---

## Project Overview

3D Ising model simulation in Rust, compiled to WASM for browser demo and CLI binaries for research.
Repo: https://github.com/faulknco/ising-rs
Live demo: https://faulknco.github.io/ising-rs

---

## Architecture

```
Rust core (src/)
  lattice.rs       — N³ cubic lattice, precomputed neighbour indices
  metropolis.rs    — Metropolis-Hastings sweep, O(N³) per call
  wolff.rs         — Wolff cluster algorithm, BFS, ~6x faster near Tc
  observables.rs   — E, |M|, M2, M4, Cv, chi per temperature point
  fitting.rs       — OLS log-log fitting for critical exponents β, α, γ
  fss.rs           — multi-N sweep runner for FSS analysis
  coarsening.rs    — domain wall density quench experiment
  wasm.rs          — WebAssembly bindings (browser demo)
  cuda/            — CUDA GPU backend (RTX 2060, feature-flagged)

CLI binaries (src/bin/)
  sweep.rs         — single sweep, CSV output
  fss.rs           — FSS sweep across multiple N, CSV per size
  coarsening.rs    — quench experiment, domain wall density CSV

Analysis (analysis/)
  fss.ipynb        — Binder cumulant, peak scaling, scaling collapse
  coarsening.ipynb — Allen-Cahn coarsening exponent fit
  data/            — CSV output (gitignored)
```

---

## Physics

**Hamiltonian:** H = -J Σ σᵢσⱼ - h Σ σᵢ
**Spin:** σ ∈ {-1, +1}
**Metropolis accept:** ΔE = 2σᵢ(J Σⱼσⱼ + h), accept if ΔE<0 else with prob exp(-βΔE)
**Wolff:** p_add = 1 - exp(-2βJ), BFS cluster flip

**3D Ising universality class theory values:**
  Tc   = 4.5115 J/kB
  β    = 0.3265  (magnetisation exponent)
  α    = 0.1096  (heat capacity exponent)
  γ    = 1.2372  (susceptibility exponent)
  ν    = 0.6301  (correlation length exponent)

**Binder cumulant:** U = 1 - <M⁴> / (3 <M²>²)
  Curves for different N cross at exactly Tc (finite-size independent)

**FSS scaling:**
  Cv_max  ~ N^(α/ν)    α/ν  = 0.1740
  χ_max   ~ N^(γ/ν)    γ/ν  = 1.9635
  Collapse: N^(-γ/ν) χ  vs  (T-Tc) N^(1/ν)

**Allen-Cahn coarsening:** ρ(t) ~ t^(-z)
  3D: z = 1/3
  2D: z = 1/2

---

## Benchmark Results (MacBook, single CPU core, release build)

### FSS Run — 2026-03-06

Parameters:
  sizes      = 8, 12, 16, 20, 24, 28
  algorithm  = Wolff
  warmup     = 2000 sweeps per temperature
  samples    = 1000 sweeps per temperature
  T range    = 3.5 → 5.5 J/kB
  steps      = 41 temperature points

Results:
  Wall time  = 1 min 24 sec  (83.5s user, 99% CPU)
  Output     = 6 CSV files, 41 rows each, 7 columns (T,E,M,M2,M4,Cv,chi)

Data quality check (N=28):
  T=3.5:  M=0.880, Cv=0.834  (ordered phase, expected M~0.88)
  T=4.45: M=0.417, Cv=1.934  (approaching Tc)
  T=4.50: M=0.197, Cv=1.682  (near Tc, sharp drop in M)
  T=4.55: M=0.089, Cv=0.949  (disordered phase)
  T=5.5:  M=0.014, Cv=0.168  (fully disordered)
  Assessment: GOOD — sharp transition near Tc=4.51, Cv peak visible

### Coarsening Run — 2026-03-06

Parameters:
  n            = 30 (cubic, 27,000 spins)
  t_high       = 10.0 (initial disorder temperature)
  t_quench     = 2.5  (well below Tc, domain walls mobile)
  warmup       = 200 sweeps at t_high
  total_steps  = 200,000
  sample_every = 10

Results:
  Wall time  = 43 sec (43.2s user, 99% CPU)
  Output     = coarsening_N30_T2.50.csv, 20,000 rows, 2 columns (t, rho)

Data quality check:
  t=0:       rho=0.350  (freshly quenched, ~35% domain walls)
  t=10:      rho=0.147  (fast early coarsening)
  t=100:     rho=0.066  (still decaying)
  t=200,000: rho=0.019  (late time, slow decay)
  Assessment: GOOD — full power-law decay visible, not frozen, late-time
              noise from finite-size (domain size approaching N=30)

### Failed parameters (for reference — do not reuse)

  t_quench=0.5, steps=50000, sample-every=100
  Problem: system froze after t=100, essentially no coarsening visible
  Reason: T=0.5 is too cold, domain walls immobile at such low T

---

## CPU vs GPU Estimates

CPU (MacBook, single core):
  FSS 6 sizes:     84 sec
  Coarsening 200k: 43 sec
  Total:           ~2 min

RTX 2060 (Windows, CUDA, estimated):
  N=28 cubic: 21,952 spins, 10,976 threads per kernel launch
  Expected speedup: 50-200x over CPU Metropolis
  FSS 6 sizes:     ~2-5 sec estimated
  Coarsening 200k: ~1-2 sec estimated

RTX 2060 enables:
  N=40: 64,000 spins  — tractable in ~10 sec
  N=50: 125,000 spins — tractable in ~30 sec
  N=64: 262,144 spins — tractable in ~2 min
  These sizes needed for publication-quality FSS

---

## Windows Machine Setup (RTX 2060)

Steps to get GPU build running:

1. Install CUDA Toolkit 12.x
   https://developer.nvidia.com/cuda-downloads
   Select: Windows > 10 > x86_64 > exe (local)

2. Set environment variable (PowerShell):
   [Environment]::SetEnvironmentVariable(
     "CUDA_PATH",
     "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x",
     "User"
   )

3. Install Rust:
   https://rustup.rs — download rustup-init.exe

4. Clone repo:
   git clone https://github.com/faulknco/ising-rs.git
   cd ising-rs

5. Build with CUDA:
   cargo build --release --features cuda --bin fss
   cargo build --release --features cuda --bin coarsening

6. Run FSS on GPU:
   cargo run --release --features cuda --bin fss -- --sizes 8,12,16,20,24,28 --wolff --gpu --outdir analysis\data

Note: RTX 2060 = Turing architecture, sm_75. This is hardcoded in build.rs.
Note: --gpu flag is currently a scaffold (prints warning, uses CPU). Full GPU
      path needs to be wired in after CUDA build is confirmed working on Windows.

---

## Next Research Steps

### Immediate (data in hand)
1. Open analysis/fss.ipynb — run all cells
   Key output: Binder cumulant crossing plot should show all N curves crossing near Tc=4.51
   Key output: Scaling collapse — all N curves collapsing onto one universal function

2. Open analysis/coarsening.ipynb — run all cells
   Key output: log-log plot of rho(t) with power-law fit z ~ 1/3

### Short-term (Windows GPU)
3. Confirm CUDA build compiles on Windows
4. Wire actual GPU sweep into --gpu flag in src/bin/fss.rs
5. Rerun FSS with N=8,12,16,20,24,28,32,40 — larger sizes for better exponents

### Research (after GPU validated)
6. Run coarsening at multiple N (20, 30, 40, 50) — check z is N-independent
7. Run coarsening at multiple T_quench values below Tc — check z doesn't depend on T
8. FSS at higher statistics: warmup=5000, samples=2000 for publication quality

### Future directions discussed
- ML phase detection: train CNN on spin snapshots, unsupervised Tc detection
  Reference: Carrasquilla & Melko, Nature Physics 13, 431 (2017)
- Finite-size scaling with N=64 — needs GPU
- Kibble-Zurek mechanism: quench rate dependence of defect density
- Frustrated magnets: triangular lattice J<0, spin liquid behaviour
- Mesh/graph geometry — arbitrary connectivity, real crystal structures (see mesh design doc)

---

## Key Commands Reference

# Build
cargo build --release

# Single sweep (test)
cargo run --release --bin sweep -- --n 20 --geometry cubic

# FSS (CPU, medium quality)
cargo run --release --bin fss -- \
  --sizes 8,12,16,20,24,28 --wolff \
  --warmup 2000 --samples 1000 \
  --tmin 3.5 --tmax 5.5 --steps 41 \
  --outdir analysis/data

# FSS (CPU, publication quality — slow)
cargo run --release --bin fss -- \
  --sizes 8,12,16,20,24,28,32,40 --wolff \
  --warmup 5000 --samples 2000 \
  --tmin 3.5 --tmax 5.5 --steps 61 \
  --outdir analysis/data

# Coarsening
cargo run --release --bin coarsening -- \
  --n 30 --t-quench 2.5 \
  --steps 200000 --sample-every 10 \
  --outdir analysis/data

# Open notebooks
cd analysis && jupyter notebook

# GPU build (Windows only)
cargo build --release --features cuda

---

## Why Mesh is Easy to Add (Architecture Note)

The current `Lattice` struct in `src/lattice.rs` already stores connectivity as:

    pub neighbours: Vec<Vec<usize>>

This is a generic adjacency list — not hardcoded to a grid. The three existing
geometries (Square2D, Triangular2D, Cubic3D) are just different ways of filling
this list at construction time. The Metropolis and Wolff algorithms never assume
a regular grid — they only ever iterate over neighbours[idx].

This means adding mesh support requires:
  1. A new Geometry variant: Geometry::Mesh
  2. A constructor that accepts an edge list (node pairs) instead of N
  3. A file format for loading graphs (JSON adjacency list or edge CSV)
  4. CLI flag: --graph path/to/graph.json

The simulation engine, observables, FSS, coarsening — all unchanged.

Design doc at: ~/ising-mesh-design.md

---

## Known Issues / Limitations

1. alpha (heat capacity exponent) is unreliable at N<50
   Reason: Cv divergence is very weak (alpha=0.11), finite-size effects dominate
   Fix: FSS peak scaling gives alpha/nu ratio, which is more reliable than direct fit

2. chi computation was fixed 2026-03-06
   Bug: was using |M| variance instead of signed M variance — suppressed chi peak
   Fix: committed in 9242c80

3. --gpu flag is a stub
   Currently prints a warning and falls back to CPU
   Fix needed: wire LatticeGpu into fss.rs and coarsening.rs sweep loops

4. Wolff algorithm does not support J<=0 or h!=0
   Falls back to Metropolis in these cases
   For frustrated magnets (J<0) use --no-wolff flag

5. Scaling collapse objective function is simple (sum of squared diffs of sorted y)
   May give poor nu fits if chi curves are noisy near Tc
   Fix: interpolate onto common x-grid before minimising
