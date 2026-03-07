# ising-rs

GPU-accelerated Monte Carlo simulation of the Ising model on arbitrary graph topologies, implemented in Rust with CUDA.

[![CI](https://github.com/faulknco/ising-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/faulknco/ising-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**[Live WebAssembly Demo →](https://faulknco.github.io/ising-rs)**

## Overview

A computational physics toolkit for studying the Ising model via Monte Carlo simulation. Supports standard cubic, BCC, FCC, diluted, and complex network lattices through a graph-agnostic architecture. Includes a full finite-size scaling (FSS) analysis pipeline with Ferrenberg-Swendsen histogram reweighting and jackknife error estimation.

### Key results (3D cubic lattice)

| Quantity | Measured | Theory | Error |
|---|---|---|---|
| T_c (Binder crossing) | 4.512(4) | 4.5115 | 0.01% |
| gamma/nu (chi peak scaling) | 1.933(30) | 1.964 | 1.6% |
| beta/nu (M at T_c) | 0.492(26) | 0.518 | 5.0% |
| nu (chi collapse) | 0.667(37) | 0.630 | 5.9% |
| 2beta/nu + gamma/nu | 2.918 | 3.0 | 2.7% |

## Features

- **Wolff cluster algorithm** — eliminates critical slowing down (z_W ~ 0.3 vs z ~ 2 for Metropolis)
- **Metropolis algorithm** — single-spin updates with GPU-accelerated checkerboard decomposition
- **Histogram reweighting** — Ferrenberg-Swendsen patchwork scheme for fine temperature interpolation near T_c
- **Arbitrary graph topologies** — cubic, BCC, FCC, bond-diluted, and custom edge-list graphs
- **Finite-size scaling** — automated T_c extraction, exponent fitting, and scaling collapse
- **Kibble-Zurek mechanism** — defect density scaling from nonequilibrium quench simulations
- **Domain coarsening** — Allen-Cahn domain wall density decay after thermal quench
- **Exchange coupling fitting** — connect simulation to experimental Curie temperatures (BCC Fe, FCC Ni)
- **WebAssembly demo** — interactive 3D visualisation in the browser (Three.js)
- **45 tests** — 38 unit tests + 7 integration tests covering all binaries

## Quick start

```bash
# Build
cargo build --release

# Run a temperature sweep (prints CSV to stdout)
cargo run --release --bin sweep -- --n 16 --geometry cubic

# Finite-size scaling with Wolff algorithm (writes CSV files)
cargo run --release --bin fss -- --wolff --sizes 8,12,16,20,24 \
    --tmin 3.5 --tmax 5.5 --steps 41 --warmup 2000 --samples 2000 \
    --outdir analysis/data

# High-stats run for histogram reweighting near T_c
cargo run --release --bin fss -- --wolff --raw --sizes 16,20,24,32,40,48 \
    --tmin 4.30 --tmax 4.70 --steps 41 --warmup 5000 --samples 10000 \
    --outdir analysis/data/hires

# Kibble-Zurek quench experiment
cargo run --release --bin kz -- --n 20 --trials 10 \
    --tau-min 100 --tau-max 100000 --tau-steps 20

# Domain coarsening after quench
cargo run --release --bin coarsening -- --n 30 --t-quench 2.5 --steps 200000

# Temperature sweep on arbitrary graph
cargo run --release --bin mesh_sweep -- --graph analysis/graphs/bcc_N12.json --j 1.0

# Run tests
cargo test

# Clippy
cargo clippy -- -D warnings
```

All binaries support `--help` for full option listings.

### GPU acceleration (optional)

Requires CUDA 12.x and the `cuda` feature:

```bash
cargo run --release --features cuda --bin fss -- --gpu --sizes 8,12,16,20,24
```

### WebAssembly demo

```bash
cargo install wasm-pack
wasm-pack build --target web --out-dir www/pkg
cd www && python3 -m http.server 8080
```

## Project structure

```
src/
  lib.rs              # Library root
  cli.rs              # Shared CLI argument parsing, validation, --help
  lattice.rs          # Lattice construction (2D/3D, PBC, arbitrary graphs)
  metropolis.rs       # Metropolis single-spin updates
  wolff.rs            # Wolff cluster algorithm
  observables.rs      # Energy, magnetisation, Cv, chi, Binder cumulant
  graph.rs            # Graph loading (CSV edge lists, JSON adjacency)
  fitting.rs          # Critical exponent fitting (OLS on log-log data)
  coarsening.rs       # Domain coarsening after thermal quench
  kibble_zurek.rs     # KZ quench experiments
  sweep.rs            # Temperature sweep driver
  fss.rs              # Finite-size scaling driver
  wasm.rs             # WebAssembly bindings
  cuda/               # GPU kernels (optional, requires --features cuda)
  bin/
    sweep.rs          # CLI: single temperature sweep
    fss.rs            # CLI: FSS across multiple lattice sizes
    kz.rs             # CLI: Kibble-Zurek experiments
    coarsening.rs     # CLI: domain coarsening quench
    mesh_sweep.rs     # CLI: sweep on arbitrary graph files

tests/
  cli.rs              # Integration tests for all 5 binaries

analysis/
  fss.ipynb           # FSS notebook: observables, Binder, reweighting, collapse
  validation.ipynb    # Validation: Onsager, exact enumeration, autocorrelation
  graphs/             # Pre-built crystal graphs (BCC, FCC)
  pub_style.py        # Publication-quality matplotlib style

scripts/
  run_fss_publication.sh    # Publication-quality FSS runs
  run_kz_publication.sh     # Publication-quality KZ runs
  run_jfit_publication.sh   # J-fitting sweep runs
  run_all_publication.sh    # Run everything

paper/
  draft.tex           # Paper: methods, validation, results
  references.bib      # Bibliography
  figures/            # Publication figures
```

## Analysis pipeline

The FSS analysis uses a two-tier data strategy:

1. **Full-range sweep** (T = 3.5-5.5, 2000 samples) — overview of all observables
2. **High-stats narrow range** (T = 4.30-4.70, 10000 samples) — histogram reweighting and exponent extraction

The Jupyter notebook `analysis/fss.ipynb` processes raw per-sample data through:
- Jackknife error estimation (20 blocks) on all observables
- Binder cumulant crossing for T_c
- Ferrenberg-Swendsen histogram reweighting on a 200-point fine temperature grid
- Peak scaling (chi_max, M(T_c), dU/dT) for gamma/nu, beta/nu, 1/nu
- Scaling collapse with adjacent-point cost optimisation

## Citing

If you use this software, please cite:

```bibtex
@software{faulkner2026ising,
  author = {Faulkner, Connor},
  title = {ising-rs: GPU-accelerated Monte Carlo simulation of the Ising model},
  year = {2026},
  url = {https://github.com/faulknco/ising-rs},
  license = {MIT}
}
```

See also [CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE)
