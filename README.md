# ising-rs

Rust + CUDA Monte Carlo toolkit for classical spin-model research on arbitrary graph topologies, targeting publication in Physical Review E.

[![CI](https://github.com/faulknco/ising-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/faulknco/ising-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**[Live WebAssembly Demo ->](https://faulknco.github.io/ising-rs)** | **[Interactive Showcase](https://faulknco.github.io/ising-rs/showcase.html)** | **[Figure Gallery](docs/gallery.md)** | **[ELI5](docs/ELI5.md)**

## Overview

`ising-rs` is a reproducible research engine for Monte Carlo simulation of classical spin models. It supports three universality classes (Ising, Heisenberg, XY) on CPU and GPU, with a full finite-size scaling analysis pipeline including Ferrenberg-Swendsen histogram reweighting and jackknife error estimation.

### Key results (3D cubic lattice)

| Quantity | Measured | Theory | Error |
|---|---|---|---|
| T_c (Binder crossing) | 4.512(4) | 4.5115 (Hasenbusch 2010) | 0.01% |
| 2D T_c (Onsager) | 2.269 | 2.2692 (exact) | <0.02% |
| KZ exponent | ~0.25 | 0.279 (theory) | ~10% |
| Fe Curie temp | 1043 K | 1043 K (experiment) | <1% |
| Ni Curie temp | ~631 K | ~627 K (experiment) | ~1% |

## Features

- **Wolff cluster algorithm** -- eliminates critical slowing down (z_W ~ 0.3 vs z ~ 2 for Metropolis)
- **Metropolis algorithm** -- single-spin updates with GPU-accelerated checkerboard decomposition
- **Three universality classes** -- Ising (Z2), Heisenberg O(3), and XY O(2) models
- **GPU parallel tempering** -- CUDA checkerboard Metropolis with replica exchange for all three models
- **Histogram reweighting** -- Ferrenberg-Swendsen patchwork scheme for fine temperature interpolation near T_c
- **Arbitrary graph topologies** -- cubic, BCC, FCC, bond-diluted, and custom edge-list graphs
- **Finite-size scaling** -- automated T_c extraction, exponent fitting, and scaling collapse
- **Kibble-Zurek mechanism** -- defect density scaling from nonequilibrium quench simulations
- **Domain coarsening** -- Allen-Cahn domain wall density decay after thermal quench
- **Exchange coupling fitting** -- connect simulation to experimental Curie temperatures (BCC Fe, FCC Ni)
- **WebAssembly demo** -- interactive 3D visualisation in the browser (Three.js)
- **87 tests** -- 74 unit tests + 13 integration tests

## Quick Start

```bash
# Bootstrap the analysis environment and verify the baseline
python scripts/bootstrap_analysis.py --verify

# Build
cargo build --release

# Run tests
cargo test

# Lint
cargo clippy -- -D warnings

# Cubic Ising sweep on CPU
cargo run --release --bin sweep -- --n 16 --geometry cubic

# Cubic finite-size scaling on CPU with Wolff dynamics
cargo run --release --bin fss -- --wolff \
  --sizes 8,12,16,20,24 \
  --tmin 3.5 --tmax 5.5 --steps 41 \
  --warmup 2000 --samples 2000 \
  --outdir analysis/data

# Raw Wolff time series for histogram reweighting
cargo run --release --bin fss -- --wolff --raw \
  --sizes 16,20,24,32,40,48 \
  --tmin 4.30 --tmax 4.70 --steps 41 \
  --warmup 5000 --samples 10000 \
  --outdir analysis/data/hires

# Kibble-Zurek quench experiment
cargo run --release --bin kz -- --n 20 --trials 10 \
  --tau-min 100 --tau-max 100000 --tau-steps 20

# Domain coarsening after quench
cargo run --release --bin coarsening -- --n 30 --t-quench 2.5 --steps 200000

# Sweep an arbitrary graph loaded from JSON
cargo run --release --bin mesh_sweep -- \
  --graph analysis/graphs/bcc_N8.json \
  --tmin 5.0 --tmax 8.0 --steps 21

# Rebuild the baseline validation pack (quick mode)
python analysis/scripts/reproduce_classical_baseline.py --quick
```

All binaries support `--help` for full option listings.

### GPU-Accelerated FSS with Parallel Tempering

Requires CUDA 12.x, an NVIDIA GPU, and the `cuda` feature.

```bash
# Build GPU binaries
cargo build --release --features cuda --bin gpu_fss --bin gpu_jfit

# Run all three universality classes (Ising, Heisenberg, XY)
python scripts/run_gpu_windows_pipeline.py --publish-on-success

# Or run a single model manually
cargo run --release --features cuda --bin gpu_fss -- \
  --model ising \
  --sizes 8,16,32,64,128 \
  --tmin 4.40 --tmax 4.62 \
  --replicas 32 \
  --warmup 5000 --samples 100000 \
  --exchange-every 10 \
  --measure-every 5 \
  --outdir analysis/data/gpu_windows_pipeline/publication

# Run FSS analysis with histogram reweighting
python analysis/scripts/analyze_gpu_fss.py
```

GPU features: checkerboard Metropolis, parallel tempering (replica exchange),
GPU-side observable reduction for the FSS path, and pre-allocated buffers.

See [analysis/REPRODUCIBILITY.md](analysis/REPRODUCIBILITY.md) for detailed
reproduction steps, parameters, and expected results.

## Repository Layout

```text
src/
  lib.rs                # library root
  cli.rs                # shared CLI argument parsing, validation, --help
  lattice.rs            # lattice construction (2D/3D, PBC, arbitrary graphs)
  metropolis.rs         # CPU Metropolis single-spin updates
  wolff.rs              # CPU Wolff cluster algorithm
  observables.rs        # energy, magnetisation, Cv, chi, Binder cumulant
  graph.rs              # graph loading from CSV edge lists and JSON adjacency
  fitting.rs            # critical exponent fitting (OLS on log-log data)
  fss.rs                # CPU finite-size scaling driver
  sweep.rs              # temperature sweep driver
  coarsening.rs         # domain coarsening after thermal quench
  kibble_zurek.rs       # KZ protocol, CPU driver, and error helpers
  parallel_tempering.rs # replica exchange framework
  wasm.rs               # WebAssembly bindings (wasm-bindgen)
  heisenberg/           # Heisenberg O(3) model: FSS, Metropolis, overrelaxation, observables
  xy/                   # XY O(2) model: FSS, Wolff, observables
  cuda/                 # cubic-lattice CUDA backend (checkerboard Metropolis, PT)
  bin/
    sweep.rs            # CLI: single temperature sweep
    fss.rs              # CLI: Ising FSS across multiple lattice sizes
    kz.rs               # CLI: Kibble-Zurek quench experiments
    coarsening.rs       # CLI: domain coarsening after quench
    mesh_sweep.rs       # CLI: sweep on arbitrary graph files
    gpu_fss.rs          # CLI: GPU FSS for Ising/Heisenberg/XY
    gpu_jfit.rs         # CLI: GPU exchange-coupling fitting
    heisenberg_fss.rs   # CLI: CPU Heisenberg FSS
    heisenberg_jfit.rs  # CLI: CPU Heisenberg J-fitting
    xy_fss.rs           # CLI: CPU XY FSS
    xy_jfit.rs          # CLI: CPU XY J-fitting

tests/
  cli.rs                # integration tests for all binaries

analysis/
  graphs/               # versioned graph inputs (BCC, FCC)
  specs/                # run/manifest schemas
  scripts/              # scripted analysis workflows
  data/                 # generated data products
  REPRODUCIBILITY.md    # GPU benchmark workflow and committed outputs

docs/
  ELI5.md               # plain-English project explainer
  gallery.md            # annotated figure gallery (32 figures)
  showcase.html         # interactive single-page demo with live WASM simulation
  reproducibility.md    # workflow and data provenance rules
  physics-validation.md # validation targets and current status
  setup.md              # fresh-machine setup guide

paper/
  draft.tex             # working manuscript (RevTeX 4-2)

results/
  published/            # versioned result packs
```

## Reproducibility

The repo uses scripted, reproducible workflows for all analysis paths.

The current scripted validation entrypoint is:

```bash
python analysis/scripts/reproduce_validation.py --quick
```

For a fresh-machine baseline rebuild, use the single entrypoint instead:

```bash
python analysis/scripts/reproduce_classical_baseline.py --quick
```

That workflow writes:

- derived comparison tables in `analysis/data/derived/validation/`
- generated figures in `analysis/figures/generated/validation/`
- a run manifest in `analysis/data/manifests/validation/`

To freeze the current validation outputs into a versioned published pack:

```bash
python analysis/scripts/promote_validation_result_pack.py \
  --pack-name classical_validation_quick_v1
```

The Kibble-Zurek CLI writes `tau_q,rho,rho_err,n_trials` and uses the same explicit protocol on
both backends: warm up at `t_start`, ramp to `t_end`, optionally snap-freeze, then measure.

The dilution workflow is also scripted:

```bash
python analysis/scripts/reproduce_dilution.py --quick
```

It generates disorder realizations, runs the corresponding mesh sweeps, and writes
disorder-averaged observables plus `T_c(p)` summaries into `analysis/data/derived/dilution/`.
Use `--max-workers` to parallelize independent realizations on CPU without changing the seeded
Monte Carlo trajectories for each job.

For larger publishing runs, the repo provides a multi-size campaign runner:

```bash
python analysis/scripts/reproduce_dilution_campaign.py \
  --output-root /path/to/publishing_data
```

The campaign accumulates both raw per-job CSVs and higher-level finite-size summaries as the run
progresses. When ready to freeze into a stable artifact:

```bash
python analysis/scripts/promote_dilution_result_pack.py \
  --campaign-root /path/to/publishing_data/dilution_publish_v1 \
  --pack-name dilution_publish_v1
```

The Kibble-Zurek analysis is also scriptable:

```bash
python analysis/scripts/reproduce_kz.py \
  --input-dir analysis/data \
  --output-dir analysis/data/derived/kz \
  --figures-dir analysis/figures/generated/kz
```

If you are using a virtualenv, activate it first. On Windows, `py -3` is the usual equivalent of
`python`.

The manifest schema lives at [analysis/specs/run-manifest.schema.json](analysis/specs/run-manifest.schema.json).

## Current Limitations

- GPU support is focused on 3D cubic-lattice FSS workflows; arbitrary graph workflows are CPU-only
- GPU Wolff cluster updates are not implemented
- Some historical CSV files predate the reproducibility cleanup; the scripted result packs are the authoritative outputs

## Contributing

Before submitting changes:

```bash
cargo fmt
cargo clippy -- -D warnings
cargo test
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use the software, cite:

```bibtex
@software{faulkner2026ising,
  author = {Faulkner, Connor},
  title = {ising-rs},
  year = {2026},
  url = {https://github.com/faulknco/ising-rs},
  license = {MIT}
}
```

## License

[MIT](LICENSE)
