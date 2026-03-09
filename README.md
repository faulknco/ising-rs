# ising-rs

Rust Monte Carlo toolkit for classical spin-model research, with a validated Ising baseline, graph-based crystal workflows, and optional CUDA acceleration for cubic-lattice Metropolis runs.

[![CI](https://github.com/faulknco/ising-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/faulknco/ising-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**[Live WebAssembly Demo ->](https://faulknco.github.io/ising-rs)**

## Scope

`ising-rs` is being cleaned up into a reproducible research engine. The immediate goal is to make the classical physics workflows trustworthy and repeatable before extending the codebase toward new models.

The current codebase includes:

- CPU Ising Monte Carlo on regular lattices and loaded graph topologies
- Wolff and Metropolis dynamics on the CPU
- Cubic-lattice CUDA Metropolis kernels
- analysis notebooks and helper scripts for FSS, validation, exchange fitting, dilution, coarsening, Kibble-Zurek, and Heisenberg studies

## Capability Matrix

### Validated baseline

- 2D and 3D classical Ising model on CPU (Wolff + Metropolis)
- CPU workflows for cubic lattices and loaded graph topologies
- graph loading for BCC, FCC, diluted, and custom edge-list inputs
- core observables: energy, magnetisation, Binder cumulant, heat capacity, susceptibility
- CLI and library test coverage via `cargo test`

### GPU FSS workflow (current branch)

- CUDA checkerboard Metropolis with parallel tempering for Ising, Heisenberg, and XY on 3D cubic lattices
- GPU-side observable reduction kernels for the publication FSS path
- scripted Windows pipeline: build -> validate -> smoke test -> publication run -> analysis
- committed summary CSVs, analysis figures, and a reproduction guide for the current GPU branch
- histogram reweighting support in `analysis/scripts/analyze_gpu_fss.py`

### Available but not yet fully packaged as reproducible research outputs

- BCC/FCC exchange-fitting workflows
- dilution studies with multi-realization averaging and propagated `T_c(p)` errors
- coarsening workflows
- Kibble-Zurek workflows with explicit ramp/freeze controls and uncertainty-aware sweep output

### Current backend limits

- GPU support is currently focused on 3D cubic-lattice finite-size-scaling workflows
- arbitrary graph workflows are CPU-first; not every analysis path is graph-native
- GPU Wolff cluster updates are not implemented

## Repository Goals

The repo is moving toward three explicit guarantees:

1. Engine correctness: algorithms and observables match the documented methods.
2. Research reproducibility: published numbers and figures can be regenerated from versioned code and inputs.
3. Physics credibility: validated benchmark physics is separated from exploratory workflows.

See [reproducibility.md](/Users/faulknco/Projects/ising-rs/docs/reproducibility.md) and [physics-validation.md](/Users/faulknco/Projects/ising-rs/docs/physics-validation.md).
Fresh-machine setup is documented in [setup.md](/Users/faulknco/Projects/ising-rs/docs/setup.md).

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
# Build GPU binary
cargo build --release --features cuda --bin gpu_fss

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
Performance depends on model, size, and measurement cadence; use
`analysis/REPRODUCIBILITY.md` for the current benchmark workflow and committed
example outputs.

See [analysis/REPRODUCIBILITY.md](analysis/REPRODUCIBILITY.md) for detailed
reproduction steps, parameters, and expected results.

## Research Layout

```text
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
  heisenberg/         # Heisenberg workflows
  cuda/               # cubic-lattice CUDA backend
  bin/                # CLI entrypoints

tests/
  cli.rs              # Integration tests for all 5 binaries

analysis/
  graphs/             # versioned graph inputs
  specs/              # run/manifest schemas
  scripts/            # scripted analysis workflows
  data/               # generated data products

docs/
  reproducibility.md  # workflow and data provenance rules
  physics-validation.md

scripts/
  run_fss_publication.sh    # Publication-quality FSS runs
  run_kz_publication.sh     # Publication-quality KZ runs
  run_jfit_publication.sh   # J-fitting sweep runs
  run_all_publication.sh    # Run everything

paper/
  draft.tex           # working manuscript

results/
  published/          # versioned result packs
```

## Reproducibility Direction

The repo is transitioning away from "notebook-only" publication workflows.

The target state is:

- raw simulation outputs with run manifests
- scripted derivation of tables and figures
- versioned result packs under `results/published/`
- a benchmark validation page with theory targets and current status

The manifest schema scaffold lives at [run-manifest.schema.json](/Users/faulknco/Projects/ising-rs/analysis/specs/run-manifest.schema.json).

The current scripted validation entrypoint is:

```bash
python analysis/scripts/reproduce_validation.py --quick
```

For a fresh-machine baseline rebuild, use the single entrypoint instead:

```bash
python analysis/scripts/reproduce_classical_baseline.py --quick
```

That workflow now writes:

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

The dilution workflow is now scripted as well:

```bash
python analysis/scripts/reproduce_dilution.py --quick
```

It generates disorder realizations, runs the corresponding mesh sweeps, and writes
disorder-averaged observables plus `T_c(p)` summaries into `analysis/data/derived/dilution/`.
Use `--max-workers` to parallelize independent realizations on CPU without changing the seeded
Monte Carlo trajectories for each job.

For larger publishing runs, the repo also provides a USB-friendly multi-size campaign runner:

```bash
python analysis/scripts/reproduce_dilution_campaign.py \
  --output-root /path/to/publishing_data
```

It writes one folder per size, plus campaign-level CSV ledgers so partial results are preserved if
the run is interrupted.

Once at least one size has finished, you can derive Binder cumulants and pairwise crossing
temperatures from the campaign outputs:

```bash
python analysis/scripts/reproduce_dilution_fss.py \
  --campaign-root /path/to/publishing_data/dilution_publish_v1
```

`reproduce_dilution_campaign.py` now refreshes this Binder/FSS analysis automatically after each
completed size, so the campaign folder accumulates both raw per-job CSVs and higher-level finite-size
summaries as the run progresses.

When a campaign is ready to freeze into a stable artifact, promote it into `results/published/`:

```bash
python analysis/scripts/promote_dilution_result_pack.py \
  --campaign-root /path/to/publishing_data/dilution_publish_v1 \
  --pack-name dilution_publish_v1
```

The promoted pack includes campaign summaries, per-size derived outputs, figures, manifests, a pack
README, and SHA-256 checksums.

To automate the last two steps while a USB campaign is still running:

```bash
python analysis/scripts/finalize_dilution_campaign.py \
  --campaign-root /path/to/publishing_data/dilution_publish_v1 \
  --pack-name dilution_publish_v1
```

That watcher reruns the Binder/FSS analysis when new size summaries appear and promotes the final
campaign into `results/published/` once every planned size has completed.

The Kibble-Zurek analysis is also scriptable:

```bash
python analysis/scripts/reproduce_kz.py \
  --input-dir analysis/data \
  --output-dir analysis/data/derived/kz \
  --figures-dir analysis/figures/generated/kz
```

If you are using a virtualenv, activate it first. On Windows, `py -3` is the usual equivalent of
`python`.

The historical `kz_N20.csv` and `kz_N30.csv` files are still too weak to pass the default screening
thresholds, so the scripted KZ path is currently best treated as a reproducible analysis scaffold
rather than a validated final result.

## Current Caveat

Some historical CSV files and manuscript claims predate the reproducibility cleanup. Until the scripted result packs are rebuilt, treat the repo as:

- trustworthy for code exploration and benchmark development
- promising for classical research
- not yet fully packaged for end-to-end paper regeneration

## Contributing

Before submitting changes:

```bash
cargo fmt
cargo clippy -- -D warnings
cargo test
```

See [CONTRIBUTING.md](/Users/faulknco/Projects/ising-rs/CONTRIBUTING.md).

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
