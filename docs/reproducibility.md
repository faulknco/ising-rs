# Reproducibility

## Purpose

This document defines how `ising-rs` should produce research outputs that can be regenerated and audited later.

The rule is simple: if a number appears in the paper or README, the repo should be able to show where it came from.

## Principles

1. Raw outputs are immutable.
2. Derived outputs are script-generated.
3. Every run has a manifest.
4. Every published figure has a provenance trail.
5. Notebooks are exploratory, not the sole source of record.

## Data Classes

### Raw

Simulation outputs written directly by a CLI or batch script.

Examples:

- temperature sweeps
- raw Wolff time series
- Kibble-Zurek sweep outputs with `rho_err` and `n_trials`
- disorder-realization outputs

### Derived

Tables created from raw outputs by deterministic scripts.

Examples:

- Binder crossing summaries
- exponent-fit tables
- disorder-averaged `T_c(p)` tables
- KZ fit parameter tables

### Figures

Plots generated from derived data or directly from raw data through a scripted pipeline.

### Published packs

Versioned bundles under `results/published/` containing the exact materials behind a result set.

## Standard Layout

```text
analysis/
  specs/
    run-manifest.schema.json
  data/
    raw/
    derived/
    manifests/
  figures/
    generated/
  scripts/

results/
  published/
    <pack-name>/
      README.md
      raw/
      derived/
      figures/
      manifests/
```

## Run Manifest Requirements

Each raw run should emit a manifest next to the data or into `analysis/data/manifests/`.

Minimum required fields:

- `schema_version`
- `run_id`
- `created_at`
- `git_commit`
- `binary`
- `subcommand_args`
- `rng`
- `backend`
- `model`
- `geometry`
- `input_files`
- `output_files`
- `environment`

The canonical schema lives at [run-manifest.schema.json](/Users/faulknco/Projects/ising-rs/analysis/specs/run-manifest.schema.json).

## Publication Rule

A result is considered publication-ready only if all of the following are true:

- the raw inputs exist
- the run manifests exist
- the analysis step is scripted
- the final table/figure is reproducible without manual notebook editing
- the result pack documents known limitations

## Notebook Rule

Notebooks may be used for:

- exploration
- interactive debugging
- trying fit variants

Notebooks should not be the only way to:

- regenerate a paper figure
- compute a quoted critical exponent
- produce a final table

## Provenance Rule

Every paper figure should be attributable to:

1. a result pack
2. one or more raw files
3. one derivation script
4. one git commit

## Suggested Workflow

1. Run the simulation CLI into `analysis/data/raw/<run-set>/`.
2. Emit a manifest for each run or batch.
3. Run an analysis script that writes into `analysis/data/derived/<run-set>/`.
4. Generate figures into `analysis/figures/generated/<run-set>/`.
5. Promote stable outputs into `results/published/<pack-name>/`.

## Fresh-Machine Entry Point

For the baseline CPU validation story, the repo now provides a single cross-platform command:

```bash
python analysis/scripts/reproduce_classical_baseline.py --quick
```

The recommended first step on a new machine is still the bootstrap script documented in
[setup.md](/Users/faulknco/Projects/ising-rs/docs/setup.md):

```bash
python scripts/bootstrap_analysis.py --verify
```

That command:

1. runs the scripted validation workflow
2. regenerates the comparison tables and figures
3. emits a validation manifest
4. promotes the result into a versioned pack unless `--skip-promotion` is used

## Current Status

The GPU FSS pipeline is the first fully scripted, reproducible workflow:

- **Pipeline**: `scripts/run_gpu_windows_pipeline.py` (build → test → publication runs)
- **Analysis**: `analysis/scripts/analyze_gpu_fss.py` (reweighting, Binder crossing, peak scaling, collapse)
- **Data**: summary CSVs committed, timeseries regenerable via pipeline (~4 hours on RTX 2060)
- **Guide**: `analysis/REPRODUCIBILITY.md` with exact parameters and expected results

The repo is still in transition overall. Historical notebook-driven CPU results still exist, but the target state is scripted and manifest-backed across all workflows.
