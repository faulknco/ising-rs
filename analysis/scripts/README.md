# Analysis Scripts

This directory is reserved for deterministic analysis workflows that replace notebook-only publication steps.

Current scripts:

- `reproduce_classical_baseline.py` — one-command entrypoint that runs the scripted classical validation workflow and can promote the resulting pack
- `ising_stats.py` — shared jackknife/error-estimation helpers for Ising raw time-series analysis
- `reproduce_fss.py` — rebuild FSS observable tables with jackknife error bars from `fss_raw_N*.csv`
- `reproduce_validation.py` — regenerate the benchmark validation suite and machine-readable summary
- `promote_validation_result_pack.py` — snapshot the validation outputs into a versioned published pack
- `reproduce_dilution.py` — run multi-realization dilution sweeps and derive disorder-averaged observables and `T_c(p)` errors
- `reproduce_dilution_campaign.py` — run a multi-size dilution campaign, suitable for USB-backed publishing datasets
- `reproduce_dilution_fss.py` — compute Binder cumulants and crossing-based `T_c` estimates from a completed or partially completed dilution campaign
- `finalize_dilution_campaign.py` — watch a running campaign, refresh Binder/FSS outputs, and promote the final pack automatically
- `promote_dilution_result_pack.py` — snapshot a campaign into a versioned publishable pack with checksums
- `reproduce_kz.py` — analyse `kz_N*.csv` sweeps, fit Kibble-Zurek scaling curves, and write derived tables/figures

`reproduce_dilution.py` supports `--max-workers` so independent `mesh_sweep` jobs can run in
parallel on CPU. This changes wall-clock time, not the per-job seeds or estimators.

`reproduce_dilution_campaign.py` writes one subdirectory per size and keeps campaign-level CSV
ledgers so partial results survive interruptions. It also refreshes the campaign-level Binder/FSS
analysis after each completed size.

`reproduce_validation.py` now writes section-level comparison tables, generated figures, and a run
manifest in addition to the metric and summary outputs.

`reproduce_classical_baseline.py` is the cross-platform top-level entrypoint for rebuilding the
baseline CPU validation story on a fresh machine.

Planned scripts:

- `reproduce_jfit.py`
- `reproduce_heisenberg.py`

Use the project analysis environment when running these scripts:

```bash
python analysis/scripts/reproduce_fss.py --help
```

or install the dependencies from `analysis/requirements.txt`.

On Windows, use `py -3` if `python` does not point at the intended interpreter.

Until the remaining scripts exist, notebook outputs should be treated as exploratory rather than final publication artifacts.
