# V2 Data Audit

This note audits the `analysis/v2` data bundle committed on the
`v2-experiment-data` branch.

## Scope

This audit focuses on the two most speculative parts of the dataset:

- Kibble-Zurek (KZ) scaling
- Domain coarsening after a quench

It does **not** re-audit the equilibrium validation and FSS tables.

## Reproducibility Note

The branch-level reproducibility note is not internally consistent:

- [analysis/v2/data/REPRODUCIBILITY.md](analysis/v2/data/REPRODUCIBILITY.md)
  says the KZ run should be `N=80`, but the committed files are
  `kz_N20.csv`, `kz_N40.csv`, and `kz_N50.csv`.
- The same note refers to `scripts/run_v2.sh`, but that file is not present
  on this branch tip.

So this branch should be treated as a historical data snapshot, not a
fully self-contained reproducibility branch.

## Kibble-Zurek Audit

The notebook states the expected KZ scaling for 3D Ising with Metropolis
dynamics as

`rho ~ tau_Q^{-d nu / (1 + z nu)} ~ tau_Q^{-0.86}`

with `d=3`, `nu~0.630`, `z~2`.

That expectation is encoded in
[analysis/v2/v2_analysis.ipynb](analysis/v2/v2_analysis.ipynb).

### Fits using the notebook's own rule

The notebook fits **all positive-rho points** in log-log space.

Resulting slopes:

- `N=20`: `-1.73 +/- 0.88`
- `N=40`: `-1.37 +/- 0.15`
- `N=50`: `-1.33 +/- 0.13`

Relative to the notebook's expected `-0.836`, that is roughly:

- `N=20`: `107%` error
- `N=40`: `64%` error
- `N=50`: `59%` error

### Interpretation

- `N=20` is not usable: it has only four positive-rho points.
- `N=40` and `N=50` show a floor / noisy tail at large `tau_Q`.
- A naive whole-range fit is too steep and does **not** support the claimed
  KZ benchmark.

If the fit is restricted to the pre-floor region only, the slope improves
somewhat, but it is still far from the notebook target.

### Verdict

The KZ dataset on this branch is **not publication-grade**.

At best, it shows a qualitative decrease of defect density with slower quenches.
It does **not** cleanly reproduce the expected KZ exponent.

## Coarsening Audit

The notebook claims Allen-Cahn late-time coarsening:

`rho(t) ~ t^{-1/2}`

and fits the **last 50% of the timeseries** to estimate the exponent.

That analysis is defined in
[analysis/v2/v2_analysis.ipynb](analysis/v2/v2_analysis.ipynb).

### Fits using the notebook's own rule

Late-time slopes from the notebook-style fit:

- `N=20`: `-0.060 +/- 0.087`
- `N=30`: `-0.00010 +/- 0.00327`
- `N=40`: `+0.00169 +/- 0.00215`
- `N=50`: `-0.00082 +/- 0.00153`

These are effectively flat, not `-1/2`.

### Interpretation

- The notebook's chosen late-time window is too late; the curves have already
  flattened toward a finite-size floor.
- There is some evidence of a more plausible **transient** power-law window
  earlier in the run, especially for `N=50`, but that is not the fit
  currently reported by the notebook.

### Verdict

The coarsening dataset, **under the notebook's own fit rule**, is also
**not publication-grade**.

It may still contain a usable transient regime, but the current analysis
does not demonstrate Allen-Cahn scaling.

## Overall Verdict

What seems usable from this branch:

- historical equilibrium/FSS data snapshot
- archival raw CSVs for re-analysis

What should **not** be treated as validated final results:

- KZ exponent extraction
- coarsening exponent extraction

## Recommended Treatment

Use this branch as:

- an archive of historical experiment outputs
- a source for re-analysis if needed

Do **not** use it as:

- a validated result pack
- a source for headline KZ or coarsening claims without rerunning and
  re-analyzing those workflows
