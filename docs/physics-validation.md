# Physics Validation

## Purpose

This page tracks which physics statements in `ising-rs` are currently treated as validated benchmarks, which are partially validated, and which remain exploratory.

It is intentionally conservative.

## Validation Status

### Benchmark baseline

| Area | Status | Notes |
|---|---|---|
| 2D Ising exact-solution comparison | In progress | Analysis notebook exists; needs scripted regeneration pack |
| Small-lattice exact enumeration | Scripted baseline exists | Included in the classical validation workflow and quick validation pack |
| 3D cubic Ising FSS on CPU Wolff path | Strongest current result | Best candidate for first fully reproducible pack |
| Core lattice/observable/unit tests | Passing locally | `cargo test` passed during repo audit |
| BCC/FCC graph construction | Plausible | Graph inputs exist and coordination counts match expectations |

### Case-study workflows

| Area | Status | Notes |
|---|---|---|
| BCC/FCC exchange fitting | Partially supported | Local data and notebooks exist; reproducible pack not complete |
| Bond dilution | Exploratory | Scripted multi-realization averaging and Binder-crossing FSS now exist; larger reruns and theory comparison are still pending |
| Kibble-Zurek | Exploratory | Backend protocol and scripted analysis now exist; larger reruns and stronger datasets are still missing |
| Coarsening | Exploratory | Useful workflow, but not yet packaged as benchmark physics |
| Heisenberg workflows | Active but not yet benchmarked here | Substantial code exists; should get its own validation track |
| Heisenberg anisotropy crossover (GPU) | Production campaign complete | 7 D values (-2..+2), sizes 16-128, 20k samples. Easy-axis/easy-plane symmetry breaking validated. Binder crossings extracted for D=0 (Tc=1.440). Component observables (Mz, Mxy) with jackknife errors. See `analysis/data/anisotropy_campaign_gpu_prod/` |

## Current Ground Truth

At present, the safest public claim is:

- `ising-rs` has a credible classical Ising baseline on CPU, with cubic FSS as the strongest validation path.

Claims that should not be treated as fully validated until rebuilt into result packs:

- final exchange-coupling fit numbers
- dilution universality claims
- Kibble-Zurek exponent claims
- any claim depending only on untracked notebook outputs

## What Counts as Validated

A physics result counts as validated in this repo only if:

1. the input data can be regenerated
2. the output data is versioned or reproducibly rebuildable
3. the analysis is scripted
4. the result is compared against theory or established literature
5. known caveats are documented

## First Validation Pack Targets

The first reproducible validation pack should cover:

- 2D Onsager comparison
- exact enumeration sanity check
- fluctuation-dissipation consistency
- 3D cubic FSS with theory comparison
- one multi-size dilution benchmark with Binder crossings and explicit disorder caveats

The current scripted validation entrypoint is:

```bash
python analysis/scripts/reproduce_validation.py --quick
```

It writes:

- `analysis/data/derived/validation/validation_metrics.csv`
- `analysis/data/derived/validation/validation_overview.csv`
- `analysis/data/derived/validation/validation_summary.json`
- comparison tables in `analysis/data/derived/validation/`
- figures in `analysis/figures/generated/validation/`
- manifests in `analysis/data/manifests/validation/`

The first frozen baseline pack now exists at:

- `results/published/classical_validation_quick_v1/`

That pack is still a quick-mode artifact, so treat it as a regression baseline rather than final
publication evidence.

## Literature Anchors

These are the current benchmark anchors for the classical validation story:

- Onsager 2D Ising exact solution
- Wolff cluster dynamics literature
- Ferrenberg-Swendsen histogram reweighting
- modern 3D Ising FSS literature such as Hasenbusch

## Maintenance Rule

This page should be updated whenever:

- a result pack is added
- a benchmark is rerun with new parameters
- a claim is downgraded because reproducibility is missing
- a previously exploratory workflow becomes validated
