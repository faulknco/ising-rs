# GPU FSS Analysis Plan for N <= 192

## Goal

Use the new `N <= 192` GPU runs to turn the current GPU benchmark story from
"good Tc estimates" into a defensible finite-size-scaling result set.

The priority is not to add more plots. The priority is to separate:

- trustworthy benchmark quantities
- quantities that are currently analysis-limited
- quantities that need a rerun rather than a new fit

## Current Status

### Already strong

- `Tc` from Binder crossings for all three models
- raw observable curves (`E`, `M`, `Cv`, `chi`)
- XY exponent story overall
- Heisenberg `gamma/nu` and `beta/nu`

### Still weak

- Ising `beta/nu`
- Heisenberg `nu` from the current collapse metric
- the current WHAM implementation

## Main Diagnosis

The weak numbers are mostly analysis problems, not obvious Monte Carlo failure.

### 1. `--method single` is acceptable for Tc, but not ideal for all exponents

`analysis/scripts/analyze_gpu_fss.py` defaults to:

```bash
python analysis/scripts/analyze_gpu_fss.py --method single
```

The single-histogram path is good enough to sharpen peak locations and Binder
crossings, but it is noticeably less stable for:

- `M(Tc)` fits
- large-size magnetisation interpolation
- exponent extraction that depends on a single fitted `Tc`

### 2. The current WHAM path is numerically unstable

The multiple-histogram path still produces runtime warnings in the weight
application step. Until that is fixed, it should not be the default or the
headline analysis path.

### 3. The collapse objective is too naive

The current `collapse_quality()` routine minimizes adjacent-point roughness of
the sorted collapsed curve. That objective is too easy to game by shrinking or
stretching the x-axis, especially for continuous-spin models.

This is the main reason the current Heisenberg `nu` result is not trustworthy.

### 4. Small sizes should not drive the final exponent fit

The `L = 8, 16` points are useful to show finite-size drift, but they should not
carry the final asymptotic exponent fit.

For the next pass, the main fit window should be:

- `L = 32, 64, 128, 192`

and the smaller sizes should be retained only for:

- observables plots
- Binder-drift plots
- sanity checks

## Analysis Rules for the N <= 192 Dataset

### Rule 1. Keep all timeseries

Do not throw away the per-replica timeseries for the new `N = 192` runs.

We need them for:

- reweighting
- blocking / autocorrelation analysis
- method-to-method comparison

### Rule 2. Fit exponents on large sizes first

For each model, compute:

- all-size fit: `8,16,32,64,128,192`
- medium/large fit: `16,32,64,128,192`
- large-only fit: `32,64,128,192`

The large-only fit should be treated as the primary estimate.

### Rule 3. Separate Tc extraction from exponent extraction

Use Binder crossings to determine `Tc`, but do not assume the same interpolation
path is optimal for all exponents.

Recommended split:

- `Tc`: Binder crossings from reweighted curves
- `gamma/nu`: susceptibility peak scaling
- `beta/nu`: magnetisation at fixed `Tc`, but only after checking large-size stability
- `nu`: Binder-slope or derivative-based estimator first, collapse second

### Rule 4. Treat collapse as a cross-check, not the only nu estimator

The next `nu` estimate should come from at least two methods:

- Binder-slope scaling, e.g. `dU/dT | Tc ~ L^(1/nu)`
- scaling collapse

If those disagree materially, report the disagreement instead of choosing one.

### Rule 5. Mask unstable Binder tails

When `M2` becomes very small, `U = 1 - M4 / (3 M2^2)` becomes numerically noisy.

The analysis should exclude temperatures where:

- `M2` is below a stability threshold, or
- the Binder curve leaves the physically expected range in an obvious way

This matters most for the largest Heisenberg and XY sizes.

## Immediate Code Tasks

### Task 1. Add fit-range control to `analyze_gpu_fss.py`

Add CLI options such as:

- `--fit-sizes 32,64,128,192`
- `--collapse-sizes 32,64,128,192`

This should be the first change. Right now the script hardwires "use all sizes".

### Task 2. Add a Binder-slope `nu` estimate

Compute `dU/dT` near `Tc` for each size and fit:

```text
dU/dT | Tc ~ L^(1/nu)
```

This gives a more defensible `nu` estimate than the current collapse heuristic
alone.

### Task 3. Stabilize the WHAM path

Before using WHAM as the main path:

- inspect the log-weight normalization
- clamp or reject invalid weight vectors
- explicitly check finite weights before matrix products
- fail closed instead of silently producing invalid moments

### Task 4. Add method-spread reporting

For each model, report exponents from:

- summary-only fit
- single-histogram reweighting
- WHAM reweighting once stable
- large-size-only fit

The spread across methods should be treated as a systematic uncertainty estimate.

### Task 5. Add a benchmark report table

For each model, write one table with:

- `Tc`
- `gamma/nu`
- `beta/nu`
- `nu`
- theory value
- absolute / relative error
- fit sizes used
- method used

This is the table that should drive the paper, not whichever plot looks best.

## Recommended Interpretation for the Next Runs

### Ising

Goal:

- confirm that `beta/nu` recovers when using large-size-only fits and improved reweighting

Expected outcome:

- `Tc` remains excellent
- `gamma/nu` remains good
- `beta/nu` should move closer to theory once the `M(Tc)` interpolation is cleaned up

### Heisenberg

Goal:

- verify that the low `nu` is a collapse-method artifact

Expected outcome:

- `Tc` remains excellent
- `gamma/nu` and `beta/nu` remain good
- Binder-slope `nu` should be more trustworthy than the current collapse-only value

### XY

Goal:

- use it as the reference case for the continuous-spin pipeline

Expected outcome:

- it should remain the cleanest full benchmark set
- if XY degrades with `N = 192`, that is a warning sign for the analysis path itself

## Minimum Acceptance Criteria

Call the GPU FSS benchmark mature only if:

- all three models keep accurate `Tc`
- `gamma/nu` is within about `5%` for all three
- `beta/nu` is within about `5%` for all three on the primary large-size fit
- `nu` is supported by at least two analysis methods
- the chosen method is not numerically unstable

If one metric fails but the rest are strong, report it as an open limitation.

## Recommended Order

1. Finish the `N = 192` runs with timeseries retained.
2. Add fit-range controls to `analyze_gpu_fss.py`.
3. Add Binder-slope `nu`.
4. Stabilize WHAM.
5. Recompute the benchmark table using `L = 32, 64, 128, 192` as the primary fit set.
6. Only then update the manuscript claims.
