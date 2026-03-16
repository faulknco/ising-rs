# Phase 2 Research Plan: Anisotropy-Driven Crossover

## Strategic Question

What is the best next research phase once the benchmark, dilution, and GPU FSS
stories are reproducible and defensible?

## Recommendation

The next phase should be a focused classical-physics study of universality
crossover in the 3D continuous-spin models already supported by the engine.

The recommended first project is:

- uniaxial anisotropy in the 3D Heisenberg model
- with the option to extend later to easy-plane / XY crossover

This is the highest-leverage next step because it:

- builds directly on the validated cubic GPU FSS path
- avoids the heavy disorder-averaging cost of a second dilution campaign
- is more interesting than "more pure-model benchmarks"
- gives a clean bridge from classical benchmark models toward more realistic and
  eventually quantum-inspired Hamiltonians

## Core Physics Question

How does a tunable anisotropy term drive crossover between the Heisenberg,
Ising, and XY universality classes in 3D, and over what finite-size window does
the asymptotic behavior become visible?

## Proposed Model

Start from the classical 3D Heisenberg Hamiltonian on the cubic lattice and add
a uniaxial anisotropy term:

```text
H = -J sum_<ij> S_i . S_j - D sum_i (S_i^z)^2
```

Interpretation:

- `D = 0`: isotropic Heisenberg limit
- large positive `D`: easy-axis tendency, expected Ising-like crossover
- large negative `D`: easy-plane tendency, expected XY-like crossover

This gives one control parameter with two scientifically useful limits.

## Why This Direction Wins

### Option A: anisotropy crossover

Pros:

- uses the current GPU continuous-spin stack directly
- stays on regular lattices, where the code is strongest
- produces a new physics question rather than another validation exercise
- creates a natural bridge to crystal-field and materials-inspired work

Cons:

- requires Hamiltonian and observable extensions
- crossover analysis is more subtle than pure critical-point benchmarking

Risk: medium

### Option B: more disorder physics

Pros:

- ties into the dilution work already done
- disorder crossover is scientifically real and publishable

Cons:

- much more expensive because of disorder averaging
- weaker fit with the current GPU regular-lattice strength
- harder to iterate quickly on one workstation

Risk: medium-high

### Option C: jump straight to quantum

Pros:

- aligns with the long-term ambition

Cons:

- highest development risk
- easiest place to overextend before the analysis layer is mature
- would slow down near-term publication momentum

Risk: high

## Phase 2 Scope

This phase should stay narrow.

Primary target:

- 3D Heisenberg model with uniaxial anisotropy on the cubic lattice

Secondary extension only if the first target is working:

- explicit easy-plane crossover analysis compared to XY benchmark behavior

Out of scope for the first paper:

- arbitrary graphs
- disorder plus anisotropy in the same campaign
- direct quantum Hamiltonians
- materials fitting

## Required Engine Work

### 1. Hamiltonian support

Add anisotropy-aware energy updates for the Heisenberg path:

- CPU Metropolis
- GPU Metropolis
- overrelaxation where still valid or approximately useful

### 2. CLI and workflow support

Add anisotropy parameters to the Heisenberg FSS workflow:

- `--anisotropy-d`
- clear sign convention in docs and CSV metadata

### 3. Observables

Add or expose:

- total magnetization magnitude
- `M_z`
- in-plane magnetization magnitude
- Binder cumulants for the relevant order parameter
- susceptibility of the relevant order parameter

### 4. Reproducibility

Every anisotropy run should produce:

- raw timeseries
- summary CSV
- run manifest
- analysis outputs
- promoted result pack

## Validation Before Physics Claims

Before starting the real campaign, validate these limiting cases:

### Limit test 1: `D = 0`

Recover the current Heisenberg benchmark behavior within error bars.

### Limit test 2: large positive `D`

Show that the relevant order parameter behaves more Ising-like as size grows.

### Limit test 3: large negative `D`

Show that the relevant order parameter behaves more XY-like as size grows.

### Limit test 4: energy update parity

Confirm CPU and GPU energies / observables agree for the same seed envelope and
parameter set.

## First Research Campaign

### Goal

Map crossover behavior with enough resolution to identify:

- which observables drift first
- where Binder crossings move
- when exponent estimates become unstable
- where the large-size window begins to resemble the target universality class

### Suggested parameter sweep

Start with a small but informative grid:

- `D / J = -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0`

Then refine only where the crossover is interesting.

### Suggested sizes

Primary fit window:

- `L = 32, 64, 96, 128, 192`

Optional smaller drift points:

- `L = 16, 24`

### Observables to publish

- Binder crossing `T_c(D)`
- `gamma/nu`
- `beta/nu`
- `nu` from Binder slope
- order-parameter component ratios, especially `M_z / |M|`

### Acceptance criteria

Call the study mature only if:

- the limiting cases behave as expected
- the crossover trends are monotone and reproducible
- the main exponents are stable across at least two fit windows
- GPU and CPU parity checks pass on representative points

## Analysis Strategy

Use the same discipline as the benchmark phase:

- separate `T_c` extraction from exponent extraction
- use large-size fits as the primary estimate
- report method spread as a systematic uncertainty
- keep collapse plots as support, not sole evidence

For crossover specifically:

- track fit drift versus minimum size
- do not force a universality-class label too early
- treat crossover length-scale estimates as part of the result

## Deliverables

### Engine deliverables

- anisotropy-capable Heisenberg CPU path
- anisotropy-capable Heisenberg GPU path
- reproducible anisotropy FSS runner

### Science deliverables

- one anisotropy validation pack
- one crossover result pack
- one benchmark-to-crossover comparison table

### Paper deliverable

A focused paper with a claim like:

"A reproducible GPU-accelerated finite-size-scaling study of anisotropy-driven
crossover in the 3D classical Heisenberg model."

## Sequencing

1. finish current `L <= 192` GPU benchmark analysis -- DONE
2. add anisotropy term and limiting-case validation -- DONE (2026-03-09)
3. run a small pilot grid in `D` -- DONE (validation campaign 2026-03-12)
4. decide where crossover is sharpest -- DONE (production campaign 2026-03-12)
5. run the full large-size campaign only in the informative region -- DONE (V2 campaign 2026-03-12, N=192 completed 2026-03-14)
6. freeze a result pack -- PENDING
7. then decide whether the next extension is materials-inspired or quantum

## V2 Campaign Results (2026-03-12 to 2026-03-14)

Campaign: `analysis/data/anisotropy_campaign_gpu_v2/`
Parameters: 7 D values (-2,-1,-0.5,0,0.5,1,2), sizes 16/32/64/96/128, 16 replicas, 20k samples
N=192 added for D=0 with fine T grid (1.42-1.48, dT=0.004), completed 2026-03-14

### Critical Exponents (D=0 Isotropic Heisenberg)

| Exponent | All sizes (16-192) | 64+ restricted | Theory |
|----------|-------------------|----------------|--------|
| gamma/nu | 1.60 | 2.55 | 1.963 |
| beta/nu | 0.24 | 0.13 | 0.519 |
| nu | 1.00 | 0.85 | 0.711 |

N=192 substantially improved fits (gamma/nu: 1.12 -> 1.60, nu: 1.33 -> 1.00).
64+ restricted fits overshoot, indicating corrections to scaling are still present.

### Anisotropic D != 0 Summary

| D | Regime | Tc (Binder) | gamma/nu | beta/nu |
|---|--------|------------|----------|---------|
| -2.0 | easy-plane | 0.571 | 0.19 | 0.001 |
| -1.0 | easy-plane | 1.059 | 0.33 | 0.003 |
| -0.5 | easy-plane | 1.039 | 0.29 | 0.003 |
| 0.0 | isotropic | 1.437 | 1.60 | 0.237 |
| +0.5 | easy-axis | 0.909 | 0.009 | 0.000 |
| +1.0 | easy-axis | 0.981 | -0.008 | 0.000 |
| +2.0 | easy-axis | 0.807 | 0.018 | 0.000 |

Easy-axis: gamma/nu ~ 0, beta/nu ~ 0 — consistent with Ising-like crossover.
Easy-plane: weak ordering with small gamma/nu — XY-like crossover.

### Known Issues

- (128,192) Binder crossing missing for D=0 — T grids don't overlap between sizes
- D=0 exponents improved but not yet publication quality
- Easy-plane Binder crossings only found for largest size pairs

### Remaining Work for Publication Quality

1. Re-run N=128 D=0 with T grid matching N=192 (1.42-1.48) to get (128,192) crossing
2. Run N=192 for D!=0 values (at least D=+2, D=-2 for limiting cases)
3. Consider N=256 for D=0 to extend asymptotic window
4. Tighter T windows for easy-plane (D<0) to improve Binder crossings
5. Systematic error analysis (autocorrelation time, thermalization checks)

## Success Metrics

- the engine supports anisotropy on CPU and GPU without breaking existing benchmarks
- the limiting cases recover known behavior
- at least one crossover region produces a stable, defensible finite-size story
- the result is interesting enough to be more than a software benchmark paper

## Short Version

Do not jump straight to quantum.

Do not repeat dilution immediately.

The best next phase is:

- anisotropy-driven crossover in the 3D Heisenberg model

It is the best balance of:

- novelty
- feasibility on current hardware
- reuse of the validated GPU path
- value as a bridge to future quantum work
