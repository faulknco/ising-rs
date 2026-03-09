# Anisotropy Crossover Implementation Checklist

## Goal

Implement the minimum engine and analysis changes needed to study uniaxial
anisotropy-driven crossover in the 3D Heisenberg model:

```text
H = -J sum_<ij> S_i . S_j - D sum_i (S_i^z)^2
```

This checklist is the execution plan for the first crossover-capable version of
the repo. It is intentionally narrow.

## Scope

In scope:

- cubic-lattice Heisenberg model
- uniaxial onsite anisotropy `D`
- CPU and GPU support
- order parameters matched to the symmetry being broken
- reproducible pilot runs and validation tests

Out of scope:

- disorder plus anisotropy
- arbitrary graphs
- publication-ready paper text
- quantum models
- materials fitting

## Phase A: CPU Hamiltonian Support

### A1. Add anisotropy parameter to the CPU Heisenberg update path

Files:

- [metropolis.rs](/Users/faulknco/Projects/ising-rs/src/heisenberg/metropolis.rs)
- [sweep.rs](/Users/faulknco/Projects/ising-rs/src/heisenberg/sweep.rs)
- [observables.rs](/Users/faulknco/Projects/ising-rs/src/heisenberg/observables.rs)
- [fss.rs](/Users/faulknco/Projects/ising-rs/src/heisenberg/fss.rs)
- [heisenberg_fss.rs](/Users/faulknco/Projects/ising-rs/src/bin/heisenberg_fss.rs)

Changes:

- add `d: f64` to the Heisenberg sweep / measure / FSS config path
- include anisotropy in the Metropolis local energy difference:
  - exchange term as today
  - onsite term `-D * sz^2`
- preserve exact current behavior when `D = 0`

Acceptance criteria:

- existing `D = 0` benchmarks still pass
- no API path silently defaults to the wrong sign convention

### A2. Decide overrelaxation policy for `D != 0`

Files:

- [overrelax.rs](/Users/faulknco/Projects/ising-rs/src/heisenberg/overrelax.rs)
- [sweep.rs](/Users/faulknco/Projects/ising-rs/src/heisenberg/sweep.rs)

Changes:

- disable overrelaxation when `D != 0` for the first version, or gate it behind
  an explicit experimental flag

Reason:

- the current overrelaxation move is microcanonical for the isotropic exchange
  field, but not obviously microcanonical once an anisotropy term is present

Acceptance criteria:

- no anisotropy run accidentally relies on an unvalidated overrelaxation step

## Phase B: Observable and Order-Parameter Support

### B1. Add anisotropy-aware observables

Files:

- [observables.rs](/Users/faulknco/Projects/ising-rs/src/heisenberg/observables.rs)
- [mod.rs](/Users/faulknco/Projects/ising-rs/src/heisenberg/mod.rs)

Add observables for each sample:

- total magnetization magnitude `|M|`
- easy-axis order parameter `|M_z| / N`
- easy-plane order parameter `sqrt(M_x^2 + M_y^2) / N`
- second and fourth moments for the relevant component observables

Derived quantities to expose:

- susceptibility for `M_z`
- susceptibility for `M_xy`
- Binder cumulant ingredients for `M_z`
- Binder cumulant ingredients for `M_xy`

Acceptance criteria:

- at `D = 0`, component-resolved observables behave sensibly and symmetrically
- at large positive `D`, `M_z` dominates over `M_xy`
- at large negative `D`, `M_xy` dominates over `M_z`

### B2. Extend CSV output

Files:

- [heisenberg_fss.rs](/Users/faulknco/Projects/ising-rs/src/bin/heisenberg_fss.rs)

Add columns for:

- `Mz`, `Mz_err`, `Mz2`, `Mz2_err`, `Mz4`, `Mz4_err`, `chi_z`, `chi_z_err`
- `Mxy`, `Mxy_err`, `Mxy2`, `Mxy2_err`, `Mxy4`, `Mxy4_err`, `chi_xy`, `chi_xy_err`
- metadata for `D`

Acceptance criteria:

- CSV schema is explicit and stable
- `D = 0` output remains easy to compare against old runs

## Phase C: GPU Hamiltonian Support

### C1. Add anisotropy to GPU Metropolis energy

Files:

- [continuous_spin_kernel.cu](/Users/faulknco/Projects/ising-rs/src/cuda/continuous_spin_kernel.cu)
- [gpu_lattice_continuous.rs](/Users/faulknco/Projects/ising-rs/src/cuda/gpu_lattice_continuous.rs)

Changes:

- pass anisotropy `D` into the continuous-spin Metropolis kernel
- add onsite anisotropy contribution to `e_old` and `e_new`
- keep XY behavior unchanged by treating `D` as a Heisenberg-only parameter in
  the first version

Acceptance criteria:

- `D = 0` reproduces current GPU behavior
- `D != 0` changes acceptance and observables in the expected direction

### C2. Add anisotropy to GPU energy measurement

Files:

- [reduce_kernel.cu](/Users/faulknco/Projects/ising-rs/src/cuda/reduce_kernel.cu)
- [reduce_gpu.rs](/Users/faulknco/Projects/ising-rs/src/cuda/reduce_gpu.rs)
- [gpu_lattice_continuous.rs](/Users/faulknco/Projects/ising-rs/src/cuda/gpu_lattice_continuous.rs)

Changes:

- include `-D * sz^2` in the continuous-spin energy reduction
- thread `D` through `measure_gpu(...)`

Acceptance criteria:

- CPU and GPU energies match on representative seeds and temperatures

### C3. Keep anisotropy runs conservative on GPU

Files:

- [gpu_fss.rs](/Users/faulknco/Projects/ising-rs/src/bin/gpu_fss.rs)

Changes:

- add `--anisotropy-d`
- disable or warn on overrelaxation when `D != 0` until validated
- record `D` in filenames, manifests, and output tables

Acceptance criteria:

- GPU anisotropy runs are explicit, not implicit

## Phase D: Validation and Parity Tests

### D1. CPU unit tests

Files:

- [metropolis.rs](/Users/faulknco/Projects/ising-rs/src/heisenberg/metropolis.rs)
- [observables.rs](/Users/faulknco/Projects/ising-rs/src/heisenberg/observables.rs)

Add tests for:

- `D = 0` no-regression behavior
- large positive `D` favors `z` alignment from a near-axis ordered start
- large negative `D` suppresses `|s_z|`
- spins remain normalized after anisotropy sweeps

### D2. CLI / integration tests

Files:

- [tests/cli.rs](/Users/faulknco/Projects/ising-rs/tests/cli.rs)

Add tests for:

- `heisenberg_fss --anisotropy-d ...` writes expected columns
- `gpu_fss --model heisenberg --anisotropy-d ...` accepts valid input
- `D = 0` and omitted `D` agree within tolerance

### D3. CPU/GPU parity test points

Add one scripted parity check at:

- `D = 0.0`
- `D = +2.0`
- `D = -2.0`

At each point compare:

- energy
- `|M|`
- `M_z`
- `M_xy`

Use small sizes first:

- `L = 8` or `L = 12`

Acceptance criteria:

- CPU/GPU agreement within defined tolerances
- no systematic sign or normalization mismatch

## Phase E: Pilot Research Campaign

### E1. Minimal pilot grid

Run:

- `D/J = -2.0, 0.0, +2.0`

Sizes:

- `L = 16, 32`

Goal:

- prove the workflow works
- verify the expected qualitative symmetry-breaking direction

### E2. First real crossover grid

Run:

- `D/J = -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0`

Primary fit sizes:

- `L = 32, 64, 96, 128, 192`

Measure:

- Binder crossing `T_c(D)` using `M_z` for easy-axis
- Binder crossing `T_c(D)` using `M_xy` for easy-plane
- `gamma/nu`
- `beta/nu`
- `nu` from Binder slope

## Phase F: Analysis Work

### F1. Add anisotropy-aware analysis entrypoint

Suggested file:

- `analysis/scripts/reproduce_heisenberg_anisotropy.py`

Responsibilities:

- run or collect anisotropy sweeps
- write manifests
- compute Binder curves for the relevant order parameter
- write crossover summary tables

### F2. Add crossover-specific summaries

Suggested outputs:

- `anisotropy_tc_summary.csv`
- `anisotropy_exponent_summary.csv`
- `anisotropy_component_ratios.csv`
- `anisotropy_summary.json`

### F3. Analysis rules

- use `M_z`-based FSS for `D > 0`
- use `M_xy`-based FSS for `D < 0`
- use total `|M|` only as a diagnostic at `D = 0`
- treat small-size drift explicitly, not as noise to hide

## File-by-File Build Order

Implement in this order:

1. [metropolis.rs](/Users/faulknco/Projects/ising-rs/src/heisenberg/metropolis.rs)
2. [sweep.rs](/Users/faulknco/Projects/ising-rs/src/heisenberg/sweep.rs)
3. [observables.rs](/Users/faulknco/Projects/ising-rs/src/heisenberg/observables.rs)
4. [fss.rs](/Users/faulknco/Projects/ising-rs/src/heisenberg/fss.rs)
5. [heisenberg_fss.rs](/Users/faulknco/Projects/ising-rs/src/bin/heisenberg_fss.rs)
6. unit tests for CPU behavior
7. [continuous_spin_kernel.cu](/Users/faulknco/Projects/ising-rs/src/cuda/continuous_spin_kernel.cu)
8. [reduce_kernel.cu](/Users/faulknco/Projects/ising-rs/src/cuda/reduce_kernel.cu)
9. [gpu_lattice_continuous.rs](/Users/faulknco/Projects/ising-rs/src/cuda/gpu_lattice_continuous.rs)
10. [gpu_fss.rs](/Users/faulknco/Projects/ising-rs/src/bin/gpu_fss.rs)
11. CLI / parity tests
12. analysis script

## Definition of Done

This phase is ready for real research only when:

- `D = 0` reproduces the Heisenberg benchmark
- anisotropy-aware CPU and GPU paths both work
- the correct order parameter is used on each side of the crossover
- reproducible pilot outputs exist for `D = -2, 0, +2`
- one larger crossover campaign can be launched from scripts without notebook-only work

## Immediate Next Step

Start with CPU support only.

The first concrete coding task should be:

- thread `d: f64` through the CPU Heisenberg sweep and measurement path
- disable overrelaxation for `D != 0`
- add `M_z` and `M_xy` observables

That is the smallest change set that can validate the physics direction before
touching CUDA.
