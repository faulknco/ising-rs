# GPU Anisotropy Port Plan

## Goal

Port the uniaxial Heisenberg anisotropy workflow from the current CPU path to
the cubic-lattice GPU path so the anisotropy crossover campaign becomes
practical at production sizes.

Target Hamiltonian:

```text
H = -J Σ_<ij> S_i · S_j - D Σ_i (S_i^z)^2
```

This is the same model already implemented on CPU.

## Why GPU First

The CPU production campaign is too expensive at the current target:

- `D/J = -2,-1,-0.5,0,0.5,1,2`
- `L = 32,64,96,128,192`
- `49` temperatures per `D`
- `4000/5000` warmup/samples

On this machine, even a single `N=32` production-style job is already several
minutes. Scaling to the full grid is a many-day CPU run.

So the next step should be to move anisotropy onto the GPU path, not to push
the CPU production campaign further.

## Strategic Recommendation

Do this in three phases:

1. **GPU anisotropy correctness port**
2. **GPU anisotropy validation and parity**
3. **GPU anisotropy production workflow**

Do **not** start with a full production campaign.

## Phase 1: Correctness Port

### Scope

Add anisotropy support to the existing continuous-spin CUDA path:

- Heisenberg first
- XY later only if we decide to add an analogous anisotropy term there

### Required GPU changes

#### 1. Continuous Metropolis kernel

In
[src/cuda/continuous_spin_kernel.cu](/Users/faulknco/Projects/ising-rs/src/cuda/continuous_spin_kernel.cu):

- extend `continuous_metropolis_kernel` to accept `D`
- change local energy from

```text
E_old = -S · h
E_new = -S' · h
```

to

```text
E_old = -S · h - D sz^2
E_new = -S' · h - D sz'^2
```

This should apply only when `n_comp == 3`.

#### 2. Continuous energy reduction

In the same GPU path, update the energy reduction so measurements include the
anisotropy term:

```text
E = -J Σ_<ij> S_i · S_j - D Σ_i sz_i^2
```

That means the continuous measurement path must accept `D`, not just `J`.

#### 3. Overrelaxation policy

The current GPU overrelaxation kernel is microcanonical only for the isotropic
exchange Hamiltonian.

With anisotropy, the existing overrelaxation move is **not** automatically
valid.

So the first GPU anisotropy version should:

- disable overrelaxation when `D != 0`
- keep current overrelaxation only for `D = 0`

This matches the safe CPU policy.

#### 4. GPU wrapper

In
[src/cuda/gpu_lattice_continuous.rs](/Users/faulknco/Projects/ising-rs/src/cuda/gpu_lattice_continuous.rs):

- thread `D` through `sweep(...)`
- thread `D` through `measure_gpu(...)`
- keep `D=0` behavior identical to the current isotropic path

## Phase 2: Validation and Parity

### CPU/GPU parity targets

Before any production run:

1. `D = 0` reproduces the current Heisenberg GPU baseline
2. `D = +2` shows easy-axis behavior (`Mz > Mxy`)
3. `D = -2` shows easy-plane behavior (`Mxy > Mz`)
4. GPU and CPU agree on:
   - `E`
   - total `M`
   - `Mz`
   - `Mxy`
   - Binder curves built from the right observable

### Minimum parity grid

Use:

- `D = -2, 0, +2`
- `L = 16, 32`
- tight windows around the Stage 1 pilot crossings

Treat parity statistically, not bitwise.

## Phase 3: Production Workflow

Once parity is clean:

1. add a GPU anisotropy runner
2. add a GPU anisotropy campaign script
3. add the same manifest / status / analysis refresh pattern used for dilution

### Production target

Then launch:

- `D/J = -2,-1,-0.5,0,0.5,1,2`
- `L = 32,64,96,128,192`
- `49-61` temperatures

Primary observables:

- `D < 0`: `Mxy`, `chi_xy`, Binder from `Mxy`
- `D > 0`: `Mz`, `chi_z`, Binder from `Mz`
- `D = 0`: total `M`, `chi`, total Binder as control

## Decision Rules

### Go to production if

- CPU/GPU parity is good at `D=-2,0,+2`
- no obvious Binder pathologies appear
- `D=0` remains close to the isotropic benchmark

### Do not go to production if

- anisotropic GPU energy does not match CPU
- easy-axis/easy-plane order parameters disagree with CPU
- overrelaxation is accidentally still influencing `D != 0` runs

## Main Risks

1. **Wrong measurement Hamiltonian**
   If the GPU reduction omits `-D sz^2`, the entire anisotropy campaign is
   invalid even if the update kernel is correct.

2. **Unsafe overrelaxation**
   This is the biggest algorithmic trap.

3. **Using the wrong order parameter in analysis**
   `M` is not the primary symmetry-breaking observable once `D != 0`.

## Recommended Order

1. GPU Metropolis anisotropy term
2. GPU energy reduction anisotropy term
3. disable overrelaxation for `D != 0`
4. CPU/GPU parity at `D=-2,0,+2`
5. GPU runner and campaign script
6. production campaign
