# GPU Anisotropy Implementation Checklist

## Objective

Add `--anisotropy-d` support to the cubic GPU Heisenberg path and make the
anisotropy crossover campaign runnable on GPU.

## File Ownership

### CUDA kernels

**Primary owner**
- [src/cuda/continuous_spin_kernel.cu](/Users/faulknco/Projects/ising-rs/src/cuda/continuous_spin_kernel.cu)

Changes:
- add `D` parameter to `continuous_metropolis_kernel`
- include `-D sz^2` in `e_old` / `e_new`
- keep XY (`n_comp == 2`) behavior unchanged
- do not enable anisotropy-aware overrelaxation yet

### GPU Rust wrapper

**Primary owner**
- [src/cuda/gpu_lattice_continuous.rs](/Users/faulknco/Projects/ising-rs/src/cuda/gpu_lattice_continuous.rs)

Changes:
- `sweep(beta, j, d, delta, n_overrelax)`
- disable overrelaxation internally when `d != 0.0`
- `measure_gpu(j, d)` includes anisotropy energy
- preserve the isotropic `d = 0` fast path

### Reduction path

**Primary owner**
- [src/cuda/reduce_gpu.rs](/Users/faulknco/Projects/ising-rs/src/cuda/reduce_gpu.rs)
- [src/cuda/reduce_kernel.cu](/Users/faulknco/Projects/ising-rs/src/cuda/reduce_kernel.cu)

Changes:
- continuous-spin energy reduction must accept `D`
- add onsite `-D sz^2` contribution for Heisenberg
- XY path remains unchanged

### CLI / runner integration

**Primary owner**
- [src/bin/gpu_fss.rs](/Users/faulknco/Projects/ising-rs/src/bin/gpu_fss.rs)

Changes:
- add `--anisotropy-d`
- pass `D` through the Heisenberg GPU path
- reject anisotropy for unsupported models if needed in v1

### Analysis / workflows

**Primary owner**
- [analysis/scripts/analyze_heisenberg_anisotropy.py](/Users/faulknco/Projects/ising-rs/analysis/scripts/analyze_heisenberg_anisotropy.py)
- new GPU runner/campaign script later

Changes:
- no major logic change needed; current analyzer already chooses:
  - `Mz` for `D>0`
  - `Mxy` for `D<0`
  - `M` for `D=0`

## Implementation Steps

### Step 1
Add `D` to the CUDA Metropolis kernel signature.

Done means:
- kernel compiles
- `D=0` path behaves exactly as before

### Step 2
Add `D` to GPU continuous energy measurement.

Done means:
- Heisenberg energy from GPU matches CPU for `D=-2,0,+2`

### Step 3
Update the Rust GPU wrapper API.

Done means:
- `ContinuousGpuLattice::sweep(...)` takes `D`
- `ContinuousGpuLattice::measure_gpu(...)` takes `D`

### Step 4
Disable overrelaxation for `D != 0`.

Done means:
- no anisotropic run performs overrelax sweeps
- `D=0` still can

### Step 5
Expose `--anisotropy-d` in `gpu_fss`.

Done means:
- `gpu_fss --model heisenberg --anisotropy-d ...` runs

### Step 6
Add parity checks.

Minimum cases:
- `D=-2.0`, `L=16`
- `D=0.0`, `L=16`
- `D=+2.0`, `L=16`

Compare CPU vs GPU:
- `E`
- `M`
- `Mz`
- `Mxy`

### Step 7
Add a GPU anisotropy campaign runner.

It should mirror the CPU campaign style:
- one directory per `D`
- `campaign_manifest.json`
- `campaign_plan.csv`
- `campaign_status.csv`
- analysis refresh after each completed dataset

## Test Plan

### Unit / smoke

- `cargo build --release --features cuda --bin gpu_fss`
- small `D=0` Heisenberg GPU smoke
- small `D=+2` Heisenberg GPU smoke
- small `D=-2` Heisenberg GPU smoke

### Physics parity

Use matched temperature windows:
- `D=-2`: easy-plane
- `D=0`: isotropic control
- `D=+2`: easy-axis

Acceptance criteria:
- GPU and CPU observables agree within statistical tolerance
- sign of symmetry response is correct

### Analysis parity

For one matched CPU/GPU mini-run:
- Binder crossings from GPU data should land near CPU values

## Non-Goals for v1

- anisotropy-aware GPU overrelaxation
- XY anisotropy extensions
- graph / mesh anisotropy on GPU
- full production run before parity is done

## Production Command Target

Once parity is complete, the intended GPU production command should look like:

```bash
cargo run --release --features cuda --bin gpu_fss -- \
  --model heisenberg \
  --anisotropy-d 1.0 \
  --sizes 32,64,96,128,192 \
  --tmin 0.95 --tmax 1.30 \
  --replicas 24 \
  --warmup 4000 \
  --samples 5000 \
  --measure-every 5 \
  --outdir analysis/data/gpu_anisotropy_campaign/dp1
```

That exact workflow should not be launched until the parity gate passes.
