# GPU Anisotropy Implementation Checklist

**Status: COMPLETE (2026-03-12)**

All steps implemented and validated. Production campaign completed with 7 D values,
sizes 16-128, 20k samples. Results in `analysis/data/anisotropy_campaign_gpu_prod/`.

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

### Step 1 -- DONE
Add `D` to the CUDA Metropolis kernel signature.

### Step 2 -- DONE
Add `D` to GPU continuous energy measurement.

### Step 3 -- DONE
Update the Rust GPU wrapper API.

### Step 4 -- DONE
Disable overrelaxation for `D != 0`.

### Step 5 -- DONE
Expose `--anisotropy-d` in `gpu_fss`.

### Step 6 -- DONE
Add parity checks. Validated D=-2, D=0, D=+2 on N=8 and N=16.

### Step 7 -- DONE
Add a GPU anisotropy campaign runner (`--gpu` flag in reproduce_heisenberg_anisotropy_campaign.py).

### Step 8 -- DONE (Phase 2 additions, 2026-03-12)
- Component observables: Mz, Mxy tracked in measurement loop with 28-column summary CSV
- `jackknife_observables_components` for Mz/Mxy/chi_z/chi_xy error bars
- Fused `reduce_mag_energy_continuous` and `reduce_mag_energy_fp16` CUDA kernels with warp-level `__shfl_down_sync`
- `--init-state {random|cold|planar}` CLI flag with cold/planar starts in all lattice types
- Auto-disable Wolff for D!=0 (broken O(n) symmetry)
- Per-replica high-T Wolff skip (beta*J*6 < 0.5)
- Heisenberg summary filename compatibility (`heisenberg_fss_N{n}.csv`)
- Windows .exe path fix in campaign script
- Production campaign: 7 D values × 4 sizes (16,32,64,128) × 16 replicas × 20k samples
- V2 campaign: 7 D values × 5 sizes (16,32,64,96,128) + N=192 for D=0
- Critical exponent extraction with OLS log-log fits (gamma/nu, beta/nu, 1/nu)
- Three fit windows: all sizes, excluding smallest, sizes>=64
- N=192 D=0 run completed 2026-03-14 (~24h on RTX 2060)

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
