# Changelog

## 2026-03-16 — V2 Campaign + N=192 + Exponent Extraction

### Added
- **Critical exponent extraction** in `analyze_heisenberg_anisotropy.py`: OLS log-log fits for gamma/nu, beta/nu, 1/nu with three fit windows (all sizes, excluding smallest, sizes>=64)
- **`ols_slope()` helper** for cleaner regression code

### Changed
- V2 campaign with 5 sizes (16,32,64,96,128) + N=192 for D=0
- Analysis now prefers `linear_interp` Binder crossings over `exact_grid`

### Production Results (V2 Campaign)
- 7 D values x 5 sizes (16,32,64,96,128) + N=192 for D=0
- D=0 isotropic with N=192: gamma/nu=1.60 (16-192), 2.55 (64+); nu=1.00 (16-192), 0.85 (64+)
- Easy-axis (D>0): gamma/nu~0, beta/nu~0 — Ising-like crossover confirmed
- Easy-plane (D<0): gamma/nu=0.05-0.33 — XY-like weak ordering
- N=192 substantially improved D=0 fits (gamma/nu: 1.12 -> 1.60, nu: 1.33 -> 1.00)
- Known issue: (128,192) Binder crossing missing — T grids don't overlap

## 2026-03-12 — GPU Anisotropy Component Observables + Performance (Phase 2)

### Added
- **Component observables**: GPU pipeline now tracks Mz (easy-axis) and Mxy (easy-plane) alongside total |M|, with 28-column summary CSV output
- **`jackknife_observables_components`**: extended jackknife analysis for Mz, Mxy, chi_z, chi_xy with proper error bars
- **Fused reduction kernels**: `reduce_mag_energy_continuous` and `reduce_mag_energy_fp16` combine magnetisation and energy reduction into a single kernel launch with warp-level `__shfl_down_sync`, reducing memory bandwidth by ~50%
- **`--init-state {random|cold|planar}`**: CLI flag for ordered initial states; cold=(0,0,1), planar=(1,0,0)
- **GPU campaign runner**: `--gpu` flag in `reproduce_heisenberg_anisotropy_campaign.py` with auto init-state selection based on D sign
- **Per-replica high-T Wolff skip**: skips Wolff cluster step when `beta*J*6 < 0.5` (clusters span entire lattice)
- **CHANGELOG.md**: this file

### Changed
- `run_continuous_fss` measurement loop stores 4-tuples `(e, |M|, Mz, Mxy)` instead of 2-tuples
- Timeseries CSV now includes Mz and Mxy columns
- Heisenberg summary files named `heisenberg_fss_N{n}.csv` (was `gpu_fss_heisenberg_N{n}_summary.csv`) for analysis script compatibility
- Wolff embedding auto-disabled when D!=0 (O(n) symmetry broken)
- Removed redundant per-temperature accumulators (sum_e, sum_m, etc.) in continuous FSS loop

### Fixed
- Windows `.exe` extension handling in campaign script binary path checks

### Production Results
- 7 D values (-2, -1, -0.5, 0, 0.5, 1, 2) x 4 sizes (16, 32, 64, 128) x 16 replicas x 20k samples
- D=0 Binder crossing: Tc = 1.440 (theory: 1.443)
- Easy-axis (D>0): Mz >> Mxy confirmed, chi_z flat with L (crossover)
- Easy-plane (D<0): Mxy >> Mz confirmed, chi_xy growing with L
- Total campaign runtime: ~2.3 hours on RTX 2060

## 2026-03-09 — GPU Anisotropy Port (Phase 1)

### Added
- Uniaxial anisotropy `-D*(S_z)^2` in CUDA Metropolis kernel for Heisenberg model
- `--anisotropy-d` CLI flag in gpu_fss
- Anisotropy-aware energy reduction in CUDA kernels (f32 and FP16)
- Automatic overrelaxation disable for D!=0
- CPU Heisenberg anisotropy support with component observables
- `reproduce_heisenberg_anisotropy_campaign.py` campaign runner
- `analyze_heisenberg_anisotropy.py` symmetry-aware analysis script
