# V2 Experiment Reproducibility

## How to reproduce all data

```bash
cd ~/ising-rs
git checkout idiot-proof-hardening   # or ensure engine fixes are on your branch
bash scripts/run_v2.sh               # Full run (~4-6 hours CPU)
bash scripts/run_v2.sh --quick       # Quick test (~5 min)
```

## Engine version
- Commit: 14c91f2 (idiot-proof-hardening branch)
- Key fixes vs v1:
  - Connected susceptibility: chi = beta * V * (<|m|^2> - <|m|>^2)
  - KZ quench endpoint: frac = step / (tau_q - 1) so T reaches t_end

## Full run parameters
| Phase | Param | Value |
|---|---|---|
| Validation (2D) | sizes | 32, 64 |
| Validation (2D) | T range | 1.5 - 3.5, 61 steps |
| Validation (2D) | warmup/samples | 5000 / 5000 |
| Validation (3D) | sizes | 8, 16 |
| Validation (3D) | T range | 0.5 - 10.0, 61 steps |
| FSS sweep | sizes | 8, 12, 16, 20, 24, 32, 40, 48 |
| FSS sweep | T range | 3.8 - 5.2, 61 steps |
| FSS sweep | warmup/samples | 5000 / 20000 |
| FSS raw | sizes | 16, 20, 24, 32, 40, 48 |
| FSS raw | T range | 4.30 - 4.70, 41 steps |
| FSS raw | warmup/samples | 5000 / 10000 |
| J-fit (BCC/FCC) | sizes | 4, 6, 8, 10, 12 |
| J-fit (BCC) | T range | 3.5 - 9.0, 41 steps |
| J-fit (FCC) | T range | 5.0 - 14.0, 41 steps |
| J-fit | warmup/samples | 2000 / 2000 |
| KZ | N | 80 |
| KZ | trials | 50 |
| KZ | tau_Q range | 100 - 100000 (25 log-spaced) |
| KZ | T range | 6.0 -> 1.0 (linear quench) |
| Coarsening | sizes | 30, 40, 50 |
| Coarsening | steps | 200000, sample every 10 |
| Coarsening | T_quench | 2.5 |

## Expected output files
```
data/
  run_info.txt
  validation/
    fss_N8.csv, fss_N16.csv       (3D cubic)
    fss_N32.csv, fss_N64.csv      (2D square)
  fss/
    fss_N{8,12,16,20,24,32,40,48}.csv
  fss_raw/
    fss_raw_N{16,20,24,32,40,48}.csv
  jfit/
    bcc_iron_N{4,6,8,10,12}_J1.00_sweep.csv
    fcc_nickel_N{4,6,8,10,12}_J1.00_sweep.csv
  kz/
    kz_N80.csv
  coarsening/
    coarsening_N{30,40,50}_T2.50.csv
```

## Post-processing
Open `analysis/v2/v2_analysis.ipynb` in Jupyter and run all cells.
Outputs:
- 10 PDF figures in `analysis/v2/figures/`
- `analysis/v2/results_summary.md` with all key numbers

## Algorithm details
- All equilibrium measurements use Wolff cluster algorithm (z_W ~ 0.3)
- KZ quench uses Metropolis single-spin updates (z ~ 2, correct dynamics)
- Histogram reweighting: Ferrenberg-Swendsen patchwork (nearest sim T)
- Error bars: Jackknife (10-20 blocks) on all fitted quantities
- Binder cumulant: U4 = 1 - <m^4>/(3<m^2>^2), crossing for Tc
- Seed: 42 for all runs (reproducible)
