# Heisenberg Anisotropy Production Campaign

## Goal

Run the first full CPU crossover campaign for the uniaxial Heisenberg model,
using the symmetry-aware anisotropy observables and the automated campaign
runner.

## Runner

Use:

```bash
python analysis/scripts/reproduce_heisenberg_anisotropy_campaign.py \
  --d-values=-2,-1,-0.5,0,0.5,1,2 \
  --sizes=32,64,96,128,192 \
  --steps=49 \
  --warmup=4000 \
  --samples=5000 \
  --output-root analysis/data/anisotropy_campaign_cpu \
  --campaign-name production_v1
```

This command:

- writes one dataset directory per anisotropy value
- writes `campaign_manifest.json`, `campaign_plan.csv`, and `campaign_status.csv`
- refreshes `analysis/anisotropy_*` summary outputs after each completed `D`

## Default Temperature Windows

The runner defaults are Stage-1-informed windows:

- `D=-2.0`: `0.55 .. 1.05`
- `D=-1.0`: `0.85 .. 1.20`
- `D=-0.5`: `0.85 .. 1.15`
- `D=0.0`: `1.25 .. 1.60`
- `D=+0.5`: `0.90 .. 1.20`
- `D=+1.0`: `0.95 .. 1.30`
- `D=+2.0`: `0.78 .. 1.05`

Override any window with:

```bash
--t-window-spec="-2:0.50:1.05;1:0.95:1.25"
```

## Analysis Contract

The downstream analysis remains symmetry-aware:

- `D < 0` uses `Mxy`, `chi_xy`, and the corresponding Binder cumulant
- `D > 0` uses `Mz`, `chi_z`, and the corresponding Binder cumulant
- `D = 0` uses total `M`, `chi`, and total Binder as the isotropic control

## Pre-Launch Gate

Before starting the full campaign:

1. confirm Stage 1B outputs are present under `analysis/data/anisotropy_stage1_cpu`
2. re-check whether `D=-2.0` needs a lower-window refinement
3. confirm `cargo test` and `cargo build --release --bin heisenberg_fss` pass

## Post-Run Review

The campaign is good enough to keep if:

- Binder crossings are interior for all `D`
- `Tc(D)` varies smoothly across the crossover grid
- `D < 0` stays easy-plane and `D > 0` stays easy-axis
- the `D=0` control remains close to the isotropic Heisenberg baseline
