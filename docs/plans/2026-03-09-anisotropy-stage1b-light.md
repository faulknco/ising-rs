# Anisotropy Stage 1B Light

## Purpose

Stage 1B Light is the reduced-cost scouting batch for the intermediate
anisotropy values after Stage 1 Batch 1 completes.

It exists because the original Stage 1 Batch 1 runs are much heavier than
expected on CPU. The goal here is not final exponent fitting. The goal is:

- locate usable temperature windows
- confirm the direction of symmetry breaking at intermediate `D`
- decide which `D` values deserve the full production campaign

## Why It Is Lighter

Compared with the current Stage 1 Batch 1 settings:

- sizes reduced from `16,32,64` to `16,32`
- temperature points reduced from `33` to `25`
- warmup reduced from `1500` to `800`
- samples reduced from `3000` to `1500`

This should cut runtime substantially while still giving enough signal for:

- `Mz` versus `Mxy`
- rough susceptibility peaks
- rough Binder-crossing placement

## D Values

Run only the intermediate points:

- `D/J = -0.5`
- `D/J = +0.5`
- `D/J = +1.0`

The outer points `D = -2, -1, 0, +2` are already covered by Stage 1 Batch 1.

## Temperature Windows

These windows are chosen by interpolation from the successful pilots:

- `D = -0.5`: `T = 0.95 .. 1.75`
- `D = +0.5`: `T = 0.95 .. 1.95`
- `D = +1.0`: `T = 0.90 .. 2.10`

These are still scouting windows, not final production windows.

## Standard Parameters

- sizes: `16,32`
- steps: `25`
- warmup: `800`
- samples: `1500`

## Launch Commands

```bash
mkdir -p analysis/data/anisotropy_stage1_cpu/dm0p5
target/release/heisenberg_fss \
  --sizes 16,32 \
  --anisotropy-d -0.5 \
  --tmin 0.95 --tmax 1.75 --steps 25 \
  --warmup 800 --samples 1500 \
  --seed 3305 \
  --outdir analysis/data/anisotropy_stage1_cpu/dm0p5
```

```bash
mkdir -p analysis/data/anisotropy_stage1_cpu/dp0p5
target/release/heisenberg_fss \
  --sizes 16,32 \
  --anisotropy-d 0.5 \
  --tmin 0.95 --tmax 1.95 --steps 25 \
  --warmup 800 --samples 1500 \
  --seed 3306 \
  --outdir analysis/data/anisotropy_stage1_cpu/dp0p5
```

```bash
mkdir -p analysis/data/anisotropy_stage1_cpu/dp1
target/release/heisenberg_fss \
  --sizes 16,32 \
  --anisotropy-d 1.0 \
  --tmin 0.90 --tmax 2.10 --steps 25 \
  --warmup 800 --samples 1500 \
  --seed 3307 \
  --outdir analysis/data/anisotropy_stage1_cpu/dp1
```

## Success Criteria

Stage 1B Light is good enough if:

- `D = -0.5` shows `Mxy > Mz` over the low-temperature side
- `D = +0.5` and `D = +1.0` show `Mz > Mxy`
- the chosen susceptibility peaks are not pinned to the scan boundary
- the Binder-crossing analysis gives a plausible `T_c(D)` trend

## Decision Rule After Stage 1B

If Stage 1B Light succeeds:

- launch the real production campaign on `L = 32,64,96,128,192`

If Stage 1B Light still shows edge-peaked windows:

- do one final temperature-window refinement before the production campaign
