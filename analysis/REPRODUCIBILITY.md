# GPU FSS Reproducibility Guide

## Overview

This directory contains finite-size scaling (FSS) analysis for three 3D universality classes
(Ising, Heisenberg, XY) using GPU-accelerated parallel tempering Monte Carlo.

## Hardware Requirements

- NVIDIA GPU with CUDA support (tested on RTX 2060 6 GB VRAM)
- CUDA Toolkit (tested with nvcc 12.x)
- MSVC build tools (Windows) or GCC (Linux) for nvcc host compiler
- ~200 MB disk space for full timeseries data

## Software Requirements

- Rust (stable, tested with 1.8x)
- Python 3.10+ with: numpy, scipy, matplotlib
- Git

## Quick Start (verify from committed data)

Summary CSVs and figures are committed to the repo. To verify:

```bash
# View pre-generated figures
ls analysis/figures/gpu_fss/

# Re-run analysis on committed summary data (no reweighting without timeseries)
python analysis/scripts/analyze_gpu_fss.py
```

## Full Reproduction (regenerate all data)

### Step 1: Build the GPU binary

```bash
# Windows: ensure cl.exe is in PATH (from VS Build Tools)
cargo build --release --features cuda --bin gpu_fss
```

### Step 2: Run the publication pipeline (~4 hours on RTX 2060)

```bash
python scripts/run_gpu_windows_pipeline.py --publish-on-success
```

This runs all three models (Ising, Heisenberg, XY) at sizes N = 8, 16, 32, 64, 128
with 20 parallel tempering replicas, 20,000 measurement samples (after 2,000 thermalisation),
and `--measure-every 5`.

Output goes to `analysis/data/gpu_windows_pipeline/publication/`:
- `gpu_fss_{model}_N{size}_summary.csv` — per-temperature averages (E, M, Cv, chi, Binder)
- `gpu_fss_{model}_N{size}_timeseries.csv` — raw per-sample measurements (for reweighting)

### Step 3: Run analysis

```bash
python analysis/scripts/analyze_gpu_fss.py
```

Produces 15 figures in `analysis/figures/gpu_fss/` and prints a results table.

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Algorithm | Wolff cluster + parallel tempering (replica exchange) |
| GPU kernels | Checkerboard Metropolis + GPU-resident reduction |
| Lattice | 3D cubic, periodic boundaries |
| Sizes | N = 8, 16, 32, 64, 128 (L = N) |
| Replicas | 20 (parallel tempering) |
| Thermalisation | 2,000 sweeps |
| Measurements | 20,000 samples |
| Measure cadence | Every 5 sweeps |
| Exchange cadence | Every sweep |

### Temperature Ranges

| Model | T_min | T_max | Points | Tc (theory) |
|-------|-------|-------|--------|-------------|
| Ising (Z2) | 4.40 | 4.62 | 32 | 4.5115 |
| Heisenberg (O(3)) | 1.35 | 1.55 | 21 | 1.4430 |
| XY (O(2)) | 2.10 | 2.30 | 21 | 2.2019 |

## Analysis Methods

1. **Binder cumulant crossing** — intersect U4(L) and U4(2L) to extract Tc
2. **Single-histogram reweighting** — interpolate observables to 200-point fine T grid
3. **Peak scaling** — chi_max ~ L^{gamma/nu}, dU4_max ~ L^{1/nu}
4. **Scaling collapse** — chi * L^{-gamma/nu} vs (T - Tc) * L^{1/nu}

## Expected Results

| Model | Tc error | gamma/nu error | nu error | beta/nu error |
|-------|----------|---------------|----------|---------------|
| Ising | 0.02% | 2.5% | 1.2% | 12.6% |
| Heisenberg | 0.04% | 1.6% | 18% | 1.9% |
| XY | 0.001% | 0.5% | 3.7% | 2.0% |

## Runtime Estimates (RTX 2060)

| Size | Ising | Heisenberg | XY |
|------|-------|------------|-----|
| N=8 | ~1 min | ~1 min | ~1 min |
| N=16 | ~2 min | ~2 min | ~2 min |
| N=32 | ~5 min | ~5 min | ~5 min |
| N=64 | ~20 min | ~15 min | ~15 min |
| N=128 | ~60 min | ~40 min | ~40 min |
| **Total** | | | **~4 hours** |

## Committed Artefacts

- `data/gpu_windows_pipeline/publication/gpu_fss_*_summary.csv` — summary statistics (15 files, ~50 KB)
- `figures/gpu_fss/*.png` — publication figures (15 files, ~1.8 MB)
- `scripts/analyze_gpu_fss.py` — analysis script

Timeseries CSVs (~128 MB) are not committed. Regenerate with the pipeline for full reweighting analysis.
