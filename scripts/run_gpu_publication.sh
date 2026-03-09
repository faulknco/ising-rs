#!/usr/bin/env bash
# Publication-quality GPU Monte Carlo runs with parallel tempering.
# Target: Windows RTX 2060 (6 GB VRAM).
#
# --measure-every N: only measure observables every N sweeps.
# This is the main speed knob — GPU sweeps without measurement are ~10x faster.
# 100K sweeps / measure-every 5 = 20K recorded measurements per temperature.
# Independence still depends on autocorrelation and should be checked from the output data.
#
# Usage:
#   bash scripts/run_gpu_publication.sh
#   bash scripts/run_gpu_publication.sh 2>&1 | tee run_gpu.log

set -e
OUTDIR="analysis/data"
mkdir -p "$OUTDIR"

echo "=== Building gpu_fss (release, CUDA) ==="
cargo build --release --features cuda --bin gpu_fss

echo "=== GPU Ising FSS: cubic lattice ==="
time cargo run --release --features cuda --bin gpu_fss -- \
  --model ising \
  --sizes 8,16,32,64,128 \
  --tmin 4.40 --tmax 4.62 --replicas 32 \
  --warmup 5000 --samples 100000 \
  --exchange-every 10 \
  --measure-every 5 \
  --outdir "$OUTDIR"

echo "=== GPU Heisenberg FSS: cubic lattice ==="
time cargo run --release --features cuda --bin gpu_fss -- \
  --model heisenberg \
  --sizes 8,16,32,64,128 \
  --tmin 1.35 --tmax 1.55 --replicas 20 \
  --warmup 5000 --samples 50000 \
  --exchange-every 10 \
  --measure-every 5 \
  --delta 0.5 --overrelax 5 \
  --outdir "$OUTDIR"

echo "=== GPU XY FSS: cubic lattice ==="
time cargo run --release --features cuda --bin gpu_fss -- \
  --model xy \
  --sizes 8,16,32,64,128 \
  --tmin 2.10 --tmax 2.30 --replicas 20 \
  --warmup 5000 --samples 50000 \
  --exchange-every 10 \
  --measure-every 5 \
  --delta 0.5 --overrelax 5 \
  --outdir "$OUTDIR"

echo "=== All GPU runs done ==="
echo "Run analysis notebooks to analyse results."
echo "Use analysis/scripts/reweighting.py for histogram reweighting on time series data."
