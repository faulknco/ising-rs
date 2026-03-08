#!/usr/bin/env bash
# Publication-quality Heisenberg Monte Carlo runs.
# Target: Windows machine (RTX 2060) — CPU is sufficient for these sizes.
# Estimated wall time: 2–4 hours on a modern CPU.
#
# Usage:
#   bash scripts/run_heisenberg_publication.sh
#   bash scripts/run_heisenberg_publication.sh 2>&1 | tee run_heisenberg.log

set -e
OUTDIR="analysis/data"
mkdir -p "$OUTDIR"

echo "=== Heisenberg FSS: cubic lattice validation ==="
cargo run --release --bin heisenberg_fss -- \
  --sizes 8,12,16,20,24,32 \
  --tmin 1.0 --tmax 2.0 --steps 41 \
  --warmup 2000 --samples 2000 \
  --overrelax 5 --delta 0.5 \
  --outdir "$OUTDIR"

echo "=== Heisenberg J-fitting: BCC iron ==="
for N in 4 6 8 10 12; do
  cargo run --release --bin heisenberg_jfit -- \
    --graph "analysis/graphs/bcc_N${N}.json" \
    --tmin 4.0 --tmax 9.0 --steps 41 \
    --warmup 2000 --samples 2000 \
    --overrelax 5 --delta 0.5 \
    --outdir "$OUTDIR"
done

echo "=== Heisenberg J-fitting: FCC nickel ==="
for N in 4 6 8 10 12; do
  cargo run --release --bin heisenberg_jfit -- \
    --graph "analysis/graphs/fcc_N${N}.json" \
    --tmin 6.0 --tmax 14.0 --steps 41 \
    --warmup 2000 --samples 2000 \
    --overrelax 5 --delta 0.5 \
    --outdir "$OUTDIR"
done

echo "=== All done ==="
echo "Run analysis/heisenberg_fss.ipynb and analysis/heisenberg_jfit.ipynb to analyse results."
