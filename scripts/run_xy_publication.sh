#!/usr/bin/env bash
# Publication-quality XY Monte Carlo runs.
# Uses the Wolff cluster algorithm — no overrelaxation or delta parameters.
# Target: Windows machine (RTX 2060) — CPU is sufficient for these sizes.
# Estimated wall time: 1–3 hours on a modern CPU.
#
# Usage:
#   bash scripts/run_xy_publication.sh
#   bash scripts/run_xy_publication.sh 2>&1 | tee run_xy.log

set -e
OUTDIR="analysis/data"
mkdir -p "$OUTDIR"

echo "=== XY FSS: cubic lattice validation ==="
cargo run --release --bin xy_fss -- \
  --sizes 8,12,16,20,24,32 \
  --tmin 1.6 --tmax 2.8 --steps 41 \
  --warmup 2000 --samples 2000 \
  --outdir "$OUTDIR"

echo "=== XY J-fitting: BCC iron ==="
# Tc_BCC(J=1) ~ 2.835 — scan 2.3 to 3.5 to straddle the transition
for N in 4 6 8 10 12; do
  cargo run --release --bin xy_jfit -- \
    --graph "analysis/graphs/bcc_N${N}.json" \
    --tmin 2.3 --tmax 3.5 --steps 41 \
    --warmup 2000 --samples 2000 \
    --outdir "$OUTDIR"
done

echo "=== XY J-fitting: FCC nickel ==="
# Tc_FCC(J=1) ~ 4.35 — scan 3.6 to 5.2 to straddle the transition
for N in 4 6 8 10 12; do
  cargo run --release --bin xy_jfit -- \
    --graph "analysis/graphs/fcc_N${N}.json" \
    --tmin 3.6 --tmax 5.2 --steps 41 \
    --warmup 2000 --samples 2000 \
    --outdir "$OUTDIR"
done

echo "=== All done ==="
echo "Run analysis/xy_fss.ipynb and analysis/xy_jfit.ipynb to analyse results."
