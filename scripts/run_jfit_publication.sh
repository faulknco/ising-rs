#!/bin/bash
# Publication-Quality J-Fitting (BCC Fe, FCC Ni) + Dilution Sweep
# ================================================================
# Runs mesh_sweep for BCC/FCC crystal graphs at multiple sizes,
# then diluted cubic at multiple concentrations.
#
# Usage:
#   ./scripts/run_jfit_publication.sh             # full run (~2 hours)
#   ./scripts/run_jfit_publication.sh --quick     # quick test (~20 min)

set -e

WARMUP=3000
SAMPLES=2000
STEPS=41
MODE="publication"

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) WARMUP=1000; SAMPLES=500; MODE="quick"; shift ;;
        *) shift ;;
    esac
done

echo "=== J-Fit ($MODE mode) ==="
echo "  Warmup:  $WARMUP sweeps"
echo "  Samples: $SAMPLES sweeps"
echo ""

# Build
cargo build --release --bin mesh_sweep 2>&1
EXE="./target/release/mesh_sweep"
OUTDIR="analysis/data/jfit"
mkdir -p "$OUTDIR"

# BCC Iron (coordination 8, Tc(J=1) ~ 6.5)
echo "--- BCC Iron ---"
for N in 4 6 8 10; do
    GRAPH="analysis/graphs/bcc_N${N}.json"
    if [ -f "$GRAPH" ]; then
        echo "  N=$N..."
        $EXE --graph "$GRAPH" --j 1.0 --tmin 5.0 --tmax 8.0 --steps "$STEPS" \
             --warmup "$WARMUP" --samples "$SAMPLES" \
             --prefix "bcc_iron_N${N}_J1.00" --outdir "$OUTDIR" 2>&1 | tail -1
    fi
done

# FCC Nickel (coordination 12, Tc(J=1) ~ 9.5)
echo "--- FCC Nickel ---"
for N in 4 6 8 10; do
    GRAPH="analysis/graphs/fcc_N${N}.json"
    if [ -f "$GRAPH" ]; then
        echo "  N=$N..."
        $EXE --graph "$GRAPH" --j 1.0 --tmin 7.0 --tmax 11.0 --steps "$STEPS" \
             --warmup "$WARMUP" --samples "$SAMPLES" \
             --prefix "fcc_nickel_N${N}_J1.00" --outdir "$OUTDIR" 2>&1 | tail -1
    fi
done

# Diluted cubic (Harris criterion study)
echo "--- Diluted Cubic ---"
for P in 00 10 30 50; do
    GRAPH="analysis/graphs/diluted_N20_p${P}.json"
    if [ -f "$GRAPH" ]; then
        echo "  p=0.${P}..."
        $EXE --graph "$GRAPH" --j 1.0 --tmin 2.0 --tmax 6.0 --steps "$STEPS" \
             --warmup "$WARMUP" --samples "$SAMPLES" \
             --prefix "diluted_p${P}_N20_J1.00" --outdir "$OUTDIR" 2>&1 | tail -1
    fi
done

echo ""
echo "=== Done ==="
ls -lh "$OUTDIR"/*.csv 2>/dev/null
echo ""
echo "Next: open analysis/fit_j.ipynb and run all cells"
