#!/bin/bash
# Publication-Quality FSS with Wolff Algorithm
# =============================================
# Usage:
#   ./scripts/run_fss_publication.sh                 # full run (overnight)
#   ./scripts/run_fss_publication.sh --quick          # quick test (~30 min)
#   ./scripts/run_fss_publication.sh --sizes 8,12,16  # custom sizes
#
# Output: analysis/data/fss_N<size>.csv

set -e

# Defaults: publication quality
SIZES="8,12,16,20,24,32,40,48"
WARMUP=5000
SAMPLES=20000
STEPS=61
TMIN=3.8
TMAX=5.2
OUTDIR="analysis/data"
MODE="publication"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)    SIZES="8,12,16,20,24"; WARMUP=2000; SAMPLES=5000; STEPS=41; MODE="quick"; shift ;;
        --sizes)    SIZES="$2"; shift 2 ;;
        --warmup)   WARMUP="$2"; shift 2 ;;
        --samples)  SAMPLES="$2"; shift 2 ;;
        --steps)    STEPS="$2"; shift 2 ;;
        --tmin)     TMIN="$2"; shift 2 ;;
        --tmax)     TMAX="$2"; shift 2 ;;
        --outdir)   OUTDIR="$2"; shift 2 ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== FSS Wolff ($MODE mode) ==="
echo "  Sizes:   $SIZES"
echo "  T range: $TMIN - $TMAX ($STEPS steps)"
echo "  Warmup:  $WARMUP Wolff steps"
echo "  Samples: $SAMPLES Wolff steps"
echo "  Output:  $OUTDIR/fss_N*.csv"
echo ""

# Build
echo "Building..."
cargo build --release --bin fss 2>&1

# Run
echo ""
echo "Running (start: $(date))..."
time ./target/release/fss --wolff \
    --sizes "$SIZES" \
    --tmin "$TMIN" --tmax "$TMAX" --steps "$STEPS" \
    --warmup "$WARMUP" --samples "$SAMPLES" \
    --outdir "$OUTDIR"

echo ""
echo "=== Done ($(date)) ==="
echo "Output files:"
ls -lh "$OUTDIR"/fss_N*.csv 2>/dev/null

echo ""
echo "Next: open analysis/fss.ipynb and run all cells"
