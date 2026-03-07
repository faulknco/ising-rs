#!/bin/bash
# Publication-Quality Kibble-Zurek with GPU
# ==========================================
# Protocol: Ramp T=6 -> Tc=4.5115, then snap-freeze.
# Runs N=50 and N=80 with 50 trials each.
#
# Usage:
#   ./scripts/run_kz_publication.sh             # full run (~1 hour GPU)
#   ./scripts/run_kz_publication.sh --quick     # quick test (~15 min)

set -e

# Environment for CUDA build
export PATH="/c/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64:$PATH:/c/Users/conno/.cargo/bin:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin"
export CUDA_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9"
export INCLUDE="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.50.35717/include;C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/ucrt"

TRIALS=50
TAU_MIN=100
TAU_MAX=50000
TAU_STEPS=20
SIZES="50 80"
MODE="publication"

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) TRIALS=10; TAU_STEPS=12; TAU_MAX=10000; MODE="quick"; shift ;;
        *) shift ;;
    esac
done

echo "=== KZ GPU ($MODE mode) ==="
echo "  Sizes:    $SIZES"
echo "  Trials:   $TRIALS per tau_q"
echo "  tau_q:    $TAU_MIN - $TAU_MAX ($TAU_STEPS steps)"
echo ""

# Build with CUDA
echo "Building with CUDA..."
cargo build --release --features cuda --bin kz 2>&1

echo ""
for N in $SIZES; do
    echo "--- N=$N (start: $(date)) ---"
    time ./target/release/kz --gpu --n "$N" \
        --trials "$TRIALS" \
        --tau-min "$TAU_MIN" --tau-max "$TAU_MAX" --tau-steps "$TAU_STEPS"
    echo ""
done

echo "=== Done ($(date)) ==="
ls -lh analysis/data/kz_N*.csv 2>/dev/null
echo ""
echo "Next: open analysis/kz.ipynb and run all cells"
