#!/bin/bash
# Run ALL Publication Experiments
# ================================
# Full overnight run: FSS + KZ + J-fit
# Expected total: 6-10 hours
#
# Usage:
#   ./scripts/run_all_publication.sh          # full overnight
#   ./scripts/run_all_publication.sh --quick   # quick test (~1 hour)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QUICK=""
[[ "$1" == "--quick" ]] && QUICK="--quick"

echo "========================================"
echo "  Ising-RS Publication Data Generation"
echo "  $(date)"
echo "========================================"
echo ""

echo "[1/3] Finite-Size Scaling (Wolff)..."
bash "$SCRIPT_DIR/run_fss_publication.sh" $QUICK
echo ""

echo "[2/3] Kibble-Zurek (GPU)..."
bash "$SCRIPT_DIR/run_kz_publication.sh" $QUICK
echo ""

echo "[3/3] J-Fitting + Dilution..."
bash "$SCRIPT_DIR/run_jfit_publication.sh" $QUICK
echo ""

echo "========================================"
echo "  ALL DONE — $(date)"
echo "========================================"
echo ""
echo "Data files generated:"
ls -lh analysis/data/fss_N*.csv 2>/dev/null
ls -lh analysis/data/kz_N*.csv 2>/dev/null
ls -lh analysis/data/jfit/*.csv 2>/dev/null
echo ""
echo "Open the notebooks to analyze:"
echo "  jupyter notebook analysis/fss.ipynb"
echo "  jupyter notebook analysis/kz.ipynb"
echo "  jupyter notebook analysis/fit_j.ipynb"
