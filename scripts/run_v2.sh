#!/usr/bin/env bash
# ============================================================================
# V2 Publication Run — Clean, Comprehensive Ising Model Experiment
# ============================================================================
#
# This script runs ALL experiments from scratch with corrected engine:
#   1. Validation (2D Onsager, exact enumeration, limits)
#   2. FSS on 3D cubic (Wolff, histogram reweighting)
#   3. J-fitting on BCC Fe and FCC Ni
#   4. Bond dilution (Harris criterion)
#   5. Kibble-Zurek mechanism
#   6. Domain coarsening
#
# Engine fixes in v2:
#   - Connected susceptibility chi = beta*N*(<|m|^2> - <|m|>^2)
#   - KZ quench reaches t_end correctly
#
# Usage:
#   bash scripts/run_v2.sh           # Full publication run (~4-6 hours CPU)
#   bash scripts/run_v2.sh --quick   # Quick test run (~5 min)
#
# Output: analysis/v2/data/
# ============================================================================

set -euo pipefail

# Ensure cargo is on PATH
export PATH="$PATH:$HOME/.cargo/bin"

OUTDIR="analysis/v2/data"
GRAPHDIR="analysis/graphs"
QUICK=false

if [[ "${1:-}" == "--quick" ]]; then
    QUICK=true
    echo "=== V2 QUICK MODE (reduced statistics for testing) ==="
fi

# Build release binary
echo "Building release binary..."
cargo build --release --bin sweep --bin fss --bin kz --bin coarsening --bin mesh_sweep

mkdir -p "$OUTDIR/validation" "$OUTDIR/fss" "$OUTDIR/fss_raw" "$OUTDIR/jfit" "$OUTDIR/kz" "$OUTDIR/coarsening"

# Record engine version
echo "Engine version:" > "$OUTDIR/run_info.txt"
git rev-parse HEAD >> "$OUTDIR/run_info.txt"
echo "Date: $(date)" >> "$OUTDIR/run_info.txt"
echo "Mode: $(if $QUICK; then echo quick; else echo full; fi)" >> "$OUTDIR/run_info.txt"
cargo --version >> "$OUTDIR/run_info.txt" 2>&1
echo "" >> "$OUTDIR/run_info.txt"

# ============================================================================
# 1. VALIDATION
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 1: Validation"
echo "=========================================="

if $QUICK; then
    VAL_WARMUP=500; VAL_SAMPLES=500; VAL_STEPS=21
else
    VAL_WARMUP=5000; VAL_SAMPLES=5000; VAL_STEPS=61
fi

# 1a. 2D Onsager validation (square lattice, known exact solution)
echo "--- 2D Onsager validation ---"
for N in 32 64; do
    echo "  N=$N..."
    cargo run --release --bin fss -- \
        --sizes $N --geometry square --wolff \
        --tmin 1.5 --tmax 3.5 --steps $VAL_STEPS \
        --warmup $VAL_WARMUP --samples $VAL_SAMPLES --seed 42 \
        --outdir "$OUTDIR/validation"
done

# 1b. 3D known limits
echo "--- 3D known limits ---"
cargo run --release --bin fss -- \
    --sizes 8,16 --geometry cubic --wolff \
    --tmin 0.5 --tmax 10.0 --steps $VAL_STEPS \
    --warmup $VAL_WARMUP --samples $VAL_SAMPLES --seed 42 \
    --outdir "$OUTDIR/validation"

echo "Validation phase complete."

# ============================================================================
# 2. FINITE-SIZE SCALING (3D CUBIC)
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 2: Finite-Size Scaling"
echo "=========================================="

if $QUICK; then
    FSS_SIZES="8,12,16,20"
    FSS_WARMUP=1000; FSS_SAMPLES=2000; FSS_STEPS=31
    RAW_SIZES="8,12,16"
    RAW_WARMUP=1000; RAW_SAMPLES=2000; RAW_STEPS=21
else
    FSS_SIZES="8,12,16,20,24,32,40,48"
    FSS_WARMUP=5000; FSS_SAMPLES=20000; FSS_STEPS=61
    RAW_SIZES="16,20,24,32,40,48"
    RAW_WARMUP=5000; RAW_SAMPLES=10000; RAW_STEPS=41
fi

# 2a. Full-range sweep
echo "--- FSS full-range sweep (T=3.8..5.2) ---"
cargo run --release --bin fss -- \
    --sizes $FSS_SIZES --geometry cubic --wolff \
    --tmin 3.8 --tmax 5.2 --steps $FSS_STEPS \
    --warmup $FSS_WARMUP --samples $FSS_SAMPLES --seed 42 \
    --outdir "$OUTDIR/fss"

# 2b. Raw time series for histogram reweighting (narrow range near Tc)
echo "--- FSS raw time series for reweighting (T=4.30..4.70) ---"
cargo run --release --bin fss -- \
    --sizes $RAW_SIZES --geometry cubic --wolff --raw \
    --tmin 4.30 --tmax 4.70 --steps $RAW_STEPS \
    --warmup $RAW_WARMUP --samples $RAW_SAMPLES --seed 42 \
    --outdir "$OUTDIR/fss_raw"

echo "FSS phase complete."

# ============================================================================
# 3. J-FITTING (BCC Fe, FCC Ni)
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 3: J-Fitting (BCC Fe, FCC Ni)"
echo "=========================================="

if $QUICK; then
    JFIT_SIZES="4 6 8"
    JFIT_WARMUP=500; JFIT_SAMPLES=500; JFIT_STEPS=31
else
    JFIT_SIZES="4 6 8 10 12"
    JFIT_WARMUP=2000; JFIT_SAMPLES=2000; JFIT_STEPS=41
fi

# Generate crystal graphs if they don't exist
for structure in bcc fcc; do
    for N in $JFIT_SIZES; do
        GRAPH="$GRAPHDIR/${structure}_N${N}.json"
        if [[ ! -f "$GRAPH" ]]; then
            echo "  Generating $structure N=$N graph..."
            python3 "$GRAPHDIR/gen_${structure}.py" $N "$GRAPH" 2>/dev/null || \
                echo "  Warning: could not generate $GRAPH (python3 or script missing)"
        fi
    done
done

# BCC Iron sweeps
echo "--- BCC Iron J-fitting ---"
for N in $JFIT_SIZES; do
    GRAPH="$GRAPHDIR/bcc_N${N}.json"
    if [[ -f "$GRAPH" ]]; then
        echo "  BCC N=$N..."
        cargo run --release --bin mesh_sweep -- \
            --graph "$GRAPH" --j 1.0 \
            --tmin 3.5 --tmax 9.0 --steps $JFIT_STEPS \
            --warmup $JFIT_WARMUP --samples $JFIT_SAMPLES --seed 42 \
            --outdir "$OUTDIR/jfit" --prefix "bcc_iron_N${N}_J1.00"
    else
        echo "  Skipping BCC N=$N (graph not found)"
    fi
done

# FCC Nickel sweeps
echo "--- FCC Nickel J-fitting ---"
for N in $JFIT_SIZES; do
    GRAPH="$GRAPHDIR/fcc_N${N}.json"
    if [[ -f "$GRAPH" ]]; then
        echo "  FCC N=$N..."
        cargo run --release --bin mesh_sweep -- \
            --graph "$GRAPH" --j 1.0 \
            --tmin 5.0 --tmax 14.0 --steps $JFIT_STEPS \
            --warmup $JFIT_WARMUP --samples $JFIT_SAMPLES --seed 42 \
            --outdir "$OUTDIR/jfit" --prefix "fcc_nickel_N${N}_J1.00"
    else
        echo "  Skipping FCC N=$N (graph not found)"
    fi
done

# Bond dilution on cubic lattice
echo "--- Bond dilution (Harris criterion) ---"
for p in 00 10 20 30 40 50; do
    GRAPH="$GRAPHDIR/diluted_cubic_p0.${p}_N20.json"
    if [[ -f "$GRAPH" ]]; then
        echo "  Dilution p=0.$p..."
        cargo run --release --bin mesh_sweep -- \
            --graph "$GRAPH" --j 1.0 \
            --tmin 2.0 --tmax 6.0 --steps $JFIT_STEPS \
            --warmup $JFIT_WARMUP --samples $JFIT_SAMPLES --seed 42 \
            --outdir "$OUTDIR/jfit" --prefix "diluted_p${p}_N20_J1.00"
    fi
done

echo "J-fitting phase complete."

# ============================================================================
# 4. KIBBLE-ZUREK MECHANISM
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 4: Kibble-Zurek Mechanism"
echo "=========================================="

if $QUICK; then
    KZ_N=20; KZ_TRIALS=5; KZ_TAU_STEPS=10
else
    KZ_N=80; KZ_TRIALS=50; KZ_TAU_STEPS=25
fi

echo "--- KZ quench (N=$KZ_N, $KZ_TRIALS trials) ---"
cargo run --release --bin kz -- \
    --n $KZ_N --geometry cubic \
    --t-start 6.0 --t-end 1.0 \
    --tau-min 100 --tau-max 100000 --tau-steps $KZ_TAU_STEPS \
    --trials $KZ_TRIALS --seed 42 \
    --outdir "$OUTDIR/kz"

echo "KZ phase complete."

# ============================================================================
# 5. DOMAIN COARSENING
# ============================================================================
echo ""
echo "=========================================="
echo "PHASE 5: Domain Coarsening"
echo "=========================================="

if $QUICK; then
    COARSEN_SIZES="20"; COARSEN_STEPS=10000; COARSEN_SAMPLE=100
else
    COARSEN_SIZES="30 40 50"; COARSEN_STEPS=200000; COARSEN_SAMPLE=10
fi

for N in $COARSEN_SIZES; do
    echo "--- Coarsening N=$N, T_quench=2.5 ---"
    cargo run --release --bin coarsening -- \
        --n $N --geometry cubic \
        --t-quench 2.5 --steps $COARSEN_STEPS --sample-every $COARSEN_SAMPLE \
        --warmup 500 --seed 42 \
        --outdir "$OUTDIR/coarsening"
done

echo "Coarsening phase complete."

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "=========================================="
echo "V2 RUN COMPLETE"
echo "=========================================="
echo "Output directory: $OUTDIR"
echo ""
echo "Data files generated:"
find "$OUTDIR" -name "*.csv" -type f | sort
echo ""
echo "Next steps:"
echo "  1. Open analysis/v2/v2_analysis.ipynb"
echo "  2. Run all cells to generate figures and results"
echo "  3. Check analysis/v2/results_summary.md for key numbers"
