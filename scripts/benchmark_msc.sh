#!/usr/bin/env bash
# Benchmark matrix: single-spin Metropolis vs MSC vs Wolff on GPU.
# Tests three temperature regimes × multiple sizes.
# Reports wall-clock time per configuration.
#
# Usage: bash scripts/benchmark_msc.sh
#
# Prerequisites:
#   - CUDA toolkit installed (nvcc on PATH or CUDA_PATH set)
#   - NVIDIA GPU available
#
# Output: results in /tmp/bench_gpu/<algo>_<regime>_N<size>/
set -e

echo "=== Building gpu_fss (release + CUDA) ==="
cargo build --release --features cuda --bin gpu_fss

ALGORITHMS="metropolis msc wolff"
# Three regimes: low T (ordered), near Tc (critical), high T (disordered)
REGIMES="low_T:3.0:3.2 near_Tc:4.4:4.6 high_T:5.8:6.2"
# MSC requires N % 32 == 0, so use 32 and 64 (128 if VRAM allows)
SIZES="32 64"
REPLICAS=20
WARMUP=2000
SAMPLES=10000

OUTBASE="/tmp/bench_gpu"
rm -rf "$OUTBASE"

echo ""
echo "=== Benchmark Matrix ==="
echo "Algorithms: $ALGORITHMS"
echo "Regimes:    low_T (3.0-3.2), near_Tc (4.4-4.6), high_T (5.8-6.2)"
echo "Sizes:      $SIZES"
echo "Replicas:   $REPLICAS"
echo "Warmup:     $WARMUP sweeps"
echo "Samples:    $SAMPLES sweeps"
echo ""

for algo in $ALGORITHMS; do
  for regime_spec in $REGIMES; do
    IFS=: read -r regime tmin tmax <<< "$regime_spec"
    for n in $SIZES; do
      outdir="$OUTBASE/${algo}_${regime}_N${n}"
      mkdir -p "$outdir"
      echo "--- $algo | $regime (T=$tmin..$tmax) | N=$n ---"
      { time target/release/gpu_fss \
        --model ising --sizes "$n" \
        --tmin "$tmin" --tmax "$tmax" --replicas "$REPLICAS" \
        --warmup "$WARMUP" --samples "$SAMPLES" \
        --exchange-every 10 --algorithm "$algo" \
        --outdir "$outdir" 2>/dev/null ; } 2>&1 | grep real
      echo ""
    done
  done
done

echo "=== Results ==="
echo ""
echo "Sample summaries (first 3 lines of each):"
echo ""
for algo in $ALGORITHMS; do
  echo "--- $algo ---"
  for regime_spec in $REGIMES; do
    IFS=: read -r regime tmin tmax <<< "$regime_spec"
    for n in $SIZES; do
      csv="$OUTBASE/${algo}_${regime}_N${n}/gpu_fss_ising_N${n}_summary.csv"
      if [ -f "$csv" ]; then
        echo "  $regime N=$n:"
        head -2 "$csv" | sed 's/^/    /'
      fi
    done
  done
  echo ""
done

echo "=== All results saved to $OUTBASE ==="
echo ""
echo "Next steps:"
echo "  1. Compare E, M values across algorithms (should agree within error)"
echo "  2. Compute ESS/sec from timeseries files for the definitive metric"
echo "  3. If GPU Wolff underperforms, fall back to CPU Wolff + GPU Metropolis"
