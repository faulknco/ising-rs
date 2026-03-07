#!/bin/bash
# Build the paper draft
# Usage: cd paper && bash build.sh

# Copy figures from analysis directory
FIGURES=(
    fss_observables.png
    fss_binder.png
    fss_peak_scaling.png
    fss_collapse.png
    fss_collapse_global.png
    fss_reweighted.png
    fss_reweighted_scaling.png
)

for fig in "${FIGURES[@]}"; do
    src="../analysis/$fig"
    if [ -f "$src" ]; then
        cp "$src" .
        echo "Copied $fig"
    else
        echo "MISSING: $fig (run the notebook first)"
    fi
done

# KZ figure lives in analysis/data/
if [ -f "../analysis/data/kz_fit.png" ]; then
    cp "../analysis/data/kz_fit.png" .
    echo "Copied kz_fit.png"
else
    echo "MISSING: kz_fit.png (run the KZ notebook first)"
fi

# Validation figures
VAL_FIGURES=(
    val_2d_onsager.png
    val_exact_enum.png
)

for fig in "${VAL_FIGURES[@]}"; do
    src="../analysis/$fig"
    if [ -f "$src" ]; then
        cp "$src" .
        echo "Copied $fig"
    else
        echo "MISSING: $fig (run the validation notebook first)"
    fi
done

# Build LaTeX (if pdflatex is available)
if command -v pdflatex &>/dev/null; then
    pdflatex draft.tex
    pdflatex draft.tex  # twice for references
    echo "Built draft.pdf"
else
    echo "pdflatex not found — install a TeX distribution to compile"
    echo "Or use Overleaf: upload draft.tex and the .png files"
fi
