# Setup

## Goal

This page is the shortest path from a fresh machine to a working CPU validation baseline.

The intended outcome is:

1. Rust builds cleanly
2. the analysis Python environment exists
3. the scripted validation workflow runs

## Prerequisites

Install these first:

- Rust via `rustup`
- Python 3.11 or newer
- Git

Rust is pinned by [rust-toolchain.toml](/Users/faulknco/Projects/ising-rs/rust-toolchain.toml).

## One-Command Bootstrap

From the repo root:

```bash
python scripts/bootstrap_analysis.py --verify
```

On Windows, if `python` is not the right launcher, use:

```powershell
py -3 scripts/bootstrap_analysis.py --verify
```

This command:

- creates `analysis/.venv` if needed
- installs `analysis/requirements.txt`
- runs `cargo build --release`
- runs the quick baseline validation check without promoting a result pack

For an already-prepared machine, a lighter verification pass is:

```bash
python scripts/bootstrap_analysis.py --skip-install --verify
```

## Manual Steps

If you prefer to do this by hand:

### 1. Build Rust code

```bash
cargo build --release
```

### 2. Create the analysis virtual environment

macOS/Linux:

```bash
python -m venv analysis/.venv
analysis/.venv/bin/python -m pip install -U pip setuptools wheel
analysis/.venv/bin/python -m pip install -r analysis/requirements.txt
```

Windows PowerShell:

```powershell
py -3 -m venv analysis/.venv
analysis\.venv\Scripts\python.exe -m pip install -U pip setuptools wheel
analysis\.venv\Scripts\python.exe -m pip install -r analysis\requirements.txt
```

### 3. Verify the baseline

```bash
python analysis/scripts/reproduce_classical_baseline.py --quick --analysis-only --skip-promotion
```

## Expected Outputs

The quick validation run should regenerate:

- `analysis/data/derived/validation/validation_metrics.csv`
- `analysis/data/derived/validation/validation_summary.json`
- figures under `analysis/figures/generated/validation/`
- a manifest under `analysis/data/manifests/validation/`

## Notes

- CUDA is optional and not part of this bootstrap path.
- The bootstrap path targets CPU reproducibility first.
- Full publication-grade reruns should be done after the quick validation check passes.
