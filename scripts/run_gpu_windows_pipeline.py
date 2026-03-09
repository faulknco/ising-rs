#!/usr/bin/env python3
"""
Windows-friendly GPU validation and publication pipeline.

Runs:
1. CPU test suite
2. CUDA-gated ignored tests
3. gpu_fss smoke runs for Ising, XY, Heisenberg
4. gpu_jfit smoke runs for Ising, XY, Heisenberg
5. Ferrenberg-Swendsen / WHAM self-test
6. Full publication GPU runs if requested

Usage:
    python scripts/run_gpu_windows_pipeline.py --validate-only
    python scripts/run_gpu_windows_pipeline.py --publish-on-success
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "analysis"
SCRIPTS_DIR = REPO_ROOT / "scripts"
GRAPH_DIR = ANALYSIS_DIR / "graphs"


def run(cmd: list[str], *, label: str, env: dict[str, str] | None = None) -> None:
    print(f"\n=== {label} ===", flush=True)
    print(" ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required tool not found on PATH: {name}")


def check_cuda_env() -> None:
    cuda_path = os.environ.get("CUDA_PATH")
    if not cuda_path:
        raise SystemExit(
            "CUDA_PATH is not set. Install the CUDA toolkit on Windows and open a shell "
            "with CUDA_PATH available before running this script."
        )
    nvcc = Path(cuda_path) / "bin" / "nvcc.exe"
    if not nvcc.exists():
        raise SystemExit(f"nvcc not found at expected location: {nvcc}")


def assert_csv_has_data(path: Path, *, min_rows: int = 1) -> None:
    if not path.exists():
        raise SystemExit(f"Expected output file missing: {path}")
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < min_rows + 1:
        raise SystemExit(f"CSV has no data rows: {path}")
    if any("NaN" in ",".join(row) for row in rows):
        raise SystemExit(f"CSV contains NaN values: {path}")


def run_gpu_fss_smoke(model: str, outdir: Path) -> None:
    run(
        [
            "cargo",
            "run",
            "--release",
            "--features",
            "cuda",
            "--bin",
            "gpu_fss",
            "--",
            "--model",
            model,
            "--sizes",
            "4",
            "--tmin",
            "4.4" if model == "ising" else ("2.10" if model == "xy" else "1.35"),
            "--tmax",
            "4.6" if model == "ising" else ("2.30" if model == "xy" else "1.55"),
            "--replicas",
            "4",
            "--warmup",
            "50",
            "--samples",
            "100",
            "--exchange-every",
            "10",
            "--delta",
            "0.5",
            "--overrelax",
            "2",
            "--outdir",
            str(outdir),
        ],
        label=f"gpu_fss smoke ({model})",
    )
    assert_csv_has_data(outdir / f"gpu_fss_{model}_N4_summary.csv")
    assert_csv_has_data(outdir / f"gpu_fss_{model}_N4_timeseries.csv")


def run_gpu_jfit_smoke(model: str, outdir: Path) -> None:
    graph = GRAPH_DIR / "bcc_N4.json"
    tmin, tmax = {
        "ising": ("6.0", "6.7"),
        "xy": ("2.1", "2.5"),
        "heisenberg": ("1.2", "1.6"),
    }[model]
    run(
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "gpu_jfit",
            "--",
            "--model",
            model,
            "--graph",
            str(graph),
            "--tmin",
            tmin,
            "--tmax",
            tmax,
            "--replicas",
            "6",
            "--warmup",
            "50",
            "--samples",
            "100",
            "--exchange-every",
            "10",
            "--outdir",
            str(outdir),
        ],
        label=f"gpu_jfit smoke ({model})",
    )
    assert_csv_has_data(outdir / f"gpu_jfit_{model}_bcc_N4.csv")


def run_publication(outdir: Path) -> None:
    publication_dir = outdir / "publication"
    publication_dir.mkdir(parents=True, exist_ok=True)

    models = [
        (
            "ising",
            {
                "tmin": "4.40",
                "tmax": "4.62",
                "replicas": "32",
                "warmup": "5000",
                "samples": "100000",
                "delta": "0.5",
                "overrelax": "0",
            },
        ),
        (
            "heisenberg",
            {
                "tmin": "1.35",
                "tmax": "1.55",
                "replicas": "20",
                "warmup": "5000",
                "samples": "50000",
                "delta": "0.5",
                "overrelax": "5",
            },
        ),
        (
            "xy",
            {
                "tmin": "2.10",
                "tmax": "2.30",
                "replicas": "20",
                "warmup": "5000",
                "samples": "50000",
                "delta": "0.5",
                "overrelax": "5",
            },
        ),
    ]

    for model, params in models:
        run(
            [
                "cargo",
                "run",
                "--release",
                "--features",
                "cuda",
                "--bin",
                "gpu_fss",
                "--",
                "--model",
                model,
                "--sizes",
                "8,16,32,64,128",
                "--tmin",
                params["tmin"],
                "--tmax",
                params["tmax"],
                "--replicas",
                params["replicas"],
                "--warmup",
                params["warmup"],
                "--samples",
                params["samples"],
                "--exchange-every",
                "10",
                "--delta",
                params["delta"],
                "--overrelax",
                params["overrelax"],
                "--measure-every",
                "5",
                "--outdir",
                str(publication_dir),
            ],
            label=f"publication run ({model})",
        )

    run(
        [sys.executable, str(ANALYSIS_DIR / "scripts" / "reweighting.py")],
        label="reweighting self-test (post-publication)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate CUDA GPU workflows on Windows and optionally run publication jobs."
    )
    parser.add_argument(
        "--outdir",
        default="analysis/data/gpu_windows_pipeline",
        help="Root output directory for smoke tests and publication outputs.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation only and stop before the publication sweep.",
    )
    mode.add_argument(
        "--publish-on-success",
        action="store_true",
        help="Run the publication sweep after validation passes.",
    )
    args = parser.parse_args()

    require_tool("cargo")
    require_tool(sys.executable)
    check_cuda_env()

    outdir = (REPO_ROOT / args.outdir).resolve()
    smoke_dir = outdir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    run(["cargo", "build", "--release", "--features", "cuda", "--bin", "gpu_fss"], label="build gpu_fss")
    run(["cargo", "build", "--release", "--bin", "gpu_jfit"], label="build gpu_jfit")
    run(["cargo", "test"], label="CPU/unit/integration tests")
    run(["cargo", "test", "--features", "cuda", "--", "--ignored"], label="CUDA ignored tests")

    for model in ("ising", "xy", "heisenberg"):
        run_gpu_fss_smoke(model, smoke_dir / f"gpu_fss_{model}")

    for model in ("ising", "xy", "heisenberg"):
        run_gpu_jfit_smoke(model, smoke_dir / f"gpu_jfit_{model}")

    run(
        [sys.executable, str(ANALYSIS_DIR / "scripts" / "reweighting.py")],
        label="reweighting self-test",
    )

    if args.publish_on_success:
        run_publication(outdir)
        print(f"\nValidation passed and publication run completed. Outputs: {outdir}")
    else:
        print(f"\nValidation passed. Smoke outputs: {smoke_dir}")


if __name__ == "__main__":
    main()
