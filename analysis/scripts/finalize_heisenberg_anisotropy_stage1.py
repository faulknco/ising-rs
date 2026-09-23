#!/usr/bin/env python3
"""Wait for anisotropy Stage 1 datasets, then regenerate the analysis outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from common import resolve_python


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "analysis" / "data" / "anisotropy_stage1_cpu"
DEFAULT_DATASETS = ("dm2", "dm1", "d0", "dp2", "dm0p5", "dp0p5", "dp1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the expected Heisenberg anisotropy scouting outputs and rerun "
            "the symmetry-aware analysis once they all exist."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing one subdirectory per anisotropy dataset.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset directories to require before running analysis.",
    )
    parser.add_argument(
        "--sizes",
        default="16,32",
        help="Comma-separated lattice sizes expected in the Stage 1B datasets.",
    )
    parser.add_argument(
        "--include-batch1-sizes",
        default="64",
        help=(
            "Comma-separated extra sizes that must exist for the Batch 1 datasets "
            "(dm2, dm1, d0, dp2)."
        ),
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=15.0,
        help="Polling interval while waiting for files to appear.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.0,
        help="Optional timeout. Use 0 to wait indefinitely.",
    )
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def expected_sizes_for_dataset(dataset: str, stage1b_sizes: list[int], batch1_extra: list[int]) -> list[int]:
    if dataset in {"dm2", "dm1", "d0", "dp2"}:
        return sorted(set(stage1b_sizes + batch1_extra))
    return stage1b_sizes


def missing_outputs(root: Path, datasets: list[str], stage1b_sizes: list[int], batch1_extra: list[int]) -> list[str]:
    missing: list[str] = []
    for dataset in datasets:
        dataset_dir = root / dataset
        if not dataset_dir.is_dir():
            missing.append(f"{dataset}/")
            continue
        for size in expected_sizes_for_dataset(dataset, stage1b_sizes, batch1_extra):
            csv_path = dataset_dir / f"heisenberg_fss_N{size}.csv"
            if not csv_path.exists():
                missing.append(str(csv_path.relative_to(root)))
    return missing


def wait_for_outputs(root: Path, datasets: list[str], stage1b_sizes: list[int], batch1_extra: list[int], poll_seconds: float, timeout_seconds: float) -> None:
    start = time.monotonic()
    while True:
        missing = missing_outputs(root, datasets, stage1b_sizes, batch1_extra)
        if not missing:
            return
        if timeout_seconds > 0 and (time.monotonic() - start) >= timeout_seconds:
            sample = ", ".join(missing[:5])
            raise SystemExit(f"timed out waiting for anisotropy outputs; still missing: {sample}")
        time.sleep(poll_seconds)


def run_analysis(root: Path) -> None:
    cmd = [
        resolve_python(),
        str(REPO_ROOT / "analysis" / "scripts" / "analyze_heisenberg_anisotropy.py"),
        "--root",
        str(root),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    stage1b_sizes = parse_int_list(args.sizes)
    batch1_extra = parse_int_list(args.include_batch1_sizes)
    wait_for_outputs(
        args.root,
        datasets,
        stage1b_sizes,
        batch1_extra,
        args.poll_seconds,
        args.timeout_seconds,
    )
    run_analysis(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
