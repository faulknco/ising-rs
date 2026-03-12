#!/usr/bin/env python3
"""Run a multi-D Heisenberg anisotropy crossover campaign and refresh analysis outputs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from common import resolve_python


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "analysis" / "data" / "anisotropy_campaign_cpu"
HEIS_FSS_BIN = REPO_ROOT / "target" / "release" / "heisenberg_fss"
ANALYSIS_SCRIPT = REPO_ROOT / "analysis" / "scripts" / "analyze_heisenberg_anisotropy.py"

# Stage-1-informed production windows. These are intended for the first real
# crossover campaign and may still need one final refinement for D=-2.0.
DEFAULT_T_WINDOWS: dict[float, tuple[float, float]] = {
    -2.0: (0.55, 1.05),
    -1.0: (0.85, 1.20),
    -0.5: (0.85, 1.15),
    0.0: (1.25, 1.60),
    0.5: (0.90, 1.20),
    1.0: (0.95, 1.30),
    2.0: (0.78, 1.05),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_d_label(d_value: float) -> str:
    if d_value == 0.0:
        return "d0"
    sign = "m" if d_value < 0 else "p"
    magnitude = str(abs(d_value)).replace(".", "p")
    return f"d{sign}{magnitude}"


def parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        values.append(float(piece))
    if not values:
        raise SystemExit("at least one D value is required")
    return values


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value <= 0:
            raise SystemExit(f"invalid integer value {value}")
        values.append(value)
    if not values:
        raise SystemExit("at least one lattice size is required")
    return values


def parse_window_spec(raw: str, d_values: list[float]) -> dict[float, tuple[float, float]]:
    windows = dict(DEFAULT_T_WINDOWS)
    if raw:
        for chunk in raw.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                d_str, tmin_str, tmax_str = chunk.split(":")
            except ValueError as exc:
                raise SystemExit(
                    f"invalid --t-window-spec entry '{chunk}', expected d:tmin:tmax"
                ) from exc
            d_value = float(d_str)
            windows[d_value] = (float(tmin_str), float(tmax_str))

    missing = [d for d in d_values if d not in windows]
    if missing:
        missing_str = ", ".join(str(d) for d in missing)
        raise SystemExit(f"no temperature windows configured for D values: {missing_str}")
    return {d: windows[d] for d in d_values}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible Heisenberg anisotropy crossover campaign. "
            "Creates one dataset directory per anisotropy value, refreshes the "
            "symmetry-aware analysis automatically, and writes campaign status files."
        )
    )
    parser.add_argument("--d-values", default="-2,-1,-0.5,0,0.5,1,2")
    parser.add_argument("--sizes", default="32,64,96,128,192")
    parser.add_argument("--steps", type=int, default=49)
    parser.add_argument("--warmup", type=int, default=4000)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--overrelax", type=int, default=5)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--j", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Campaign root directory.",
    )
    parser.add_argument(
        "--campaign-name",
        default="",
        help="Optional subdirectory name under output-root.",
    )
    parser.add_argument(
        "--t-window-spec",
        default="",
        help="Optional overrides as 'd:tmin:tmax;d:tmin:tmax'.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--heisenberg-fss-bin",
        type=Path,
        default=HEIS_FSS_BIN,
        help="Path to the heisenberg_fss binary to run.",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip all simulations and only regenerate the anisotropy analysis.",
    )
    return parser.parse_args()


def append_campaign_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not path.exists()
    fieldnames = [
        "label",
        "d_value",
        "status",
        "started_at",
        "finished_at",
        "dataset_root",
        "stderr_tail",
    ]
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if header_needed:
            writer.writeheader()
        writer.writerow(row)


def write_campaign_manifest(
    campaign_root: Path,
    d_values: list[float],
    sizes: list[int],
    windows: dict[float, tuple[float, float]],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "generated_at": utc_now_iso(),
        "repo_root": str(REPO_ROOT),
        "campaign_root": str(campaign_root),
        "d_values": d_values,
        "sizes": sizes,
        "temperature_windows": {str(d): {"tmin": tmin, "tmax": tmax} for d, (tmin, tmax) in windows.items()},
        "steps": args.steps,
        "warmup": args.warmup,
        "samples": args.samples,
        "overrelax": args.overrelax,
        "delta": args.delta,
        "j": args.j,
        "seed": args.seed,
    }
    (campaign_root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def write_campaign_plan(
    campaign_root: Path,
    d_values: list[float],
    sizes: list[int],
    windows: dict[float, tuple[float, float]],
    args: argparse.Namespace,
) -> None:
    rows: list[dict[str, object]] = []
    for d_value in d_values:
        tmin, tmax = windows[d_value]
        rows.append(
            {
                "label": format_d_label(d_value),
                "d_value": d_value,
                "sizes": ",".join(str(n) for n in sizes),
                "tmin": tmin,
                "tmax": tmax,
                "steps": args.steps,
                "warmup": args.warmup,
                "samples": args.samples,
                "delta": args.delta,
                "overrelax": args.overrelax,
                "j": args.j,
                "seed": args.seed,
            }
        )
    with (campaign_root / "campaign_plan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(campaign_root: Path) -> None:
    subprocess.run(
        [
            resolve_python(),
            str(ANALYSIS_SCRIPT),
            "--root",
            str(campaign_root),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def run_dataset(
    campaign_root: Path,
    d_value: float,
    sizes: list[int],
    tmin: float,
    tmax: float,
    args: argparse.Namespace,
) -> None:
    label = format_d_label(d_value)
    dataset_root = campaign_root / label
    expected_paths = [dataset_root / f"heisenberg_fss_N{n}.csv" for n in sizes]
    if args.skip_existing and all(path.exists() for path in expected_paths):
        append_campaign_row(
            campaign_root / "campaign_status.csv",
            {
                "label": label,
                "d_value": d_value,
                "status": "skipped_existing",
                "finished_at": utc_now_iso(),
                "dataset_root": str(dataset_root),
            },
        )
        run_analysis(campaign_root)
        return

    dataset_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.heisenberg_fss_bin),
        "--sizes",
        ",".join(str(n) for n in sizes),
        "--anisotropy-d",
        str(d_value),
        "--tmin",
        str(tmin),
        "--tmax",
        str(tmax),
        "--steps",
        str(args.steps),
        "--warmup",
        str(args.warmup),
        "--samples",
        str(args.samples),
        "--delta",
        str(args.delta),
        "--overrelax",
        str(args.overrelax),
        "--j",
        str(args.j),
        "--seed",
        str(args.seed),
        "--outdir",
        str(dataset_root),
    ]

    started_at = utc_now_iso()
    append_campaign_row(
        campaign_root / "campaign_status.csv",
        {
            "label": label,
            "d_value": d_value,
            "status": "started",
            "started_at": started_at,
            "dataset_root": str(dataset_root),
        },
    )

    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        append_campaign_row(
            campaign_root / "campaign_status.csv",
            {
                "label": label,
                "d_value": d_value,
                "status": "failed",
                "started_at": started_at,
                "finished_at": utc_now_iso(),
                "dataset_root": str(dataset_root),
                "stderr_tail": "\n".join(result.stderr.strip().splitlines()[-10:]),
            },
        )
        raise RuntimeError(
            f"anisotropy dataset {label} failed\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )

    append_campaign_row(
        campaign_root / "campaign_status.csv",
        {
            "label": label,
            "d_value": d_value,
            "status": "completed",
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "dataset_root": str(dataset_root),
        },
    )
    run_analysis(campaign_root)


def main() -> int:
    args = parse_args()
    d_values = parse_float_list(args.d_values)
    sizes = parse_int_list(args.sizes)
    windows = parse_window_spec(args.t_window_spec, d_values)

    campaign_root = (
        args.output_root / args.campaign_name if args.campaign_name else args.output_root
    )
    campaign_root.mkdir(parents=True, exist_ok=True)

    write_campaign_manifest(campaign_root, d_values, sizes, windows, args)
    write_campaign_plan(campaign_root, d_values, sizes, windows, args)

    if args.analysis_only:
        run_analysis(campaign_root)
        return 0

    if not args.heisenberg_fss_bin.exists():
        raise SystemExit(
            f"{args.heisenberg_fss_bin} not found. Build it first with: cargo build --release --bin heisenberg_fss"
        )

    for d_value in d_values:
        tmin, tmax = windows[d_value]
        run_dataset(campaign_root, d_value, sizes, tmin, tmax, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
