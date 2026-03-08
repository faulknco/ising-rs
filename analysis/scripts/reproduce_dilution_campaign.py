#!/usr/bin/env python3
"""Run a multi-size dilution publishing campaign with USB-backed outputs."""

from __future__ import annotations

import argparse
import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common import default_publishing_root, resolve_python


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = default_publishing_root(REPO_ROOT)
SCRIPT_PATH = REPO_ROOT / "analysis" / "scripts" / "reproduce_dilution.py"
FSS_SCRIPT_PATH = REPO_ROOT / "analysis" / "scripts" / "reproduce_dilution_fss.py"

DEFAULT_T_WINDOWS = {
    0.0: (3.8, 5.2),
    0.1: (3.5, 4.8),
    0.2: (3.0, 4.3),
    0.3: (2.4, 3.8),
    0.4: (1.8, 3.2),
    0.5: (1.3, 2.7),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_sizes(raw: str) -> list[int]:
    sizes = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value <= 0:
            raise SystemExit(f"invalid size {value}")
        sizes.append(value)
    if not sizes:
        raise SystemExit("at least one size is required")
    return sizes


def parse_p_values(raw: str) -> list[float]:
    values = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        values.append(float(piece))
    if not values:
        raise SystemExit("at least one dilution value is required")
    return values


def build_t_window_spec(p_values: list[float]) -> str:
    entries = []
    for p_removed in p_values:
        if p_removed not in DEFAULT_T_WINDOWS:
            raise SystemExit(f"no default temperature window configured for p={p_removed}")
        tmin, tmax = DEFAULT_T_WINDOWS[p_removed]
        entries.append(f"{p_removed}:{tmin}:{tmax}")
    return ";".join(entries)


def build_realizations_spec(undiluted_realizations: int) -> str:
    return f"0.0:{undiluted_realizations}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-size dilution campaign into a USB-backed publishing_data folder. "
            "Each size gets its own subdirectory with raw, derived, figure, and manifest outputs."
        )
    )
    parser.add_argument("--sizes", default="24,28,32", help="Comma-separated lattice sizes.")
    parser.add_argument("--p-values", default="0.0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--realizations", type=int, default=12)
    parser.add_argument("--undiluted-realizations", type=int, default=4)
    parser.add_argument("--steps", type=int, default=51)
    parser.add_argument("--warmup", type=int, default=4000)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--j", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Base output directory. Set this to an external drive path if you want USB-backed outputs.",
    )
    parser.add_argument(
        "--campaign-name",
        default="",
        help="Optional subdirectory name under output-root. Defaults to a timestamped campaign folder.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def append_campaign_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not path.exists()
    fieldnames = [
        "size",
        "status",
        "started_at",
        "finished_at",
        "size_root",
        "stderr_tail",
    ]
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if header_needed:
            writer.writeheader()
        writer.writerow(row)


def run_size(args: argparse.Namespace, campaign_root: Path, size: int, t_window_spec: str, p_values: list[float]) -> None:
    size_root = campaign_root / f"size_N{size}"
    python_exe = resolve_python()
    cmd = [
        python_exe,
        str(SCRIPT_PATH),
        "--n",
        str(size),
        "--p-values",
        ",".join(str(p) for p in p_values),
        "--realizations",
        str(args.realizations),
        "--realizations-spec",
        build_realizations_spec(args.undiluted_realizations),
        "--t-window-spec",
        t_window_spec,
        "--steps",
        str(args.steps),
        "--warmup",
        str(args.warmup),
        "--samples",
        str(args.samples),
        "--j",
        str(args.j),
        "--seed",
        str(args.seed),
        "--max-workers",
        str(args.max_workers),
        "--output-root",
        str(size_root),
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.quick:
        cmd.append("--quick")

    started_at = utc_now_iso()
    append_campaign_row(
        campaign_root / "campaign_status.csv",
        {
            "size": size,
            "status": "started",
            "started_at": started_at,
            "size_root": str(size_root),
        },
    )
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        append_campaign_row(
            campaign_root / "campaign_status.csv",
            {
                "size": size,
                "status": "failed",
                "started_at": started_at,
                "finished_at": utc_now_iso(),
                "size_root": str(size_root),
                "stderr_tail": "\n".join(result.stderr.strip().splitlines()[-10:]),
            },
        )
        raise RuntimeError(
            f"size N={size} failed\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )

    append_campaign_row(
        campaign_root / "campaign_status.csv",
        {
            "size": size,
            "status": "completed",
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "size_root": str(size_root),
        },
    )

    tc_summary_path = size_root / "derived" / "dilution" / "dilution_tc_summary.csv"
    if tc_summary_path.exists():
        tc_summary = pd.read_csv(tc_summary_path)
        tc_summary.insert(0, "n", size)
        tc_summary.to_csv(
            campaign_root / f"campaign_tc_summary_N{size}.csv",
            index=False,
        )

    aggregate_campaign_summaries(campaign_root)
    update_fss_analysis(campaign_root)


def aggregate_campaign_summaries(campaign_root: Path) -> None:
    summary_paths = sorted(campaign_root.glob("campaign_tc_summary_N*.csv"))
    if not summary_paths:
        return
    frames = [pd.read_csv(path) for path in summary_paths]
    combined = pd.concat(frames, ignore_index=True).sort_values(["n", "p_removed"])
    combined.to_csv(campaign_root / "campaign_tc_summary_all_sizes.csv", index=False)


def update_fss_analysis(campaign_root: Path) -> None:
    if not FSS_SCRIPT_PATH.exists():
        return
    python_exe = resolve_python()
    result = subprocess.run(
        [python_exe, str(FSS_SCRIPT_PATH), "--campaign-root", str(campaign_root)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to update dilution Binder/FSS analysis\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )


def write_campaign_plan(args: argparse.Namespace, campaign_root: Path, sizes: list[int], p_values: list[float]) -> None:
    plan_rows = []
    for size in sizes:
        for p_removed in p_values:
            tmin, tmax = DEFAULT_T_WINDOWS[p_removed]
            n_realizations = args.undiluted_realizations if p_removed == 0.0 else args.realizations
            plan_rows.append(
                {
                    "n": size,
                    "p_removed": p_removed,
                    "realizations": n_realizations,
                    "tmin": tmin,
                    "tmax": tmax,
                    "steps": args.steps,
                    "warmup": args.warmup,
                    "samples": args.samples,
                }
            )
    pd.DataFrame(plan_rows).to_csv(campaign_root / "campaign_plan.csv", index=False)


def main() -> None:
    args = parse_args()
    sizes = parse_sizes(args.sizes)
    p_values = parse_p_values(args.p_values)

    if args.quick:
        args.realizations = min(args.realizations, 2)
        args.undiluted_realizations = min(args.undiluted_realizations, 2)
        args.warmup = min(args.warmup, 200)
        args.samples = min(args.samples, 200)
        args.steps = min(args.steps, 11)

    output_root = args.output_root.expanduser().resolve()
    campaign_name = args.campaign_name or f"dilution_campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    campaign_root = output_root / campaign_name
    campaign_root.mkdir(parents=True, exist_ok=True)

    write_campaign_plan(args, campaign_root, sizes, p_values)
    t_window_spec = build_t_window_spec(p_values)

    for size in sizes:
        print(f"running dilution campaign for N={size}")
        run_size(args, campaign_root, size, t_window_spec, p_values)

    print(f"campaign outputs written to {campaign_root}")


if __name__ == "__main__":
    main()
