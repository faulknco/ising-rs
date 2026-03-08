#!/usr/bin/env python3
"""Watch a dilution campaign, refresh FSS outputs, and promote a final result pack."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import pandas as pd

from common import resolve_python


REPO_ROOT = Path(__file__).resolve().parents[2]
FSS_SCRIPT = REPO_ROOT / "analysis" / "scripts" / "reproduce_dilution_fss.py"
PROMOTE_SCRIPT = REPO_ROOT / "analysis" / "scripts" / "promote_dilution_result_pack.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watch a dilution campaign directory, refresh Binder/FSS outputs when new size summaries "
            "appear, and promote the campaign into results/published when all planned sizes complete."
        )
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--pack-name", required=True)
    parser.add_argument(
        "--pack-output-root",
        type=Path,
        default=REPO_ROOT / "results" / "published",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--once", action="store_true", help="Run one check cycle and exit.")
    return parser.parse_args()


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)


def planned_sizes(campaign_root: Path) -> set[int]:
    plan_path = campaign_root / "campaign_plan.csv"
    if not plan_path.exists():
        return set()
    plan = pd.read_csv(plan_path)
    if "n" not in plan.columns:
        return set()
    return {int(value) for value in plan["n"].dropna().unique()}


def completed_sizes(campaign_root: Path) -> set[int]:
    status_path = campaign_root / "campaign_status.csv"
    if not status_path.exists():
        return set()
    status = pd.read_csv(status_path)
    if status.empty:
        return set()
    completed = status.loc[status["status"] == "completed", "size"]
    return {int(value) for value in completed.dropna().unique()}


def newest_summary_mtime(campaign_root: Path) -> float | None:
    summary_paths = sorted(campaign_root.glob("campaign_tc_summary_N*.csv"))
    if not summary_paths:
        return None
    return max(path.stat().st_mtime for path in summary_paths)


def fss_summary_mtime(campaign_root: Path) -> float | None:
    fss_summary = campaign_root / "analysis_fss" / "dilution_fss_summary.json"
    if not fss_summary.exists():
        return None
    return fss_summary.stat().st_mtime


def maybe_refresh_fss(campaign_root: Path) -> None:
    summary_mtime = newest_summary_mtime(campaign_root)
    if summary_mtime is None:
        return
    existing_fss_mtime = fss_summary_mtime(campaign_root)
    if existing_fss_mtime is not None and existing_fss_mtime >= summary_mtime:
        return

    result = run_command([resolve_python(), str(FSS_SCRIPT), "--campaign-root", str(campaign_root)])
    if result.returncode != 0:
        raise RuntimeError(
            "failed to refresh dilution FSS outputs\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    print(result.stdout.strip())


def maybe_promote_pack(args: argparse.Namespace) -> bool:
    pack_root = args.pack_output_root.expanduser().resolve() / args.pack_name
    if pack_root.exists():
        return True

    plan_sizes = planned_sizes(args.campaign_root)
    done_sizes = completed_sizes(args.campaign_root)
    if not plan_sizes or plan_sizes != done_sizes:
        return False

    cmd = [
        resolve_python(),
        str(PROMOTE_SCRIPT),
        "--campaign-root",
        str(args.campaign_root),
        "--pack-name",
        args.pack_name,
        "--output-root",
        str(args.pack_output_root),
    ]
    if args.include_raw:
        cmd.append("--include-raw")
    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            "failed to promote dilution result pack\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    print(result.stdout.strip())
    return True


def main() -> None:
    args = parse_args()
    args.campaign_root = args.campaign_root.expanduser().resolve()

    while True:
        maybe_refresh_fss(args.campaign_root)
        promoted = maybe_promote_pack(args)
        if args.once or promoted:
            return
        time.sleep(max(args.poll_seconds, 1))


if __name__ == "__main__":
    main()
