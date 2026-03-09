#!/usr/bin/env python3
"""One-command entrypoint for the classical CPU validation baseline."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import resolve_python


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_SCRIPT = REPO_ROOT / "analysis" / "scripts" / "reproduce_validation.py"
PROMOTE_SCRIPT = REPO_ROOT / "analysis" / "scripts" / "promote_validation_result_pack.py"


def run_command(cmd: list[str]) -> None:
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    if result.stdout.strip():
        print(result.stdout.strip())


def default_pack_name(quick: bool) -> str:
    return "classical_validation_quick_v1" if quick else "classical_validation_full_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the classical CPU validation baseline and optionally promote it into "
            "a versioned result pack."
        )
    )
    parser.add_argument("--quick", action="store_true", help="Run the quick validation configuration.")
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip raw data generation and only analyse the current validation datasets.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing raw CSVs when the validation script supports it.",
    )
    parser.add_argument(
        "--skip-promotion",
        action="store_true",
        help="Do not create/update a published result pack.",
    )
    parser.add_argument(
        "--pack-name",
        default="",
        help="Optional pack name override. Defaults to a mode-specific validation pack name.",
    )
    parser.add_argument(
        "--pack-output-root",
        type=Path,
        default=None,
        help="Optional output root for the promoted validation pack.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include raw validation CSVs when promoting the pack.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python_exe = resolve_python()

    validation_cmd = [python_exe, str(VALIDATION_SCRIPT)]
    if args.quick:
        validation_cmd.append("--quick")
    if args.analysis_only:
        validation_cmd.append("--analysis-only")
    if args.skip_existing:
        validation_cmd.append("--skip-existing")
    run_command(validation_cmd)

    if args.skip_promotion:
        return

    pack_name = args.pack_name or default_pack_name(args.quick)
    promote_cmd = [
        python_exe,
        str(PROMOTE_SCRIPT),
        "--pack-name",
        pack_name,
    ]
    if args.pack_output_root is not None:
        promote_cmd.extend(["--output-root", str(args.pack_output_root)])
    if args.include_raw:
        promote_cmd.append("--include-raw")
    run_command(promote_cmd)


if __name__ == "__main__":
    main()
