#!/usr/bin/env python3
"""Cross-platform bootstrap for the analysis environment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV = REPO_ROOT / "analysis" / ".venv"
REQUIREMENTS = REPO_ROOT / "analysis" / "requirements.txt"
BASELINE_SCRIPT = REPO_ROOT / "analysis" / "scripts" / "reproduce_classical_baseline.py"


def venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> None:
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the analysis virtual environment, install Python dependencies, "
            "and optionally verify the baseline validation workflow."
        )
    )
    parser.add_argument(
        "--venv-dir",
        type=Path,
        default=DEFAULT_VENV,
        help="Virtual environment path. Defaults to analysis/.venv.",
    )
    parser.add_argument(
        "--skip-pip-upgrade",
        action="store_true",
        help="Skip upgrading pip/setuptools/wheel in the virtual environment.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip installing Python requirements into the virtual environment.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip cargo build verification.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run the quick classical baseline check after installing dependencies.",
    )
    return parser.parse_args()


def ensure_rust() -> None:
    for binary in (["rustc", "--version"], ["cargo", "--version"]):
        result = subprocess.run(binary, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            name = binary[0]
            raise SystemExit(
                f"missing required Rust tool '{name}'. Install Rust via https://rustup.rs before bootstrapping."
            )
        print(result.stdout.strip())


def main() -> None:
    args = parse_args()
    ensure_rust()

    venv_dir = args.venv_dir.expanduser().resolve()
    if not venv_dir.exists():
        print(f"creating virtual environment at {venv_dir}")
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(venv_dir)
    else:
        print(f"using existing virtual environment at {venv_dir}")

    python_bin = venv_python_path(venv_dir)
    if not python_bin.exists():
        raise SystemExit(f"virtual environment python not found: {python_bin}")

    if not args.skip_install:
        if not args.skip_pip_upgrade:
            run([str(python_bin), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        run([str(python_bin), "-m", "pip", "install", "-r", str(REQUIREMENTS)])

    if not args.skip_build:
        run(["cargo", "build", "--release"])

    if args.verify:
        run([str(python_bin), str(BASELINE_SCRIPT), "--quick", "--analysis-only", "--skip-promotion"])

    print("")
    print("bootstrap complete")
    print(f"venv python: {python_bin}")
    print("next steps:")
    print(f"  {python_bin} {BASELINE_SCRIPT} --quick")


if __name__ == "__main__":
    main()
