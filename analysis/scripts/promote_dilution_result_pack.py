#!/usr/bin/env python3
"""Promote a dilution campaign into a versioned publishable result pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACK_ROOT = REPO_ROOT / "results" / "published"


def git_commit() -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the derived outputs of a dilution campaign into a versioned result pack "
            "under results/published/."
        )
    )
    parser.add_argument("--campaign-root", type=Path, required=True, help="Completed or partial dilution campaign root.")
    parser.add_argument("--pack-name", required=True, help="Directory name for the published pack.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_PACK_ROOT,
        help="Base directory that will receive the published result pack.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Also copy raw graph and sweep files into the pack. Default is metadata-only for raw inputs.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_hidden_artifact(path: Path) -> bool:
    return any(part.startswith("._") for part in path.parts)


def collect_files(campaign_root: Path, include_raw: bool) -> list[Path]:
    patterns = [
        "campaign_plan.csv",
        "campaign_status.csv",
        "campaign_tc_summary_N*.csv",
        "campaign_tc_summary_all_sizes.csv",
        "analysis_fss/*.csv",
        "analysis_fss/*.json",
        "analysis_fss/*.png",
        "size_N*/derived/dilution/*.csv",
        "size_N*/figures/dilution/*.png",
        "size_N*/manifests/dilution/*.json",
    ]
    if include_raw:
        patterns.extend(
            [
                "size_N*/raw/dilution/graphs/**/*.json",
                "size_N*/raw/dilution/sweeps/**/*.csv",
            ]
        )

    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in campaign_root.glob(pattern) if path.is_file() and not is_hidden_artifact(path))
    return sorted(set(files))


def copy_files(files: list[Path], source_root: Path, pack_root: Path) -> list[dict[str, object]]:
    inventory = []
    for source in files:
        rel = source.relative_to(source_root)
        dest = pack_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        inventory.append(
            {
                "relative_path": str(rel),
                "size_bytes": dest.stat().st_size,
                "sha256": sha256_file(dest),
            }
        )
    return inventory


def write_pack_readme(pack_root: Path, campaign_root: Path, include_raw: bool, inventory: list[dict[str, object]]) -> Path:
    readme_path = pack_root / "README.md"
    lines = [
        "# Dilution Result Pack",
        "",
        f"- source campaign: `{campaign_root}`",
        f"- promoted at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- source git commit: `{git_commit()}`",
        f"- includes raw data: `{'yes' if include_raw else 'no'}`",
        f"- file count: `{len(inventory)}`",
        "",
        "## Scope",
        "",
        "This pack snapshots the derived outputs of a dilution campaign into a stable folder that can be cited,",
        "reviewed, or copied elsewhere without depending on the original USB layout.",
        "",
        "## Contents",
        "",
        "- campaign-level ledgers and Tc summaries",
        "- Binder/FSS analysis outputs when available",
        "- per-size derived tables, figures, and manifests",
        "- raw files only when `--include-raw` is used",
        "",
        "## Caveats",
        "",
        "- A partial campaign is still promotable; inspect `campaign_status.csv` before treating the pack as final.",
        "- Raw graph and sweep files are excluded by default to keep packs small.",
        "- The inventory and checksums live in `pack_manifest.json`.",
        "",
    ]
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    return readme_path


def write_pack_manifest(
    pack_root: Path,
    campaign_root: Path,
    include_raw: bool,
    inventory: list[dict[str, object]],
    readme_path: Path,
) -> Path:
    manifest = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_campaign_root": str(campaign_root),
        "source_git_commit": git_commit(),
        "pack_root": str(pack_root),
        "includes_raw": include_raw,
        "readme": str(readme_path.relative_to(pack_root)),
        "files": inventory,
    }
    manifest_path = pack_root / "pack_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.expanduser().resolve()
    if not campaign_root.exists():
        raise SystemExit(f"campaign root does not exist: {campaign_root}")

    pack_root = args.output_root.expanduser().resolve() / args.pack_name
    if pack_root.exists():
        raise SystemExit(f"pack already exists: {pack_root}")
    pack_root.mkdir(parents=True, exist_ok=False)

    files = collect_files(campaign_root, include_raw=args.include_raw)
    if not files:
        raise SystemExit(f"no promotable campaign files found in {campaign_root}")

    inventory = copy_files(files, campaign_root, pack_root)
    readme_path = write_pack_readme(pack_root, campaign_root, args.include_raw, inventory)
    manifest_path = write_pack_manifest(pack_root, campaign_root, args.include_raw, inventory, readme_path)

    print(f"promoted {len(inventory)} files into {pack_root}")
    print(f"wrote {readme_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
