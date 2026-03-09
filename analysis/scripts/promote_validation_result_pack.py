#!/usr/bin/env python3
"""Promote the classical validation outputs into a versioned result pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACK_ROOT = REPO_ROOT / "results" / "published"
RAW_ROOT = REPO_ROOT / "analysis" / "data" / "raw" / "validation"
DERIVED_ROOT = REPO_ROOT / "analysis" / "data" / "derived" / "validation"
FIGURE_ROOT = REPO_ROOT / "analysis" / "figures" / "generated" / "validation"
MANIFEST_ROOT = REPO_ROOT / "analysis" / "data" / "manifests" / "validation"


def git_commit() -> str:
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
        description="Copy validation outputs into a versioned result pack under results/published/."
    )
    parser.add_argument("--pack-name", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_PACK_ROOT,
        help="Base directory for promoted packs.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Also copy the raw validation CSV datasets into the pack.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(include_raw: bool) -> list[Path]:
    files = []
    for root in [DERIVED_ROOT, FIGURE_ROOT, MANIFEST_ROOT]:
        files.extend(path for path in root.rglob("*") if path.is_file())
    if include_raw:
        files.extend(path for path in RAW_ROOT.rglob("*.csv") if path.is_file())
    return sorted(set(files))


def copy_with_inventory(files: list[Path], pack_root: Path) -> list[dict[str, object]]:
    inventory = []
    for source in files:
        rel = source.relative_to(REPO_ROOT)
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


def write_readme(pack_root: Path, include_raw: bool, inventory: list[dict[str, object]]) -> Path:
    readme = pack_root / "README.md"
    lines = [
        "# Classical Validation Result Pack",
        "",
        f"- promoted at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- source git commit: `{git_commit()}`",
        f"- includes raw datasets: `{'yes' if include_raw else 'no'}`",
        f"- file count: `{len(inventory)}`",
        "",
        "## Scope",
        "",
        "This pack contains the scripted baseline validation artifacts for the classical Ising engine:",
        "",
        "- Onsager comparison in 2D",
        "- exact enumeration on a 4x4 lattice",
        "- fluctuation-dissipation consistency",
        "- Wolff autocorrelation scaling summary",
        "- known low- and high-temperature limits",
        "",
        "## Caveats",
        "",
        "- Quick-mode packs are useful for regression checking, not final publication numbers.",
        "- If raw datasets are omitted, reproducibility still depends on the tracked raw validation tree in the repo.",
        "- File hashes are recorded in `pack_manifest.json`.",
        "",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")
    return readme


def write_manifest(pack_root: Path, include_raw: bool, inventory: list[dict[str, object]], readme: Path) -> Path:
    manifest = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": git_commit(),
        "pack_root": str(pack_root),
        "includes_raw": include_raw,
        "readme": str(readme.relative_to(pack_root)),
        "files": inventory,
    }
    manifest_path = pack_root / "pack_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def main() -> None:
    args = parse_args()
    pack_root = args.output_root.expanduser().resolve() / args.pack_name
    if pack_root.exists():
        raise SystemExit(f"pack already exists: {pack_root}")
    pack_root.mkdir(parents=True, exist_ok=False)

    files = collect_files(include_raw=args.include_raw)
    if not files:
        raise SystemExit("no validation outputs found to promote")
    inventory = copy_with_inventory(files, pack_root)
    readme = write_readme(pack_root, args.include_raw, inventory)
    manifest = write_manifest(pack_root, args.include_raw, inventory, readme)

    print(f"promoted {len(inventory)} files into {pack_root}")
    print(f"wrote {readme}")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
