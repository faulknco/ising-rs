"""Shared helpers for cross-platform analysis scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def resolve_python() -> str:
    """Reuse the current interpreter when chaining analysis scripts."""
    if sys.executable:
        return sys.executable
    return "python3"


def default_publishing_root(repo_root: Path) -> Path:
    """Cross-platform default for large generated publishing datasets."""
    return repo_root / "publishing_data"
