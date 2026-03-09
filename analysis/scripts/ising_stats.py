"""Shared statistics helpers for Ising-model analysis workflows.

This module provides deterministic, script-friendly error estimation so
publication-critical workflows do not depend on notebook-only utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def jackknife_blocks(data: np.ndarray, n_blocks: int) -> np.ndarray:
    """Trim data and reshape into equal-sized jackknife blocks."""
    if data.ndim != 1:
        raise ValueError("jackknife_blocks expects a 1D array")
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least 2")

    block_size = len(data) // n_blocks
    if block_size < 1:
        raise ValueError("not enough samples for the requested number of blocks")

    trimmed = data[: n_blocks * block_size]
    return trimmed.reshape(n_blocks, block_size)


def jackknife_error(data: np.ndarray, func: Callable[[np.ndarray], float], n_blocks: int = 20) -> float:
    """Return the jackknife standard error of ``func(data)``."""
    blocks = jackknife_blocks(np.asarray(data), n_blocks)
    full = func(blocks.reshape(-1))
    estimates = np.zeros(len(blocks), dtype=float)

    for i in range(len(blocks)):
        subset = np.delete(blocks, i, axis=0).reshape(-1)
        estimates[i] = func(subset)

    variance = (len(blocks) - 1) / len(blocks) * np.sum((estimates - full) ** 2)
    return float(np.sqrt(variance))


@dataclass
class IsingObservableRow:
    temperature: float
    energy: float
    energy_err: float
    magnetisation: float
    magnetisation_err: float
    m2: float
    m2_err: float
    m4: float
    m4_err: float
    heat_capacity: float
    heat_capacity_err: float
    susceptibility: float
    susceptibility_err: float
    binder: float
    binder_err: float
    n_samples: int


def _binder(m2: float, m4: float) -> float:
    if m2 <= 1e-15:
        return 0.0
    return 1.0 - m4 / (3.0 * m2 * m2)


def compute_observables_from_raw(raw_df: pd.DataFrame, linear_size: int, n_blocks: int = 20) -> pd.DataFrame:
    """Compute observables with jackknife errors from raw Wolff time-series CSV data.

    Expected columns:
    - ``T``
    - ``e`` or ``e_per_spin``
    - ``m_abs``
    - ``m_signed``
    """
    if "e" in raw_df.columns:
        e_col = "e"
    elif "e_per_spin" in raw_df.columns:
        e_col = "e_per_spin"
    else:
        raise ValueError("raw_df must contain 'e' or 'e_per_spin'")

    required = {"T", "m_abs", "m_signed"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"raw_df missing required columns: {sorted(missing)}")

    volume = linear_size ** 3
    rows: list[IsingObservableRow] = []

    for temperature in sorted(raw_df["T"].unique()):
        sub = raw_df[np.isclose(raw_df["T"], temperature)].copy()
        e = sub[e_col].to_numpy(dtype=float)
        m_abs = sub["m_abs"].to_numpy(dtype=float)
        m_signed = sub["m_signed"].to_numpy(dtype=float)

        blocks = jackknife_blocks(e, n_blocks)
        trimmed_len = blocks.size
        e = e[:trimmed_len]
        m_abs = m_abs[:trimmed_len]
        m_signed = m_signed[:trimmed_len]

        beta = 1.0 / float(temperature)

        energy = float(np.mean(e))
        magnetisation = float(np.mean(m_abs))
        m2 = float(np.mean(m_abs ** 2))
        m4 = float(np.mean(m_abs ** 4))
        heat_capacity = float(beta * beta * volume * np.var(e, ddof=0))
        susceptibility = float(beta * volume * (np.mean(m_signed ** 2) - np.mean(m_signed) ** 2))
        binder = _binder(m2, m4)

        energy_err = jackknife_error(e, np.mean, n_blocks=n_blocks)
        magnetisation_err = jackknife_error(m_abs, np.mean, n_blocks=n_blocks)
        m2_err = jackknife_error(m_abs, lambda x: np.mean(x ** 2), n_blocks=n_blocks)
        m4_err = jackknife_error(m_abs, lambda x: np.mean(x ** 4), n_blocks=n_blocks)
        heat_capacity_err = jackknife_error(
            e,
            lambda x: beta * beta * volume * np.var(x, ddof=0),
            n_blocks=n_blocks,
        )
        susceptibility_err = jackknife_error(
            m_signed,
            lambda x: beta * volume * (np.mean(x ** 2) - np.mean(x) ** 2),
            n_blocks=n_blocks,
        )
        binder_err = jackknife_error(
            m_abs,
            lambda x: _binder(float(np.mean(x ** 2)), float(np.mean(x ** 4))),
            n_blocks=n_blocks,
        )

        rows.append(
            IsingObservableRow(
                temperature=float(temperature),
                energy=energy,
                energy_err=energy_err,
                magnetisation=magnetisation,
                magnetisation_err=magnetisation_err,
                m2=m2,
                m2_err=m2_err,
                m4=m4,
                m4_err=m4_err,
                heat_capacity=heat_capacity,
                heat_capacity_err=heat_capacity_err,
                susceptibility=susceptibility,
                susceptibility_err=susceptibility_err,
                binder=binder,
                binder_err=binder_err,
                n_samples=trimmed_len,
            )
        )

    return pd.DataFrame([row.__dict__ for row in rows])


def infer_linear_size(path: Path) -> int:
    """Infer ``N`` from filenames like ``fss_raw_N16.csv``."""
    stem = path.stem
    if "N" not in stem:
        raise ValueError(f"could not infer linear size from {path}")
    return int(stem.split("N")[-1])
