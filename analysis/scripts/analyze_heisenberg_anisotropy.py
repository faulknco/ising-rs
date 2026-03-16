#!/usr/bin/env python3
"""Analyse Heisenberg anisotropy pilot and crossover sweeps."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DatasetSpec:
    label: str
    d_value: float
    path: Path


@dataclass
class CrossingRow:
    label: str
    d_value: float
    observable: str
    size_a: int
    size_b: int
    tc_crossing: float
    binder_crossing: float
    method: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse Heisenberg anisotropy FSS outputs and choose the symmetry-aware "
            "order parameter automatically from the sign of D."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "analysis" / "data" / "anisotropy_pilot_cpu",
        help="Root directory containing one subdirectory per anisotropy run.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Optional explicit dataset in the form label:path:d_value. "
            "If omitted, subdirectories under --root are auto-discovered."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for derived tables and plots. Defaults to <root>/analysis.",
    )
    return parser.parse_args()


def infer_d_from_name(name: str) -> float:
    if re.fullmatch(r"d0(?:_.*)?", name):
        return 0.0

    match = re.fullmatch(r"d([mp])([0-9]+(?:p[0-9]+)?)(?:_.*)?", name)
    if match:
        sign = -1.0 if match.group(1) == "m" else 1.0
        magnitude = float(match.group(2).replace("p", "."))
        return sign * magnitude

    match = re.fullmatch(r"d(-?[0-9]+(?:\.[0-9]+)?)(?:_.*)?", name)
    if match:
        return float(match.group(1))

    raise ValueError(f"could not infer anisotropy D from directory name '{name}'")


def discover_datasets(root: Path, explicit: list[str]) -> list[DatasetSpec]:
    if explicit:
        specs: list[DatasetSpec] = []
        for item in explicit:
            try:
                label, path_str, d_str = item.split(":", 2)
            except ValueError as exc:
                raise SystemExit(f"invalid --dataset '{item}', expected label:path:d_value") from exc
            specs.append(DatasetSpec(label=label, path=Path(path_str), d_value=float(d_str)))
        return specs

    specs = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        try:
            d_value = infer_d_from_name(path.name)
        except ValueError:
            continue
        specs.append(DatasetSpec(label=path.name, path=path, d_value=d_value))
    if not specs:
        raise SystemExit(f"no anisotropy datasets found in {root}")
    return specs


def choose_observable(d_value: float) -> tuple[str, str, str, str]:
    if d_value > 0.0:
        return ("z", "Mz", "chi_z", "easy_axis")
    if d_value < 0.0:
        return ("xy", "Mxy", "chi_xy", "easy_plane")
    return ("total", "M", "chi", "isotropic")


def binder_from_columns(df: pd.DataFrame, observable_key: str) -> pd.Series:
    if observable_key == "z":
        m2 = df["Mz2"]
        m4 = df["Mz4"]
    elif observable_key == "xy":
        m2 = df["Mxy2"]
        m4 = df["Mxy4"]
    else:
        m2 = df["M2"]
        m4 = df["M4"]

    denom = 3.0 * np.square(m2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - m4 / denom


def load_dataset(spec: DatasetSpec) -> dict[int, pd.DataFrame]:
    tables: dict[int, pd.DataFrame] = {}
    for path in sorted(spec.path.glob("heisenberg_fss_N*.csv")):
        size = int(path.stem.split("_N")[1])
        df = pd.read_csv(path)
        observable_key, _, _, _ = choose_observable(spec.d_value)
        df["binder_rel"] = binder_from_columns(df, observable_key)
        tables[size] = df
    if not tables:
        raise SystemExit(f"no heisenberg_fss_N*.csv files found in {spec.path}")
    return tables


def linear_crossing(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    value_col: str = "binder_rel",
) -> tuple[float, float, str] | None:
    merged = df_a[["T", value_col]].merge(df_b[["T", value_col]], on="T", suffixes=("_a", "_b"))
    if len(merged) < 2:
        return None

    diff = merged[f"{value_col}_a"] - merged[f"{value_col}_b"]
    for idx in range(len(merged) - 1):
        d0 = float(diff.iloc[idx])
        d1 = float(diff.iloc[idx + 1])
        if d0 == 0.0:
            return (float(merged.iloc[idx]["T"]), float(merged.iloc[idx][f"{value_col}_a"]), "exact_grid")
        if d0 * d1 < 0.0:
            t0 = float(merged.iloc[idx]["T"])
            t1 = float(merged.iloc[idx + 1]["T"])
            u0 = float(merged.iloc[idx][f"{value_col}_a"])
            u1 = float(merged.iloc[idx + 1][f"{value_col}_a"])
            tc = t0 - d0 * (t1 - t0) / (d1 - d0)
            uc = u0 + (u1 - u0) * (tc - t0) / (t1 - t0)
            return (tc, uc, "linear_interp")
    return None


def summarise_dataset(spec: DatasetSpec, tables: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    observable_key, order_col, chi_col, regime = choose_observable(spec.d_value)
    peaks: list[dict[str, object]] = []
    crossings: list[CrossingRow] = []

    sizes = sorted(tables)
    for n in sizes:
        df = tables[n]
        peak_row = df.loc[df[chi_col].idxmax()]
        peaks.append(
            {
                "label": spec.label,
                "d_value": spec.d_value,
                "regime": regime,
                "n": n,
                "observable": order_col,
                "chi_column": chi_col,
                "peak_T": float(peak_row["T"]),
                "peak_chi": float(peak_row[chi_col]),
                "peak_M": float(peak_row["M"]),
                "peak_Mz": float(peak_row["Mz"]),
                "peak_Mxy": float(peak_row["Mxy"]),
                "lowT_M": float(df.iloc[0]["M"]),
                "lowT_Mz": float(df.iloc[0]["Mz"]),
                "lowT_Mxy": float(df.iloc[0]["Mxy"]),
                "highT_M": float(df.iloc[-1]["M"]),
                "highT_Mz": float(df.iloc[-1]["Mz"]),
                "highT_Mxy": float(df.iloc[-1]["Mxy"]),
            }
        )

    for size_a, size_b in zip(sizes, sizes[1:]):
        crossing = linear_crossing(tables[size_a], tables[size_b], "binder_rel")
        if crossing is None:
            crossings.append(
                CrossingRow(
                    label=spec.label,
                    d_value=spec.d_value,
                    observable=order_col,
                    size_a=size_a,
                    size_b=size_b,
                    tc_crossing=math.nan,
                    binder_crossing=math.nan,
                    method="no_crossing",
                )
            )
            continue
        tc, uc, method = crossing
        crossings.append(
            CrossingRow(
                label=spec.label,
                d_value=spec.d_value,
                observable=order_col,
                size_a=size_a,
                size_b=size_b,
                tc_crossing=tc,
                binder_crossing=uc,
                method=method,
            )
        )

    return pd.DataFrame(peaks), pd.DataFrame(asdict(row) for row in crossings)


def plot_component_curves(spec: DatasetSpec, tables: dict[int, pd.DataFrame], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    sizes = sorted(tables)
    for n in sizes:
        df = tables[n]
        axes[0].plot(df["T"], df["Mz"], label=f"N={n}")
        axes[1].plot(df["T"], df["Mxy"], label=f"N={n}")

    axes[0].set_title(f"{spec.label}: Mz(T)")
    axes[1].set_title(f"{spec.label}: Mxy(T)")
    for ax in axes:
        ax.set_xlabel("T")
        ax.set_ylabel("order parameter")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_binder_curves(spec: DatasetSpec, tables: dict[int, pd.DataFrame], output_path: Path) -> None:
    observable_key, order_col, _, regime = choose_observable(spec.d_value)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for n in sorted(tables):
        df = tables[n]
        ax.plot(df["T"], df["binder_rel"], label=f"N={n}")
    ax.set_title(f"{spec.label}: Binder ({order_col}, {regime})")
    ax.set_xlabel("T")
    ax.set_ylabel("U")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def extract_exponents(
    spec: DatasetSpec,
    tables: dict[int, pd.DataFrame],
    crossings_df: pd.DataFrame,
) -> dict[str, object]:
    """Extract gamma/nu from chi peak scaling and beta/nu from M(Tc) scaling.

    gamma/nu: log(chi_peak) vs log(L) slope
    beta/nu: -log(M(Tc)) vs log(L) slope (from Binder crossing temperature)
    """
    observable_key, order_col, chi_col, regime = choose_observable(spec.d_value)
    sizes = sorted(tables)
    result: dict[str, object] = {
        "label": spec.label,
        "d_value": spec.d_value,
        "regime": regime,
    }

    # gamma/nu from chi_peak ~ L^(gamma/nu)
    log_L = []
    log_chi_peak = []
    for n in sizes:
        df = tables[n]
        peak_chi = float(df[chi_col].max())
        if peak_chi > 0:
            log_L.append(np.log(n))
            log_chi_peak.append(np.log(peak_chi))

    def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
        A = np.vstack([x, np.ones(len(x))]).T
        return float(np.linalg.lstsq(A, y, rcond=None)[0][0])

    if len(log_L) >= 3:
        log_L_arr = np.array(log_L)
        log_chi_arr = np.array(log_chi_peak)
        result["gamma_over_nu"] = round(ols_slope(log_L_arr, log_chi_arr), 4)
        result["gamma_over_nu_sizes"] = f"{min(sizes)}-{max(sizes)}"

        # Fit excluding smallest size
        if len(log_L) >= 4:
            result["gamma_over_nu_large"] = round(ols_slope(log_L_arr[1:], log_chi_arr[1:]), 4)
            result["gamma_over_nu_large_sizes"] = f"{sizes[1]}-{max(sizes)}"

        # Fit using only sizes >= 64 (asymptotic window)
        mask_64 = np.array([s >= 64 for s in sizes if s in [sizes[i] for i, _ in enumerate(log_L)]])
        sizes_with_chi = [s for s in sizes if tables[s][chi_col].max() > 0]
        idx_64 = [i for i, s in enumerate(sizes_with_chi) if s >= 64]
        if len(idx_64) >= 2:
            ll64 = log_L_arr[idx_64]
            lc64 = log_chi_arr[idx_64]
            result["gamma_over_nu_64plus"] = round(ols_slope(ll64, lc64), 4)
            result["gamma_over_nu_64plus_sizes"] = ",".join(str(sizes_with_chi[i]) for i in idx_64)
    else:
        result["gamma_over_nu"] = None

    # beta/nu from M(Tc) ~ L^(-beta/nu)
    # Use the best Binder crossing Tc: prefer linear_interp from largest pair
    spec_crossings = crossings_df[
        (crossings_df["label"] == spec.label) & (crossings_df["method"] != "no_crossing")
    ]
    interp_crossings = spec_crossings[spec_crossings["method"] == "linear_interp"]
    best_source = interp_crossings if not interp_crossings.empty else spec_crossings
    if not best_source.empty:
        best = best_source.iloc[-1]  # largest pair with preferred method
        tc = float(best["tc_crossing"])
        result["tc_used"] = round(tc, 6)

        log_L_m = []
        log_M_tc = []
        for n in sizes:
            df = tables[n]
            # Interpolate M at Tc
            if df["T"].min() <= tc <= df["T"].max():
                m_at_tc = float(np.interp(tc, df["T"], df[order_col]))
                if m_at_tc > 0:
                    log_L_m.append(np.log(n))
                    log_M_tc.append(np.log(m_at_tc))

        if len(log_L_m) >= 3:
            log_L_m_arr = np.array(log_L_m)
            log_M_arr = np.array(log_M_tc)
            result["beta_over_nu"] = round(-ols_slope(log_L_m_arr, log_M_arr), 4)
            result["beta_over_nu_sizes"] = f"{min(sizes)}-{max(sizes)}"

            # Restricted fit: sizes >= 64
            sizes_m = [s for s in sizes if tables[s]["T"].min() <= tc <= tables[s]["T"].max()
                       and float(np.interp(tc, tables[s]["T"], tables[s][order_col])) > 0]
            idx_64m = [i for i, s in enumerate(sizes_m) if s >= 64]
            if len(idx_64m) >= 2:
                ll64m = log_L_m_arr[idx_64m]
                lm64 = log_M_arr[idx_64m]
                result["beta_over_nu_64plus"] = round(-ols_slope(ll64m, lm64), 4)
                result["beta_over_nu_64plus_sizes"] = ",".join(str(sizes_m[i]) for i in idx_64m)
        else:
            result["beta_over_nu"] = None
    else:
        result["tc_used"] = None
        result["beta_over_nu"] = None

    # nu from Binder slope: |dU/dT|_max ~ L^(1/nu)
    log_L_slope = []
    log_binder_slope = []
    for n in sizes:
        df = tables[n]
        binder = df["binder_rel"].values
        T = df["T"].values
        if len(T) >= 3:
            # Numerical derivative, take max absolute slope
            dU_dT = np.gradient(binder, T)
            max_slope = float(np.max(np.abs(dU_dT)))
            if max_slope > 0:
                log_L_slope.append(np.log(n))
                log_binder_slope.append(np.log(max_slope))

    if len(log_L_slope) >= 3:
        log_L_s = np.array(log_L_slope)
        log_s = np.array(log_binder_slope)
        slope_val = ols_slope(log_L_s, log_s)
        result["one_over_nu"] = round(slope_val, 4)
        result["nu"] = round(1.0 / slope_val, 4) if abs(slope_val) > 0.01 else None
        result["nu_sizes"] = f"{min(sizes)}-{max(sizes)}"

        # Restricted fit: sizes >= 64
        sizes_slope = [s for s in sizes if len(tables[s]) >= 3]
        idx_64s = [i for i, s in enumerate(sizes_slope) if s >= 64]
        if len(idx_64s) >= 2:
            ll64s = log_L_s[idx_64s]
            ls64 = log_s[idx_64s]
            slope_64 = ols_slope(ll64s, ls64)
            result["one_over_nu_64plus"] = round(slope_64, 4)
            result["nu_64plus"] = round(1.0 / slope_64, 4) if abs(slope_64) > 0.01 else None
            result["nu_64plus_sizes"] = ",".join(str(sizes_slope[i]) for i in idx_64s)
    else:
        result["one_over_nu"] = None
        result["nu"] = None

    return result


def plot_susceptibility_curves(spec: DatasetSpec, tables: dict[int, pd.DataFrame], output_path: Path) -> None:
    _, order_col, chi_col, regime = choose_observable(spec.d_value)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for n in sorted(tables):
        df = tables[n]
        ax.plot(df["T"], df[chi_col], label=f"N={n}")
    ax.set_title(f"{spec.label}: {chi_col}(T) ({order_col}, {regime})")
    ax.set_xlabel("T")
    ax.set_ylabel(chi_col)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.root / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = discover_datasets(args.root, args.dataset)

    peak_frames = []
    crossing_frames = []
    summary_rows: list[dict[str, object]] = []
    all_tables: dict[str, dict[int, pd.DataFrame]] = {}

    for spec in specs:
        tables = load_dataset(spec)
        all_tables[spec.label] = tables
        peaks, crossings = summarise_dataset(spec, tables)
        peak_frames.append(peaks)
        crossing_frames.append(crossings)

        plot_component_curves(spec, tables, output_dir / f"{spec.label}_components.png")
        plot_binder_curves(spec, tables, output_dir / f"{spec.label}_binder.png")
        plot_susceptibility_curves(spec, tables, output_dir / f"{spec.label}_chi.png")

        observable_key, order_col, chi_col, regime = choose_observable(spec.d_value)
        summary_rows.append(
            {
                "label": spec.label,
                "d_value": spec.d_value,
                "regime": regime,
                "observable": order_col,
                "chi_column": chi_col,
                "sizes": sorted(tables.keys()),
            }
        )

    peaks_df = pd.concat(peak_frames, ignore_index=True).sort_values(["d_value", "n"])
    crossings_df = pd.concat(crossing_frames, ignore_index=True).sort_values(
        ["d_value", "size_a", "size_b"]
    )

    # Extract critical exponents for each dataset
    exponent_rows = []
    for spec in specs:
        tables = all_tables[spec.label]
        exponents = extract_exponents(spec, tables, crossings_df)
        exponent_rows.append(exponents)
    exponents_df = pd.DataFrame(exponent_rows).sort_values("d_value")

    peaks_path = output_dir / "anisotropy_peak_summary.csv"
    crossings_path = output_dir / "anisotropy_binder_crossings.csv"
    exponents_path = output_dir / "anisotropy_exponents.csv"
    summary_path = output_dir / "anisotropy_summary.json"

    peaks_df.to_csv(peaks_path, index=False)
    crossings_df.to_csv(crossings_path, index=False)
    exponents_df.to_csv(exponents_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "datasets": summary_rows,
                "peak_summary_csv": str(peaks_path),
                "binder_crossings_csv": str(crossings_path),
                "exponents_csv": str(exponents_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote {peaks_path}")
    print(f"wrote {crossings_path}")
    print(f"wrote {exponents_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
