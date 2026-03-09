#!/usr/bin/env python3
"""Binder-based finite-size analysis for multi-size dilution campaigns."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CrossingResult:
    p_removed: float
    size_a: int
    size_b: int
    tc_crossing: float
    binder_crossing: float
    method: str


def stderr(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def binder_from_moments(m2: np.ndarray, m4: np.ndarray) -> np.ndarray:
    denom = 3.0 * np.square(m2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - m4 / denom


def find_crossings(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[tuple[float, float, str]]:
    merged = df_a.merge(df_b, on="T", suffixes=("_a", "_b"))
    if len(merged) < 2:
        return []
    diff = merged["binder_a"] - merged["binder_b"]
    crossings: list[tuple[float, float, str]] = []
    for idx in range(len(merged) - 1):
        d0 = float(diff.iloc[idx])
        d1 = float(diff.iloc[idx + 1])
        if d0 == 0.0:
            crossings.append(
                (float(merged.iloc[idx]["T"]), float(merged.iloc[idx]["binder_a"]), "exact_grid")
            )
            continue
        if d0 * d1 < 0.0:
            t0 = float(merged.iloc[idx]["T"])
            t1 = float(merged.iloc[idx + 1]["T"])
            u0 = float(merged.iloc[idx]["binder_a"])
            u1 = float(merged.iloc[idx + 1]["binder_a"])
            tc = t0 - d0 * (t1 - t0) / (d1 - d0)
            u_cross = u0 + (u1 - u0) * (tc - t0) / (t1 - t0)
            crossings.append((tc, u_cross, "linear_interp"))
    return crossings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse multi-size dilution campaign outputs by computing Binder cumulants "
            "and pairwise crossing temperatures."
        )
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        required=True,
        help="Campaign root containing size_N*/derived/dilution outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for derived Binder/crossing outputs. Defaults to <campaign-root>/analysis_fss.",
    )
    return parser.parse_args()


def load_size_tables(campaign_root: Path) -> dict[int, pd.DataFrame]:
    tables: dict[int, pd.DataFrame] = {}
    for path in sorted(campaign_root.glob("size_N*/derived/dilution/dilution_manifest_summary.csv")):
        size = int(path.parts[-4].split("N")[1])
        manifest_df = pd.read_csv(path)
        frames = []
        for row in manifest_df.itertuples(index=False):
            sweep_path = Path(row.sweep_file)
            sweep_df = pd.read_csv(sweep_path, usecols=["T", "M2", "M4"])
            frames.append(
                pd.DataFrame(
                    {
                        "T": sweep_df["T"].to_numpy(dtype=float),
                        "p_removed": float(row.p_removed),
                        "realization": int(row.realization),
                        "binder": binder_from_moments(
                            sweep_df["M2"].to_numpy(dtype=float),
                            sweep_df["M4"].to_numpy(dtype=float),
                        ),
                    }
                )
            )
        if not frames:
            continue
        all_rows = pd.concat(frames, ignore_index=True)
        tables[size] = (
            all_rows.groupby(["p_removed", "T"], as_index=False)
            .agg(
                binder=("binder", "mean"),
                binder_err=("binder", stderr),
                n_realizations=("realization", "nunique"),
            )
            .sort_values(["p_removed", "T"])
            .reset_index(drop=True)
        )
    if not tables:
        raise SystemExit(
            f"no size_N*/derived/dilution/dilution_manifest_summary.csv files found in {campaign_root}"
        )
    return tables


def load_peak_summary(campaign_root: Path) -> pd.DataFrame:
    combined_path = campaign_root / "campaign_tc_summary_all_sizes.csv"
    if combined_path.exists():
        return pd.read_csv(combined_path)

    summary_paths = sorted(campaign_root.glob("campaign_tc_summary_N*.csv"))
    if not summary_paths:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(path) for path in summary_paths), ignore_index=True)


def build_binder_table(size_tables: dict[int, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for size, df in size_tables.items():
        frame = df.copy()
        frame.insert(0, "n", size)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["p_removed", "n", "T"]).reset_index(drop=True)


def peak_tc_reference(
    peak_summary: pd.DataFrame,
    p_removed: float,
    size_a: int,
    size_b: int,
) -> float | None:
    if peak_summary.empty or "n" not in peak_summary.columns:
        return None
    subset = peak_summary.loc[
        (peak_summary["p_removed"] == p_removed) & (peak_summary["n"].isin([size_a, size_b]))
    ]
    if len(subset) != 2:
        return None
    return float(subset["tc_mean"].mean())


def choose_crossing(
    candidates: list[tuple[float, float, str]],
    reference_tc: float | None,
) -> tuple[float, float, str] | None:
    if not candidates:
        return None
    if reference_tc is None or math.isnan(reference_tc):
        return candidates[0]
    return min(candidates, key=lambda item: abs(item[0] - reference_tc))


def build_crossings(size_tables: dict[int, pd.DataFrame], peak_summary: pd.DataFrame) -> pd.DataFrame:
    sizes = sorted(size_tables)
    p_values = sorted({float(p) for df in size_tables.values() for p in df["p_removed"].unique()})
    rows = []

    for p_removed in p_values:
        available_sizes = [n for n in sizes if p_removed in set(size_tables[n]["p_removed"])]
        for size_a, size_b in zip(available_sizes, available_sizes[1:]):
            df_a = size_tables[size_a].loc[size_tables[size_a]["p_removed"] == p_removed, ["T", "binder"]].copy()
            df_b = size_tables[size_b].loc[size_tables[size_b]["p_removed"] == p_removed, ["T", "binder"]].copy()
            candidates = find_crossings(df_a, df_b)
            reference_tc = peak_tc_reference(peak_summary, p_removed, size_a, size_b)
            crossing = choose_crossing(candidates, reference_tc)
            if crossing is None:
                rows.append(
                    asdict(
                        CrossingResult(
                            p_removed=p_removed,
                            size_a=size_a,
                            size_b=size_b,
                            tc_crossing=math.nan,
                            binder_crossing=math.nan,
                            method="no_crossing",
                        )
                    )
                )
                continue
            tc_crossing, binder_crossing, method = crossing
            if reference_tc is not None:
                method = f"{method}_peak_guided"
            rows.append(
                asdict(
                    CrossingResult(
                        p_removed=p_removed,
                        size_a=size_a,
                        size_b=size_b,
                        tc_crossing=tc_crossing,
                        binder_crossing=binder_crossing,
                        method=method,
                    )
                )
            )

    columns = ["p_removed", "size_a", "size_b", "tc_crossing", "binder_crossing", "method"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["p_removed", "size_a", "size_b"]).reset_index(drop=True)


def summarise_crossings(crossings: pd.DataFrame) -> pd.DataFrame:
    valid = crossings.replace([np.inf, -np.inf], np.nan).dropna(subset=["tc_crossing"])
    if valid.empty:
        return pd.DataFrame(
            columns=["p_removed", "tc_binder_mean", "tc_binder_err", "u_cross_mean", "u_cross_err", "n_pairs"]
        )

    summary = (
        valid.groupby("p_removed", as_index=False)
        .agg(
            tc_binder_mean=("tc_crossing", "mean"),
            tc_binder_err=("tc_crossing", stderr),
            u_cross_mean=("binder_crossing", "mean"),
            u_cross_err=("binder_crossing", stderr),
            n_pairs=("tc_crossing", "size"),
        )
        .sort_values("p_removed")
        .reset_index(drop=True)
    )
    return summary


def plot_binder_curves(binder_table: pd.DataFrame, output_path: Path) -> None:
    p_values = sorted(binder_table["p_removed"].unique())
    fig, axes = plt.subplots(
        len(p_values),
        1,
        figsize=(7, max(3.2, 2.6 * len(p_values))),
        constrained_layout=True,
        sharex=False,
    )
    if len(p_values) == 1:
        axes = [axes]

    for ax, p_removed in zip(axes, p_values):
        subset = binder_table.loc[binder_table["p_removed"] == p_removed]
        for n, df in subset.groupby("n"):
            ax.plot(df["T"], df["binder"], label=f"N={n}", linewidth=1.3)
            if "binder_err" in df.columns:
                ax.fill_between(
                    df["T"],
                    df["binder"] - df["binder_err"],
                    df["binder"] + df["binder_err"],
                    alpha=0.12,
                )
        ax.set_title(f"Binder cumulant, p={p_removed:.3f}")
        ax.set_xlabel("T (J/k_B)")
        ax.set_ylabel("U")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=3)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_crossing_summary(crossing_summary: pd.DataFrame, tc_peak_summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    if not crossing_summary.empty:
        ax.errorbar(
            crossing_summary["p_removed"],
            crossing_summary["tc_binder_mean"],
            yerr=crossing_summary["tc_binder_err"],
            fmt="o-",
            capsize=4,
            label="Binder crossings",
        )
    if not tc_peak_summary.empty:
        if "n" in tc_peak_summary.columns:
            latest_size = int(tc_peak_summary["n"].max())
            tc_peak_summary = (
                tc_peak_summary.loc[tc_peak_summary["n"] == latest_size]
                .sort_values("p_removed")
                .reset_index(drop=True)
            )
            peak_label = f"chi-peak summary (N={latest_size})"
        else:
            peak_label = "chi-peak summary"
        ax.errorbar(
            tc_peak_summary["p_removed"],
            tc_peak_summary["tc_mean"],
            yerr=tc_peak_summary["tc_err"],
            fmt="s--",
            capsize=4,
            label=peak_label,
        )

    ax.set_xlabel("Removed-bond fraction p")
    ax.set_ylabel("Tc")
    ax.set_title("Dilution finite-size Tc estimates")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.campaign_root / "analysis_fss")
    output_dir.mkdir(parents=True, exist_ok=True)

    size_tables = load_size_tables(args.campaign_root)
    binder_table = build_binder_table(size_tables)
    peak_summary = load_peak_summary(args.campaign_root)
    crossings = build_crossings(size_tables, peak_summary)
    crossing_summary = summarise_crossings(crossings)

    binder_table_path = output_dir / "dilution_binder_curves.csv"
    crossings_path = output_dir / "dilution_binder_crossings.csv"
    summary_path = output_dir / "dilution_binder_crossing_summary.csv"
    binder_fig = output_dir / "dilution_binder_curves.png"
    tc_compare_fig = output_dir / "dilution_tc_compare.png"

    binder_table.to_csv(binder_table_path, index=False)
    crossings.to_csv(crossings_path, index=False)
    crossing_summary.to_csv(summary_path, index=False)
    plot_binder_curves(binder_table, binder_fig)
    plot_crossing_summary(crossing_summary, peak_summary, tc_compare_fig)

    meta = {
        "campaign_root": str(args.campaign_root),
        "n_sizes_loaded": len(size_tables),
        "sizes_loaded": sorted(size_tables),
        "p_values_loaded": sorted(float(p) for p in binder_table["p_removed"].unique()),
        "n_valid_crossing_pairs": int(crossings["tc_crossing"].notna().sum()) if not crossings.empty else 0,
        "binder_table_csv": str(binder_table_path),
        "crossings_csv": str(crossings_path),
        "crossing_summary_csv": str(summary_path),
        "binder_figure": str(binder_fig),
        "tc_compare_figure": str(tc_compare_fig),
    }
    meta_path = output_dir / "dilution_fss_summary.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {binder_table_path}")
    print(f"wrote {crossings_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {binder_fig}")
    print(f"wrote {tc_compare_fig}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
