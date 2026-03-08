#!/usr/bin/env python3
"""Deterministic Kibble-Zurek analysis from KZ sweep CSV files."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "analysis" / "data"
DEFAULT_DERIVED = REPO_ROOT / "analysis" / "data" / "derived" / "kz"
DEFAULT_FIGURES = REPO_ROOT / "analysis" / "figures" / "generated" / "kz"

NU_3D_ISING = 0.6301
Z_METROPOLIS = 2.0
KZM_EXPONENT = NU_3D_ISING / (1.0 + NU_3D_ISING * Z_METROPOLIS)


@dataclass
class FitResult:
    dataset: str
    n: int
    n_rows: int
    n_fit_points: int
    kappa_measured: float
    kappa_err: float
    kappa_theory: float
    pct_error: float
    rho_eq: float
    rho_eq_err: float
    amplitude: float
    amplitude_err: float
    r2: float
    weighted_fit: bool


def kz_model(tau_q: np.ndarray, amplitude: float, kappa: float, rho_eq: float) -> np.ndarray:
    return amplitude * np.power(tau_q, -kappa) + rho_eq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load Kibble-Zurek sweep CSV files, fit the defect-density scaling curve, "
            "and write deterministic summaries and figures."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help="Directory containing kz_N*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DERIVED,
        help="Directory for derived KZ tables and summary JSON.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES,
        help="Directory for generated KZ figures.",
    )
    parser.add_argument(
        "--min-positive-rho",
        type=float,
        default=0.01,
        help="Minimum rho counted as a useful nonzero point when screening datasets.",
    )
    parser.add_argument(
        "--min-positive-points",
        type=int,
        default=8,
        help="Minimum number of rho points above --min-positive-rho to keep a dataset.",
    )
    return parser.parse_args()


def infer_n_from_name(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.split("N", 1)[1])
    except (IndexError, ValueError) as exc:
        raise SystemExit(f"cannot infer N from filename {path.name}") from exc


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"tau_q", "rho"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} missing columns: {sorted(missing)}")
    if "rho_err" not in df.columns:
        df["rho_err"] = 0.0
    if "n_trials" not in df.columns:
        df["n_trials"] = np.nan
    return df.sort_values("tau_q").reset_index(drop=True)


def fit_dataset(dataset_name: str, n: int, df: pd.DataFrame) -> FitResult:
    tau = df["tau_q"].to_numpy(dtype=float)
    rho = df["rho"].to_numpy(dtype=float)
    rho_err = df["rho_err"].fillna(0.0).to_numpy(dtype=float)

    rho_eq_guess = float(np.mean(rho[-3:]))
    amplitude_guess = max(float(rho[0] - rho_eq_guess), 1e-6) * tau[0] ** 0.3
    sigma = None
    weighted_fit = bool(np.any(rho_err > 0.0))
    if weighted_fit:
        sigma = np.where(rho_err > 0.0, rho_err, np.min(rho_err[rho_err > 0.0]))

    popt, pcov = curve_fit(
        kz_model,
        tau,
        rho,
        p0=[amplitude_guess, 0.3, max(rho_eq_guess, 0.0)],
        bounds=([0.0, 0.01, 0.0], [np.inf, 3.0, 0.5]),
        sigma=sigma,
        absolute_sigma=weighted_fit,
        maxfev=10000,
    )
    perr = np.sqrt(np.diag(pcov))
    rho_pred = kz_model(tau, *popt)
    ss_res = float(np.sum((rho - rho_pred) ** 2))
    ss_tot = float(np.sum((rho - rho.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return FitResult(
        dataset=dataset_name,
        n=n,
        n_rows=len(df),
        n_fit_points=len(df),
        kappa_measured=float(popt[1]),
        kappa_err=float(perr[1]),
        kappa_theory=KZM_EXPONENT,
        pct_error=float(abs(popt[1] - KZM_EXPONENT) / KZM_EXPONENT * 100.0),
        rho_eq=float(popt[2]),
        rho_eq_err=float(perr[2]),
        amplitude=float(popt[0]),
        amplitude_err=float(perr[0]),
        r2=r2,
        weighted_fit=weighted_fit,
    )


def plot_raw(datasets: dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis
    markers = ["o", "s", "D", "^", "v", "P", "X"]
    colors = {key: cmap(i / max(len(datasets) - 1, 1)) for i, key in enumerate(datasets)}

    for i, (key, df) in enumerate(datasets.items()):
        n = infer_n_from_name(Path(key))
        if (df["rho_err"] > 0).any():
            ax.errorbar(
                df["tau_q"],
                df["rho"],
                yerr=df["rho_err"],
                fmt=markers[i % len(markers)] + "-",
                color=colors[key],
                label=f"N={n}",
                markersize=5,
                capsize=2,
                alpha=0.85,
            )
        else:
            ax.loglog(
                df["tau_q"],
                df["rho"],
                markers[i % len(markers)] + "-",
                color=colors[key],
                label=f"N={n}",
                alpha=0.85,
                markersize=5,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tau_Q (sweeps)")
    ax.set_ylabel("rho (domain wall density)")
    ax.set_title("Kibble-Zurek defect density vs quench time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_fit(
    datasets: dict[str, pd.DataFrame],
    fit_results: dict[str, FitResult],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    cmap = plt.cm.viridis
    markers = ["o", "s", "D", "^", "v", "P", "X"]
    colors = {key: cmap(i / max(len(datasets) - 1, 1)) for i, key in enumerate(datasets)}

    for i, (key, df) in enumerate(datasets.items()):
        fit = fit_results[key]
        n = fit.n
        tau = df["tau_q"].to_numpy(dtype=float)
        rho = df["rho"].to_numpy(dtype=float)
        axes[0].loglog(
            tau,
            rho,
            markers[i % len(markers)],
            color=colors[key],
            label=f"N={n} data",
            alpha=0.7,
            markersize=6,
        )
        tau_smooth = np.logspace(np.log10(tau.min()), np.log10(tau.max()), 200)
        rho_fit = kz_model(tau_smooth, fit.amplitude, fit.kappa_measured, fit.rho_eq)
        axes[0].loglog(
            tau_smooth,
            rho_fit,
            "-",
            color=colors[key],
            linewidth=2,
            label=f"N={n} fit (kappa={fit.kappa_measured:.3f})",
        )

        excess = rho - fit.rho_eq
        mask = excess > 0
        if mask.sum() >= 3:
            axes[1].loglog(
                tau[mask],
                excess[mask],
                markers[i % len(markers)],
                color=colors[key],
                label=f"N={n}",
                alpha=0.75,
                markersize=6,
            )

    if fit_results:
        reference = fit_results[sorted(fit_results, key=lambda key: fit_results[key].n)[-1]]
        tau_ref = np.logspace(2, 5, 200)
        excess_theory = reference.amplitude * tau_ref ** (-KZM_EXPONENT)
        axes[1].loglog(
            tau_ref,
            excess_theory,
            "k--",
            linewidth=1.7,
            label=f"3D Ising theory (kappa={KZM_EXPONENT:.3f})",
        )

    axes[0].set_xlabel("tau_Q (sweeps)")
    axes[0].set_ylabel("rho")
    axes[0].set_title("KZ data and nonlinear fit")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("tau_Q (sweeps)")
    axes[1].set_ylabel("rho - rho_eq")
    axes[1].set_title("Excess defect density")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths = sorted(args.input_dir.glob("kz_N*.csv"))
    if not dataset_paths:
        raise SystemExit(f"no kz_N*.csv files found in {args.input_dir}")

    datasets: dict[str, pd.DataFrame] = {}
    skipped_rows = []
    for path in dataset_paths:
        df = load_dataset(path)
        n = infer_n_from_name(path)
        n_positive = int((df["rho"] > args.min_positive_rho).sum())
        keep = n_positive >= args.min_positive_points
        skipped_rows.append(
            {
                "dataset": path.stem,
                "n": n,
                "n_rows": len(df),
                "n_positive_points": n_positive,
                "min_positive_rho": args.min_positive_rho,
                "kept": keep,
                "input_file": str(path),
            }
        )
        if keep:
            datasets[path.stem] = df

    screening_path = args.output_dir / "kz_dataset_screening.csv"
    pd.DataFrame(skipped_rows).sort_values("n").to_csv(screening_path, index=False)
    if not datasets:
        raise SystemExit("no KZ datasets passed the screening thresholds")

    fit_results = []
    fit_result_map: dict[str, FitResult] = {}
    for key, df in datasets.items():
        n = infer_n_from_name(Path(key))
        fit = fit_dataset(key, n, df)
        fit_results.append(asdict(fit))
        fit_result_map[key] = fit

    fit_summary = pd.DataFrame(fit_results).sort_values("n").reset_index(drop=True)
    fit_summary_path = args.output_dir / "kz_fit_summary.csv"
    fit_summary.to_csv(fit_summary_path, index=False)

    raw_plot_path = args.figures_dir / "kz_raw.png"
    fit_plot_path = args.figures_dir / "kz_fit.png"
    plot_raw(datasets, raw_plot_path)
    plot_fit(datasets, fit_result_map, fit_plot_path)

    summary = {
        "theory": {
            "nu": NU_3D_ISING,
            "z": Z_METROPOLIS,
            "kappa": KZM_EXPONENT,
        },
        "n_input_datasets": len(dataset_paths),
        "n_kept_datasets": len(datasets),
        "kept_datasets": sorted(datasets),
        "screening_csv": str(screening_path),
        "fit_summary_csv": str(fit_summary_path),
        "raw_figure": str(raw_plot_path),
        "fit_figure": str(fit_plot_path),
    }
    summary_path = args.output_dir / "kz_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {screening_path}")
    print(f"wrote {fit_summary_path}")
    print(f"wrote {raw_plot_path}")
    print(f"wrote {fit_plot_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
