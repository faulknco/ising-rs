#!/usr/bin/env python3
"""Run and analyse the classical Ising validation suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ellipk


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO_ROOT / "analysis" / "data" / "raw" / "validation"
DERIVED_ROOT = REPO_ROOT / "analysis" / "data" / "derived" / "validation"
FIGURE_ROOT = REPO_ROOT / "analysis" / "figures" / "generated" / "validation"
MANIFEST_ROOT = REPO_ROOT / "analysis" / "data" / "manifests" / "validation"


@dataclass
class ValidationMetric:
    name: str
    status: str
    measured: float | None
    reference: float | None
    details: str


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


def run_command(cmd: list[str]) -> None:
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )


def run_fss(outdir: Path, args: list[str], skip_existing: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if skip_existing and any(outdir.glob("*.csv")):
        return
    cmd = ["cargo", "run", "--release", "--bin", "fss", "--", *args, "--outdir", str(outdir)]
    run_command(cmd)


def onsager_tc(j: float = 1.0) -> float:
    return 2 * j / np.log(1 + np.sqrt(2))


def onsager_energy(temperature: float, j: float = 1.0) -> float:
    beta = 1.0 / temperature
    k = beta * j
    elliptic_k = 2 * np.sinh(2 * k) / np.cosh(2 * k) ** 2
    return -j * (1.0 / np.tanh(2 * k)) * (
        1 + (2 / np.pi) * (2 * np.tanh(2 * k) ** 2 - 1) * ellipk(elliptic_k**2)
    )


def onsager_magnetization(temperature: float, j: float = 1.0) -> float:
    if temperature >= onsager_tc(j):
        return 0.0
    beta = 1.0 / temperature
    k = beta * j
    return float((1 - np.sinh(2 * k) ** (-4)) ** (1 / 8))


def exact_enumeration_2d(linear_size: int, temperatures: np.ndarray, j: float = 1.0) -> pd.DataFrame:
    n_spins = linear_size * linear_size
    n_states = 2**n_spins

    neighbours = []
    for i in range(linear_size):
        for j_idx in range(linear_size):
            neighbours.append(
                [
                    ((i + 1) % linear_size) * linear_size + j_idx,
                    i * linear_size + (j_idx + 1) % linear_size,
                ]
            )

    energies = np.zeros(n_states)
    magnetizations = np.zeros(n_states)
    for state in range(n_states):
        spins = np.array([(((state >> k) & 1) * 2 - 1) for k in range(n_spins)], dtype=int)
        magnetizations[state] = np.sum(spins)
        energy = 0.0
        for idx in range(n_spins):
            for nb_idx in neighbours[idx]:
                energy += -j * spins[idx] * spins[nb_idx]
        energies[state] = energy

    rows = []
    for temperature in temperatures:
        beta = 1.0 / temperature
        log_weights = -beta * energies
        shift = np.max(log_weights)
        weights = np.exp(log_weights - shift)
        weights /= np.sum(weights)

        avg_e = np.sum(weights * energies) / n_spins
        avg_e2 = np.sum(weights * energies**2) / n_spins**2
        avg_m_abs = np.sum(weights * np.abs(magnetizations)) / n_spins
        avg_m2 = np.sum(weights * magnetizations**2) / n_spins**2
        avg_m4 = np.sum(weights * magnetizations**4) / n_spins**4
        avg_m_signed = np.sum(weights * magnetizations) / n_spins
        avg_m_signed2 = np.sum(weights * magnetizations**2) / n_spins**2
        rows.append(
            {
                "T": temperature,
                "E": avg_e,
                "M": avg_m_abs,
                "M2": avg_m2,
                "M4": avg_m4,
                "Cv": beta**2 * n_spins * (avg_e2 - avg_e**2),
                "chi_conn": beta * n_spins * (avg_m2 - avg_m_abs**2),
                "chi_signed": beta * n_spins * (avg_m_signed2 - avg_m_signed**2),
            }
        )
    return pd.DataFrame(rows)


def autocorrelation(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if max_lag is None:
        max_lag = n // 4
    variance = np.var(x)
    if variance < 1e-15:
        return np.zeros(max_lag)
    acf = np.correlate(x, x, mode="full")[n - 1 : n - 1 + max_lag]
    acf /= variance * np.arange(n, n - max_lag, -1)
    return acf


def integrated_autocorrelation_time(acf: np.ndarray) -> float:
    tau = 0.5
    for t in range(1, len(acf)):
        tau += acf[t]
        if t >= 6 * tau:
            break
    return float(tau)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def require_files(paths: list[Path], label: str) -> list[Path]:
    if not paths:
        raise SystemExit(
            f"missing required validation dataset for '{label}'. "
            "Run without --analysis-only or generate the raw validation data first."
        )
    return paths


def status_from_threshold(value: float, good: float, warn: float) -> str:
    if value <= good:
        return "ok"
    if value <= warn:
        return "warn"
    return "bad"


def prepare_datasets(quick: bool, skip_existing: bool) -> None:
    if quick:
        onsager_sizes = "16,32"
        onsager_samples = "2000"
        onsager_warmup = "1000"
        autocorr_sizes = "16,32"
        autocorr_samples = "5000"
        limits_sizes_3d = "8"
        limits_sizes_2d = "16"
    else:
        onsager_sizes = "32,64,128"
        onsager_samples = "10000"
        onsager_warmup = "5000"
        autocorr_sizes = "16,32,64"
        autocorr_samples = "20000"
        limits_sizes_3d = "8,16"
        limits_sizes_2d = "16,32"

    run_fss(
        RAW_ROOT / "val_2d",
        [
            "--sizes",
            onsager_sizes,
            "--geometry",
            "square",
            "--tmin",
            "1.0",
            "--tmax",
            "3.5",
            "--steps",
            "51",
            "--warmup",
            onsager_warmup,
            "--samples",
            onsager_samples,
            "--wolff",
        ],
        skip_existing,
    )
    run_fss(
        RAW_ROOT / "val_enum",
        [
            "--sizes",
            "4",
            "--geometry",
            "square",
            "--tmin",
            "1.0",
            "--tmax",
            "4.0",
            "--steps",
            "31",
            "--warmup",
            "10000" if not quick else "2000",
            "--samples",
            "50000" if not quick else "10000",
            "--wolff",
        ],
        skip_existing,
    )
    run_fss(
        RAW_ROOT / "val_autocorr_wolff",
        [
            "--sizes",
            autocorr_sizes,
            "--geometry",
            "square",
            "--tmin",
            "2.27",
            "--tmax",
            "2.27",
            "--steps",
            "1",
            "--warmup",
            "5000",
            "--samples",
            autocorr_samples,
            "--wolff",
            "--raw",
        ],
        skip_existing,
    )
    run_fss(
        RAW_ROOT / "val_limits_3d",
        [
            "--sizes",
            limits_sizes_3d,
            "--tmin",
            "0.1",
            "--tmax",
            "50.0",
            "--steps",
            "21",
            "--warmup",
            "5000" if not quick else "1000",
            "--samples",
            "5000" if not quick else "1000",
            "--wolff",
        ],
        skip_existing,
    )
    run_fss(
        RAW_ROOT / "val_limits_2d",
        [
            "--sizes",
            limits_sizes_2d,
            "--geometry",
            "square",
            "--tmin",
            "0.1",
            "--tmax",
            "50.0",
            "--steps",
            "21",
            "--warmup",
            "5000" if not quick else "1000",
            "--samples",
            "5000" if not quick else "1000",
            "--wolff",
        ],
        skip_existing,
    )


def analyse_onsager() -> tuple[list[ValidationMetric], dict[str, object]]:
    metrics: list[ValidationMetric] = []
    val_2d_files = require_files(sorted((RAW_ROOT / "val_2d").glob("fss_N*.csv")), "val_2d")
    frames = []
    for path in val_2d_files:
        size = int(path.stem.split("N")[1])
        df = load_csv(path).copy()
        df.insert(0, "n", size)
        df["E_exact"] = df["T"].map(onsager_energy)
        df["M_exact"] = df["T"].map(onsager_magnetization)
        df["E_abs_err"] = (df["E"] - df["E_exact"]).abs()
        df["M_abs_err"] = (df["M"] - df["M_exact"]).abs()
        frames.append(df)

    comparison = pd.concat(frames, ignore_index=True).sort_values(["n", "T"]).reset_index(drop=True)
    largest = comparison["n"].max()
    largest_df = comparison.loc[comparison["n"] == largest].copy()
    mean_abs_energy_error = float(largest_df["E_abs_err"].mean())
    tc_2d = onsager_tc()
    mag_mask = largest_df["T"].to_numpy() < (tc_2d - 0.2)
    mean_abs_mag_error = float(largest_df.loc[mag_mask, "M_abs_err"].mean())
    metrics.append(
        ValidationMetric(
            name="onsager_energy_mean_abs_error",
            status=status_from_threshold(mean_abs_energy_error, good=0.01, warn=0.02),
            measured=mean_abs_energy_error,
            reference=0.01,
            details=f"largest lattice file fss_N{largest}.csv",
        )
    )
    metrics.append(
        ValidationMetric(
            name="onsager_magnetization_mean_abs_error",
            status=status_from_threshold(mean_abs_mag_error, good=0.03, warn=0.06),
            measured=mean_abs_mag_error,
            reference=0.03,
            details=f"largest lattice file fss_N{largest}.csv, restricted to T < Tc - 0.2",
        )
    )
    return metrics, {"comparison": comparison, "largest_size": int(largest), "tc_exact": float(tc_2d)}


def analyse_exact_enumeration() -> tuple[list[ValidationMetric], dict[str, object]]:
    metrics: list[ValidationMetric] = []
    mc_enum = load_csv(RAW_ROOT / "val_enum" / "fss_N4.csv").copy()
    exact_enum = exact_enumeration_2d(4, mc_enum["T"].to_numpy())
    beta = 1.0 / mc_enum["T"].to_numpy()
    mc_enum["chi_conn"] = beta * 16 * (mc_enum["M2"] - mc_enum["M"] ** 2)
    comparison = mc_enum.merge(exact_enum, on="T", suffixes=("_mc", "_exact"))
    for observable in ["E", "M", "Cv", "chi_conn"]:
        comparison[f"{observable}_abs_err"] = (comparison[f"{observable}_mc"] - comparison[f"{observable}_exact"]).abs()

    thresholds = [
        ("exact_enum_energy_max_abs_error", "E_abs_err", 0.02, 0.05),
        ("exact_enum_magnetization_max_abs_error", "M_abs_err", 0.01, 0.03),
        ("exact_enum_cv_max_abs_error", "Cv_abs_err", 0.2, 0.5),
        ("exact_enum_chi_max_abs_error", "chi_conn_abs_err", 0.2, 0.5),
    ]
    for name, column, good, warn in thresholds:
        value = float(comparison[column].max())
        metrics.append(
            ValidationMetric(
                name=name,
                status=status_from_threshold(value, good=good, warn=warn),
                measured=value,
                reference=good,
                details="4x4 exact enumeration comparison",
            )
        )
    return metrics, {"comparison": comparison}


def analyse_fluctuation_dissipation() -> tuple[list[ValidationMetric], dict[str, object]]:
    val_2d_files = require_files(sorted((RAW_ROOT / "val_2d").glob("fss_N*.csv")), "val_2d")
    fd_target = val_2d_files[-1]
    fd_data = load_csv(fd_target).copy()
    fd_data["dE_dT"] = np.gradient(fd_data["E"], fd_data["T"])
    fd_data["Cv_minus_dE_dT"] = fd_data["Cv"] - fd_data["dE_dT"]
    temperatures = fd_data["T"].to_numpy()[2:-2]
    residual = fd_data["Cv_minus_dE_dT"].to_numpy()[2:-2]
    mask_away = (temperatures < 2.0) | (temperatures > 2.6)
    rel_err_away = np.abs(residual[mask_away]) / np.maximum(np.abs(fd_data["Cv"].to_numpy()[2:-2][mask_away]), 1e-10)
    mean_rel_err = float(np.mean(rel_err_away))
    metrics = [
        ValidationMetric(
            name="fluctuation_dissipation_mean_relative_error_away_from_tc",
            status=status_from_threshold(mean_rel_err, good=0.2, warn=0.35),
            measured=mean_rel_err,
            reference=0.2,
            details=f"2D square {fd_target.name}, excluding critical window",
        )
    ]
    return metrics, {"comparison": fd_data, "largest_file": fd_target.name}


def analyse_autocorrelation() -> tuple[list[ValidationMetric], dict[str, object]]:
    sizes = []
    taus = []
    rows = []
    for raw_file in sorted((RAW_ROOT / "val_autocorr_wolff").glob("fss_raw_N*.csv")):
        with raw_file.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            e_samples = np.array([float(row["e"]) for row in reader], dtype=float)
        tau_int = integrated_autocorrelation_time(autocorrelation(e_samples, max_lag=min(2000, len(e_samples) // 4)))
        size = int(raw_file.stem.split("N")[1])
        sizes.append(size)
        taus.append(tau_int)
        rows.append({"n": size, "tau_int_e": tau_int, "n_samples": len(e_samples), "raw_file": raw_file.name})

    summary = pd.DataFrame(rows).sort_values("n").reset_index(drop=True)
    metrics: list[ValidationMetric] = []
    fit = {"z_eff": None, "z_w": None, "intercept": None}
    if len(sizes) >= 2:
        log_l = np.log(np.array(sizes, dtype=float))
        log_tau = np.log(np.array(taus, dtype=float))
        z_eff, intercept = np.polyfit(log_l, log_tau, 1)
        d_f_2d = 2.0 - 1.0 / 8.0
        z_w = float(z_eff - d_f_2d)
        fit = {"z_eff": float(z_eff), "z_w": z_w, "intercept": float(intercept)}
        metrics.append(
            ValidationMetric(
                name="wolff_dynamic_exponent_estimate",
                status="ok" if -0.5 <= z_w <= 0.5 else "warn",
                measured=z_w,
                reference=0.25,
                details="2D Wolff estimate after fractal-dimension correction",
            )
        )
    return metrics, {"summary": summary, "fit": fit}


def analyse_known_limits() -> tuple[list[ValidationMetric], dict[str, object]]:
    metrics: list[ValidationMetric] = []
    rows = []
    for path, expected_e, label in [
        (RAW_ROOT / "val_limits_3d", -3.0, "3d"),
        (RAW_ROOT / "val_limits_2d", -2.0, "2d"),
    ]:
        for csv_path in require_files(sorted(path.glob("fss_N*.csv")), path.name):
            df = load_csv(csv_path)
            size = int(csv_path.stem.split("N")[1])
            low_e = float(df["E"].iloc[0])
            low_m = float(df["M"].iloc[0])
            high_e = float(df["E"].iloc[-1])
            high_m = float(df["M"].iloc[-1])
            rows.append(
                {
                    "geometry": label,
                    "n": size,
                    "expected_low_temp_energy": expected_e,
                    "low_temp_energy": low_e,
                    "low_temp_magnetisation": low_m,
                    "high_temp_energy": high_e,
                    "high_temp_magnetisation": high_m,
                }
            )
            metrics.extend(
                [
                    ValidationMetric(
                        name=f"{label}_{csv_path.stem}_low_temp_energy",
                        status=status_from_threshold(abs(low_e - expected_e), good=0.05, warn=0.15),
                        measured=low_e,
                        reference=expected_e,
                        details="T=0.1 limit",
                    ),
                    ValidationMetric(
                        name=f"{label}_{csv_path.stem}_low_temp_magnetisation",
                        status=status_from_threshold(abs(low_m - 1.0), good=0.05, warn=0.15),
                        measured=low_m,
                        reference=1.0,
                        details="T=0.1 limit",
                    ),
                    ValidationMetric(
                        name=f"{label}_{csv_path.stem}_high_temp_energy",
                        status=status_from_threshold(abs(high_e), good=0.2, warn=0.5),
                        measured=high_e,
                        reference=0.0,
                        details="T=50 limit",
                    ),
                ]
            )
    return metrics, {"summary": pd.DataFrame(rows).sort_values(["geometry", "n"]).reset_index(drop=True)}


def analyse_validation() -> tuple[list[ValidationMetric], dict[str, dict[str, object]]]:
    DERIVED_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

    metrics: list[ValidationMetric] = []
    sections: dict[str, dict[str, object]] = {}
    for name, fn in [
        ("onsager", analyse_onsager),
        ("exact_enumeration", analyse_exact_enumeration),
        ("fluctuation_dissipation", analyse_fluctuation_dissipation),
        ("autocorrelation", analyse_autocorrelation),
        ("known_limits", analyse_known_limits),
    ]:
        section_metrics, section_payload = fn()
        metrics.extend(section_metrics)
        sections[name] = section_payload
    return metrics, sections


def plot_onsager(section: dict[str, object], output_path: Path) -> None:
    comparison = section["comparison"]
    largest_size = section["largest_size"]
    tc_exact = section["tc_exact"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    for n, df in comparison.groupby("n"):
        axes[0].plot(df["T"], df["E"], marker="o", ms=2.5, lw=0.9, label=f"L={n}")
        axes[1].plot(df["T"], df["M"], marker="o", ms=2.5, lw=0.9, label=f"L={n}")

    largest_df = comparison.loc[comparison["n"] == largest_size]
    axes[0].plot(largest_df["T"], largest_df["E_exact"], color="black", lw=1.4, label="Onsager")
    axes[1].plot(largest_df["T"], largest_df["M_exact"], color="black", lw=1.4, label="Onsager")
    axes[2].plot(largest_df["T"], largest_df["E"] - largest_df["E_exact"], marker="o", ms=2.5, lw=0.9)
    axes[2].axhline(0.0, color="black", ls="--", lw=0.8)
    axes[2].axvline(tc_exact, color="gray", ls=":", lw=0.8)

    axes[0].set_title("Energy vs Onsager")
    axes[1].set_title("Magnetisation vs Onsager")
    axes[2].set_title(f"Energy residual, L={largest_size}")
    axes[0].set_ylabel("Observable")
    for ax in axes:
        ax.set_xlabel("T (J/k_B)")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_exact_enumeration(section: dict[str, object], output_path: Path) -> None:
    comparison = section["comparison"]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5), constrained_layout=True)
    mapping = [
        ("E", "Energy per spin"),
        ("M", "Magnetisation"),
        ("Cv", "Heat capacity"),
        ("chi_conn", "Connected susceptibility"),
    ]
    for ax, (column, title) in zip(axes.flat, mapping):
        ax.plot(comparison["T"], comparison[f"{column}_mc"], marker="o", ms=2.4, lw=0.9, label="Monte Carlo")
        ax.plot(comparison["T"], comparison[f"{column}_exact"], color="black", lw=1.2, label="Exact")
        ax.set_title(title)
        ax.set_xlabel("T (J/k_B)")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_fluctuation_dissipation(section: dict[str, object], output_path: Path) -> None:
    comparison = section["comparison"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    axes[0].plot(comparison["T"], comparison["Cv"], marker="o", ms=2.3, lw=0.9, label="Cv from fluctuations")
    axes[0].plot(comparison["T"], comparison["dE_dT"], color="black", lw=1.2, label="dE/dT")
    axes[1].plot(comparison["T"], comparison["Cv_minus_dE_dT"], marker="o", ms=2.3, lw=0.9)
    axes[1].axhline(0.0, color="black", ls="--", lw=0.8)
    axes[0].set_title("Fluctuation-dissipation check")
    axes[1].set_title("Residual: Cv - dE/dT")
    for ax in axes:
        ax.set_xlabel("T (J/k_B)")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_autocorrelation(section: dict[str, object], output_path: Path) -> None:
    summary = section["summary"]
    fit = section["fit"]
    fig, ax = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
    ax.loglog(summary["n"], summary["tau_int_e"], marker="o", ms=4, lw=0.9, label="Measured")
    if fit["z_eff"] is not None:
        x = np.array(summary["n"], dtype=float)
        y = np.exp(fit["intercept"]) * x ** fit["z_eff"]
        ax.loglog(x, y, color="black", lw=1.2, label=f"fit z_eff={fit['z_eff']:.3f}")
    ax.set_xlabel("L")
    ax.set_ylabel("tau_int(E)")
    ax.set_title("Wolff autocorrelation scaling")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_known_limits(section: dict[str, object], output_path: Path) -> None:
    summary = section["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    for geometry, df in summary.groupby("geometry"):
        axes[0].plot(df["n"], df["low_temp_energy"], marker="o", ms=4, lw=0.9, label=geometry)
        axes[1].plot(df["n"], df["high_temp_energy"], marker="o", ms=4, lw=0.9, label=geometry)
        axes[0].hlines(df["expected_low_temp_energy"].iloc[0], xmin=df["n"].min(), xmax=df["n"].max(), colors="black", linestyles="--", lw=0.8)
    axes[1].axhline(0.0, color="black", ls="--", lw=0.8)
    axes[0].set_title("Low-temperature energy limit")
    axes[1].set_title("High-temperature energy limit")
    for ax in axes:
        ax.set_xlabel("L")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Energy per spin")
    axes[0].legend(fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_overview(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prefix, label in [
        ("onsager_", "Onsager 2D"),
        ("exact_enum_", "Exact enumeration"),
        ("fluctuation_dissipation_", "Fluctuation-dissipation"),
        ("wolff_", "Autocorrelation"),
        ("2d_", "2D limits"),
        ("3d_", "3D limits"),
    ]:
        subset = metrics_df.loc[metrics_df["name"].str.startswith(prefix)]
        if subset.empty:
            continue
        status = "ok"
        if (subset["status"] == "bad").any():
            status = "bad"
        elif (subset["status"] == "warn").any():
            status = "warn"
        rows.append(
            {
                "section": label,
                "status": status,
                "n_metrics": int(len(subset)),
                "n_ok": int((subset["status"] == "ok").sum()),
                "n_warn": int((subset["status"] == "warn").sum()),
                "n_bad": int((subset["status"] == "bad").sum()),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    metrics: list[ValidationMetric],
    sections: dict[str, dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, list[str] | str]:
    DERIVED_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

    generated_tables: list[str] = []
    generated_figures: list[str] = []

    metrics_df = pd.DataFrame([asdict(metric) for metric in metrics])
    metrics_path = DERIVED_ROOT / "validation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    generated_tables.append(str(metrics_path))

    overview_df = build_overview(metrics_df)
    overview_path = DERIVED_ROOT / "validation_overview.csv"
    overview_df.to_csv(overview_path, index=False)
    generated_tables.append(str(overview_path))

    onsager_path = DERIVED_ROOT / "onsager_comparison.csv"
    sections["onsager"]["comparison"].to_csv(onsager_path, index=False)
    generated_tables.append(str(onsager_path))

    exact_path = DERIVED_ROOT / "exact_enumeration_comparison.csv"
    sections["exact_enumeration"]["comparison"].to_csv(exact_path, index=False)
    generated_tables.append(str(exact_path))

    fd_path = DERIVED_ROOT / "fluctuation_dissipation.csv"
    sections["fluctuation_dissipation"]["comparison"].to_csv(fd_path, index=False)
    generated_tables.append(str(fd_path))

    autocorr_path = DERIVED_ROOT / "autocorrelation_summary.csv"
    sections["autocorrelation"]["summary"].to_csv(autocorr_path, index=False)
    generated_tables.append(str(autocorr_path))

    limits_path = DERIVED_ROOT / "known_limits_summary.csv"
    sections["known_limits"]["summary"].to_csv(limits_path, index=False)
    generated_tables.append(str(limits_path))

    plot_onsager(sections["onsager"], FIGURE_ROOT / "onsager_comparison.png")
    plot_exact_enumeration(sections["exact_enumeration"], FIGURE_ROOT / "exact_enumeration.png")
    plot_fluctuation_dissipation(sections["fluctuation_dissipation"], FIGURE_ROOT / "fluctuation_dissipation.png")
    plot_autocorrelation(sections["autocorrelation"], FIGURE_ROOT / "autocorrelation_scaling.png")
    plot_known_limits(sections["known_limits"], FIGURE_ROOT / "known_limits.png")
    generated_figures.extend(str(path) for path in sorted(FIGURE_ROOT.glob("*.png")))

    overall_status = "ok"
    if any(metric.status == "bad" for metric in metrics):
        overall_status = "bad"
    elif any(metric.status == "warn" for metric in metrics):
        overall_status = "warn"

    summary = {
        "overall_status": overall_status,
        "mode": "quick" if args.quick else "full",
        "n_metrics": len(metrics),
        "n_ok": sum(metric.status == "ok" for metric in metrics),
        "n_warn": sum(metric.status == "warn" for metric in metrics),
        "n_bad": sum(metric.status == "bad" for metric in metrics),
        "metrics": [asdict(metric) for metric in metrics],
        "sections": sections["autocorrelation"]["fit"],
    }
    summary_path = DERIVED_ROOT / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "workflow": "analysis/scripts/reproduce_validation.py",
        "mode": "quick" if args.quick else "full",
        "analysis_only": args.analysis_only,
        "skip_existing": args.skip_existing,
        "raw_root": str(RAW_ROOT),
        "derived_root": str(DERIVED_ROOT),
        "figure_root": str(FIGURE_ROOT),
        "input_files": sorted(str(path) for path in RAW_ROOT.glob("*/*.csv")),
        "generated_tables": generated_tables + [str(summary_path)],
        "generated_figures": generated_figures,
    }
    manifest_path = MANIFEST_ROOT / f"validation_{'quick' if args.quick else 'full'}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "metrics": str(metrics_path),
        "overview": str(overview_path),
        "summary": str(summary_path),
        "manifest": str(manifest_path),
        "tables": generated_tables,
        "figures": generated_figures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and analyse the Ising validation suite.")
    parser.add_argument("--quick", action="store_true", help="Use reduced lattice sizes and sampling.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing raw CSVs if they are already present.",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip data generation and only analyse the current validation datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.analysis_only:
        prepare_datasets(quick=args.quick, skip_existing=args.skip_existing)
    metrics, sections = analyse_validation()
    outputs = write_outputs(metrics, sections, args)

    print(f"wrote {outputs['metrics']}")
    print(f"wrote {outputs['overview']}")
    print(f"wrote {outputs['summary']}")
    print(f"wrote {outputs['manifest']}")
    for figure in outputs["figures"]:
        print(f"wrote {figure}")
    for metric in metrics:
        print(f"[{metric.status}] {metric.name}: measured={metric.measured} reference={metric.reference} {metric.details}")


if __name__ == "__main__":
    main()
