#!/usr/bin/env python3
"""
GPU FSS Publication Analysis — Ising, Heisenberg, XY

Reads summary + timeseries CSVs from gpu_fss publication runs, performs:
1. Raw observable plots (E, M, Cv, chi) for all sizes
2. Binder cumulant crossing → Tc
3. WHAM histogram reweighting → fine T grid near Tc
4. Reweighted peak scaling → gamma/nu, beta/nu
5. Scaling collapse → nu
6. Summary table with theory comparison

Usage:
    python analysis/scripts/analyze_gpu_fss.py
    python analysis/scripts/analyze_gpu_fss.py --data-dir path/to/publication
    python analysis/scripts/analyze_gpu_fss.py --model ising
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

# Add analysis/scripts to path for reweighting module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reweighting import wham_reweight

# ── Theory values ──────────────────────────────────────────────────────────
THEORY = {
    "ising": {"Tc": 4.5115, "nu": 0.6301, "gamma": 1.2372, "beta": 0.3265},
    "heisenberg": {"Tc": 1.4430, "nu": 0.7112, "gamma": 1.3960, "beta": 0.3689},
    "xy": {"Tc": 2.2019, "nu": 0.6717, "gamma": 1.3178, "beta": 0.3486},
}


def load_model(data_dir: Path, model: str) -> dict[int, pd.DataFrame]:
    """Load summary CSVs for a model, return {size: DataFrame}."""
    dfs = {}
    for p in sorted(data_dir.glob(f"gpu_fss_{model}_N*_summary.csv")):
        n = int(p.stem.split("_N")[1].split("_")[0])
        df = pd.read_csv(p)
        # Compute Binder cumulant: U = 1 - M4 / (3 * M2^2)
        df["U"] = 1.0 - df["M4"] / (3.0 * df["M2"] ** 2)
        # Compute connected susceptibility: chi_conn = beta * V * (M2 - M^2)
        V = n ** 3
        df["chi_conn"] = (1.0 / df["T"]) * V * (df["M2"] - df["M"] ** 2)
        dfs[n] = df
    return dfs


def load_timeseries(data_dir: Path, model: str, n: int, temperatures: np.ndarray):
    """Load timeseries CSV and split into per-replica energy/mag lists.

    Returns (energy_lists, mag_lists, beta_list) for WHAM input.
    energy_lists[k] = total energies for replica k
    mag_lists[k] = |m| per spin for replica k
    """
    p = data_dir / f"gpu_fss_{model}_N{n}_timeseries.csv"
    if not p.exists():
        return None, None, None
    df = pd.read_csv(p)
    V = n ** 3
    n_temps = len(temperatures)
    energy_lists = []
    mag_lists = []
    beta_list = []
    for t_idx in range(n_temps):
        sub = df[df["temp_idx"] == t_idx]
        if len(sub) == 0:
            continue
        energy_lists.append(sub["E"].values * V)  # total energy
        mag_lists.append(sub["M"].values)          # |m| per spin
        beta_list.append(1.0 / temperatures[t_idx])
    return energy_lists, mag_lists, np.array(beta_list)


def reweight_single_histogram(energy_lists, mag_lists, beta_list, T_fine, n: int):
    """Fast single-histogram reweighting to fine T grid.

    For each target T, picks the closest simulation temperature and reweights.
    Much faster than WHAM (no iterative solve), good enough for peak finding.
    """
    V = n ** 3
    sim_temps = 1.0 / beta_list
    rows = []
    for T in T_fine:
        beta = 1.0 / T
        # Find closest simulation temperature
        k_best = int(np.argmin(np.abs(sim_temps - T)))
        e_sim = energy_lists[k_best]
        m_sim = mag_lists[k_best]
        delta_beta = beta - beta_list[k_best]

        log_w = -delta_beta * e_sim
        log_w -= log_w.max()
        w = np.exp(log_w)
        w /= w.sum()

        avg_m = np.sum(w * m_sim)
        avg_m2 = np.sum(w * m_sim ** 2)
        avg_m4 = np.sum(w * m_sim ** 4)
        avg_e = np.sum(w * e_sim)
        avg_e2 = np.sum(w * e_sim ** 2)

        chi_conn = beta * V * (avg_m2 - avg_m ** 2)
        cv = beta ** 2 * (avg_e2 - avg_e ** 2) / V
        U = 1.0 - avg_m4 / (3.0 * avg_m2 ** 2) if avg_m2 > 1e-15 else 0.0

        # Jackknife error (fast: only over samples from one replica)
        n_blocks = 20
        n_s = len(e_sim)
        bs = n_s // n_blocks
        jk_chi = np.zeros(n_blocks)
        jk_m_arr = np.zeros(n_blocks)
        jk_U_arr = np.zeros(n_blocks)
        for b in range(n_blocks):
            idx = np.concatenate([np.arange(0, b * bs), np.arange((b + 1) * bs, n_blocks * bs)])
            ej, mj = e_sim[idx], m_sim[idx]
            lw = -delta_beta * ej
            lw -= lw.max()
            wj = np.exp(lw)
            wj /= wj.sum()
            jm = np.sum(wj * mj)
            jm2 = np.sum(wj * mj ** 2)
            jm4 = np.sum(wj * mj ** 4)
            jk_chi[b] = beta * V * (jm2 - jm ** 2)
            jk_m_arr[b] = jm
            jk_U_arr[b] = 1.0 - jm4 / (3.0 * jm2 ** 2) if jm2 > 1e-15 else 0.0

        jk_f = (n_blocks - 1.0) / n_blocks
        chi_err = np.sqrt(jk_f * np.sum((jk_chi - chi_conn) ** 2))
        m_err = np.sqrt(jk_f * np.sum((jk_m_arr - avg_m) ** 2))
        U_err = np.sqrt(jk_f * np.sum((jk_U_arr - U) ** 2))

        rows.append({
            "T": T, "M": avg_m, "M2": avg_m2, "M4": avg_m4,
            "chi_conn": chi_conn, "chi_err": chi_err,
            "Cv": cv, "U": U, "E": avg_e / V,
            "M_err": m_err, "U_err": U_err,
        })
    return pd.DataFrame(rows)


def find_binder_crossings(dfs: dict[int, pd.DataFrame]) -> list[float]:
    """Find Tc from Binder cumulant crossings between adjacent sizes."""
    sizes = sorted(dfs.keys())
    crossings = []
    for i in range(len(sizes) - 1):
        n1, n2 = sizes[i], sizes[i + 1]
        df1, df2 = dfs[n1], dfs[n2]
        t_lo = max(df1["T"].min(), df2["T"].min())
        t_hi = min(df1["T"].max(), df2["T"].max())
        T_common = np.linspace(t_lo, t_hi, 2000)
        try:
            f1 = interp1d(df1["T"], df1["U"], kind="cubic")
            f2 = interp1d(df2["T"], df2["U"], kind="cubic")
        except ValueError:
            continue
        diff = f1(T_common) - f2(T_common)
        for j in range(len(diff) - 1):
            if diff[j] * diff[j + 1] < 0:
                tc = T_common[j] - diff[j] * (T_common[j + 1] - T_common[j]) / (diff[j + 1] - diff[j])
                crossings.append((n1, n2, tc))
                break
    return crossings


def peak_scaling(dfs: dict[int, pd.DataFrame], tc: float):
    """Extract gamma/nu from chi_conn peak and beta/nu from M(Tc)."""
    sizes = sorted(dfs.keys())
    log_n = np.log(sizes)

    # chi_conn peak
    chi_peaks = []
    for n in sizes:
        df = dfs[n]
        chi_peaks.append(df["chi_conn"].max())
    slope_chi, _ = np.polyfit(log_n, np.log(chi_peaks), 1)

    # M(Tc)
    m_at_tc = []
    for n in sizes:
        df = dfs[n]
        try:
            f_m = interp1d(df["T"], df["M"], kind="cubic")
            m_at_tc.append(float(f_m(tc)))
        except (ValueError, KeyError):
            m_at_tc.append(np.nan)

    valid = [(n, m) for n, m in zip(sizes, m_at_tc) if not np.isnan(m) and m > 0]
    if len(valid) >= 2:
        vn, vm = zip(*valid)
        slope_m, _ = np.polyfit(np.log(vn), np.log(vm), 1)
        beta_nu = -slope_m
    else:
        beta_nu = np.nan

    return slope_chi, beta_nu, sizes, chi_peaks, m_at_tc


def collapse_quality(nu, dfs, tc, gamma_nu):
    """Adjacent-point cost for chi_conn scaling collapse."""
    sizes = sorted(dfs.keys())
    all_x, all_y = [], []
    for n in sizes:
        df = dfs[n]
        x = (df["T"].values - tc) * n ** (1.0 / nu)
        y = df["chi_conn"].values / n ** gamma_nu
        mask = (x > -15) & (x < 15)
        all_x.extend(x[mask])
        all_y.extend(y[mask])
    if len(all_x) < 10:
        return 1e10
    order = np.argsort(all_x)
    ys = np.array(all_y)[order]
    rng = ys.max() - ys.min()
    if rng < 1e-15:
        return 1e10
    return float(np.sum(np.diff(ys) ** 2)) / rng ** 2


def analyze_model(model: str, dfs: dict[int, pd.DataFrame], out_dir: Path,
                   data_dir: Path):
    """Full FSS analysis for one model."""
    th = THEORY[model]
    sizes = sorted(dfs.keys())
    colors = cm.viridis(np.linspace(0.15, 0.85, len(sizes)))

    print(f"\n{'=' * 60}")
    print(f"  {model.upper()} — 3D FSS Analysis")
    print(f"{'=' * 60}")
    print(f"Sizes: {sizes}")
    print(f"T range: [{dfs[sizes[0]]['T'].min():.4f}, {dfs[sizes[0]]['T'].max():.4f}]")
    print(f"Temperatures per size: {len(dfs[sizes[0]])}")

    # ── 1. Raw observables ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    cols = ["E", "M", "Cv", "chi_conn"]
    errs = ["E_err", "M_err", "Cv_err", "chi_err"]
    labels = [r"$\langle e \rangle / J$", r"$|m|$", r"$C_v$", r"$\chi_{\mathrm{conn}}$"]
    panels = ["a", "b", "c", "d"]

    for ax, col, err_col, lab, pan in zip(axes.flat, cols, errs, labels, panels):
        for (n, df), c in zip(sorted(dfs.items()), colors):
            ax.plot(df["T"], df[col], "-", color=c, lw=0.8, label=f"L={n}")
            if err_col in df.columns:
                ax.fill_between(df["T"], df[col] - df[err_col], df[col] + df[err_col],
                                color=c, alpha=0.12)
        ax.axvline(th["Tc"], color="red", lw=0.6, ls="--")
        ax.set_xlabel(r"$T \; [J/k_B]$")
        ax.set_ylabel(lab)
        ax.text(0.05, 0.92, f"({pan})", transform=ax.transAxes, fontsize=9, fontweight="bold")

    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"{model.upper()} — Raw Observables", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / f"{model}_observables.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {model}_observables.png")

    # ── 2. Binder cumulant crossing → Tc ───────────────────────────────────
    crossings = find_binder_crossings(dfs)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for (n, df), c in zip(sorted(dfs.items()), colors):
        ax.plot(df["T"], df["U"], color=c, lw=0.8, label=f"L={n}")
    ax.axvline(th["Tc"], color="red", lw=0.6, ls="--", label=f"Tc(theory)={th['Tc']}")
    ax.set_xlabel(r"$T \; [J/k_B]$")
    ax.set_ylabel("Binder cumulant $U$")
    ax.legend(fontsize=6, ncol=2)
    fig.suptitle(f"{model.upper()} — Binder Cumulant", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / f"{model}_binder.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {model}_binder.png")

    if crossings:
        # Use last (largest-size) crossings
        best = crossings[-min(3, len(crossings)):]
        tc_meas = np.mean([c[2] for c in best])
        tc_err = np.std([c[2] for c in best]) if len(best) > 1 else 0.0
        print(f"\n  Binder crossings:")
        for n1, n2, tc in crossings:
            print(f"    L={n1}/{n2}: Tc = {tc:.4f}")
        print(f"  Tc (Binder) = {tc_meas:.4f} +/- {tc_err:.4f}")
        print(f"  Tc (theory) = {th['Tc']:.4f}")
        print(f"  Error: {abs(tc_meas - th['Tc']) / th['Tc'] * 100:.2f}%")
    else:
        tc_meas = th["Tc"]
        tc_err = 0.0
        print("  No Binder crossings found — using theory Tc")

    # ── 3. WHAM histogram reweighting → fine T grid ─────────────────────────
    print(f"\n  Histogram reweighting (WHAM)...")
    t_lo = dfs[sizes[0]]["T"].min()
    t_hi = dfs[sizes[0]]["T"].max()
    T_fine = np.linspace(t_lo, t_hi, 200)

    rw_dfs = {}
    for n in sizes:
        temperatures = dfs[n]["T"].values
        e_lists, m_lists, beta_list = load_timeseries(data_dir, model, n, temperatures)
        if e_lists is None:
            print(f"    N={n}: no timeseries, skipping reweighting")
            continue
        print(f"    N={n}: reweighting ({sum(len(e) for e in e_lists)} samples, "
              f"{len(e_lists)} replicas)...", end=" ", flush=True)
        rw = reweight_single_histogram(e_lists, m_lists, beta_list, T_fine, n)
        rw_dfs[n] = rw
        idx = rw["chi_conn"].idxmax()
        print(f"chi_peak={rw.loc[idx, 'chi_conn']:.1f} at T={rw.loc[idx, 'T']:.4f}")

    # Use reweighted data for peak scaling if available, else fall back to summary
    use_rw = len(rw_dfs) >= 2
    peak_src = rw_dfs if use_rw else dfs
    src_label = "reweighted" if use_rw else "summary"

    # ── 3b. Reweighted observables plot ────────────────────────────────────
    if use_rw:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        rw_colors = cm.viridis(np.linspace(0.15, 0.85, len(rw_dfs)))

        ax = axes[0]
        for (n, rw), c in zip(sorted(rw_dfs.items()), rw_colors):
            ax.plot(rw["T"], rw["chi_conn"], color=c, lw=0.8, label=f"L={n}")
            if "chi_err" in rw.columns:
                ax.fill_between(rw["T"], rw["chi_conn"] - rw["chi_err"],
                                rw["chi_conn"] + rw["chi_err"], color=c, alpha=0.15)
        ax.axvline(th["Tc"], color="red", ls="--", lw=0.6)
        ax.set_xlabel(r"$T$"); ax.set_ylabel(r"$\chi_{\mathrm{conn}}$")
        ax.legend(fontsize=5, ncol=2)

        ax = axes[1]
        for (n, rw), c in zip(sorted(rw_dfs.items()), rw_colors):
            ax.plot(rw["T"], rw["Cv"], color=c, lw=0.8, label=f"L={n}")
        ax.axvline(th["Tc"], color="red", ls="--", lw=0.6)
        ax.set_xlabel(r"$T$"); ax.set_ylabel(r"$C_v$")

        ax = axes[2]
        for (n, rw), c in zip(sorted(rw_dfs.items()), rw_colors):
            ax.plot(rw["T"], rw["U"], color=c, lw=0.8, label=f"L={n}")
            if "U_err" in rw.columns:
                ax.fill_between(rw["T"], rw["U"] - rw["U_err"],
                                rw["U"] + rw["U_err"], color=c, alpha=0.15)
        ax.axvline(th["Tc"], color="red", ls="--", lw=0.6)
        ax.set_xlabel(r"$T$"); ax.set_ylabel("Binder $U$")

        fig.suptitle(f"{model.upper()} — WHAM Reweighted Observables", fontsize=11)
        plt.tight_layout()
        fig.savefig(out_dir / f"{model}_reweighted.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {model}_reweighted.png")

    # ── 3c. Reweighted Binder crossing (finer Tc) ─────────────────────────
    if use_rw:
        rw_crossings = find_binder_crossings(rw_dfs)
        if rw_crossings:
            best_rw = rw_crossings[-min(3, len(rw_crossings)):]
            tc_rw = np.mean([c[2] for c in best_rw])
            tc_rw_err = np.std([c[2] for c in best_rw]) if len(best_rw) > 1 else 0.0
            print(f"\n  Reweighted Binder crossings:")
            for n1, n2, tc in rw_crossings:
                print(f"    L={n1}/{n2}: Tc = {tc:.4f}")
            print(f"  Tc (WHAM Binder) = {tc_rw:.4f} +/- {tc_rw_err:.4f}")
            print(f"  Error: {abs(tc_rw - th['Tc']) / th['Tc'] * 100:.3f}%")
            tc_meas = tc_rw  # use better estimate
            tc_err = tc_rw_err

    # ── 4. Peak scaling from reweighted data → gamma/nu, beta/nu ──────────
    rw_sizes = sorted(peak_src.keys())
    log_n = np.log(rw_sizes)

    chi_peaks = []
    chi_peak_errs = []
    for n in rw_sizes:
        df = peak_src[n]
        idx = df["chi_conn"].idxmax()
        chi_peaks.append(df.loc[idx, "chi_conn"])
        chi_peak_errs.append(df.loc[idx, "chi_err"] if "chi_err" in df.columns else 0.0)

    slope_chi, _ = np.polyfit(log_n, np.log(chi_peaks), 1)
    gamma_nu = slope_chi
    th_gamma_nu = th["gamma"] / th["nu"]

    m_at_tc = []
    m_at_tc_errs = []
    for n in rw_sizes:
        df = peak_src[n]
        try:
            f_m = interp1d(df["T"], df["M"], kind="cubic")
            m_at_tc.append(float(f_m(tc_meas)))
            if "M_err" in df.columns:
                f_e = interp1d(df["T"], df["M_err"], kind="linear")
                m_at_tc_errs.append(float(f_e(tc_meas)))
            else:
                m_at_tc_errs.append(0.0)
        except (ValueError, KeyError):
            m_at_tc.append(np.nan)
            m_at_tc_errs.append(0.0)

    valid = [(n, m) for n, m in zip(rw_sizes, m_at_tc) if not np.isnan(m) and m > 0]
    if len(valid) >= 2:
        vn, vm = zip(*valid)
        slope_m, _ = np.polyfit(np.log(vn), np.log(vm), 1)
        beta_nu = -slope_m
    else:
        beta_nu = np.nan
    th_beta_nu = th["beta"] / th["nu"]

    print(f"\n  Peak scaling ({src_label}):")
    print(f"    gamma/nu = {gamma_nu:.4f}  (theory = {th_gamma_nu:.4f}, error = {abs(gamma_nu - th_gamma_nu) / th_gamma_nu * 100:.1f}%)")
    if not np.isnan(beta_nu):
        print(f"    beta/nu  = {beta_nu:.4f}  (theory = {th_beta_nu:.4f}, error = {abs(beta_nu - th_beta_nu) / th_beta_nu * 100:.1f}%)")
        hyperscaling = 2 * beta_nu + gamma_nu
        print(f"    2*beta/nu + gamma/nu = {hyperscaling:.4f}  (should = 3, error = {abs(hyperscaling - 3) / 3 * 100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    n_fit = np.linspace(min(rw_sizes) * 0.9, max(rw_sizes) * 1.1, 100)

    ax = axes[0]
    ax.errorbar(rw_sizes, chi_peaks, yerr=chi_peak_errs, fmt="o", ms=5, color="C0",
                capsize=3, zorder=5)
    c0 = np.log(chi_peaks[0]) - gamma_nu * np.log(rw_sizes[0])
    ax.plot(n_fit, np.exp(gamma_nu * np.log(n_fit) + c0), "--", color="C1", lw=0.8,
            label=rf"$\gamma/\nu={gamma_nu:.3f}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$L$"); ax.set_ylabel(r"$\chi_{\mathrm{max}}$")
    ax.legend(fontsize=8)

    ax = axes[1]
    valid_idx = [i for i, m in enumerate(m_at_tc) if not np.isnan(m) and m > 0]
    if valid_idx:
        vn = [rw_sizes[i] for i in valid_idx]
        vm = [m_at_tc[i] for i in valid_idx]
        ve = [m_at_tc_errs[i] for i in valid_idx]
        ax.errorbar(vn, vm, yerr=ve, fmt="o", ms=5, color="C0", capsize=3, zorder=5)
        if not np.isnan(beta_nu):
            c0m = np.log(vm[0]) + beta_nu * np.log(vn[0])
            ax.plot(n_fit, np.exp(-beta_nu * np.log(n_fit) + c0m), "--", color="C1", lw=0.8,
                    label=rf"$\beta/\nu={beta_nu:.3f}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$L$"); ax.set_ylabel(r"$M(T_c)$")
    ax.legend(fontsize=8)

    fig.suptitle(f"{model.upper()} — Peak Scaling ({src_label})", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / f"{model}_peak_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {model}_peak_scaling.png")

    # ── 5. Scaling collapse → nu ───────────────────────────────────────────
    result = minimize_scalar(
        collapse_quality, bounds=(0.3, 1.2), method="bounded",
        args=(peak_src, tc_meas, gamma_nu),
    )
    nu_collapse = result.x
    print(f"\n  Scaling collapse:")
    print(f"    nu (chi collapse) = {nu_collapse:.4f}  (theory = {th['nu']:.4f}, error = {abs(nu_collapse - th['nu']) / th['nu'] * 100:.1f}%)")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for i, n in enumerate(sorted(peak_src.keys())):
        df = peak_src[n]
        x = (df["T"] - tc_meas) * n ** (1.0 / nu_collapse)
        y = df["chi_conn"] / n ** gamma_nu
        mask = (x > -15) & (x < 15)
        ax.plot(x[mask], y[mask], "o-", color=colors[i], markersize=2, lw=0.6, label=f"L={n}")
    ax.set_xlabel(r"$(T - T_c) L^{1/\nu}$")
    ax.set_ylabel(r"$\chi_{\mathrm{conn}} / L^{\gamma/\nu}$")
    ax.legend(fontsize=6, ncol=2)
    ax.set_xlim(-15, 15)
    fig.suptitle(f"{model.upper()} — Scaling Collapse (nu={nu_collapse:.3f})", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / f"{model}_collapse.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {model}_collapse.png")

    # ── 6. Summary table ───────────────────────────────────────────────────
    print(f"\n  {'Quantity':<24} {'Measured':>10} {'Theory':>10} {'Error':>8}")
    print(f"  {'-' * 55}")
    rows = [
        ("Tc (Binder)", tc_meas, th["Tc"]),
        ("gamma/nu (peak)", gamma_nu, th_gamma_nu),
        ("nu (collapse)", nu_collapse, th["nu"]),
    ]
    if not np.isnan(beta_nu):
        rows.append(("beta/nu (M@Tc)", beta_nu, th_beta_nu))
        rows.append(("2b/n + g/n", 2 * beta_nu + gamma_nu, 3.0))

    for name, meas, theory in rows:
        err = abs(meas - theory) / abs(theory) * 100
        status = "OK" if err < 5 else "~" if err < 15 else "!"
        print(f"  {name:<24} {meas:>10.4f} {theory:>10.4f} {err:>7.1f}%  {status}")

    return {
        "model": model,
        "Tc": tc_meas, "Tc_err": tc_err,
        "gamma_nu": gamma_nu,
        "beta_nu": beta_nu,
        "nu": nu_collapse,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze GPU FSS publication data")
    parser.add_argument("--data-dir", default="analysis/data/gpu_windows_pipeline/publication",
                        help="Directory containing gpu_fss_*_summary.csv files")
    parser.add_argument("--out-dir", default="analysis/figures/gpu_fss",
                        help="Output directory for figures")
    parser.add_argument("--model", default=None, choices=["ising", "heisenberg", "xy"],
                        help="Analyze only one model (default: all)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = (repo_root / args.data_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)

    models = [args.model] if args.model else ["ising", "heisenberg", "xy"]
    results = []

    for model in models:
        dfs = load_model(data_dir, model)
        if not dfs:
            print(f"\nNo data found for {model}, skipping.")
            continue
        res = analyze_model(model, dfs, out_dir, data_dir)
        results.append(res)

    # ── Cross-model comparison ─────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print(f"  CROSS-MODEL COMPARISON")
        print(f"{'=' * 60}")
        print(f"  {'Model':<14} {'Tc':>8} {'gamma/nu':>10} {'beta/nu':>10} {'nu':>8}")
        print(f"  {'-' * 52}")
        for r in results:
            bn = f"{r['beta_nu']:.4f}" if not np.isnan(r["beta_nu"]) else "N/A"
            print(f"  {r['model']:<14} {r['Tc']:>8.4f} {r['gamma_nu']:>10.4f} {bn:>10} {r['nu']:>8.4f}")
        print(f"\n  Theory:")
        for model in models:
            if model in THEORY:
                th = THEORY[model]
                print(f"  {model:<14} {th['Tc']:>8.4f} {th['gamma']/th['nu']:>10.4f} {th['beta']/th['nu']:>10.4f} {th['nu']:>8.4f}")

    print(f"\nFigures saved to: {out_dir}")


if __name__ == "__main__":
    main()
