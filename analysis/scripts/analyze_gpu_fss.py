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


def wham_reweight_fine(energy_lists, mag_lists, beta_list, T_fine, n: int,
                       n_iter=100, tol=1e-8):
    """WHAM (multiple-histogram) reweighting to fine T grid.

    Unlike single-histogram, this uses ALL replicas simultaneously for each
    target T, giving better accuracy especially near Tc where overlap matters.

    The expensive WHAM denominator (log_denom) is computed once and reused
    for all target temperatures.

    Error bars use single-histogram jackknife from the closest replica
    (fast approximation — full WHAM jackknife is too expensive for large datasets).
    """
    V = n ** 3
    K = len(beta_list)
    N_k = np.array([len(e) for e in energy_lists])
    all_energies = np.concatenate(energy_lists)
    all_mags = np.concatenate(mag_lists)

    # Solve WHAM equations iteratively
    f = np.zeros(K)
    for _ in range(n_iter):
        # Vectorised: log_denom_terms[n, k] = log(N_k) + f_k - beta_k * E_n
        # Use broadcasting: (N,1) op (K,) → (N,K)
        log_denom_terms = (np.log(N_k) + f)[np.newaxis, :] - \
                          beta_list[np.newaxis, :] * all_energies[:, np.newaxis]
        log_denom = np.logaddexp.reduce(log_denom_terms, axis=1)

        f_new = np.zeros(K)
        for k in range(K):
            log_terms = -beta_list[k] * all_energies - log_denom
            f_new[k] = -np.logaddexp.reduce(log_terms)
        f_new -= f_new[0]

        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new

    # Precompute mag powers
    all_mags2 = all_mags ** 2
    all_mags4 = all_mags ** 4
    all_energies2 = all_energies ** 2

    # Compute observables at all target temperatures (vectorised weight application)
    sim_temps = 1.0 / beta_list
    rows = []
    for T in T_fine:
        beta_t = 1.0 / T
        log_w = -beta_t * all_energies - log_denom
        log_w -= log_w.max()
        w = np.exp(log_w)
        w /= w.sum()

        avg_m = w @ all_mags
        avg_m2 = w @ all_mags2
        avg_m4 = w @ all_mags4
        avg_e = w @ all_energies
        avg_e2 = w @ all_energies2

        chi_conn = beta_t * V * (avg_m2 - avg_m ** 2)
        cv = beta_t ** 2 * (avg_e2 - avg_e ** 2) / V
        U = 1.0 - avg_m4 / (3.0 * avg_m2 ** 2) if avg_m2 > 1e-15 else 0.0

        # Fast error bars via single-histogram jackknife from closest replica
        k_best = int(np.argmin(np.abs(sim_temps - T)))
        e_sim = energy_lists[k_best]
        m_sim = mag_lists[k_best]
        delta_beta = beta_t - beta_list[k_best]
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
            jk_chi[b] = beta_t * V * (jm2 - jm ** 2)
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


def binder_slope_nu(dfs: dict[int, pd.DataFrame], tc: float, fit_sizes: list[int] | None = None):
    """Estimate nu from Binder slope scaling: dU/dT|Tc ~ L^{1/nu}.

    Computes dU/dT at Tc via cubic interpolation derivative for each size,
    then fits log(|dU/dT|) vs log(L) to extract 1/nu.

    Returns (nu, sizes_used, slopes) or (nan, [], []) on failure.
    """
    sizes = sorted(fit_sizes if fit_sizes else dfs.keys())
    log_L = []
    log_dUdT = []
    slopes_out = []

    for n in sizes:
        if n not in dfs:
            continue
        df = dfs[n]
        T = df["T"].values
        U = df["U"].values

        # Mask out unstable Binder tails where M2 is very small
        if "M2" in df.columns:
            stable = df["M2"].values > 1e-10
            T = T[stable]
            U = U[stable]

        if len(T) < 4:
            continue

        # Check Tc is within the data range
        if tc < T.min() or tc > T.max():
            continue

        try:
            f_U = interp1d(T, U, kind="cubic")
        except ValueError:
            continue

        # Numerical derivative at Tc via central difference
        dT = (T.max() - T.min()) / len(T)  # ~spacing
        h = max(dT * 0.5, 1e-5)
        # Ensure we don't go out of bounds
        if tc - h < T.min() or tc + h > T.max():
            h = min(tc - T.min(), T.max() - tc) * 0.9
        if h < 1e-6:
            continue

        dUdT = (f_U(tc + h) - f_U(tc - h)) / (2 * h)
        slopes_out.append((n, dUdT))

        if abs(dUdT) > 1e-10:
            log_L.append(np.log(n))
            log_dUdT.append(np.log(abs(dUdT)))

    if len(log_L) < 2:
        return np.nan, [], slopes_out

    log_L = np.array(log_L)
    log_dUdT = np.array(log_dUdT)
    inv_nu, _ = np.polyfit(log_L, log_dUdT, 1)
    nu = 1.0 / inv_nu if abs(inv_nu) > 1e-10 else np.nan

    return nu, [s[0] for s in slopes_out], slopes_out


def analyze_model(model: str, dfs: dict[int, pd.DataFrame], out_dir: Path,
                   data_dir: Path, method: str = "wham",
                   fit_sizes: list[int] | None = None,
                   collapse_sizes: list[int] | None = None):
    """Full FSS analysis for one model.

    Parameters
    ----------
    fit_sizes : list[int] or None
        Sizes to use for exponent fitting. If None, use all available sizes.
    collapse_sizes : list[int] or None
        Sizes to use for scaling collapse. If None, same as fit_sizes.
    """
    th = THEORY[model]
    sizes = sorted(dfs.keys())
    colors = cm.viridis(np.linspace(0.15, 0.85, len(sizes)))

    # Determine fit and collapse size sets
    if fit_sizes is None:
        fit_sizes_actual = sizes
    else:
        fit_sizes_actual = sorted(n for n in fit_sizes if n in dfs)
    if collapse_sizes is None:
        collapse_sizes_actual = fit_sizes_actual
    else:
        collapse_sizes_actual = sorted(n for n in collapse_sizes if n in dfs)

    print(f"\n{'=' * 60}")
    print(f"  {model.upper()} — 3D FSS Analysis")
    print(f"{'=' * 60}")
    print(f"Sizes (all): {sizes}")
    if fit_sizes_actual != sizes:
        print(f"Sizes (fit): {fit_sizes_actual}")
    if collapse_sizes_actual != fit_sizes_actual:
        print(f"Sizes (collapse): {collapse_sizes_actual}")
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

    # ── 3. Histogram reweighting → fine T grid ──────────────────────────────
    print(f"\n  Histogram reweighting ({method})...")
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
        n_samples = sum(len(e) for e in e_lists)
        print(f"    N={n}: {method} reweighting ({n_samples} samples, "
              f"{len(e_lists)} replicas)...", end=" ", flush=True)
        if method == "wham":
            rw = wham_reweight_fine(e_lists, m_lists, beta_list, T_fine, n)
        else:
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
    # Use fit_sizes for exponent extraction, but collect data for all sizes for plots
    all_rw_sizes = sorted(peak_src.keys())
    fit_rw_sizes = sorted(n for n in fit_sizes_actual if n in peak_src)
    if len(fit_rw_sizes) < 2:
        fit_rw_sizes = all_rw_sizes  # fallback

    # Collect chi peaks for ALL sizes (for plotting)
    all_chi_peaks = {}
    all_chi_peak_errs = {}
    for n in all_rw_sizes:
        df = peak_src[n]
        idx = df["chi_conn"].idxmax()
        all_chi_peaks[n] = df.loc[idx, "chi_conn"]
        all_chi_peak_errs[n] = df.loc[idx, "chi_err"] if "chi_err" in df.columns else 0.0

    # Fit on fit_rw_sizes only
    fit_log_n = np.log(fit_rw_sizes)
    fit_chi_peaks = [all_chi_peaks[n] for n in fit_rw_sizes]
    slope_chi, _ = np.polyfit(fit_log_n, np.log(fit_chi_peaks), 1)
    gamma_nu = slope_chi
    th_gamma_nu = th["gamma"] / th["nu"]

    # M(Tc) for all sizes (for plotting)
    all_m_at_tc = {}
    all_m_at_tc_errs = {}
    for n in all_rw_sizes:
        df = peak_src[n]
        try:
            f_m = interp1d(df["T"], df["M"], kind="cubic")
            all_m_at_tc[n] = float(f_m(tc_meas))
            if "M_err" in df.columns:
                f_e = interp1d(df["T"], df["M_err"], kind="linear")
                all_m_at_tc_errs[n] = float(f_e(tc_meas))
            else:
                all_m_at_tc_errs[n] = 0.0
        except (ValueError, KeyError):
            all_m_at_tc[n] = np.nan
            all_m_at_tc_errs[n] = 0.0

    # Fit beta/nu on fit_rw_sizes only
    valid = [(n, all_m_at_tc[n]) for n in fit_rw_sizes
             if not np.isnan(all_m_at_tc.get(n, np.nan)) and all_m_at_tc.get(n, 0) > 0]
    if len(valid) >= 2:
        vn, vm = zip(*valid)
        slope_m, _ = np.polyfit(np.log(vn), np.log(vm), 1)
        beta_nu = -slope_m
    else:
        beta_nu = np.nan
    th_beta_nu = th["beta"] / th["nu"]

    fit_label = f"{src_label}, fit L={fit_rw_sizes}" if fit_rw_sizes != all_rw_sizes else src_label
    print(f"\n  Peak scaling ({fit_label}):")
    print(f"    gamma/nu = {gamma_nu:.4f}  (theory = {th_gamma_nu:.4f}, error = {abs(gamma_nu - th_gamma_nu) / th_gamma_nu * 100:.1f}%)")
    if not np.isnan(beta_nu):
        print(f"    beta/nu  = {beta_nu:.4f}  (theory = {th_beta_nu:.4f}, error = {abs(beta_nu - th_beta_nu) / th_beta_nu * 100:.1f}%)")
        hyperscaling = 2 * beta_nu + gamma_nu
        print(f"    2*beta/nu + gamma/nu = {hyperscaling:.4f}  (should = 3, error = {abs(hyperscaling - 3) / 3 * 100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    n_fit_line = np.linspace(min(all_rw_sizes) * 0.9, max(all_rw_sizes) * 1.1, 100)

    ax = axes[0]
    # Plot all sizes; highlight fit sizes
    for n in all_rw_sizes:
        is_fit = n in fit_rw_sizes
        ax.errorbar(n, all_chi_peaks[n], yerr=all_chi_peak_errs[n],
                     fmt="o" if is_fit else "s", ms=5 if is_fit else 4,
                     color="C0" if is_fit else "C7", capsize=3, zorder=5 if is_fit else 3,
                     alpha=1.0 if is_fit else 0.4)
    c0 = np.log(fit_chi_peaks[0]) - gamma_nu * np.log(fit_rw_sizes[0])
    ax.plot(n_fit_line, np.exp(gamma_nu * np.log(n_fit_line) + c0), "--", color="C1", lw=0.8,
            label=rf"$\gamma/\nu={gamma_nu:.3f}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$L$"); ax.set_ylabel(r"$\chi_{\mathrm{max}}$")
    ax.legend(fontsize=8)

    ax = axes[1]
    for n in all_rw_sizes:
        m = all_m_at_tc.get(n, np.nan)
        if np.isnan(m) or m <= 0:
            continue
        is_fit = n in fit_rw_sizes
        ax.errorbar(n, m, yerr=all_m_at_tc_errs.get(n, 0),
                     fmt="o" if is_fit else "s", ms=5 if is_fit else 4,
                     color="C0" if is_fit else "C7", capsize=3, zorder=5 if is_fit else 3,
                     alpha=1.0 if is_fit else 0.4)
    if not np.isnan(beta_nu) and len(valid) >= 2:
        c0m = np.log(valid[0][1]) + beta_nu * np.log(valid[0][0])
        ax.plot(n_fit_line, np.exp(-beta_nu * np.log(n_fit_line) + c0m), "--", color="C1", lw=0.8,
                label=rf"$\beta/\nu={beta_nu:.3f}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$L$"); ax.set_ylabel(r"$M(T_c)$")
    ax.legend(fontsize=8)

    fig.suptitle(f"{model.upper()} — Peak Scaling ({fit_label})", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / f"{model}_peak_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {model}_peak_scaling.png")

    # ── 5. Scaling collapse → nu ───────────────────────────────────────────
    # Use collapse_sizes for the fit, but plot all sizes
    collapse_src = {n: peak_src[n] for n in collapse_sizes_actual if n in peak_src}
    if len(collapse_src) < 2:
        collapse_src = peak_src  # fallback

    result = minimize_scalar(
        collapse_quality, bounds=(0.3, 1.2), method="bounded",
        args=(collapse_src, tc_meas, gamma_nu),
    )
    nu_collapse = result.x
    collapse_label = f"L={sorted(collapse_src.keys())}" if sorted(collapse_src.keys()) != all_rw_sizes else "all"
    print(f"\n  Scaling collapse ({collapse_label}):")
    print(f"    nu (chi collapse) = {nu_collapse:.4f}  (theory = {th['nu']:.4f}, error = {abs(nu_collapse - th['nu']) / th['nu'] * 100:.1f}%)")

    # ── 5b. Binder-slope nu ──────────────────────────────────────────────
    nu_binder, binder_sizes_used, binder_slopes = binder_slope_nu(
        peak_src, tc_meas, fit_sizes=fit_rw_sizes,
    )
    if not np.isnan(nu_binder):
        print(f"    nu (Binder slope) = {nu_binder:.4f}  (theory = {th['nu']:.4f}, error = {abs(nu_binder - th['nu']) / th['nu'] * 100:.1f}%)")
        print(f"      dU/dT|Tc: ", end="")
        for n, s in binder_slopes:
            print(f"L={n}: {s:.2f}  ", end="")
        print()
    else:
        print(f"    nu (Binder slope) = N/A (insufficient data)")

    # Choose best nu: prefer Binder slope if reasonable, else collapse
    if not np.isnan(nu_binder) and 0.3 < nu_binder < 1.5:
        nu_best = nu_binder
        nu_method = "Binder slope"
    else:
        nu_best = nu_collapse
        nu_method = "collapse"

    fig, ax = plt.subplots(figsize=(5, 3.5))
    # Plot all sizes for visual context
    for i, n in enumerate(sorted(peak_src.keys())):
        df = peak_src[n]
        x = (df["T"] - tc_meas) * n ** (1.0 / nu_collapse)
        y = df["chi_conn"] / n ** gamma_nu
        mask = (x > -15) & (x < 15)
        in_collapse = n in collapse_src
        ax.plot(x[mask], y[mask], "o-", color=colors[i], markersize=2, lw=0.6,
                label=f"L={n}", alpha=1.0 if in_collapse else 0.3)
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
    print(f"\n  {'Quantity':<28} {'Measured':>10} {'Theory':>10} {'Error':>8} {'Sizes'}")
    print(f"  {'-' * 75}")
    table_rows = [
        ("Tc (Binder)", tc_meas, th["Tc"], "all"),
        ("gamma/nu (peak)", gamma_nu, th_gamma_nu, str(fit_rw_sizes)),
        ("nu (collapse)", nu_collapse, th["nu"], str(sorted(collapse_src.keys()))),
    ]
    if not np.isnan(nu_binder):
        table_rows.append(("nu (Binder slope)", nu_binder, th["nu"], str(binder_sizes_used)))
    if not np.isnan(beta_nu):
        table_rows.append(("beta/nu (M@Tc)", beta_nu, th_beta_nu, str(fit_rw_sizes)))
        table_rows.append(("2b/n + g/n", 2 * beta_nu + gamma_nu, 3.0, ""))

    for name, meas, theory, sz in table_rows:
        err = abs(meas - theory) / abs(theory) * 100
        status = "OK" if err < 5 else "~" if err < 15 else "!"
        print(f"  {name:<28} {meas:>10.4f} {theory:>10.4f} {err:>7.1f}%  {status}  {sz}")

    return {
        "model": model,
        "Tc": tc_meas, "Tc_err": tc_err,
        "gamma_nu": gamma_nu,
        "beta_nu": beta_nu,
        "nu_collapse": nu_collapse,
        "nu_binder": nu_binder,
        "nu_best": nu_best,
        "nu_method": nu_method,
        "fit_sizes": fit_rw_sizes,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze GPU FSS publication data")
    parser.add_argument("--data-dir", default="analysis/data/gpu_windows_pipeline/publication",
                        help="Directory containing gpu_fss_*_summary.csv files")
    parser.add_argument("--out-dir", default="analysis/figures/gpu_fss",
                        help="Output directory for figures")
    parser.add_argument("--model", default=None, choices=["ising", "heisenberg", "xy"],
                        help="Analyze only one model (default: all)")
    parser.add_argument("--method", default="single", choices=["wham", "single"],
                        help="Reweighting method: wham (multi-histogram) or single (default: single)")
    parser.add_argument("--fit-sizes", default=None, type=str,
                        help="Comma-separated sizes for exponent fitting (e.g., 32,64,128,192). Default: all.")
    parser.add_argument("--collapse-sizes", default=None, type=str,
                        help="Comma-separated sizes for scaling collapse (e.g., 32,64,128). Default: same as --fit-sizes.")
    args = parser.parse_args()

    # Parse size lists
    fit_sizes = [int(x) for x in args.fit_sizes.split(",")] if args.fit_sizes else None
    collapse_sizes = [int(x) for x in args.collapse_sizes.split(",")] if args.collapse_sizes else None

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
        res = analyze_model(model, dfs, out_dir, data_dir, method=args.method,
                            fit_sizes=fit_sizes, collapse_sizes=collapse_sizes)
        results.append(res)

    # ── Cross-model comparison ─────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print(f"  CROSS-MODEL COMPARISON")
        print(f"{'=' * 60}")
        print(f"  {'Model':<14} {'Tc':>8} {'gamma/nu':>10} {'beta/nu':>10} {'nu(col)':>8} {'nu(Bnd)':>8} {'nu*':>8} {'method':>10}")
        print(f"  {'-' * 82}")
        for r in results:
            bn = f"{r['beta_nu']:.4f}" if not np.isnan(r["beta_nu"]) else "N/A"
            nc = f"{r['nu_collapse']:.4f}"
            nb = f"{r['nu_binder']:.4f}" if not np.isnan(r["nu_binder"]) else "N/A"
            nu_best = f"{r['nu_best']:.4f}"
            print(f"  {r['model']:<14} {r['Tc']:>8.4f} {r['gamma_nu']:>10.4f} {bn:>10} {nc:>8} {nb:>8} {nu_best:>8} {r['nu_method']:>10}")
        print(f"\n  Theory:")
        for model in models:
            if model in THEORY:
                th = THEORY[model]
                print(f"  {model:<14} {th['Tc']:>8.4f} {th['gamma']/th['nu']:>10.4f} {th['beta']/th['nu']:>10.4f} {th['nu']:>8.4f}")

    print(f"\nFigures saved to: {out_dir}")


if __name__ == "__main__":
    main()
