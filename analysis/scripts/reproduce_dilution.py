#!/usr/bin/env python3
"""Generate and analyse multi-realization bond-dilution sweeps."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import resolve_python


REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_DILUTED = REPO_ROOT / "analysis" / "graphs" / "gen_diluted.py"
CSV_WRITE_LOCK = threading.Lock()


@dataclass
class SweepJob:
    n: int
    p_removed: float
    realization: int
    graph_path: Path
    csv_path: Path
    seed: int
    mean_degree: float
    n_edges: int
    tmin: float
    tmax: float
    steps: int
    warmup: int
    samples: int


@dataclass(frozen=True)
class PathsConfig:
    raw_root: Path
    derived_root: Path
    figure_root: Path
    manifest_root: Path


def default_max_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, 8))


def run_command(cmd: list[str], cwd: Path = REPO_ROOT) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )


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
        description=(
            "Generate diluted cubic realizations, run mesh_sweep for each, and "
            "produce disorder-averaged observable and Tc summary tables."
        )
    )
    parser.add_argument("--n", type=int, default=20, help="Linear cubic size.")
    parser.add_argument(
        "--p-values",
        default="0.0,0.1,0.3,0.5",
        help="Comma-separated removed-bond fractions.",
    )
    parser.add_argument(
        "--realizations",
        type=int,
        default=8,
        help="Number of disorder realizations per dilution fraction.",
    )
    parser.add_argument("--tmin", type=float, default=2.0)
    parser.add_argument("--tmax", type=float, default=6.0)
    parser.add_argument("--steps", type=int, default=41)
    parser.add_argument("--warmup", type=int, default=3000)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--j", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Base directory for raw/derived/figure/manifest outputs. "
            "Defaults to the repository analysis layout."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use lighter defaults suitable for smoke tests.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip graph/sweep generation when the target file already exists.",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Do not generate graphs or run sweeps; derive outputs from existing raw files only.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=default_max_workers(),
        help="Maximum number of concurrent mesh_sweep jobs.",
    )
    parser.add_argument(
        "--t-window-spec",
        default="",
        help=(
            "Optional semicolon-separated temperature windows by dilution, "
            "for example '0.0:3.8:5.2;0.3:2.4:3.8'."
        ),
    )
    parser.add_argument(
        "--realizations-spec",
        default="",
        help=(
            "Optional semicolon-separated realization counts by dilution, "
            "for example '0.0:4;0.1:12'."
        ),
    )
    return parser.parse_args()


def parse_p_values(raw: str) -> list[float]:
    values = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = float(piece)
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"invalid dilution fraction {value}; expected 0 <= p <= 1")
        values.append(value)
    if not values:
        raise SystemExit("at least one dilution fraction is required")
    return values


def parse_realizations_spec(raw: str) -> dict[float, int]:
    if not raw.strip():
        return {}
    mapping = {}
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        try:
            p_raw, count_raw = item.split(":")
        except ValueError as exc:
            raise SystemExit(f"bad --realizations-spec entry '{item}'") from exc
        p_removed = float(p_raw)
        count = int(count_raw)
        if count <= 0:
            raise SystemExit(f"realization count must be positive in '{item}'")
        mapping[p_removed] = count
    return mapping


def parse_t_window_spec(raw: str) -> dict[float, tuple[float, float]]:
    if not raw.strip():
        return {}
    mapping = {}
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        try:
            p_raw, tmin_raw, tmax_raw = item.split(":")
        except ValueError as exc:
            raise SystemExit(f"bad --t-window-spec entry '{item}'") from exc
        p_removed = float(p_raw)
        tmin = float(tmin_raw)
        tmax = float(tmax_raw)
        if tmax <= tmin:
            raise SystemExit(f"invalid temperature window in '{item}'")
        mapping[p_removed] = (tmin, tmax)
    return mapping


def stderr(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def format_p_tag(p_removed: float) -> str:
    return f"p{int(round(p_removed * 1000)):03d}"


def resolve_paths(args: argparse.Namespace) -> PathsConfig:
    if args.output_root is None:
        base = REPO_ROOT / "analysis"
        return PathsConfig(
            raw_root=base / "data" / "raw" / "dilution",
            derived_root=base / "data" / "derived" / "dilution",
            figure_root=base / "figures" / "generated" / "dilution",
            manifest_root=base / "data" / "manifests" / "dilution",
        )

    base = args.output_root.expanduser().resolve()
    return PathsConfig(
        raw_root=base / "raw" / "dilution",
        derived_root=base / "derived" / "dilution",
        figure_root=base / "figures" / "dilution",
        manifest_root=base / "manifests" / "dilution",
    )


def resolve_mesh_sweep() -> Path:
    exe = REPO_ROOT / "target" / "release" / "mesh_sweep"
    if exe.exists():
        return exe
    exe_windows = exe.with_suffix(".exe")
    if exe_windows.exists():
        return exe_windows
    run_command(["cargo", "build", "--release", "--bin", "mesh_sweep"])
    if exe.exists():
        return exe
    if exe_windows.exists():
        return exe_windows
    raise RuntimeError("failed to build mesh_sweep binary")


def load_graph_metadata(path: Path) -> tuple[float, int]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    metadata = data.get("metadata", {})
    mean_degree = float(metadata.get("mean_degree", 2.0 * len(data["edges"]) / data["n_nodes"]))
    n_edges = int(metadata.get("n_edges_kept", len(data["edges"])))
    return mean_degree, n_edges


def append_csv_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with CSV_WRITE_LOCK:
        header_needed = not path.exists()
        df = pd.DataFrame([row])
        df.to_csv(path, mode="a", header=header_needed, index=False)


def generate_graph(
    python_exe: str,
    n: int,
    p_removed: float,
    realization: int,
    graph_path: Path,
    seed: int,
    skip_existing: bool,
) -> tuple[float, int]:
    if graph_path.exists() and skip_existing:
        return load_graph_metadata(graph_path)

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        str(GEN_DILUTED),
        "--n",
        str(n),
        "--p",
        str(p_removed),
        "--seed",
        str(seed),
        "--realization",
        str(realization),
        "--out",
        str(graph_path),
    ]
    run_command(cmd)
    return load_graph_metadata(graph_path)


def run_mesh_sweep(
    mesh_sweep: Path,
    job: SweepJob,
    args: argparse.Namespace,
) -> None:
    if job.csv_path.exists() and args.skip_existing:
        return

    job.csv_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = job.csv_path.stem.replace("_sweep", "")
    cmd = [
        str(mesh_sweep),
        "--graph",
        str(job.graph_path),
        "--j",
        str(args.j),
        "--tmin",
        str(job.tmin),
        "--tmax",
        str(job.tmax),
        "--steps",
        str(job.steps),
        "--warmup",
        str(job.warmup),
        "--samples",
        str(job.samples),
        "--seed",
        str(job.seed),
        "--prefix",
        prefix,
        "--outdir",
        str(job.csv_path.parent),
    ]
    run_command(cmd)


def stream_completed_job(job: SweepJob, paths: PathsConfig) -> None:
    job_row = {
        "n": job.n,
        "p_removed": job.p_removed,
        "realization": job.realization,
        "seed": job.seed,
        "tmin": job.tmin,
        "tmax": job.tmax,
        "steps": job.steps,
        "warmup": job.warmup,
        "samples": job.samples,
        "mean_degree": job.mean_degree,
        "n_edges": job.n_edges,
        "graph_file": str(job.graph_path),
        "sweep_file": str(job.csv_path),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    append_csv_row(paths.derived_root / "dilution_job_progress.csv", job_row)

    df = pd.read_csv(job.csv_path)
    tc, chi_peak, method = quadratic_peak_temperature(df)
    partial_row = {
        "n": job.n,
        "p_removed": job.p_removed,
        "realization": job.realization,
        "tc_chi_peak": tc,
        "chi_peak": chi_peak,
        "peak_method": method,
        "mean_degree": job.mean_degree,
        "n_edges": job.n_edges,
        "graph_file": str(job.graph_path),
        "sweep_file": str(job.csv_path),
    }
    append_csv_row(paths.derived_root / "dilution_tc_partial.csv", partial_row)


def run_mesh_sweeps_parallel(
    mesh_sweep: Path,
    jobs: list[SweepJob],
    args: argparse.Namespace,
    paths: PathsConfig,
) -> None:
    pending_jobs = [job for job in jobs if not (job.csv_path.exists() and args.skip_existing)]
    if not pending_jobs:
        return

    max_workers = min(args.max_workers, len(pending_jobs))
    if max_workers <= 1:
        for job in pending_jobs:
            run_mesh_sweep(mesh_sweep, job, args)
            stream_completed_job(job, paths)
        return

    print(f"running {len(pending_jobs)} dilution sweeps with max_workers={max_workers}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(run_mesh_sweep, mesh_sweep, job, args): job for job in pending_jobs
        }
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            try:
                future.result()
                stream_completed_job(job, paths)
            except Exception as exc:
                raise RuntimeError(
                    f"mesh_sweep failed for p={job.p_removed:.3f}, realization={job.realization}"
                ) from exc


def quadratic_peak_temperature(df: pd.DataFrame, column: str = "chi") -> tuple[float, float, str]:
    idx = int(df[column].idxmax())
    t_grid = float(df.loc[idx, "T"])
    y_grid = float(df.loc[idx, column])

    if idx == 0 or idx == len(df) - 1:
        return t_grid, y_grid, "grid_edge"

    local = df.iloc[idx - 1 : idx + 2]
    x = local["T"].to_numpy(dtype=float)
    y = local[column].to_numpy(dtype=float)
    a, b, c = np.polyfit(x, y, deg=2)
    if abs(a) < 1e-12 or a >= 0.0:
        return t_grid, y_grid, "grid_nonconcave"

    t_peak = -b / (2.0 * a)
    if t_peak < x.min() or t_peak > x.max():
        return t_grid, y_grid, "grid_outside_window"

    y_peak = a * t_peak**2 + b * t_peak + c
    return float(t_peak), float(y_peak), "quadratic"


def load_jobs(
    args: argparse.Namespace,
    p_values: list[float],
    paths: PathsConfig,
    t_windows: dict[float, tuple[float, float]],
    realizations_by_p: dict[float, int],
) -> list[SweepJob]:
    jobs = []
    for p_removed in p_values:
        n_realizations = realizations_by_p.get(p_removed, args.realizations)
        tmin, tmax = t_windows.get(p_removed, (args.tmin, args.tmax))
        p_tag = format_p_tag(p_removed)
        graph_dir = paths.raw_root / "graphs" / f"N{args.n}" / p_tag
        sweep_dir = paths.raw_root / "sweeps" / f"N{args.n}" / p_tag
        for realization in range(n_realizations):
            graph_path = graph_dir / f"diluted_N{args.n}_{p_tag}_r{realization:03d}.json"
            csv_path = sweep_dir / f"diluted_N{args.n}_{p_tag}_r{realization:03d}_sweep.csv"
            sweep_seed = args.seed + 1_000_000 + int(round(p_removed * 10_000)) + realization * 1009
            mean_degree, n_edges = load_graph_metadata(graph_path)
            jobs.append(
                SweepJob(
                    n=args.n,
                    p_removed=p_removed,
                    realization=realization,
                    graph_path=graph_path,
                    csv_path=csv_path,
                    seed=sweep_seed,
                    mean_degree=mean_degree,
                    n_edges=n_edges,
                    tmin=tmin,
                    tmax=tmax,
                    steps=args.steps,
                    warmup=args.warmup,
                    samples=args.samples,
                )
            )
    return jobs


def prepare_inputs(
    args: argparse.Namespace,
    p_values: list[float],
    paths: PathsConfig,
    t_windows: dict[float, tuple[float, float]],
    realizations_by_p: dict[float, int],
) -> list[SweepJob]:
    python_exe = resolve_python()
    mesh_sweep = resolve_mesh_sweep()
    jobs = []

    for p_removed in p_values:
        n_realizations = realizations_by_p.get(p_removed, args.realizations)
        tmin, tmax = t_windows.get(p_removed, (args.tmin, args.tmax))
        p_tag = format_p_tag(p_removed)
        graph_dir = paths.raw_root / "graphs" / f"N{args.n}" / p_tag
        sweep_dir = paths.raw_root / "sweeps" / f"N{args.n}" / p_tag
        for realization in range(n_realizations):
            graph_path = graph_dir / f"diluted_N{args.n}_{p_tag}_r{realization:03d}.json"
            csv_path = sweep_dir / f"diluted_N{args.n}_{p_tag}_r{realization:03d}_sweep.csv"
            graph_seed = args.seed + int(round(p_removed * 10_000)) + realization * 101
            sweep_seed = args.seed + 1_000_000 + int(round(p_removed * 10_000)) + realization * 1009
            mean_degree, n_edges = generate_graph(
                python_exe,
                args.n,
                p_removed,
                realization,
                graph_path,
                graph_seed,
                args.skip_existing,
            )
            job = SweepJob(
                n=args.n,
                p_removed=p_removed,
                realization=realization,
                graph_path=graph_path,
                csv_path=csv_path,
                seed=sweep_seed,
                mean_degree=mean_degree,
                n_edges=n_edges,
                tmin=tmin,
                tmax=tmax,
                steps=args.steps,
                warmup=args.warmup,
                samples=args.samples,
            )
            jobs.append(job)

    run_mesh_sweeps_parallel(mesh_sweep, jobs, args, paths)
    return jobs


def read_all_sweeps(jobs: list[SweepJob]) -> pd.DataFrame:
    frames = []
    for job in jobs:
        if not job.csv_path.exists():
            raise SystemExit(f"missing sweep CSV: {job.csv_path}")
        df = pd.read_csv(job.csv_path)
        df["p_removed"] = job.p_removed
        df["realization"] = job.realization
        df["graph_file"] = str(job.graph_path)
        df["sweep_file"] = str(job.csv_path)
        df["mean_degree"] = job.mean_degree
        df["n_edges"] = job.n_edges
        frames.append(df)
    if not frames:
        raise SystemExit("no dilution sweeps available")
    return pd.concat(frames, ignore_index=True)


def build_observable_summary(all_sweeps: pd.DataFrame) -> pd.DataFrame:
    agg = (
        all_sweeps.groupby(["p_removed", "T"], as_index=False)
        .agg(
            E=("E", "mean"),
            E_err=("E", stderr),
            M=("M", "mean"),
            M_err=("M", stderr),
            M2=("M2", "mean"),
            M2_err=("M2", stderr),
            M4=("M4", "mean"),
            M4_err=("M4", stderr),
            Cv=("Cv", "mean"),
            Cv_err=("Cv", stderr),
            chi=("chi", "mean"),
            chi_err=("chi", stderr),
            mean_degree=("mean_degree", "mean"),
            mean_degree_err=("mean_degree", stderr),
            n_realizations=("realization", "nunique"),
        )
        .sort_values(["p_removed", "T"])
    )
    return agg


def build_tc_tables(all_sweeps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (p_removed, realization), df in all_sweeps.groupby(["p_removed", "realization"], as_index=False):
        df = df.sort_values("T").reset_index(drop=True)
        tc, chi_peak, method = quadratic_peak_temperature(df)
        rows.append(
            {
                "p_removed": p_removed,
                "realization": realization,
                "tc_chi_peak": tc,
                "chi_peak": chi_peak,
                "peak_method": method,
                "mean_degree": float(df["mean_degree"].iloc[0]),
                "n_edges": int(df["n_edges"].iloc[0]),
                "graph_file": df["graph_file"].iloc[0],
                "sweep_file": df["sweep_file"].iloc[0],
            }
        )

    tc_realizations = pd.DataFrame(rows).sort_values(["p_removed", "realization"]).reset_index(drop=True)
    tc_summary = (
        tc_realizations.groupby("p_removed", as_index=False)
        .agg(
            tc_mean=("tc_chi_peak", "mean"),
            tc_err=("tc_chi_peak", stderr),
            chi_peak_mean=("chi_peak", "mean"),
            chi_peak_err=("chi_peak", stderr),
            mean_degree=("mean_degree", "mean"),
            mean_degree_err=("mean_degree", stderr),
            n_realizations=("realization", "nunique"),
        )
        .sort_values("p_removed")
        .reset_index(drop=True)
    )

    if (tc_summary["p_removed"] == 0.0).any():
        baseline = tc_summary.loc[tc_summary["p_removed"] == 0.0].iloc[0]
        tc0 = float(baseline["tc_mean"])
        tc0_err = float(baseline["tc_err"])
        ratios = []
        ratio_errs = []
        for _, row in tc_summary.iterrows():
            ratio = float(row["tc_mean"]) / tc0 if tc0 else math.nan
            if row["p_removed"] == 0.0:
                ratio_err = 0.0
            elif ratio and tc0 and row["tc_mean"] > 0.0:
                ratio_err = ratio * math.sqrt(
                    (float(row["tc_err"]) / float(row["tc_mean"])) ** 2 + (tc0_err / tc0) ** 2
                )
            else:
                ratio_err = math.nan
            ratios.append(ratio)
            ratio_errs.append(ratio_err)
        tc_summary["tc_over_tc0"] = ratios
        tc_summary["tc_over_tc0_err"] = ratio_errs

    return tc_realizations, tc_summary


def plot_outputs(
    observable_summary: pd.DataFrame,
    tc_summary: pd.DataFrame,
    paths: PathsConfig,
) -> list[Path]:
    paths.figure_root.mkdir(parents=True, exist_ok=True)
    out_paths = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for p_removed, df in observable_summary.groupby("p_removed"):
        label = f"p={p_removed:.3f}"
        axes[0].plot(df["T"], df["M"], marker="o", ms=3, label=label)
        axes[0].fill_between(df["T"], df["M"] - df["M_err"], df["M"] + df["M_err"], alpha=0.15)
        axes[1].plot(df["T"], df["chi"], marker="o", ms=3, label=label)
        axes[1].fill_between(
            df["T"],
            df["chi"] - df["chi_err"],
            df["chi"] + df["chi_err"],
            alpha=0.15,
        )

    axes[0].set_xlabel("T (J/k_B)")
    axes[0].set_ylabel("|M|")
    axes[0].set_title("Disorder-averaged magnetisation")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("T (J/k_B)")
    axes[1].set_ylabel("chi")
    axes[1].set_title("Disorder-averaged susceptibility")
    axes[1].grid(alpha=0.25)
    axes[1].legend(title="Removed-bond fraction", fontsize=8)

    obs_fig = paths.figure_root / "dilution_observables.png"
    fig.savefig(obs_fig, dpi=180)
    plt.close(fig)
    out_paths.append(obs_fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.errorbar(
        tc_summary["p_removed"],
        tc_summary["tc_mean"],
        yerr=tc_summary["tc_err"],
        fmt="o-",
        capsize=4,
        lw=1.2,
    )
    ax.set_xlabel("Removed-bond fraction p")
    ax.set_ylabel("Tc from susceptibility peak")
    ax.set_title("Dilution trend with disorder averaging")
    ax.grid(alpha=0.25)

    tc_fig = paths.figure_root / "dilution_tc_summary.png"
    fig.savefig(tc_fig, dpi=180)
    plt.close(fig)
    out_paths.append(tc_fig)
    return out_paths


def write_manifest(
    args: argparse.Namespace,
    jobs: list[SweepJob],
    paths: PathsConfig,
    observable_path: Path,
    realization_path: Path,
    tc_summary_path: Path,
    manifest_summary_path: Path,
    figure_paths: list[Path],
) -> Path:
    paths.manifest_root.mkdir(parents=True, exist_ok=True)
    run_id = f"dilution_n{args.n}_r{args.realizations}_seed{args.seed}"
    manifest = {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "binary": "mesh_sweep",
        "workflow": "analysis/scripts/reproduce_dilution.py",
        "model": "classical_ising",
        "geometry": "cubic3d_bond_diluted",
        "backend": "cpu_metropolis_mesh",
        "subcommand_args": {
            "n": args.n,
            "p_values": parse_p_values(args.p_values),
            "realizations": args.realizations,
            "max_workers": args.max_workers,
            "tmin": args.tmin,
            "tmax": args.tmax,
            "steps": args.steps,
            "warmup": args.warmup,
            "samples": args.samples,
            "j": args.j,
            "seed": args.seed,
            "output_root": str(args.output_root) if args.output_root else None,
            "t_window_spec": args.t_window_spec,
            "realizations_spec": args.realizations_spec,
        },
        "rng": {
            "graph_seed_base": args.seed,
            "sweep_seed_base": args.seed + 1_000_000,
        },
        "input_files": sorted(str(job.graph_path) for job in jobs),
        "output_files": {
            "raw_sweeps": sorted(str(job.csv_path) for job in jobs),
            "derived": [
                str(observable_path),
                str(realization_path),
                str(tc_summary_path),
                str(manifest_summary_path),
            ],
            "figures": [str(path) for path in figure_paths],
        },
        "environment": {
            "python": resolve_python(),
            "repo_root": str(REPO_ROOT),
        },
        "jobs": [asdict(job) | {"graph_path": str(job.graph_path), "csv_path": str(job.csv_path)} for job in jobs],
    }
    manifest_path = paths.manifest_root / f"{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def main() -> None:
    args = parse_args()
    if args.quick:
        args.realizations = min(args.realizations, 3)
        args.warmup = min(args.warmup, 500)
        args.samples = min(args.samples, 250)
        args.steps = min(args.steps, 21)

    if args.n <= 0:
        raise SystemExit("--n must be positive")
    if args.realizations <= 0:
        raise SystemExit("--realizations must be positive")
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive")
    if args.steps <= 1:
        raise SystemExit("--steps must be at least 2")
    if args.tmax <= args.tmin:
        raise SystemExit("--tmax must be greater than --tmin")

    p_values = parse_p_values(args.p_values)
    t_windows = parse_t_window_spec(args.t_window_spec)
    realizations_by_p = parse_realizations_spec(args.realizations_spec)
    paths = resolve_paths(args)

    if args.analysis_only:
        jobs = load_jobs(args, p_values, paths, t_windows, realizations_by_p)
    else:
        jobs = prepare_inputs(args, p_values, paths, t_windows, realizations_by_p)

    all_sweeps = read_all_sweeps(jobs)
    observable_summary = build_observable_summary(all_sweeps)
    tc_realizations, tc_summary = build_tc_tables(all_sweeps)

    paths.derived_root.mkdir(parents=True, exist_ok=True)
    observable_path = paths.derived_root / "dilution_observables_by_p.csv"
    realization_path = paths.derived_root / "dilution_tc_by_realization.csv"
    tc_summary_path = paths.derived_root / "dilution_tc_summary.csv"
    all_sweeps_path = paths.derived_root / "dilution_manifest_summary.csv"

    observable_summary.to_csv(observable_path, index=False)
    tc_realizations.to_csv(realization_path, index=False)
    tc_summary.to_csv(tc_summary_path, index=False)
    pd.DataFrame(
        [
            {
                "p_removed": job.p_removed,
                "n": job.n,
                "realization": job.realization,
                "graph_file": str(job.graph_path),
                "sweep_file": str(job.csv_path),
                "mean_degree": job.mean_degree,
                "n_edges": job.n_edges,
                "seed": job.seed,
                "tmin": job.tmin,
                "tmax": job.tmax,
                "steps": job.steps,
                "warmup": job.warmup,
                "samples": job.samples,
            }
            for job in jobs
        ]
    ).sort_values(["n", "p_removed", "realization"]).to_csv(all_sweeps_path, index=False)

    figure_paths = plot_outputs(observable_summary, tc_summary, paths)
    manifest_path = write_manifest(
        args,
        jobs,
        paths,
        observable_path,
        realization_path,
        tc_summary_path,
        all_sweeps_path,
        figure_paths,
    )

    print(f"wrote {observable_path}")
    print(f"wrote {realization_path}")
    print(f"wrote {tc_summary_path}")
    print(f"wrote {all_sweeps_path}")
    for path in figure_paths:
        print(f"wrote {path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
