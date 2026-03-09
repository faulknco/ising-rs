#!/usr/bin/env python3
"""Rebuild Ising FSS observable tables with jackknife error bars from raw data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ising_stats import compute_observables_from_raw, infer_linear_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Ising FSS observables with jackknife errors from fss_raw_N*.csv files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("analysis/data/hires"),
        help="Directory containing raw FSS time-series CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/data/derived/fss"),
        help="Directory for derived observable tables.",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=20,
        help="Number of jackknife blocks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.input_dir.glob("fss_raw_N*.csv"))
    if not paths:
        raise SystemExit(f"no raw FSS files found in {args.input_dir}")

    summary_rows = []
    for path in paths:
        linear_size = infer_linear_size(path)
        raw_df = pd.read_csv(path)
        obs_df = compute_observables_from_raw(raw_df, linear_size=linear_size, n_blocks=args.n_blocks)

        out_path = args.output_dir / f"fss_obs_N{linear_size}.csv"
        obs_df.to_csv(out_path, index=False)
        summary_rows.append(
            {
                "linear_size": linear_size,
                "input_file": str(path),
                "output_file": str(out_path),
                "n_temperatures": len(obs_df),
                "n_blocks": args.n_blocks,
            }
        )
        print(f"wrote {out_path}")

    pd.DataFrame(summary_rows).sort_values("linear_size").to_csv(
        args.output_dir / "manifest_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
