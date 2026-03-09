#!/usr/bin/env python3
"""
Generate a diluted cubic lattice by randomly removing a fraction of cubic bonds.

Usage:
  python gen_diluted.py --n 20 --p 0.1 --seed 42 --out diluted_N20_p10_r000.json
"""
import argparse
import json
import random


def cubic_edges(n):
    edges = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                idx = i * n * n + j * n + k
                edges.append((idx, ((i + 1) % n) * n * n + j * n + k))
                edges.append((idx, i * n * n + ((j + 1) % n) * n + k))
                edges.append((idx, i * n * n + j * n + (k + 1) % n))
    return edges


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--p", type=float, default=0.1, help="fraction of bonds to remove")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--realization",
        type=int,
        default=0,
        help="realization index recorded in the output metadata",
    )
    p.add_argument("--out", default="diluted.json")
    args = p.parse_args()

    if not 0.0 <= args.p <= 1.0:
        raise SystemExit("--p must lie in [0, 1]")

    random.seed(args.seed)
    edges = cubic_edges(args.n)
    n_remove = int(len(edges) * args.p)
    remove_idx = set(random.sample(range(len(edges)), n_remove))
    kept = [e for i, e in enumerate(edges) if i not in remove_idx]

    data = {
        "n_nodes": args.n**3,
        "edges": kept,
        "metadata": {
            "generator": "analysis/graphs/gen_diluted.py",
            "geometry": "cubic3d_bond_diluted",
            "linear_size": args.n,
            "removed_fraction": args.p,
            "kept_fraction": 1.0 - args.p,
            "seed": args.seed,
            "realization": args.realization,
            "n_edges_original": len(edges),
            "n_edges_kept": len(kept),
            "mean_degree": 2.0 * len(kept) / (args.n**3),
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    print(
        f"N={args.n}^3={args.n**3} nodes, {len(edges)} original bonds, "
        f"{len(kept)} kept ({args.p*100:.0f}% removed), realization={args.realization} "
        f"-> {args.out}"
    )


if __name__ == "__main__":
    main()
