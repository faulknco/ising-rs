#!/usr/bin/env python3
"""
Generate a diluted cubic lattice by randomly removing fraction p of bonds.

Usage:
  python gen_diluted.py --n 20 --p 0.1 --seed 42 --out diluted_N20_p10.json
"""
import argparse
import json
import random

def cubic_edges(n):
    edges = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                idx = i*n*n + j*n + k
                edges.append((idx, ((i+1)%n)*n*n + j*n + k))
                edges.append((idx, i*n*n + ((j+1)%n)*n + k))
                edges.append((idx, i*n*n + j*n + (k+1)%n))
    return edges

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--p', type=float, default=0.1, help='fraction of bonds to remove')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='diluted.json')
    args = p.parse_args()

    random.seed(args.seed)
    edges = cubic_edges(args.n)
    n_remove = int(len(edges) * args.p)
    remove_idx = set(random.sample(range(len(edges)), n_remove))
    kept = [e for i, e in enumerate(edges) if i not in remove_idx]

    data = {'n_nodes': args.n ** 3, 'edges': kept}
    with open(args.out, 'w') as f:
        json.dump(data, f)

    print(f"N={args.n}^3={args.n**3} nodes, {len(edges)} original bonds, "
          f"{len(kept)} kept ({args.p*100:.0f}% removed) -> {args.out}")

if __name__ == '__main__':
    main()
