#!/usr/bin/env python3
"""
Generate BCC crystal structure (8 nearest neighbours per node).
2*n^3 nodes total (corner + body-centre atoms per unit cell).

Usage:
  python gen_bcc.py --n 10 --out bcc_N10.json
"""
import argparse
import json

def bcc_edges(n):
    n3 = n ** 3

    def corner_idx(i, j, k):
        return ((i % n) * n + (j % n)) * n + (k % n)

    def body_idx(i, j, k):
        return n3 + ((i % n) * n + (j % n)) * n + (k % n)

    edges = set()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                b = body_idx(i, j, k)
                for di in [0, 1]:
                    for dj in [0, 1]:
                        for dk in [0, 1]:
                            c = corner_idx(i + di - 1, j + dj - 1, k + dk - 1)
                            edges.add((min(b, c), max(b, c)))
    return list(edges)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--out', default='bcc.json')
    args = p.parse_args()

    edges = bcc_edges(args.n)
    n_nodes = 2 * args.n ** 3
    data = {'n_nodes': n_nodes, 'edges': edges}
    with open(args.out, 'w') as f:
        json.dump(data, f)
    print(f"BCC: n={args.n}, {n_nodes} nodes, {len(edges)} edges -> {args.out}")

if __name__ == '__main__':
    main()
