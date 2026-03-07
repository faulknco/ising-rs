#!/usr/bin/env python3
"""
Generate FCC crystal structure (12 nearest neighbours per node).

FCC unit cell: 4 atoms per cell in Cartesian coords (units of a/2).
  - Corner:   (2i,   2j,   2k  )
  - xy-face:  (2i+1, 2j+1, 2k  )
  - xz-face:  (2i+1, 2j,   2k+1)
  - yz-face:  (2i,   2j+1, 2k+1)

12 NN offsets: all permutations of (±1, ±1, 0).
With n^3 unit cells and periodic boundary: 4*n^3 nodes total.

Usage:
  python gen_fcc.py --n 8 --out fcc_N8.json
  python gen_fcc.py --n 10 --out fcc_N10.json
"""
import argparse
import json

# 12 nearest-neighbour offsets in units of a/2
NN_OFFSETS = [
    (s1, s2, 0) for s1 in (1,-1) for s2 in (1,-1)
] + [
    (s1, 0, s2) for s1 in (1,-1) for s2 in (1,-1)
] + [
    (0, s1, s2) for s1 in (1,-1) for s2 in (1,-1)
]  # 12 offsets


def fcc_edges(n):
    """Build FCC adjacency list for n^3 unit cells with PBC."""
    L = 2 * n  # grid side in half-lattice units

    # Build position -> index map
    pos_to_idx = {}
    positions = []

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for atom in [
                    (2*i,   2*j,   2*k  ),   # corner
                    (2*i+1, 2*j+1, 2*k  ),   # xy-face
                    (2*i+1, 2*j,   2*k+1),   # xz-face
                    (2*i,   2*j+1, 2*k+1),   # yz-face
                ]:
                    pos_to_idx[atom] = len(positions)
                    positions.append(atom)

    edges = set()
    for pos, idx_a in pos_to_idx.items():
        x, y, z = pos
        for dx, dy, dz in NN_OFFSETS:
            nb = ((x + dx) % L, (y + dy) % L, (z + dz) % L)
            if nb in pos_to_idx:
                idx_b = pos_to_idx[nb]
                edges.add((min(idx_a, idx_b), max(idx_a, idx_b)))

    return list(edges)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=8,
                   help='unit cells per side (total nodes = 4*n^3)')
    p.add_argument('--out', default='fcc.json')
    args = p.parse_args()

    edges = fcc_edges(args.n)
    n_nodes = 4 * args.n ** 3
    mean_deg = 2 * len(edges) / n_nodes

    data = {'n_nodes': n_nodes, 'edges': edges}
    with open(args.out, 'w') as f:
        json.dump(data, f)

    print(f"FCC: n={args.n}, {n_nodes} nodes, {len(edges)} edges, "
          f"mean degree={mean_deg:.1f} (expect 12) -> {args.out}")


if __name__ == '__main__':
    main()
