# Mesh Geometry Feature — Design Document
Date: 2026-03-06
Status: PLANNED (not yet implemented)

---

## Motivation

The current simulation supports three regular lattice geometries:
  - Square2D     (4 neighbours per spin, periodic boundary)
  - Triangular2D (6 neighbours per spin, periodic boundary)
  - Cubic3D      (6 neighbours per spin, periodic boundary)

These are ideal for studying universal properties (critical exponents, universality
classes) because they are analytically tractable and well-studied in the literature.

However, real physical systems have irregular connectivity:
  - BCC/FCC crystal structures have 8 or 12 nearest neighbours
  - Amorphous magnets have a distribution of coordination numbers
  - Biological spin models (Hopfield networks, protein contacts) have arbitrary topology
  - Diluted magnets (random bond removal) are between regular and random graph

The architecture already supports this — adding mesh geometry is a small, contained change.

---

## Why it is Easy

src/lattice.rs stores connectivity as:

    pub neighbours: Vec<Vec<usize>>

This is a generic adjacency list. The simulation engine (metropolis.rs, wolff.rs,
observables.rs, coarsening.rs, fss.rs) never assumes a regular grid. Every algorithm
only ever does:

    for &nb in &lattice.neighbours[idx] { ... }

The three existing geometries are just different ways to fill neighbours[] at
construction time. A mesh is a fourth way — load from file instead of computing
from coordinates.

No changes needed to:
  - metropolis.rs
  - wolff.rs
  - observables.rs
  - fss.rs
  - coarsening.rs
  - wasm.rs
  - Any analysis notebooks

---

## Design

### New Geometry Variant

In src/lattice.rs:

    pub enum Geometry {
        Square2D,
        Triangular2D,
        Cubic3D,
        Mesh,           // <-- new
    }

### New Constructor

    impl Lattice {
        /// Load a lattice from an arbitrary edge list.
        /// edges: list of (i, j) pairs, 0-indexed, undirected.
        /// n_nodes: total number of spins.
        pub fn from_edges(n_nodes: usize, edges: &[(usize, usize)]) -> Self {
            let spins = vec![1i8; n_nodes];
            let mut neighbours = vec![Vec::new(); n_nodes];
            for &(i, j) in edges {
                neighbours[i].push(j);
                neighbours[j].push(i);   // undirected
            }
            Self {
                n: n_nodes,   // n = total nodes (not a side length)
                spins,
                neighbours,
                geometry: Geometry::Mesh,
            }
        }
    }

Note: for Mesh geometry, n means total node count, not side length.
This is already fine — size() returns spins.len() which is correct.

### File Format

Two options, both simple to parse:

Option A — Edge list CSV (simplest):
    # comment lines ignored
    0,1
    0,2
    1,3
    ...

Option B — JSON adjacency list:
    {
      "n_nodes": 1000,
      "edges": [[0,1],[0,2],[1,3],...]
    }

Recommendation: support both. CSV for hand-crafted graphs, JSON for
programmatically generated ones (e.g. from Python NetworkX).

### New src/graph.rs Module

    pub struct GraphDef {
        pub n_nodes: usize,
        pub edges: Vec<(usize, usize)>,
    }

    impl GraphDef {
        pub fn from_edge_csv(path: &str) -> Result<Self, ...>
        pub fn from_json(path: &str) -> Result<Self, ...>
        pub fn into_lattice(self) -> Lattice {
            Lattice::from_edges(self.n_nodes, &self.edges)
        }
    }

### CLI Integration

In src/bin/sweep.rs and src/bin/fss.rs, add:

    --graph path/to/graph.json   Load mesh from file (overrides --geometry and --n)

Example:
    cargo run --release --bin sweep -- --graph graphs/bcc_iron.json --j 1.0

### Graph Generation Scripts

In analysis/graphs/ directory (Python, not in critical path):

    gen_bcc.py       — BCC crystal structure, N³ unit cells, periodic boundary
    gen_fcc.py       — FCC crystal structure
    gen_random.py    — Erdos-Renyi random graph, given N and mean degree
    gen_diluted.py   — Take cubic lattice, remove fraction p of bonds randomly
    gen_smallworld.py— Watts-Strogatz small-world network

Output: JSON edge list files that can be passed to --graph flag.

---

## Physical Applications

### 1. Real Crystal Structures
BCC iron: 8 nearest neighbours per spin (body-centred cubic)
  Expected Tc higher than simple cubic — more bonds = stronger ordering
  Can compare simulated Tc to experiment: Tc(Fe) = 1043 K

FCC nickel: 12 nearest neighbours
  Tc(Ni) = 627 K (experiment)
  With J fitted to experiment, simulation should reproduce this

### 2. Diluted Magnets
Start with cubic lattice, randomly remove fraction p of bonds.
At critical dilution pc (percolation threshold), long-range order vanishes.
For 3D cubic: pc ≈ 0.75 (bond percolation threshold)
  p < pc: Ising universality class preserved (Harris criterion: ν > 2/d)
  p > pc: no ferromagnetic order possible

### 3. Random Graphs (Mean-Field regime)
Erdos-Renyi graph with mean degree k:
  For large k, approaches mean-field Ising (Bethe lattice)
  Mean-field exponents: β=0.5, γ=1.0, α=0 (above upper critical dimension)
  Interesting crossover from 3D to mean-field as k increases

### 4. Small-World Networks
Watts-Strogatz: start with ring lattice, rewire fraction p of edges randomly
  p=0: 1D ring (no long-range order)
  p=1: random graph (mean-field)
  p intermediate: small-world — high clustering + short path length
  Tc appears at surprisingly small p — small-world topology strongly promotes ordering

---

## What Changes vs Current Code

| Component         | Change needed | Notes |
|-------------------|---------------|-------|
| src/lattice.rs    | Add Mesh variant, from_edges() | ~20 lines |
| src/graph.rs      | New file — CSV/JSON loader | ~60 lines |
| src/bin/sweep.rs  | Add --graph flag | ~10 lines |
| src/bin/fss.rs    | Add --graph flag | ~10 lines |
| src/bin/coarsening.rs | Add --graph flag | ~10 lines |
| analysis/graphs/  | Python graph generators | Optional |
| src/cuda/kernels.cu | NOT needed — GPU kernel hardcoded 3D cubic | Separate concern |
| Everything else   | No changes | Physics engine is geometry-agnostic |

Total: ~100 lines of new Rust code, all contained in lattice.rs and graph.rs.

---

## Limitations of Mesh Approach

1. FSS is harder — no clean N³ scaling, must use total node count
   For irregular graphs, "system size" is not a single number
   Alternative: use mean coordination number + node count together

2. Wolff algorithm requires uniform J — works for mesh if all bonds have same J
   For bond-disordered systems (varying J per edge), must use Metropolis only
   Extension: store J_ij per edge in GraphDef for disorder studies

3. CUDA checkerboard decomposition does not apply to irregular graphs
   Graph colouring needed (NP-hard in general, but polynomial for sparse graphs)
   For GPU + mesh: use graph colouring preprocessing, then colour-by-colour updates
   This is a significant extension — out of scope for initial implementation

4. Periodic boundary conditions do not apply to finite graphs
   For crystal structures: either use open boundaries (surface effects) or
   construct a supercell with periodic images

5. Binder cumulant crossing to find Tc still works — N is just node count
   Peak scaling (Cv_max ~ N^(alpha/nu)) still works if N is varied by scaling the graph

---

## Implementation Order (when ready to build)

Task 1: Add Geometry::Mesh variant and Lattice::from_edges() to lattice.rs
Task 2: Create src/graph.rs with CSV + JSON loaders
Task 3: Add --graph flag to src/bin/sweep.rs (simplest binary, good test)
Task 4: Write analysis/graphs/gen_diluted.py — diluted cubic, easiest physics test
Task 5: Run sweep on diluted cubic at p=0.1, 0.3, 0.5 — verify Tc decreases with p
Task 6: Add --graph flag to src/bin/fss.rs and src/bin/coarsening.rs
Task 7: gen_bcc.py + compare simulated Tc to iron experiment
Task 8: FSS on random graphs — observe mean-field exponents at large k

---

## References

- Harris criterion: A.B. Harris, J. Phys. C 7, 1671 (1974)
  Disorder is irrelevant if ν > 2/d — 3D Ising is marginally stable to dilution
- Watts & Strogatz, Nature 393, 440 (1998) — small-world networks
- Dorogovtsev, Goltsev & Mendes, Rev. Mod. Phys. 80, 1275 (2008)
  Ising model on complex networks — comprehensive review
- Newman & Barkema, Monte Carlo Methods in Statistical Physics (1999)
  Chapter 10: disordered systems
