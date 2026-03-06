/// Geometry of the lattice
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Geometry {
    Square2D,
    Triangular2D,
    Cubic3D,
    Mesh,
}

/// A flat spin lattice with precomputed neighbour indices.
///
/// Spins are stored as i8 in a flat Vec, indexed row-major:
///   2D: idx = i*N + j
///   3D: idx = i*N*N + j*N + k
///
/// Neighbours are precomputed once at construction — O(1) lookup per flip.
pub struct Lattice {
    pub n: usize,
    pub spins: Vec<i8>,
    /// neighbours[idx] = list of neighbour flat indices (4 or 6 entries)
    pub neighbours: Vec<Vec<usize>>,
    pub geometry: Geometry,
}

impl Lattice {
    pub fn new(n: usize, geometry: Geometry) -> Self {
        let size = match geometry {
            Geometry::Cubic3D => n * n * n,
            _ => n * n,
        };

        // Random initialisation: all +1 (ordered start, faster convergence at low T)
        let spins = vec![1i8; size];

        let neighbours = match geometry {
            Geometry::Square2D => Self::build_neighbours_2d_square(n),
            Geometry::Triangular2D => Self::build_neighbours_2d_triangular(n),
            Geometry::Cubic3D => Self::build_neighbours_3d_cubic(n),
            Geometry::Mesh => panic!("Use Lattice::from_edges() for Mesh geometry"),
        };

        Self { n, spins, neighbours, geometry }
    }

    /// Load a lattice from an arbitrary undirected edge list.
    /// n_nodes: total number of spins.
    /// edges: list of (i, j) pairs, 0-indexed.
    pub fn from_edges(n_nodes: usize, edges: &[(usize, usize)]) -> Self {
        let spins = vec![1i8; n_nodes];
        let mut neighbours = vec![Vec::new(); n_nodes];
        for &(i, j) in edges {
            assert!(i < n_nodes && j < n_nodes, "edge ({i},{j}) out of range for n_nodes={n_nodes}");
            neighbours[i].push(j);
            neighbours[j].push(i);
        }
        Self {
            n: n_nodes,
            spins,
            neighbours,
            geometry: Geometry::Mesh,
        }
    }

    /// Randomise spins uniformly ±1
    pub fn randomise(&mut self, rng: &mut impl rand::Rng) {
        for s in self.spins.iter_mut() {
            *s = if rng.gen_bool(0.5) { 1 } else { -1 };
        }
    }

    pub fn size(&self) -> usize {
        self.spins.len()
    }

    // ── Neighbour builders ────────────────────────────────────────────────────

    fn build_neighbours_2d_square(n: usize) -> Vec<Vec<usize>> {
        let size = n * n;
        let mut nb = vec![Vec::with_capacity(4); size];
        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                nb[idx] = vec![
                    ((i + n - 1) % n) * n + j, // up
                    ((i + 1) % n) * n + j,     // down
                    i * n + (j + n - 1) % n,   // left
                    i * n + (j + 1) % n,        // right
                ];
            }
        }
        nb
    }

    fn build_neighbours_2d_triangular(n: usize) -> Vec<Vec<usize>> {
        // Square grid with diagonal neighbours added based on row parity,
        // matching Andrew's construction (offset even rows to the right).
        let size = n * n;
        let mut nb = vec![Vec::with_capacity(6); size];
        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                let up = (i + n - 1) % n;
                let down = (i + 1) % n;
                let left = (j + n - 1) % n;
                let right = (j + 1) % n;

                let mut neighbours = vec![
                    up * n + j,
                    down * n + j,
                    i * n + left,
                    i * n + right,
                ];

                // Extra diagonal neighbours depend on row parity
                if i % 2 == 0 {
                    neighbours.push(down * n + left);
                    neighbours.push(up * n + left);
                } else {
                    neighbours.push(down * n + right);
                    neighbours.push(up * n + right);
                }

                nb[idx] = neighbours;
            }
        }
        nb
    }

    fn build_neighbours_3d_cubic(n: usize) -> Vec<Vec<usize>> {
        let size = n * n * n;
        let mut nb = vec![Vec::with_capacity(6); size];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let idx = i * n * n + j * n + k;
                    nb[idx] = vec![
                        ((i + n - 1) % n) * n * n + j * n + k, // -x
                        ((i + 1) % n) * n * n + j * n + k,     // +x
                        i * n * n + ((j + n - 1) % n) * n + k, // -y
                        i * n * n + ((j + 1) % n) * n + k,     // +y
                        i * n * n + j * n + (k + n - 1) % n,   // -z
                        i * n * n + j * n + (k + 1) % n,       // +z
                    ];
                }
            }
        }
        nb
    }
}
