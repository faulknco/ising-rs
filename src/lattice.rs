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

        Self {
            n,
            spins,
            neighbours,
            geometry,
        }
    }

    /// Load a lattice from an arbitrary undirected edge list.
    /// n_nodes: total number of spins.
    /// edges: list of (i, j) pairs, 0-indexed.
    pub fn from_edges(n_nodes: usize, edges: &[(usize, usize)]) -> Self {
        let spins = vec![1i8; n_nodes];
        let mut neighbours = vec![Vec::new(); n_nodes];
        for &(i, j) in edges {
            assert!(
                i < n_nodes && j < n_nodes,
                "edge ({i},{j}) out of range for n_nodes={n_nodes}"
            );
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
                    i * n + (j + 1) % n,       // right
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

                let mut neighbours = vec![up * n + j, down * n + j, i * n + left, i * n + right];

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_3d_size() {
        let lat = Lattice::new(4, Geometry::Cubic3D);
        assert_eq!(lat.size(), 64);
        assert_eq!(lat.spins.len(), 64);
    }

    #[test]
    fn square_2d_size() {
        let lat = Lattice::new(8, Geometry::Square2D);
        assert_eq!(lat.size(), 64);
    }

    #[test]
    fn cubic_3d_neighbour_count() {
        let lat = Lattice::new(6, Geometry::Cubic3D);
        for nb in &lat.neighbours {
            assert_eq!(nb.len(), 6, "3D cubic should have z=6 neighbours");
        }
    }

    #[test]
    fn square_2d_neighbour_count() {
        let lat = Lattice::new(8, Geometry::Square2D);
        for nb in &lat.neighbours {
            assert_eq!(nb.len(), 4, "2D square should have z=4 neighbours");
        }
    }

    #[test]
    fn triangular_2d_neighbour_count() {
        let lat = Lattice::new(8, Geometry::Triangular2D);
        for nb in &lat.neighbours {
            assert_eq!(nb.len(), 6, "2D triangular should have z=6 neighbours");
        }
    }

    #[test]
    fn periodic_boundary_2d() {
        // Corner spin (0,0) should wrap to (N-1,*) and (*,N-1)
        let n = 5;
        let lat = Lattice::new(n, Geometry::Square2D);
        let nb0 = &lat.neighbours[0]; // site (0,0)
        assert!(nb0.contains(&((n - 1) * n)), "should wrap up to row N-1");
        assert!(nb0.contains(&(n - 1)), "should wrap left to col N-1");
        assert!(nb0.contains(&n), "should have (1,0) below");
        assert!(nb0.contains(&1), "should have (0,1) right");
    }

    #[test]
    fn periodic_boundary_3d() {
        let n = 4;
        let lat = Lattice::new(n, Geometry::Cubic3D);
        // site (0,0,0) idx=0
        let nb = &lat.neighbours[0];
        assert_eq!(nb.len(), 6);
        // -x wraps to (3,0,0) = 3*16 = 48
        assert!(nb.contains(&((n - 1) * n * n)));
    }

    #[test]
    fn no_self_neighbours() {
        let lat = Lattice::new(6, Geometry::Cubic3D);
        for (idx, nb) in lat.neighbours.iter().enumerate() {
            assert!(
                !nb.contains(&idx),
                "site {idx} should not be its own neighbour"
            );
        }
    }

    #[test]
    fn neighbour_symmetry() {
        // If j is a neighbour of i, then i must be a neighbour of j
        let lat = Lattice::new(5, Geometry::Cubic3D);
        for (i, nbs) in lat.neighbours.iter().enumerate() {
            for &j in nbs {
                assert!(
                    lat.neighbours[j].contains(&i),
                    "site {j} should have {i} as neighbour (symmetry)"
                );
            }
        }
    }

    #[test]
    fn from_edges_triangle() {
        let edges = vec![(0, 1), (1, 2), (0, 2)];
        let lat = Lattice::from_edges(3, &edges);
        assert_eq!(lat.size(), 3);
        assert_eq!(lat.neighbours[0].len(), 2);
        assert!(lat.neighbours[0].contains(&1));
        assert!(lat.neighbours[0].contains(&2));
    }

    #[test]
    fn randomise_produces_both_spins() {
        let mut lat = Lattice::new(10, Geometry::Square2D);
        let mut rng = rand::thread_rng();
        lat.randomise(&mut rng);
        let has_up = lat.spins.iter().any(|&s| s == 1);
        let has_down = lat.spins.iter().any(|&s| s == -1);
        assert!(
            has_up && has_down,
            "randomise should produce both +1 and -1"
        );
    }

    #[test]
    fn all_spins_initialized_up() {
        let lat = Lattice::new(4, Geometry::Cubic3D);
        assert!(
            lat.spins.iter().all(|&s| s == 1),
            "initial state should be all +1"
        );
    }
}
