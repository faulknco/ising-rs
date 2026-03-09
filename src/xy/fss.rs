use crate::lattice::{Geometry, Lattice};
use crate::xy::{
    observables::{measure, XyObservables},
    XyLattice,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Configuration for an XY FSS run over multiple lattice sizes.
pub struct XyFssConfig {
    /// Linear lattice sizes to simulate.
    pub sizes: Vec<usize>,
    /// Lattice geometry (Cubic3D for O(2) validation).
    pub geometry: Geometry,
    /// Exchange coupling J.
    pub j: f64,
    /// Minimum temperature (units J/k_B).
    pub t_min: f64,
    /// Maximum temperature (units J/k_B).
    pub t_max: f64,
    /// Number of temperature steps (uniformly spaced).
    pub t_steps: usize,
    /// Wolff cluster sweeps to discard before measuring.
    pub warmup_sweeps: usize,
    /// Wolff cluster sweeps to measure (must be divisible by 20).
    pub sample_sweeps: usize,
    /// Base RNG seed (each size uses seed.wrapping_add(n)).
    pub seed: u64,
}

impl Default for XyFssConfig {
    fn default() -> Self {
        Self {
            sizes: vec![8, 12, 16, 20, 24, 32],
            geometry: Geometry::Cubic3D,
            j: 1.0,
            t_min: 1.8,
            t_max: 2.7,
            t_steps: 41,
            warmup_sweeps: 2000,
            sample_sweeps: 2000,
            seed: 42,
        }
    }
}

/// Run XY FSS temperature sweeps for each lattice size.
///
/// For each size N, builds a cubic lattice of N³ spins, randomises it,
/// and sweeps through t_steps temperatures from t_min to t_max.
/// Each size uses a distinct RNG seed (base_seed + N) for independence.
///
/// Returns `Vec<(N, Vec<XyObservables>)>` ordered from t_min to t_max.
pub fn run_xy_fss(config: &XyFssConfig) -> Vec<(usize, Vec<XyObservables>)> {
    config
        .sizes
        .iter()
        .map(|&n| {
            eprintln!("XY FSS: N={n}");
            let seed = config.seed.wrapping_add(n as u64);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

            let ising_lat = Lattice::new(n, config.geometry);
            let mut lat = XyLattice::new(ising_lat.neighbours.clone());
            lat.randomise(&mut rng);

            let temps: Vec<f64> = (0..config.t_steps)
                .map(|i| {
                    config.t_min
                        + (config.t_max - config.t_min) * i as f64 / (config.t_steps - 1) as f64
                })
                .collect();

            let results: Vec<XyObservables> = temps
                .iter()
                .map(|&t| {
                    let beta = 1.0 / t;
                    measure(
                        &mut lat,
                        beta,
                        config.j,
                        config.warmup_sweeps,
                        config.sample_sweeps,
                        &mut rng,
                    )
                })
                .collect();

            (n, results)
        })
        .collect()
}
