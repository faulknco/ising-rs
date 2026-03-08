use crate::heisenberg::{HeisenbergLattice, observables::{measure, HeisenbergObservables}};
use crate::lattice::{Geometry, Lattice};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Configuration for a Heisenberg FSS run over multiple lattice sizes.
pub struct HeisFssConfig {
    /// Linear lattice sizes to simulate (e.g. [8, 12, 16, 20, 24]).
    pub sizes: Vec<usize>,
    /// Lattice geometry (Cubic3D for validation).
    pub geometry: Geometry,
    /// Exchange coupling J.
    pub j: f64,
    /// Minimum temperature (units J/k_B).
    pub t_min: f64,
    /// Maximum temperature (units J/k_B).
    pub t_max: f64,
    /// Number of temperature steps (uniformly spaced).
    pub t_steps: usize,
    /// Equilibration sweeps per temperature point.
    pub warmup_sweeps: usize,
    /// Measurement sweeps per temperature point (must be divisible by 20).
    pub sample_sweeps: usize,
    /// Over-relaxation sweeps per Metropolis sweep.
    pub n_overrelax: usize,
    /// Metropolis cap angle in radians.
    pub delta: f64,
    /// Base RNG seed (each size uses seed.wrapping_add(n)).
    pub seed: u64,
}

impl Default for HeisFssConfig {
    fn default() -> Self {
        Self {
            sizes: vec![8, 12, 16, 20, 24],
            geometry: Geometry::Cubic3D,
            j: 1.0,
            t_min: 0.8,
            t_max: 2.0,
            t_steps: 41,
            warmup_sweeps: 500,
            sample_sweeps: 500,
            n_overrelax: 5,
            delta: 0.5,
            seed: 42,
        }
    }
}

/// Run Heisenberg FSS temperature sweeps for each lattice size.
///
/// For each size N, builds a cubic lattice of N³ spins, randomises it,
/// and sweeps through t_steps temperatures from t_min to t_max.
/// Each size uses a distinct RNG seed (base_seed + N) for independence.
///
/// Returns `Vec<(N, Vec<HeisenbergObservables>)>` — one entry per size,
/// with observables ordered from t_min to t_max.
pub fn run_heisenberg_fss(config: &HeisFssConfig) -> Vec<(usize, Vec<HeisenbergObservables>)> {
    config.sizes.iter().map(|&n| {
        eprintln!("Heisenberg FSS: N={n}");
        let seed = config.seed.wrapping_add(n as u64);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

        // Build lattice from existing Lattice infrastructure
        let ising_lat = Lattice::new(n, config.geometry);
        let mut lat = HeisenbergLattice::new(ising_lat.neighbours.clone());
        lat.randomise(&mut rng);

        let temps: Vec<f64> = (0..config.t_steps).map(|i| {
            config.t_min + (config.t_max - config.t_min) * i as f64 / (config.t_steps - 1) as f64
        }).collect();

        let results: Vec<HeisenbergObservables> = temps.iter().map(|&t| {
            let beta = 1.0 / t;
            measure(
                &mut lat, beta, config.j, config.delta,
                config.n_overrelax, config.warmup_sweeps, config.sample_sweeps,
                &mut rng,
            )
        }).collect();

        (n, results)
    }).collect()
}
