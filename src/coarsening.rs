use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::lattice::{Geometry, Lattice};
use crate::metropolis::{sweep, warm_up};

pub struct CoarseningConfig {
    pub n: usize,
    pub geometry: Geometry,
    pub j: f64,
    pub t_high: f64,
    pub t_quench: f64,
    pub warmup_sweeps: usize,
    pub total_steps: usize,
    pub sample_every: usize,
    pub seed: u64,
}

impl Default for CoarseningConfig {
    fn default() -> Self {
        Self {
            n: 30,
            geometry: Geometry::Cubic3D,
            j: 1.0,
            t_high: 10.0,
            t_quench: 0.5,
            warmup_sweeps: 200,
            total_steps: 50_000,
            sample_every: 100,
            seed: 42,
        }
    }
}

pub struct CoarseningPoint {
    pub step: usize,
    pub rho: f64,
}

/// Fraction of nearest-neighbour pairs with opposite spins.
/// Each bond is counted once (only when nb > idx).
pub fn domain_wall_density(lattice: &Lattice) -> f64 {
    let mut walls = 0usize;
    let mut bonds = 0usize;
    for (idx, &spin) in lattice.spins.iter().enumerate() {
        for &nb in &lattice.neighbours[idx] {
            if nb > idx {
                bonds += 1;
                if lattice.spins[nb] != spin {
                    walls += 1;
                }
            }
        }
    }
    if bonds == 0 {
        return 0.0;
    }
    walls as f64 / bonds as f64
}

/// Run a quench experiment. Returns time series of domain wall density.
pub fn run_coarsening(config: &CoarseningConfig) -> Vec<CoarseningPoint> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(config.seed);
    let mut lattice = Lattice::new(config.n, config.geometry);
    lattice.randomise(&mut rng);

    let beta_high = 1.0 / config.t_high;
    warm_up(
        &mut lattice,
        beta_high,
        config.j,
        0.0,
        config.warmup_sweeps,
        &mut rng,
    );

    let beta_quench = 1.0 / f64::max(config.t_quench, 0.01);
    let mut results = Vec::new();

    for step in 0..config.total_steps {
        sweep(&mut lattice, beta_quench, config.j, 0.0, &mut rng);
        if step % config.sample_every == 0 {
            results.push(CoarseningPoint {
                step,
                rho: domain_wall_density(&lattice),
            });
        }
    }

    results
}
