use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::coarsening::domain_wall_density;
use crate::lattice::{Geometry, Lattice};
use crate::metropolis::{sweep, warm_up};

pub struct KzConfig {
    pub n: usize,
    pub geometry: Geometry,
    pub j: f64,
    pub t_start: f64, // start temperature (disordered, > Tc)
    pub t_end: f64,   // end temperature (ordered, < Tc)
    pub tau_q: usize, // quench time in sweeps
    pub seed: u64,
}

impl Default for KzConfig {
    fn default() -> Self {
        Self {
            n: 20,
            geometry: Geometry::Cubic3D,
            j: 1.0,
            t_start: 6.0,
            t_end: 1.0,
            tau_q: 1000,
            seed: 42,
        }
    }
}

pub struct KzResult {
    pub tau_q: usize,
    pub rho_final: f64,
}

/// Run one KZ quench: cool linearly from t_start to t_end over tau_q sweeps.
/// Returns domain wall density at end of quench.
pub fn run_kz(config: &KzConfig) -> KzResult {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(config.seed);
    let mut lattice = Lattice::new(config.n, config.geometry);
    lattice.randomise(&mut rng);

    // Warmup at t_start to get into disordered phase
    let beta_start = 1.0 / config.t_start;
    warm_up(&mut lattice, beta_start, config.j, 0.0, 200, &mut rng);

    // Linear ramp T from t_start to t_end over tau_q sweeps
    for step in 0..config.tau_q {
        let frac = step as f64 / config.tau_q as f64;
        let t = config.t_start + (config.t_end - config.t_start) * frac;
        let beta = 1.0 / t.max(0.01);
        sweep(&mut lattice, beta, config.j, 0.0, &mut rng);
    }

    KzResult {
        tau_q: config.tau_q,
        rho_final: domain_wall_density(&lattice),
    }
}

/// Run KZ experiment over a range of quench times.
/// Returns Vec<(tau_q, rho_final)> averaged over n_trials trials.
#[allow(clippy::too_many_arguments)]
pub fn run_kz_sweep(
    n: usize,
    geometry: Geometry,
    j: f64,
    t_start: f64,
    t_end: f64,
    tau_q_values: &[usize],
    n_trials: usize,
    base_seed: u64,
) -> Vec<(usize, f64)> {
    tau_q_values
        .iter()
        .map(|&tau_q| {
            let rho_avg: f64 = (0..n_trials)
                .map(|trial| {
                    let config = KzConfig {
                        n,
                        geometry,
                        j,
                        t_start,
                        t_end,
                        tau_q,
                        seed: base_seed
                            .wrapping_add(tau_q as u64)
                            .wrapping_add(trial as u64),
                    };
                    run_kz(&config).rho_final
                })
                .sum::<f64>()
                / n_trials as f64;
            (tau_q, rho_avg)
        })
        .collect()
}
