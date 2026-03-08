use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::coarsening::domain_wall_density;
use crate::lattice::{Geometry, Lattice};
use crate::metropolis::{sweep, warm_up};

const MIN_TEMPERATURE: f64 = 0.01;

#[derive(Debug, Clone, Copy)]
pub struct KzProtocol {
    pub warmup_sweeps: usize,
    pub freeze_sweeps: usize,
    pub freeze_temperature: f64,
}

impl Default for KzProtocol {
    fn default() -> Self {
        Self {
            warmup_sweeps: 200,
            freeze_sweeps: 0,
            freeze_temperature: MIN_TEMPERATURE,
        }
    }
}

impl KzProtocol {
    pub fn describe(&self) -> String {
        format!(
            "warmup={} sweeps, ramp endpoint measurement, freeze={} sweeps at T={:.4}",
            self.warmup_sweeps, self.freeze_sweeps, self.freeze_temperature
        )
    }
}

pub struct KzConfig {
    pub n: usize,
    pub geometry: Geometry,
    pub j: f64,
    pub t_start: f64, // start temperature (disordered, > Tc)
    pub t_end: f64,   // end temperature (ordered, < Tc)
    pub tau_q: usize, // quench time in sweeps
    pub protocol: KzProtocol,
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
            protocol: KzProtocol::default(),
            seed: 42,
        }
    }
}

pub struct KzResult {
    pub tau_q: usize,
    pub rho_final: f64,
}

#[derive(Debug, Clone)]
pub struct KzSweepPoint {
    pub tau_q: usize,
    pub rho: f64,
    pub rho_err: f64,
    pub n_trials: usize,
}

/// Inclusive linear ramp temperature after `step` sweeps.
///
/// For `tau_q > 0`, the final sweep lands exactly at `t_end`.
pub fn linear_ramp_temperature(t_start: f64, t_end: f64, step: usize, tau_q: usize) -> f64 {
    if tau_q == 0 {
        return t_end;
    }
    let frac = (step + 1) as f64 / tau_q as f64;
    t_start + (t_end - t_start) * frac
}

pub fn mean_and_std_err(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if values.len() == 1 {
        return (mean, 0.0);
    }

    let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_err = var.sqrt() / n.sqrt();
    (mean, std_err)
}

pub fn trial_seed(base_seed: u64, tau_q: usize, trial: usize) -> u64 {
    base_seed
        .wrapping_add(tau_q as u64)
        .wrapping_add(trial as u64)
}

/// Run one KZ quench: warm up at `t_start`, ramp linearly to `t_end` over `tau_q` sweeps,
/// then optionally snap-freeze for a small number of sweeps before measuring domain walls.
pub fn run_kz(config: &KzConfig) -> KzResult {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(config.seed);
    let mut lattice = Lattice::new(config.n, config.geometry);
    lattice.randomise(&mut rng);

    // Warmup at t_start to get into disordered phase
    let beta_start = 1.0 / config.t_start;
    warm_up(
        &mut lattice,
        beta_start,
        config.j,
        0.0,
        config.protocol.warmup_sweeps,
        &mut rng,
    );

    // Linear ramp T from t_start to t_end over tau_q sweeps.
    for step in 0..config.tau_q {
        let t = linear_ramp_temperature(config.t_start, config.t_end, step, config.tau_q);
        let beta = 1.0 / t.max(MIN_TEMPERATURE);
        sweep(&mut lattice, beta, config.j, 0.0, &mut rng);
    }

    if config.protocol.freeze_sweeps > 0 {
        let beta_freeze = 1.0 / config.protocol.freeze_temperature.max(MIN_TEMPERATURE);
        for _ in 0..config.protocol.freeze_sweeps {
            sweep(&mut lattice, beta_freeze, config.j, 0.0, &mut rng);
        }
    }

    KzResult {
        tau_q: config.tau_q,
        rho_final: domain_wall_density(&lattice),
    }
}

/// Run KZ experiment over a range of quench times.
/// Returns mean domain-wall density and its standard error over `n_trials`.
#[allow(clippy::too_many_arguments)]
pub fn run_kz_sweep(
    n: usize,
    geometry: Geometry,
    j: f64,
    t_start: f64,
    t_end: f64,
    tau_q_values: &[usize],
    n_trials: usize,
    protocol: KzProtocol,
    base_seed: u64,
) -> Vec<KzSweepPoint> {
    tau_q_values
        .iter()
        .map(|&tau_q| {
            let rho_samples: Vec<f64> = (0..n_trials)
                .map(|trial| {
                    let config = KzConfig {
                        n,
                        geometry,
                        j,
                        t_start,
                        t_end,
                        tau_q,
                        protocol,
                        seed: trial_seed(base_seed, tau_q, trial),
                    };
                    run_kz(&config).rho_final
                })
                .collect();
            let (rho, rho_err) = mean_and_std_err(&rho_samples);
            KzSweepPoint {
                tau_q,
                rho,
                rho_err,
                n_trials,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_ramp_reaches_endpoint() {
        let start = 6.0;
        let end = 1.0;
        let tau_q = 5;
        let temps: Vec<f64> = (0..tau_q)
            .map(|step| linear_ramp_temperature(start, end, step, tau_q))
            .collect();

        assert!((temps[0] - 5.0).abs() < 1e-12);
        assert!((temps[tau_q - 1] - end).abs() < 1e-12);
    }

    #[test]
    fn zero_length_ramp_uses_end_temperature() {
        assert_eq!(linear_ramp_temperature(6.0, 1.0, 0, 0), 1.0);
    }

    #[test]
    fn mean_and_std_err_matches_known_values() {
        let (mean, err) = mean_and_std_err(&[1.0, 3.0, 5.0, 7.0]);
        assert!((mean - 4.0).abs() < 1e-12);
        let expected = (20.0_f64 / 3.0).sqrt() / 2.0;
        assert!((err - expected).abs() < 1e-12);
    }

    #[test]
    fn single_sample_has_zero_standard_error() {
        let (mean, err) = mean_and_std_err(&[0.25]);
        assert!((mean - 0.25).abs() < 1e-12);
        assert_eq!(err, 0.0);
    }
}
