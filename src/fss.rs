use crate::lattice::Geometry;
use crate::observables::Observables;
use crate::sweep::{run, Algorithm, SweepConfig};

/// Configuration for a finite-size scaling run.
pub struct FssConfig {
    pub sizes: Vec<usize>,
    pub geometry: Geometry,
    pub j: f64,
    pub h: f64,
    pub t_min: f64,
    pub t_max: f64,
    pub t_steps: usize,
    pub warmup_sweeps: usize,
    pub sample_sweeps: usize,
    pub seed: u64,
    pub algorithm: Algorithm,
}

impl Default for FssConfig {
    fn default() -> Self {
        Self {
            sizes: vec![8, 12, 16, 20, 24, 28],
            geometry: Geometry::Cubic3D,
            j: 1.0,
            h: 0.0,
            t_min: 3.5,
            t_max: 5.5,
            t_steps: 41,
            warmup_sweeps: 500,
            sample_sweeps: 200,
            seed: 42,
            algorithm: Algorithm::Wolff,
        }
    }
}

/// Run a temperature sweep for each lattice size in config.sizes.
/// Returns Vec of (n, observables_per_temperature).
pub fn run_fss(config: &FssConfig) -> Vec<(usize, Vec<Observables>)> {
    config.sizes.iter().map(|&n| {
        eprintln!("FSS: N={n}");
        let sweep_cfg = SweepConfig {
            n,
            geometry: config.geometry,
            j: config.j,
            h: config.h,
            t_min: config.t_min,
            t_max: config.t_max,
            t_steps: config.t_steps,
            warmup_sweeps: config.warmup_sweeps,
            sample_sweeps: config.sample_sweeps,
            seed: config.seed,
            algorithm: config.algorithm,
        };
        let obs = run(&sweep_cfg);
        (n, obs)
    }).collect()
}
