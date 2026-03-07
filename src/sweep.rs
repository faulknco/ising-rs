use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::lattice::{Geometry, Lattice};
use crate::metropolis::warm_up as metropolis_warm_up;
use crate::observables::{measure, measure_wolff, measure_wolff_raw, Observables, RawSamples};
use crate::wolff::warm_up as wolff_warm_up;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Algorithm {
    Metropolis,
    Wolff,
}

/// Configuration for a temperature sweep.
pub struct SweepConfig {
    pub n: usize,
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

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            n: 20,
            geometry: Geometry::Square2D,
            j: 1.0,
            h: 0.0,
            t_min: 0.5,
            t_max: 5.0,
            t_steps: 46,
            warmup_sweeps: 2000,
            sample_sweeps: 500,
            seed: 42,
            algorithm: Algorithm::Metropolis,
        }
    }
}

/// Run a full temperature sweep from T_max down to T_min, returning one
/// Observables per temperature point.
///
/// Sweeping from high T → low T lets the system anneal naturally:
///   - At high T the lattice disorders quickly (easy to equilibrate)
///   - As T drops, the ordered state grows from the previous configuration
///   - This avoids getting stuck in random-start metastable domains at low T
pub fn run(config: &SweepConfig) -> Vec<Observables> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(config.seed);
    let mut lattice = Lattice::new(config.n, config.geometry);

    // Start disordered at high T
    lattice.randomise(&mut rng);

    let mut results = Vec::with_capacity(config.t_steps);

    // Sweep high → low
    for step in (0..config.t_steps).rev() {
        let t = config.t_min
            + (config.t_max - config.t_min) * step as f64 / (config.t_steps - 1) as f64;
        let beta = 1.0 / t;

        // Warm up at this temperature, carrying state from previous T
        match config.algorithm {
            Algorithm::Metropolis => metropolis_warm_up(
                &mut lattice,
                beta,
                config.j,
                config.h,
                config.warmup_sweeps,
                &mut rng,
            ),
            Algorithm::Wolff => wolff_warm_up(
                &mut lattice,
                beta,
                config.j,
                config.h,
                config.warmup_sweeps,
                &mut rng,
            ),
        };

        // Measure (using same algorithm as warmup for decorrelation)
        let obs = match config.algorithm {
            Algorithm::Wolff => measure_wolff(
                &mut lattice,
                beta,
                config.j,
                config.h,
                config.sample_sweeps,
                &mut rng,
            ),
            Algorithm::Metropolis => measure(
                &mut lattice,
                beta,
                config.j,
                config.h,
                config.sample_sweeps,
                &mut rng,
            ),
        };
        results.push(obs);
    }

    // Reverse so results are ordered T_min → T_max
    results.reverse();
    results
}

/// Run a temperature sweep collecting raw (E, M) time series at each T.
/// Used for histogram reweighting. Only supports Wolff algorithm.
pub fn run_raw(config: &SweepConfig) -> Vec<RawSamples> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(config.seed);
    let mut lattice = Lattice::new(config.n, config.geometry);
    lattice.randomise(&mut rng);

    let mut results = Vec::with_capacity(config.t_steps);

    for step in (0..config.t_steps).rev() {
        let t = config.t_min
            + (config.t_max - config.t_min) * step as f64 / (config.t_steps - 1) as f64;
        let beta = 1.0 / t;

        wolff_warm_up(
            &mut lattice,
            beta,
            config.j,
            config.h,
            config.warmup_sweeps,
            &mut rng,
        );

        let raw = measure_wolff_raw(
            &mut lattice,
            beta,
            config.j,
            config.h,
            config.sample_sweeps,
            &mut rng,
        );
        results.push(raw);
    }

    results.reverse();
    results
}
