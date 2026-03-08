use crate::heisenberg::HeisenbergLattice;
use crate::heisenberg::metropolis;
use crate::heisenberg::overrelax;
use rand::Rng;

/// Configuration for a Heisenberg temperature sweep.
pub struct HeisSweepConfig {
    /// Over-relaxation sweeps per Metropolis sweep (typically 5).
    pub n_overrelax: usize,
    /// Metropolis cap angle in radians — tune to ~50% acceptance (~0.5 for cubic near Tc).
    pub delta: f64,
    pub j: f64,
    pub t_min: f64,
    pub t_max: f64,
    pub t_steps: usize,
    pub warmup_sweeps: usize,
    pub sample_sweeps: usize,
    pub seed: u64,
}

impl Default for HeisSweepConfig {
    fn default() -> Self {
        Self {
            n_overrelax: 5,
            delta: 0.5,
            j: 1.0,
            t_min: 0.8,
            t_max: 2.0,
            t_steps: 41,
            warmup_sweeps: 500,
            sample_sweeps: 500,
            seed: 42,
        }
    }
}

/// One combined sweep: 1 Metropolis + n_overrelax over-relaxation sweeps.
///
/// This is the standard algorithm for classical Heisenberg MC:
/// a single Metropolis pass (ergodic, satisfies detailed balance) followed by
/// several over-relaxation passes (energy-conserving, reduces autocorrelation time).
/// Reference: Peczak, Ferrenberg, Landau, PRB 1991.
pub fn combined_sweep(
    lat: &mut HeisenbergLattice,
    beta: f64,
    j: f64,
    delta: f64,
    n_overrelax: usize,
    rng: &mut impl Rng,
) {
    metropolis::sweep(lat, beta, j, delta, rng);
    for _ in 0..n_overrelax {
        overrelax::sweep(lat, j);
    }
}
