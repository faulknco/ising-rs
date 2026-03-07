use crate::lattice::Lattice;

/// All four Ising observables at a given temperature.
#[derive(Debug, Clone)]
pub struct Observables {
    pub temperature: f64,
    pub energy: f64,         // ⟨E⟩ per spin
    pub magnetisation: f64,  // |⟨M⟩| per spin
    pub heat_capacity: f64,  // Cv
    pub susceptibility: f64, // χ
    pub m2: f64,             // ⟨M²⟩ per spin² — needed for Binder cumulant
    pub m4: f64,             // ⟨M⁴⟩ per spin⁴ — needed for Binder cumulant
}

/// Accumulate E, M statistics and build Observables.
#[allow(clippy::too_many_arguments)]
fn finalize(
    beta: f64,
    n2: f64,
    samples: usize,
    sum_e: f64,
    sum_e2: f64,
    sum_m: f64,
    sum_m2: f64,
    sum_m4: f64,
    sum_m_signed: f64,
    sum_m_signed2: f64,
) -> Observables {
    let s = samples as f64;
    let avg_e = sum_e / s;
    let avg_e2 = sum_e2 / s;
    let avg_m = sum_m / s;
    let avg_m2 = sum_m2 / s;
    let avg_m4 = sum_m4 / s;
    let avg_m_signed = sum_m_signed / s;
    let avg_m_signed2 = sum_m_signed2 / s;

    let t = 1.0 / beta;
    let cv = beta * beta * (avg_e2 - avg_e * avg_e) * n2;
    let chi = beta * (avg_m_signed2 - avg_m_signed * avg_m_signed) * n2;

    Observables {
        temperature: t,
        energy: avg_e,
        magnetisation: avg_m,
        heat_capacity: cv,
        susceptibility: chi,
        m2: avg_m2,
        m4: avg_m4,
    }
}

/// Accumulate E, M samples and return Observables.
///
/// `step_fn` is called once per sample to advance the lattice state.
/// It captures the rng by closure so downstream functions keep their `impl Rng` bounds.
fn measure_with(
    lattice: &mut Lattice,
    j: f64,
    h: f64,
    beta: f64,
    samples: usize,
    mut step_fn: impl FnMut(&mut Lattice),
) -> Observables {
    let n2 = lattice.size() as f64;
    let (mut sum_e, mut sum_e2, mut sum_m, mut sum_m2, mut sum_m4) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut sum_ms, mut sum_ms2) = (0.0, 0.0);

    for _ in 0..samples {
        step_fn(lattice);
        let (e, m) = energy_magnetisation(lattice, j, h);
        let e_per = e / n2;
        let m_per = (m / n2).abs();
        let m_signed = m / n2;
        sum_e += e_per;
        sum_e2 += e_per * e_per;
        sum_m += m_per;
        sum_m2 += m_per * m_per;
        sum_m4 += m_per.powi(4);
        sum_ms += m_signed;
        sum_ms2 += m_signed * m_signed;
    }

    finalize(
        beta, n2, samples, sum_e, sum_e2, sum_m, sum_m2, sum_m4, sum_ms, sum_ms2,
    )
}

/// Compute observables by averaging over `samples` Metropolis sweeps.
pub fn measure(
    lattice: &mut Lattice,
    beta: f64,
    j: f64,
    h: f64,
    samples: usize,
    rng: &mut impl rand::Rng,
) -> Observables {
    use crate::metropolis::sweep;
    measure_with(lattice, j, h, beta, samples, |lat| {
        sweep(lat, beta, j, h, rng);
    })
}

/// Compute observables using Wolff cluster steps between measurements.
pub fn measure_wolff(
    lattice: &mut Lattice,
    beta: f64,
    j: f64,
    h: f64,
    samples: usize,
    rng: &mut impl rand::Rng,
) -> Observables {
    use crate::metropolis::sweep as metro_sweep;
    use crate::wolff::step as wolff_step;
    measure_with(lattice, j, h, beta, samples, |lat| {
        wolff_step(lat, beta, j, rng);
        if h.abs() > 1e-9 {
            metro_sweep(lat, beta, j, h, rng);
        }
    })
}

/// Raw per-sample data for histogram reweighting.
#[derive(Debug, Clone)]
pub struct RawSamples {
    pub temperature: f64,
    pub e_per_spin: Vec<f64>, // energy per spin each sample
    pub m_abs: Vec<f64>,      // |m| per spin each sample
    pub m_signed: Vec<f64>,   // m per spin each sample (signed)
}

/// Collect raw (E, M) time series using Wolff steps — for histogram reweighting.
pub fn measure_wolff_raw(
    lattice: &mut Lattice,
    beta: f64,
    j: f64,
    h: f64,
    samples: usize,
    rng: &mut impl rand::Rng,
) -> RawSamples {
    use crate::metropolis::sweep as metro_sweep;
    use crate::wolff::step as wolff_step;

    let n2 = lattice.size() as f64;
    let mut e_per_spin = Vec::with_capacity(samples);
    let mut m_abs = Vec::with_capacity(samples);
    let mut m_signed = Vec::with_capacity(samples);

    for _ in 0..samples {
        wolff_step(lattice, beta, j, rng);
        if h.abs() > 1e-9 {
            metro_sweep(lattice, beta, j, h, rng);
        }
        let (e, m) = energy_magnetisation(lattice, j, h);
        e_per_spin.push(e / n2);
        m_abs.push((m / n2).abs());
        m_signed.push(m / n2);
    }

    RawSamples {
        temperature: 1.0 / beta,
        e_per_spin,
        m_abs,
        m_signed,
    }
}

/// Compute total energy and total magnetisation of the current configuration.
///
/// Energy: E = −J Σ_{⟨i,j⟩} σᵢσⱼ − h Σᵢ σᵢ
/// The /2 avoids double-counting each pair.
pub fn energy_magnetisation(lattice: &Lattice, j: f64, h: f64) -> (f64, f64) {
    let mut e = 0.0_f64;
    let mut m = 0.0_f64;

    for (idx, &spin) in lattice.spins.iter().enumerate() {
        let s = spin as f64;
        let neighbour_sum: f64 = lattice.neighbours[idx]
            .iter()
            .map(|&k| lattice.spins[k] as f64)
            .sum();

        e += -j * s * neighbour_sum / 2.0 - h * s;
        m += s;
    }

    (e, m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{Geometry, Lattice};

    #[test]
    fn ground_state_energy_3d() {
        // All spins +1, J=1, h=0: E = -J * z/2 * N = -3N (z=6, each bond counted once)
        let lat = Lattice::new(4, Geometry::Cubic3D);
        let n = lat.size() as f64;
        let (e, m) = energy_magnetisation(&lat, 1.0, 0.0);
        let e_per_spin = e / n;
        assert!(
            (e_per_spin - (-3.0)).abs() < 1e-10,
            "3D ground state energy should be -3J per spin, got {e_per_spin}"
        );
        assert!(
            (m - n).abs() < 1e-10,
            "all-up magnetisation should be N, got {m}"
        );
    }

    #[test]
    fn ground_state_energy_2d() {
        // All spins +1, J=1, h=0: E = -J * z/2 * N = -2N (z=4)
        let lat = Lattice::new(8, Geometry::Square2D);
        let n = lat.size() as f64;
        let (e, _) = energy_magnetisation(&lat, 1.0, 0.0);
        let e_per_spin = e / n;
        assert!(
            (e_per_spin - (-2.0)).abs() < 1e-10,
            "2D ground state energy should be -2J per spin, got {e_per_spin}"
        );
    }

    #[test]
    fn field_energy() {
        // All spins +1, h=1: field contribution = -h * N
        let lat = Lattice::new(4, Geometry::Square2D);
        let n = lat.size() as f64;
        let (e_no_h, _) = energy_magnetisation(&lat, 1.0, 0.0);
        let (e_with_h, _) = energy_magnetisation(&lat, 1.0, 1.0);
        assert!(
            (e_with_h - e_no_h - (-n)).abs() < 1e-10,
            "field energy should add -h*N for all-up spins"
        );
    }

    #[test]
    fn antiferromagnet_energy() {
        // All spins +1, J=-1: E = +3N for 3D (antiferromagnet is frustrated on all-up)
        let lat = Lattice::new(4, Geometry::Cubic3D);
        let n = lat.size() as f64;
        let (e, _) = energy_magnetisation(&lat, -1.0, 0.0);
        let e_per_spin = e / n;
        assert!(
            (e_per_spin - 3.0).abs() < 1e-10,
            "antiferromagnet all-up should have +3J per spin, got {e_per_spin}"
        );
    }

    #[test]
    fn measure_low_temp_ordered() {
        // At very low T, Metropolis should preserve the ordered ground state
        use rand::SeedableRng;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = Lattice::new(4, Geometry::Cubic3D);
        let beta = 100.0; // T = 0.01
        let obs = measure(&mut lat, beta, 1.0, 0.0, 100, &mut rng);
        assert!(
            (obs.energy - (-3.0)).abs() < 0.01,
            "at T≈0, energy should be ~-3J, got {}",
            obs.energy
        );
        assert!(
            (obs.magnetisation - 1.0).abs() < 0.01,
            "at T≈0, |m| should be ~1, got {}",
            obs.magnetisation
        );
    }

    #[test]
    fn measure_wolff_low_temp() {
        use rand::SeedableRng;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = Lattice::new(4, Geometry::Cubic3D);
        let beta = 100.0;
        let obs = measure_wolff(&mut lat, beta, 1.0, 0.0, 100, &mut rng);
        assert!(
            (obs.energy - (-3.0)).abs() < 0.01,
            "Wolff at T≈0: energy should be ~-3J, got {}",
            obs.energy
        );
        assert!(
            (obs.magnetisation - 1.0).abs() < 0.01,
            "Wolff at T≈0: |m| should be ~1, got {}",
            obs.magnetisation
        );
    }

    #[test]
    fn measure_high_temp_disordered() {
        // At very high T, |m| should be small (≈ 1/sqrt(N))
        use rand::SeedableRng;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = Lattice::new(8, Geometry::Cubic3D);
        lat.randomise(&mut rng);
        let beta = 0.01; // T = 100
        let obs = measure(&mut lat, beta, 1.0, 0.0, 500, &mut rng);
        assert!(
            obs.magnetisation < 0.2,
            "at T→∞, |m| should be small, got {}",
            obs.magnetisation
        );
    }

    #[test]
    fn binder_cumulant_ground_state() {
        // At T→0, all m samples = 1, so M2=1, M4=1, U = 1 - 1/(3*1) = 2/3
        use rand::SeedableRng;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = Lattice::new(4, Geometry::Cubic3D);
        let beta = 100.0;
        let obs = measure_wolff(&mut lat, beta, 1.0, 0.0, 200, &mut rng);
        let u = 1.0 - obs.m4 / (3.0 * obs.m2 * obs.m2);
        assert!(
            (u - 2.0 / 3.0).abs() < 0.01,
            "Binder cumulant at T→0 should be 2/3, got {u}"
        );
    }

    #[test]
    fn raw_samples_length() {
        use rand::SeedableRng;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = Lattice::new(4, Geometry::Cubic3D);
        let raw = measure_wolff_raw(&mut lat, 0.5, 1.0, 0.0, 50, &mut rng);
        assert_eq!(raw.e_per_spin.len(), 50);
        assert_eq!(raw.m_abs.len(), 50);
        assert_eq!(raw.m_signed.len(), 50);
        assert!((raw.temperature - 2.0).abs() < 1e-10);
    }

    #[test]
    fn raw_samples_m_abs_nonnegative() {
        use rand::SeedableRng;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = Lattice::new(6, Geometry::Cubic3D);
        lat.randomise(&mut rng);
        let raw = measure_wolff_raw(&mut lat, 0.2, 1.0, 0.0, 100, &mut rng);
        assert!(
            raw.m_abs.iter().all(|&m| m >= 0.0),
            "|m| should always be non-negative"
        );
    }
}
