use crate::lattice::Lattice;

/// All four Ising observables at a given temperature.
#[derive(Debug, Clone)]
pub struct Observables {
    pub temperature: f64,
    pub energy: f64,        // ⟨E⟩ per spin
    pub magnetisation: f64, // |⟨M⟩| per spin
    pub heat_capacity: f64, // Cv
    pub susceptibility: f64,// χ
    pub m2: f64,            // ⟨M²⟩ per spin² — needed for Binder cumulant
    pub m4: f64,            // ⟨M⁴⟩ per spin⁴ — needed for Binder cumulant
}

/// Accumulate E, M statistics and build Observables.
fn finalize(beta: f64, n2: f64, samples: usize,
            sum_e: f64, sum_e2: f64, sum_m: f64, sum_m2: f64, sum_m4: f64,
            sum_m_signed: f64, sum_m_signed2: f64) -> Observables {
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

    let n2 = lattice.size() as f64;
    let (mut sum_e, mut sum_e2, mut sum_m, mut sum_m2, mut sum_m4) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut sum_ms, mut sum_ms2) = (0.0, 0.0);

    for _ in 0..samples {
        sweep(lattice, beta, j, h, rng);
        let (e, m) = energy_magnetisation(lattice, j, h);
        let e_per = e / n2;
        let m_per = (m / n2).abs();
        let m_signed = m / n2;
        sum_e += e_per; sum_e2 += e_per * e_per;
        sum_m += m_per; sum_m2 += m_per * m_per; sum_m4 += m_per.powi(4);
        sum_ms += m_signed; sum_ms2 += m_signed * m_signed;
    }

    finalize(beta, n2, samples, sum_e, sum_e2, sum_m, sum_m2, sum_m4, sum_ms, sum_ms2)
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
    use crate::wolff::step as wolff_step;
    use crate::metropolis::sweep as metro_sweep;

    let n2 = lattice.size() as f64;
    let (mut sum_e, mut sum_e2, mut sum_m, mut sum_m2, mut sum_m4) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut sum_ms, mut sum_ms2) = (0.0, 0.0);

    for _ in 0..samples {
        wolff_step(lattice, beta, j, rng);
        if h.abs() > 1e-9 {
            metro_sweep(lattice, beta, j, h, rng);
        }
        let (e, m) = energy_magnetisation(lattice, j, h);
        let e_per = e / n2;
        let m_per = (m / n2).abs();
        let m_signed = m / n2;
        sum_e += e_per; sum_e2 += e_per * e_per;
        sum_m += m_per; sum_m2 += m_per * m_per; sum_m4 += m_per.powi(4);
        sum_ms += m_signed; sum_ms2 += m_signed * m_signed;
    }

    finalize(beta, n2, samples, sum_e, sum_e2, sum_m, sum_m2, sum_m4, sum_ms, sum_ms2)
}

/// Raw per-sample data for histogram reweighting.
#[derive(Debug, Clone)]
pub struct RawSamples {
    pub temperature: f64,
    pub e_per_spin: Vec<f64>,   // energy per spin each sample
    pub m_abs: Vec<f64>,        // |m| per spin each sample
    pub m_signed: Vec<f64>,     // m per spin each sample (signed)
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
    use crate::wolff::step as wolff_step;
    use crate::metropolis::sweep as metro_sweep;

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
fn energy_magnetisation(lattice: &Lattice, j: f64, h: f64) -> (f64, f64) {
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
