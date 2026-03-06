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

/// Compute observables by averaging over `samples` sweeps.
///
/// Collects E and M after each sweep, then derives Cv and χ
/// from the variance: Cv = β²(⟨E²⟩ − ⟨E⟩²)N, χ = β(⟨M²⟩ − ⟨M⟩²)N
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
    let mut sum_e = 0.0_f64;
    let mut sum_e2 = 0.0_f64;
    let mut sum_m = 0.0_f64;
    let mut sum_m2 = 0.0_f64;
    let mut sum_m4 = 0.0_f64;

    for _ in 0..samples {
        sweep(lattice, beta, j, h, rng);

        let (e, m) = energy_magnetisation(lattice, j, h);
        let e_per = e / n2;
        let m_per = (m / n2).abs();

        sum_e += e_per;
        sum_e2 += e_per * e_per;
        sum_m += m_per;
        sum_m2 += m_per * m_per;
        sum_m4 += m_per * m_per * m_per * m_per;
    }

    let s = samples as f64;
    let avg_e = sum_e / s;
    let avg_e2 = sum_e2 / s;
    let avg_m = sum_m / s;
    let avg_m2 = sum_m2 / s;
    let avg_m4 = sum_m4 / s;

    let t = 1.0 / beta;
    let cv = beta * beta * (avg_e2 - avg_e * avg_e) * n2;
    let chi = beta * (avg_m2 - avg_m * avg_m) * n2;

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
