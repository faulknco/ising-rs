use crate::xy::wolff::sweep;
use crate::xy::{energy_magnetisation, XyLattice};
use rand::Rng;

/// All measured observables for one temperature point, with jackknife error bars.
#[derive(Debug, Clone)]
pub struct XyObservables {
    pub temperature: f64,
    /// Mean energy per spin E/N.
    pub energy: f64,
    pub energy_err: f64,
    /// Mean |M|/N (scalar magnetisation per spin).
    pub magnetisation: f64,
    pub magnetisation_err: f64,
    /// Total heat capacity C = β²N(⟨(E/N)²⟩ − ⟨E/N⟩²).
    pub heat_capacity: f64,
    pub heat_capacity_err: f64,
    /// Magnetic susceptibility χ = βN(⟨m²⟩ − ⟨m⟩²).
    pub susceptibility: f64,
    pub susceptibility_err: f64,
    /// ⟨m²⟩ (needed for Binder cumulant).
    pub m2: f64,
    pub m2_err: f64,
    /// ⟨m⁴⟩ (needed for Binder cumulant).
    pub m4: f64,
    pub m4_err: f64,
}

/// Equilibrate a lattice and measure observables with 20-block jackknife error estimation.
///
/// One call to `wolff::sweep()` = one cluster flip attempt, which near Tc decorrelates
/// the system as efficiently as O(N) Metropolis sweeps. The measure() function therefore
/// collects one observable sample per sweep call (no thinning needed).
///
/// # Arguments
/// - `lat`: lattice (mutated in place; callers may re-use across temperatures)
/// - `beta`: inverse temperature 1/T
/// - `j`: exchange coupling
/// - `warmup`: equilibration Wolff sweeps (discarded)
/// - `samples`: measurement Wolff sweeps (must be ≥ 20 for jackknife; divisible by 20)
pub fn measure(
    lat: &mut XyLattice,
    beta: f64,
    j: f64,
    warmup: usize,
    samples: usize,
    rng: &mut impl Rng,
) -> XyObservables {
    let n = lat.size() as f64;

    for _ in 0..warmup {
        sweep(lat, beta, j, rng);
    }

    let mut e_series = Vec::with_capacity(samples);
    let mut m_series = Vec::with_capacity(samples);

    for _ in 0..samples {
        sweep(lat, beta, j, rng);
        let (e, mv) = energy_magnetisation(lat, j);
        let m_abs = (mv[0] * mv[0] + mv[1] * mv[1]).sqrt() / n;
        e_series.push(e / n);
        m_series.push(m_abs);
    }

    let avg_e = mean(&e_series);
    let avg_m = mean(&m_series);
    let avg_e2 = mean_of_sq(&e_series);
    let avg_m2 = mean_of_sq(&m_series);
    let avg_m4 = mean_of_pow4(&m_series);

    let cv = beta * beta * n * (avg_e2 - avg_e * avg_e);
    let chi = beta * n * (avg_m2 - avg_m * avg_m);

    let n_blocks = 20;
    let block_size = samples / n_blocks;
    debug_assert!(
        samples.is_multiple_of(n_blocks),
        "samples ({samples}) is not divisible by n_blocks ({n_blocks}); \
         {} samples will be silently discarded",
        samples % n_blocks
    );

    let (e_err, m_err, cv_err, chi_err, m2_err, m4_err) =
        jackknife_errors(&e_series, &m_series, beta, n, n_blocks, block_size);

    XyObservables {
        temperature: 1.0 / beta,
        energy: avg_e,
        energy_err: e_err,
        magnetisation: avg_m,
        magnetisation_err: m_err,
        heat_capacity: cv,
        heat_capacity_err: cv_err,
        susceptibility: chi,
        susceptibility_err: chi_err,
        m2: avg_m2,
        m2_err,
        m4: avg_m4,
        m4_err,
    }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn mean_of_sq(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64
}

fn mean_of_pow4(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x * x * x).sum::<f64>() / v.len() as f64
}

fn jackknife_errors(
    e: &[f64],
    m: &[f64],
    beta: f64,
    n: f64,
    n_blocks: usize,
    block_size: usize,
) -> (f64, f64, f64, f64, f64, f64) {
    let mut jk_e = Vec::with_capacity(n_blocks);
    let mut jk_m = Vec::with_capacity(n_blocks);
    let mut jk_cv = Vec::with_capacity(n_blocks);
    let mut jk_chi = Vec::with_capacity(n_blocks);
    let mut jk_m2 = Vec::with_capacity(n_blocks);
    let mut jk_m4 = Vec::with_capacity(n_blocks);

    let total = n_blocks * block_size;

    for b in 0..n_blocks {
        let lo = b * block_size;
        let hi = lo + block_size;

        let e_jk: Vec<f64> = e[..lo].iter().chain(&e[hi..total]).copied().collect();
        let m_jk: Vec<f64> = m[..lo].iter().chain(&m[hi..total]).copied().collect();

        let ae = mean(&e_jk);
        let am = mean(&m_jk);
        let ae2 = mean_of_sq(&e_jk);
        let am2 = mean_of_sq(&m_jk);
        let am4 = mean_of_pow4(&m_jk);

        jk_e.push(ae);
        jk_m.push(am);
        jk_cv.push(beta * beta * n * (ae2 - ae * ae));
        jk_chi.push(beta * n * (am2 - am * am));
        jk_m2.push(am2);
        jk_m4.push(am4);
    }

    (
        jackknife_std(&jk_e),
        jackknife_std(&jk_m),
        jackknife_std(&jk_cv),
        jackknife_std(&jk_chi),
        jackknife_std(&jk_m2),
        jackknife_std(&jk_m4),
    )
}

fn jackknife_std(jk: &[f64]) -> f64 {
    let nb = jk.len() as f64;
    let mean = jk.iter().sum::<f64>() / nb;
    let var = jk.iter().map(|x| (x - mean).powi(2)).sum::<f64>() * (nb - 1.0) / nb;
    var.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xy::XyLattice;
    use rand::SeedableRng;

    fn cubic3d_neighbours(n: usize) -> Vec<Vec<usize>> {
        (0..n * n * n)
            .map(|idx| {
                let z = idx / (n * n);
                let y = (idx / n) % n;
                let x = idx % n;
                vec![
                    ((x + 1) % n) + y * n + z * n * n,
                    ((x + n - 1) % n) + y * n + z * n * n,
                    x + ((y + 1) % n) * n + z * n * n,
                    x + ((y + n - 1) % n) * n + z * n * n,
                    x + y * n + ((z + 1) % n) * n * n,
                    x + y * n + ((z + n - 1) % n) * n * n,
                ]
            })
            .collect()
    }

    #[test]
    fn measure_low_temp_ordered() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = cubic3d_neighbours(4);
        let mut lat = XyLattice::new(nb);
        let obs = measure(&mut lat, 0.5_f64.recip(), 1.0, 200, 200, &mut rng);
        assert!(
            obs.magnetisation > 0.85,
            "at T=0.5 |m| should be >0.85, got {}",
            obs.magnetisation
        );
    }

    #[test]
    fn measure_high_temp_disordered() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = cubic3d_neighbours(6);
        let mut lat = XyLattice::new(nb);
        lat.randomise(&mut rng);
        let obs = measure(&mut lat, (10.0_f64).recip(), 1.0, 200, 500, &mut rng);
        assert!(
            obs.magnetisation < 0.3,
            "at T=10 |m| should be <0.3, got {}",
            obs.magnetisation
        );
    }

    #[test]
    fn binder_ground_state() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = cubic3d_neighbours(4);
        let mut lat = XyLattice::new(nb);
        let obs = measure(&mut lat, (0.1_f64).recip(), 1.0, 500, 500, &mut rng);
        let u = 1.0 - obs.m4 / (3.0 * obs.m2 * obs.m2);
        assert!(
            (u - 2.0 / 3.0).abs() < 0.1,
            "Binder cumulant at T→0 should be ≈2/3, got {u}"
        );
    }

    #[test]
    fn jackknife_errors_are_positive() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = cubic3d_neighbours(4);
        let mut lat = XyLattice::new(nb);
        lat.randomise(&mut rng);
        let obs = measure(&mut lat, (2.2_f64).recip(), 1.0, 100, 200, &mut rng);
        assert!(obs.energy_err >= 0.0);
        assert!(obs.magnetisation_err >= 0.0);
        assert!(obs.heat_capacity_err >= 0.0);
        assert!(obs.susceptibility_err >= 0.0);
    }
}
