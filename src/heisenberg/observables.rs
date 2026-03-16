use crate::heisenberg::sweep::combined_sweep;
use crate::heisenberg::{energy_magnetisation_anisotropy, HeisenbergLattice};
use rand::Rng;

/// All measured observables for one temperature point, with jackknife error bars.
///
/// Quantities are per-spin (energy, magnetisation) or extensive (heat_capacity,
/// susceptibility). For anisotropic runs, `mz` and `mxy` are the symmetry-aware
/// order parameters for easy-axis and easy-plane behavior.
#[derive(Debug, Clone)]
pub struct HeisenbergObservables {
    pub temperature: f64,
    pub anisotropy_d: f64,
    /// Mean energy per spin E/N.
    pub energy: f64,
    pub energy_err: f64,
    /// Mean |M|/N (scalar magnetisation per spin).
    pub magnetisation: f64,
    pub magnetisation_err: f64,
    /// Total heat capacity C = β²N(⟨(E/N)²⟩ − ⟨E/N⟩²), where E/N is energy per spin.
    pub heat_capacity: f64,
    pub heat_capacity_err: f64,
    /// Pseudo-susceptibility from the total magnetisation magnitude.
    pub susceptibility: f64,
    pub susceptibility_err: f64,
    /// ⟨m²⟩ (needed for Binder cumulant).
    pub m2: f64,
    pub m2_err: f64,
    /// ⟨m⁴⟩ (needed for Binder cumulant).
    pub m4: f64,
    pub m4_err: f64,
    /// Mean easy-axis order parameter |Mz|/N.
    pub mz: f64,
    pub mz_err: f64,
    pub mz2: f64,
    pub mz2_err: f64,
    pub mz4: f64,
    pub mz4_err: f64,
    pub chi_z: f64,
    pub chi_z_err: f64,
    /// Mean easy-plane order parameter sqrt(Mx² + My²)/N.
    pub mxy: f64,
    pub mxy_err: f64,
    pub mxy2: f64,
    pub mxy2_err: f64,
    pub mxy4: f64,
    pub mxy4_err: f64,
    pub chi_xy: f64,
    pub chi_xy_err: f64,
}

/// Equilibrate a lattice and measure observables with 20-block jackknife error estimation.
#[allow(clippy::too_many_arguments)]
pub fn measure(
    lat: &mut HeisenbergLattice,
    beta: f64,
    j: f64,
    d: f64,
    delta: f64,
    n_overrelax: usize,
    warmup: usize,
    samples: usize,
    rng: &mut impl Rng,
) -> HeisenbergObservables {
    let n = lat.size() as f64;

    for _ in 0..warmup {
        combined_sweep(lat, beta, j, d, delta, n_overrelax, rng);
    }

    let mut e_series = Vec::with_capacity(samples);
    let mut m_series = Vec::with_capacity(samples);
    let mut mz_series = Vec::with_capacity(samples);
    let mut mxy_series = Vec::with_capacity(samples);

    for _ in 0..samples {
        combined_sweep(lat, beta, j, d, delta, n_overrelax, rng);
        let (e, mv) = energy_magnetisation_anisotropy(lat, j, d);
        let m_abs = (mv[0] * mv[0] + mv[1] * mv[1] + mv[2] * mv[2]).sqrt() / n;
        let mz_abs = mv[2].abs() / n;
        let mxy_abs = (mv[0] * mv[0] + mv[1] * mv[1]).sqrt() / n;
        e_series.push(e / n);
        m_series.push(m_abs);
        mz_series.push(mz_abs);
        mxy_series.push(mxy_abs);
    }

    let avg_e = mean(&e_series);
    let avg_m = mean(&m_series);
    let avg_e2 = mean_of_sq(&e_series);
    let avg_m2 = mean_of_sq(&m_series);
    let avg_m4 = mean_of_pow4(&m_series);
    let avg_mz = mean(&mz_series);
    let avg_mz2 = mean_of_sq(&mz_series);
    let avg_mz4 = mean_of_pow4(&mz_series);
    let avg_mxy = mean(&mxy_series);
    let avg_mxy2 = mean_of_sq(&mxy_series);
    let avg_mxy4 = mean_of_pow4(&mxy_series);

    let cv = beta * beta * n * (avg_e2 - avg_e * avg_e);
    let chi = beta * n * (avg_m2 - avg_m * avg_m);
    let chi_z = beta * n * (avg_mz2 - avg_mz * avg_mz);
    let chi_xy = beta * n * (avg_mxy2 - avg_mxy * avg_mxy);

    let n_blocks = 20;
    let block_size = samples / n_blocks;
    debug_assert!(
        samples.is_multiple_of(n_blocks),
        "samples ({samples}) is not divisible by n_blocks ({n_blocks}); {} samples will be silently discarded",
        samples % n_blocks
    );

    let errs = jackknife_errors(
        &e_series,
        &m_series,
        &mz_series,
        &mxy_series,
        beta,
        n,
        n_blocks,
        block_size,
    );

    HeisenbergObservables {
        temperature: 1.0 / beta,
        anisotropy_d: d,
        energy: avg_e,
        energy_err: errs.e_err,
        magnetisation: avg_m,
        magnetisation_err: errs.m_err,
        heat_capacity: cv,
        heat_capacity_err: errs.cv_err,
        susceptibility: chi,
        susceptibility_err: errs.chi_err,
        m2: avg_m2,
        m2_err: errs.m2_err,
        m4: avg_m4,
        m4_err: errs.m4_err,
        mz: avg_mz,
        mz_err: errs.mz_err,
        mz2: avg_mz2,
        mz2_err: errs.mz2_err,
        mz4: avg_mz4,
        mz4_err: errs.mz4_err,
        chi_z,
        chi_z_err: errs.chi_z_err,
        mxy: avg_mxy,
        mxy_err: errs.mxy_err,
        mxy2: avg_mxy2,
        mxy2_err: errs.mxy2_err,
        mxy4: avg_mxy4,
        mxy4_err: errs.mxy4_err,
        chi_xy,
        chi_xy_err: errs.chi_xy_err,
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

struct JackknifeErrors {
    e_err: f64,
    m_err: f64,
    cv_err: f64,
    chi_err: f64,
    m2_err: f64,
    m4_err: f64,
    mz_err: f64,
    chi_z_err: f64,
    mz2_err: f64,
    mz4_err: f64,
    mxy_err: f64,
    chi_xy_err: f64,
    mxy2_err: f64,
    mxy4_err: f64,
}

fn jackknife_errors(
    e: &[f64],
    m: &[f64],
    mz: &[f64],
    mxy: &[f64],
    beta: f64,
    n: f64,
    n_blocks: usize,
    block_size: usize,
) -> JackknifeErrors {
    let mut jk_e = Vec::with_capacity(n_blocks);
    let mut jk_m = Vec::with_capacity(n_blocks);
    let mut jk_cv = Vec::with_capacity(n_blocks);
    let mut jk_chi = Vec::with_capacity(n_blocks);
    let mut jk_m2 = Vec::with_capacity(n_blocks);
    let mut jk_m4 = Vec::with_capacity(n_blocks);
    let mut jk_mz = Vec::with_capacity(n_blocks);
    let mut jk_chi_z = Vec::with_capacity(n_blocks);
    let mut jk_mz2 = Vec::with_capacity(n_blocks);
    let mut jk_mz4 = Vec::with_capacity(n_blocks);
    let mut jk_mxy = Vec::with_capacity(n_blocks);
    let mut jk_chi_xy = Vec::with_capacity(n_blocks);
    let mut jk_mxy2 = Vec::with_capacity(n_blocks);
    let mut jk_mxy4 = Vec::with_capacity(n_blocks);

    let total = n_blocks * block_size;

    for b in 0..n_blocks {
        let lo = b * block_size;
        let hi = lo + block_size;

        let e_jk: Vec<f64> = e[..lo].iter().chain(&e[hi..total]).copied().collect();
        let m_jk: Vec<f64> = m[..lo].iter().chain(&m[hi..total]).copied().collect();
        let mz_jk: Vec<f64> = mz[..lo].iter().chain(&mz[hi..total]).copied().collect();
        let mxy_jk: Vec<f64> = mxy[..lo].iter().chain(&mxy[hi..total]).copied().collect();

        let ae = mean(&e_jk);
        let am = mean(&m_jk);
        let ae2 = mean_of_sq(&e_jk);
        let am2 = mean_of_sq(&m_jk);
        let am4 = mean_of_pow4(&m_jk);
        let amz = mean(&mz_jk);
        let amz2 = mean_of_sq(&mz_jk);
        let amz4 = mean_of_pow4(&mz_jk);
        let amxy = mean(&mxy_jk);
        let amxy2 = mean_of_sq(&mxy_jk);
        let amxy4 = mean_of_pow4(&mxy_jk);

        jk_e.push(ae);
        jk_m.push(am);
        jk_cv.push(beta * beta * n * (ae2 - ae * ae));
        jk_chi.push(beta * n * (am2 - am * am));
        jk_m2.push(am2);
        jk_m4.push(am4);
        jk_mz.push(amz);
        jk_chi_z.push(beta * n * (amz2 - amz * amz));
        jk_mz2.push(amz2);
        jk_mz4.push(amz4);
        jk_mxy.push(amxy);
        jk_chi_xy.push(beta * n * (amxy2 - amxy * amxy));
        jk_mxy2.push(amxy2);
        jk_mxy4.push(amxy4);
    }

    JackknifeErrors {
        e_err: jackknife_std(&jk_e),
        m_err: jackknife_std(&jk_m),
        cv_err: jackknife_std(&jk_cv),
        chi_err: jackknife_std(&jk_chi),
        m2_err: jackknife_std(&jk_m2),
        m4_err: jackknife_std(&jk_m4),
        mz_err: jackknife_std(&jk_mz),
        chi_z_err: jackknife_std(&jk_chi_z),
        mz2_err: jackknife_std(&jk_mz2),
        mz4_err: jackknife_std(&jk_mz4),
        mxy_err: jackknife_std(&jk_mxy),
        chi_xy_err: jackknife_std(&jk_chi_xy),
        mxy2_err: jackknife_std(&jk_mxy2),
        mxy4_err: jackknife_std(&jk_mxy4),
    }
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
    use crate::heisenberg::HeisenbergLattice;
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
        let mut lat = HeisenbergLattice::new(nb);
        let obs = measure(
            &mut lat,
            0.5_f64.recip(),
            1.0,
            0.0,
            0.3,
            5,
            200,
            200,
            &mut rng,
        );
        assert!(obs.magnetisation > 0.85);
    }

    #[test]
    fn measure_high_temp_disordered() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = cubic3d_neighbours(6);
        let mut lat = HeisenbergLattice::new(nb);
        lat.randomise(&mut rng);
        let obs = measure(
            &mut lat,
            (10.0_f64).recip(),
            1.0,
            0.0,
            0.5,
            5,
            200,
            500,
            &mut rng,
        );
        assert!(obs.magnetisation < 0.3);
    }

    #[test]
    fn binder_ground_state() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = cubic3d_neighbours(4);
        let mut lat = HeisenbergLattice::new(nb);
        let obs = measure(
            &mut lat,
            (0.1_f64).recip(),
            1.0,
            0.0,
            0.1,
            5,
            500,
            500,
            &mut rng,
        );
        let u = 1.0 - obs.m4 / (3.0 * obs.m2 * obs.m2);
        assert!((u - 2.0 / 3.0).abs() < 0.05, "got {u}");
    }

    #[test]
    fn jackknife_errors_are_nonnegative() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = cubic3d_neighbours(4);
        let mut lat = HeisenbergLattice::new(nb);
        lat.randomise(&mut rng);
        let obs = measure(
            &mut lat,
            (2.0_f64).recip(),
            1.0,
            0.0,
            0.5,
            5,
            100,
            200,
            &mut rng,
        );
        assert!(obs.energy_err >= 0.0);
        assert!(obs.magnetisation_err >= 0.0);
        assert!(obs.heat_capacity_err >= 0.0);
        assert!(obs.susceptibility_err >= 0.0);
        assert!(obs.mz_err >= 0.0);
        assert!(obs.mxy_err >= 0.0);
    }

    #[test]
    fn easy_axis_and_easy_plane_observables_respond_to_d() {
        let mut rng_axis = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(101);
        let mut rng_plane = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(202);
        let nb = cubic3d_neighbours(4);
        let mut lat_axis = HeisenbergLattice::new(nb.clone());
        let mut lat_plane = HeisenbergLattice::new(nb);
        lat_axis.randomise(&mut rng_axis);
        lat_plane.randomise(&mut rng_plane);

        let axis = measure(
            &mut lat_axis,
            (0.8_f64).recip(),
            1.0,
            4.0,
            0.6,
            0,
            400,
            400,
            &mut rng_axis,
        );
        let plane = measure(
            &mut lat_plane,
            (0.8_f64).recip(),
            1.0,
            -4.0,
            0.6,
            0,
            400,
            400,
            &mut rng_plane,
        );

        assert!(axis.mz > axis.mxy, "easy-axis should favor Mz over Mxy");
        assert!(plane.mxy > plane.mz, "easy-plane should favor Mxy over Mz");
    }
}
