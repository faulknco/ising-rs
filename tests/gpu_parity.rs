//! GPU statistical parity tests: verify optimized paths produce correct physics.
//!
//! Run with: cargo test --features cuda -- --ignored
//!
//! These are STATISTICAL parity tests. MSC and Wolff use different data layouts
//! and RNG consumption patterns than single-spin Metropolis. The correct gate is:
//!   - same detailed-balance target distribution
//!   - equilibrium observables agree within statistical tolerance
//!
//! Each test runs enough sweeps to equilibrate, then measures observables and
//! compares against the baseline single-spin Metropolis path.

#[cfg(feature = "cuda")]
mod gpu_parity_tests {
    use cudarc::driver::CudaDevice;
    use ising_rs::cuda::lattice_gpu::LatticeGpu;
    use ising_rs::cuda::msc_lattice::MscLattice;
    use ising_rs::cuda::wolff_gpu::WolffGpuLattice;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    /// Compute mean and standard error of a slice.
    fn mean_stderr(data: &[f64]) -> (f64, f64) {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, (var / n).sqrt())
    }

    /// Assert two means agree within combined 3-sigma tolerance.
    fn assert_statistical_match(name: &str, a_mean: f64, a_err: f64, b_mean: f64, b_err: f64) {
        let combined_err = (a_err * a_err + b_err * b_err).sqrt();
        let diff = (a_mean - b_mean).abs();
        let tolerance = 3.0 * combined_err;
        assert!(
            diff < tolerance,
            "{name}: |{a_mean:.6} - {b_mean:.6}| = {diff:.6} > 3sigma = {tolerance:.6}"
        );
    }

    /// Run single-spin Metropolis (LatticeGpu) and collect per-sweep (E/spin, |M|/spin).
    fn collect_single_spin(
        n: usize,
        beta: f32,
        j: f32,
        warmup: usize,
        measure: usize,
        seed: u64,
    ) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
        let mut gpu = LatticeGpu::new(n, seed)?;
        gpu.warm_up(beta, j, 0.0, warmup)?;

        let mut energies = Vec::with_capacity(measure);
        let mut mags = Vec::with_capacity(measure);
        for _ in 0..measure {
            gpu.step(beta, j, 0.0)?;
            let (e, m) = gpu.measure_gpu(j)?;
            energies.push(e);
            mags.push(m);
        }
        Ok((energies, mags))
    }

    /// Run MSC Metropolis and collect per-sweep (E/spin, |M|/spin).
    fn collect_msc(
        n: usize,
        beta: f32,
        j: f32,
        warmup: usize,
        measure: usize,
        seed: u64,
    ) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
        let device = CudaDevice::new(0)?;
        let mut msc = MscLattice::new(n, seed, device)?;
        msc.randomise()?;
        msc.warm_up(beta, j, warmup)?;

        let mut energies = Vec::with_capacity(measure);
        let mut mags = Vec::with_capacity(measure);
        for _ in 0..measure {
            msc.step(beta, j)?;
            let (e, m) = msc.measure_gpu(j)?;
            energies.push(e);
            mags.push(m);
        }
        Ok((energies, mags))
    }

    /// Run GPU Wolff and collect per-step (E/spin, |M|/spin).
    fn collect_wolff_gpu(
        n: usize,
        beta: f32,
        j: f32,
        warmup: usize,
        measure: usize,
        seed: u64,
    ) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
        let device = CudaDevice::new(0)?;
        let mut wolff = WolffGpuLattice::new(n, seed, device)?;
        let mut rng = SmallRng::seed_from_u64(seed);
        wolff.warm_up(beta, j, warmup, &mut rng)?;

        let mut energies = Vec::with_capacity(measure);
        let mut mags = Vec::with_capacity(measure);
        for _ in 0..measure {
            wolff.step(beta, j, &mut rng)?;
            let (e, m) = wolff.measure_gpu(j)?;
            energies.push(e);
            mags.push(m);
        }
        Ok((energies, mags))
    }

    // -----------------------------------------------------------------------
    // MSC parity tests at three temperatures
    // -----------------------------------------------------------------------

    #[test]
    #[ignore] // Requires --features cuda and GPU hardware
    fn msc_statistical_parity_near_tc() {
        // MSC and single-spin Metropolis should produce the same equilibrium
        // observables at T ~ 4.51 (near Tc for 3D Ising).
        let n = 32;
        let t = 4.51_f32;
        let beta = 1.0 / t;
        let j = 1.0_f32;
        let warmup = 3000;
        let measure = 10000;

        let (e_ss, m_ss) = collect_single_spin(n, beta, j, warmup, measure, 42).unwrap();
        let (e_msc, m_msc) = collect_msc(n, beta, j, warmup, measure, 123).unwrap();

        let (e_ss_mean, e_ss_err) = mean_stderr(&e_ss);
        let (e_msc_mean, e_msc_err) = mean_stderr(&e_msc);
        let (m_ss_mean, m_ss_err) = mean_stderr(&m_ss);
        let (m_msc_mean, m_msc_err) = mean_stderr(&m_msc);

        assert_statistical_match("E/spin near Tc", e_ss_mean, e_ss_err, e_msc_mean, e_msc_err);
        assert_statistical_match("|M|/spin near Tc", m_ss_mean, m_ss_err, m_msc_mean, m_msc_err);
    }

    #[test]
    #[ignore]
    fn msc_statistical_parity_low_t() {
        // T = 3.0 (ordered phase): both paths should find near-ground-state
        // energy E ~ -3J per spin.
        let n = 32;
        let t = 3.0_f32;
        let beta = 1.0 / t;
        let j = 1.0_f32;
        let warmup = 3000;
        let measure = 10000;

        let (e_ss, m_ss) = collect_single_spin(n, beta, j, warmup, measure, 42).unwrap();
        let (e_msc, m_msc) = collect_msc(n, beta, j, warmup, measure, 123).unwrap();

        let (e_ss_mean, e_ss_err) = mean_stderr(&e_ss);
        let (e_msc_mean, e_msc_err) = mean_stderr(&e_msc);
        let (m_ss_mean, m_ss_err) = mean_stderr(&m_ss);
        let (m_msc_mean, m_msc_err) = mean_stderr(&m_msc);

        assert_statistical_match("E/spin low T", e_ss_mean, e_ss_err, e_msc_mean, e_msc_err);
        assert_statistical_match("|M|/spin low T", m_ss_mean, m_ss_err, m_msc_mean, m_msc_err);
    }

    #[test]
    #[ignore]
    fn msc_statistical_parity_high_t() {
        // T = 6.0 (disordered phase): both paths should find E ~ 0, |M| ~ 0.
        let n = 32;
        let t = 6.0_f32;
        let beta = 1.0 / t;
        let j = 1.0_f32;
        let warmup = 3000;
        let measure = 10000;

        let (e_ss, m_ss) = collect_single_spin(n, beta, j, warmup, measure, 42).unwrap();
        let (e_msc, m_msc) = collect_msc(n, beta, j, warmup, measure, 123).unwrap();

        let (e_ss_mean, e_ss_err) = mean_stderr(&e_ss);
        let (e_msc_mean, e_msc_err) = mean_stderr(&e_msc);
        let (m_ss_mean, m_ss_err) = mean_stderr(&m_ss);
        let (m_msc_mean, m_msc_err) = mean_stderr(&m_msc);

        assert_statistical_match("E/spin high T", e_ss_mean, e_ss_err, e_msc_mean, e_msc_err);
        assert_statistical_match("|M|/spin high T", m_ss_mean, m_ss_err, m_msc_mean, m_msc_err);
    }

    // -----------------------------------------------------------------------
    // GPU Wolff parity
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn wolff_gpu_statistical_parity() {
        // GPU Wolff and single-spin Metropolis should sample the same equilibrium
        // distribution. Use smaller lattice (N=8) since Wolff steps are expensive.
        let n = 8;
        let t = 4.51_f32;
        let beta = 1.0 / t;
        let j = 1.0_f32;
        let warmup = 5000;
        let measure = 10000;

        let (e_ss, m_ss) = collect_single_spin(n, beta, j, warmup, measure, 42).unwrap();
        let (e_wolff, m_wolff) = collect_wolff_gpu(n, beta, j, warmup, measure, 99).unwrap();

        let (e_ss_mean, e_ss_err) = mean_stderr(&e_ss);
        let (e_w_mean, e_w_err) = mean_stderr(&e_wolff);
        let (m_ss_mean, m_ss_err) = mean_stderr(&m_ss);
        let (m_w_mean, m_w_err) = mean_stderr(&m_wolff);

        assert_statistical_match("E/spin Wolff", e_ss_mean, e_ss_err, e_w_mean, e_w_err);
        assert_statistical_match("|M|/spin Wolff", m_ss_mean, m_ss_err, m_w_mean, m_w_err);
    }

    // -----------------------------------------------------------------------
    // Philox RNG sanity check
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn philox_rng_produces_valid_metropolis() {
        // After Philox swap, LatticeGpu should produce physically reasonable
        // observables. Not comparing against old RNG (different sequence),
        // just sanity checking ranges.
        let n = 8;
        let t = 4.51_f32;
        let beta = 1.0 / t;
        let j = 1.0_f32;
        let warmup = 2000;
        let measure = 5000;

        let (energies, mags) = collect_single_spin(n, beta, j, warmup, measure, 42).unwrap();
        let (e_mean, _) = mean_stderr(&energies);
        let (m_mean, _) = mean_stderr(&mags);

        // E/spin must be in [-3, 0] for 3D Ising with J=1
        assert!(
            e_mean > -3.0 && e_mean < 0.0,
            "E/spin = {e_mean:.4} outside [-3, 0]"
        );
        // |M|/spin must be in [0, 1]
        assert!(
            m_mean > 0.0 && m_mean < 1.0,
            "|M|/spin = {m_mean:.4} outside [0, 1]"
        );
    }

    // -----------------------------------------------------------------------
    // MSC ground state (exact, no statistics)
    // -----------------------------------------------------------------------

    #[test]
    #[ignore]
    fn msc_ground_state_energy() {
        // All spins initialized to +1 (init_spins sets all bits to 1).
        // At T -> 0 with no sweeps, E = -3J per spin exactly
        // (each of N^3 sites has 6 neighbours; forward-counting gives
        // 3 bonds per site, each contributing -J).
        let n = 32;
        let j = 1.0_f32;

        let device = CudaDevice::new(0).unwrap();
        let mut msc = MscLattice::new(n, 42, device).unwrap();
        // init_spins sets all bits to 1 (all spins up) -- no randomisation
        let (e_per_spin, m_per_spin) = msc.measure_gpu(j).unwrap();

        let e_tol = 1e-6;
        assert!(
            (e_per_spin - (-3.0)).abs() < e_tol,
            "Ground state E/spin = {e_per_spin:.6}, expected -3.0"
        );
        assert!(
            (m_per_spin - 1.0).abs() < e_tol,
            "Ground state |M|/spin = {m_per_spin:.6}, expected 1.0"
        );
    }
}
