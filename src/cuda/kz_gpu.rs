use super::lattice_gpu::LatticeGpu;

const TC_3D: f64 = 4.5115;

/// Run a single KZ quench on GPU.
/// Protocol: (1) warm up at t_start, (2) linear ramp from t_start to Tc over tau_q sweeps,
/// (3) snap-freeze at very low T for a few sweeps to remove thermal noise without coarsening.
/// Returns domain wall density after freeze.
pub fn run_kz_gpu(
    n: usize,
    j: f64,
    t_start: f64,
    tau_q: usize,
    warmup_sweeps: usize,
    seed: u64,
) -> anyhow::Result<f64> {
    let mut gpu = LatticeGpu::new(n, seed)?;

    // Warmup at t_start (disordered phase)
    let beta_start = 1.0 / t_start;
    gpu.warm_up(beta_start as f32, j as f32, 0.0, warmup_sweeps)?;

    // Linear temperature ramp from t_start down to Tc
    for step in 0..tau_q {
        let frac = step as f64 / tau_q as f64;
        let t = t_start + (TC_3D - t_start) * frac;
        let beta = 1.0 / t.max(0.01);
        gpu.step(beta as f32, j as f32, 0.0)?;
    }

    // Snap-freeze: 5 sweeps at very low T to clean thermal noise
    let beta_freeze = 100.0_f32; // T = 0.01
    for _ in 0..5 {
        gpu.step(beta_freeze, j as f32, 0.0)?;
    }

    gpu.domain_wall_density()
}

/// Run KZ sweep over multiple tau_q values, averaging over n_trials.
/// Returns Vec<(tau_q, avg_rho)>.
pub fn run_kz_sweep_gpu(
    n: usize,
    j: f64,
    t_start: f64,
    _t_end: f64,
    tau_q_values: &[usize],
    n_trials: usize,
    warmup_sweeps: usize,
    base_seed: u64,
) -> Vec<(usize, f64)> {
    eprintln!("  Protocol: ramp T={t_start} -> Tc={TC_3D}, then snap-freeze");
    tau_q_values
        .iter()
        .map(|&tau_q| {
            eprintln!("  tau_q = {tau_q} ...");
            let rho_avg: f64 = (0..n_trials)
                .map(|trial| {
                    let seed = base_seed
                        .wrapping_add(tau_q as u64)
                        .wrapping_add(trial as u64);
                    run_kz_gpu(n, j, t_start, tau_q, warmup_sweeps, seed)
                        .expect("GPU KZ quench failed")
                })
                .sum::<f64>()
                / n_trials as f64;
            eprintln!("  tau_q = {tau_q}: rho = {rho_avg:.6}");
            (tau_q, rho_avg)
        })
        .collect()
}
