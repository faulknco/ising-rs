use super::lattice_gpu::LatticeGpu;
use crate::kibble_zurek::{
    linear_ramp_temperature, mean_and_std_err, trial_seed, KzConfig, KzResult, KzSweepPoint,
};
use crate::lattice::Geometry;

/// Run a single KZ quench on GPU.
/// Returns domain wall density after the optional freeze stage.
pub fn run_kz_gpu(config: &KzConfig) -> anyhow::Result<KzResult> {
    if config.geometry != Geometry::Cubic3D {
        anyhow::bail!("GPU KZ only supports cubic geometry");
    }

    let mut gpu = LatticeGpu::new(config.n, config.seed)?;

    // Warmup at t_start (disordered phase)
    let beta_start = 1.0 / config.t_start;
    gpu.warm_up(
        beta_start as f32,
        config.j as f32,
        0.0,
        config.protocol.warmup_sweeps,
    )?;

    // Linear temperature ramp from t_start to t_end.
    for step in 0..config.tau_q {
        let t = linear_ramp_temperature(config.t_start, config.t_end, step, config.tau_q);
        let beta = 1.0 / t.max(0.01);
        gpu.step(beta as f32, config.j as f32, 0.0)?;
    }

    if config.protocol.freeze_sweeps > 0 {
        let beta_freeze = (1.0 / config.protocol.freeze_temperature.max(0.01)) as f32;
        for _ in 0..config.protocol.freeze_sweeps {
            gpu.step(beta_freeze, config.j as f32, 0.0)?;
        }
    }

    Ok(KzResult {
        tau_q: config.tau_q,
        rho_final: gpu.domain_wall_density()?,
    })
}

/// Run KZ sweep over multiple tau_q values, averaging over n_trials.
/// Returns mean domain-wall density and its standard error over `n_trials`.
pub fn run_kz_sweep_gpu(
    n: usize,
    geometry: Geometry,
    j: f64,
    t_start: f64,
    t_end: f64,
    tau_q_values: &[usize],
    n_trials: usize,
    protocol: crate::kibble_zurek::KzProtocol,
    base_seed: u64,
) -> anyhow::Result<Vec<KzSweepPoint>> {
    if geometry != Geometry::Cubic3D {
        anyhow::bail!("GPU KZ only supports cubic geometry");
    }

    eprintln!(
        "  Protocol: ramp T={t_start} -> T={t_end}, {}",
        protocol.describe()
    );
    tau_q_values
        .iter()
        .map(|&tau_q| {
            eprintln!("  tau_q = {tau_q} ...");
            let rho_samples: Vec<f64> = (0..n_trials)
                .map(|trial| {
                    let config = KzConfig {
                        n,
                        geometry,
                        j,
                        t_start,
                        t_end,
                        tau_q,
                        protocol,
                        seed: trial_seed(base_seed, tau_q, trial),
                    };
                    run_kz_gpu(&config).expect("GPU KZ quench failed").rho_final
                })
                .collect();
            let (rho, rho_err) = mean_and_std_err(&rho_samples);
            eprintln!("  tau_q = {tau_q}: rho = {rho:.6} +/- {rho_err:.6}");
            Ok(KzSweepPoint {
                tau_q,
                rho,
                rho_err,
                n_trials,
            })
        })
        .collect()
}
