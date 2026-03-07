use crate::cuda::lattice_gpu::LatticeGpu;
use crate::coarsening::{CoarseningConfig, CoarseningPoint};

/// Run a quench coarsening experiment on the GPU.
/// Same logic as the CPU `run_coarsening`, but uses LatticeGpu.
pub fn run_coarsening_gpu(config: &CoarseningConfig) -> Vec<CoarseningPoint> {
    let mut gpu = LatticeGpu::new(config.n, config.seed)
        .expect("failed to create GPU lattice");

    // Warm up at high T (disorder)
    let beta_high = (1.0 / config.t_high) as f32;
    gpu.warm_up(beta_high, config.j as f32, 0.0, config.warmup_sweeps)
        .expect("GPU warm-up failed");

    // Quench
    let beta_quench = (1.0 / f64::max(config.t_quench, 0.01)) as f32;
    let mut results = Vec::new();

    for step in 0..config.total_steps {
        gpu.step(beta_quench, config.j as f32, 0.0)
            .expect("GPU step failed");

        if step % config.sample_every == 0 {
            let rho = gpu.domain_wall_density()
                .expect("GPU domain_wall_density failed");
            results.push(CoarseningPoint { step, rho });
        }
    }

    results
}
