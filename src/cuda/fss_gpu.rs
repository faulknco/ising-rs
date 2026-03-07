use crate::cuda::lattice_gpu::LatticeGpu;
use crate::fss::FssConfig;
use crate::observables::Observables;

/// Run FSS temperature sweeps on the GPU.
/// Same logic as the CPU `run_fss`, but uses LatticeGpu (CUDA checkerboard Metropolis).
pub fn run_fss_gpu(config: &FssConfig) -> Vec<(usize, Vec<Observables>)> {
    config.sizes.iter().map(|&n| {
        eprintln!("FSS GPU: N={n}");
        let seed = config.seed.wrapping_add(n as u64);
        let mut gpu = LatticeGpu::new(n, seed)
            .expect("failed to create GPU lattice");

        let mut results = Vec::with_capacity(config.t_steps);

        // Sweep high T → low T (annealing)
        for step in (0..config.t_steps).rev() {
            let t = config.t_min
                + (config.t_max - config.t_min) * step as f64 / (config.t_steps - 1) as f64;
            let beta = (1.0 / t) as f32;

            // Warm up
            gpu.warm_up(beta, config.j as f32, config.h as f32, config.warmup_sweeps)
                .expect("GPU warm-up failed");

            // Measure
            let obs = gpu.measure(t, config.j, config.h, config.sample_sweeps)
                .expect("GPU measure failed");
            results.push(obs);
        }

        results.reverse();
        (n, results)
    }).collect()
}
