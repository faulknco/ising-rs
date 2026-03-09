use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use crate::cuda::reduce_gpu;

const CONTINUOUS_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/continuous_spin_kernel.ptx"));
const BLOCK_SIZE: u32 = 256;

pub struct ContinuousGpuLattice {
    pub n: usize,
    pub n_comp: usize, // 2 = XY, 3 = Heisenberg
    device: Arc<CudaDevice>,
    pub spins: CudaSlice<f32>,
    rng_states: CudaSlice<u8>,
    n_threads: u32,
}

impl ContinuousGpuLattice {
    pub fn new(
        n: usize,
        n_comp: usize,
        seed: u64,
        device: Arc<CudaDevice>,
    ) -> anyhow::Result<Self> {
        device.load_ptx(
            CONTINUOUS_PTX.into(),
            "continuous",
            &[
                "continuous_metropolis_kernel",
                "continuous_overrelax_kernel",
                "init_continuous_rng_kernel",
            ],
        )?;
        reduce_gpu::load_reduce_kernels(&device)?;

        let n_sites = n * n * n;
        let n_threads = (n_sites / 2) as u32;

        // Random initial spins on host (unit vectors)
        let mut host_spins = vec![0.0f32; n_sites * n_comp];
        let mut rng = rand::thread_rng();
        use rand::Rng;
        for i in 0..n_sites {
            if n_comp == 3 {
                // Random point on S2
                let z: f32 = rng.gen_range(-1.0..1.0);
                let phi: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
                let r = (1.0 - z * z).sqrt();
                host_spins[i * n_comp] = r * phi.cos();
                host_spins[i * n_comp + 1] = r * phi.sin();
                host_spins[i * n_comp + 2] = z;
            } else {
                let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
                host_spins[i * n_comp] = angle.cos();
                host_spins[i * n_comp + 1] = angle.sin();
            }
        }

        let spins = device.htod_sync_copy(&host_spins)?;
        let rng_states = device.alloc_zeros::<u8>((n_threads as usize) * 48)?;

        let mut lat = Self {
            n,
            n_comp,
            device,
            spins,
            rng_states,
            n_threads,
        };
        lat.init_rng(seed)?;
        Ok(lat)
    }

    fn init_rng(&mut self, seed: u64) -> anyhow::Result<()> {
        let grid = (self.n_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let f = self
            .device
            .get_func("continuous", "init_continuous_rng_kernel")
            .unwrap();
        unsafe {
            f.launch(
                LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut self.rng_states, seed, self.n_threads as i32),
            )?;
        }
        self.device.synchronize()?;
        Ok(())
    }

    /// One Metropolis sweep (black + white) with n_or overrelaxation sweeps interleaved.
    pub fn sweep(
        &mut self,
        beta: f32,
        j: f32,
        delta: f32,
        n_overrelax: usize,
    ) -> anyhow::Result<()> {
        let grid = (self.n_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let n = self.n as i32;
        let nc = self.n_comp as i32;

        // Overrelaxation sweeps (no RNG, deterministic)
        for _ in 0..n_overrelax {
            for parity in [0i32, 1i32] {
                let f = self
                    .device
                    .get_func("continuous", "continuous_overrelax_kernel")
                    .unwrap();
                unsafe {
                    f.launch(
                        LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&mut self.spins, n, nc, j, parity),
                    )?;
                }
            }
        }

        // Metropolis sweep
        for parity in [0i32, 1i32] {
            let f = self
                .device
                .get_func("continuous", "continuous_metropolis_kernel")
                .unwrap();
            unsafe {
                f.launch(
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (BLOCK_SIZE, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &mut self.spins,
                        &mut self.rng_states,
                        n,
                        nc,
                        beta,
                        j,
                        delta,
                        parity,
                    ),
                )?;
            }
        }

        self.device.synchronize()?;
        Ok(())
    }

    /// Compute (E, mx, my, mz) using GPU reduction for mag, host for energy.
    pub fn measure_raw(&self) -> anyhow::Result<(f64, f64, f64, f64)> {
        let n_sites = self.n * self.n * self.n;
        let (mx, my, mz) =
            reduce_gpu::reduce_continuous_mag(&self.device, &self.spins, n_sites, self.n_comp)?;
        // Energy: compute on host (reduction kernel for continuous energy is complex)
        let host_spins = self.device.dtoh_sync_copy(&self.spins)?;
        let e = energy_continuous_host(&host_spins, self.n, self.n_comp, 1.0);
        Ok((e, mx, my, mz))
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

fn energy_continuous_host(spins: &[f32], n: usize, n_comp: usize, j: f64) -> f64 {
    let mut e = 0.0_f64;
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let idx = iz * n * n + iy * n + ix;
                // Forward neighbours only
                let fwd = [
                    iz * n * n + iy * n + (ix + 1) % n,
                    iz * n * n + ((iy + 1) % n) * n + ix,
                    ((iz + 1) % n) * n * n + iy * n + ix,
                ];
                for &nb in &fwd {
                    let mut dot = 0.0_f64;
                    for c in 0..n_comp {
                        dot += spins[idx * n_comp + c] as f64 * spins[nb * n_comp + c] as f64;
                    }
                    e -= j * dot;
                }
            }
        }
    }
    e
}
