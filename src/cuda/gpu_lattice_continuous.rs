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
    // Pre-allocated reduction buffers (avoid per-sample allocation)
    reduce_partial_mx: CudaSlice<f32>,
    reduce_partial_my: CudaSlice<f32>,
    reduce_partial_mz: CudaSlice<f32>,
    reduce_partial_e: CudaSlice<f32>,
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
        // sizeof(curandStatePhilox4_32_10) = 64 bytes per thread
        let rng_states = device.alloc_zeros::<u8>((n_threads as usize) * 64)?;

        // Pre-allocate reduction buffers
        let n_blocks = ((n_sites as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let reduce_partial_mx = device.alloc_zeros::<f32>(n_blocks as usize)?;
        let reduce_partial_my = device.alloc_zeros::<f32>(n_blocks as usize)?;
        let reduce_partial_mz = device.alloc_zeros::<f32>(n_blocks as usize)?;
        let reduce_partial_e = device.alloc_zeros::<f32>(n_blocks as usize)?;

        let mut lat = Self {
            n,
            n_comp,
            device,
            spins,
            rng_states,
            n_threads,
            reduce_partial_mx,
            reduce_partial_my,
            reduce_partial_mz,
            reduce_partial_e,
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
    /// Does NOT synchronize — caller should sync only when needed (before measurement).
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

        // No synchronize here — reduction kernels or dtoh_sync_copy will
        // implicitly wait for the sweep kernels to finish on the same stream.
        Ok(())
    }

    /// Compute (E, mx, my, mz) using GPU reduction with pre-allocated buffers.
    /// No host↔device spin transfer.
    pub fn measure_gpu(&mut self, j: f32) -> anyhow::Result<(f64, f64, f64, f64)> {
        let n_sites = self.n * self.n * self.n;
        let n_blocks = ((n_sites as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // --- Magnetisation (3-component) ---
        let shared_mag = BLOCK_SIZE as u32 * 4 * 3; // 3 float arrays in shared mem
        let f_mag = self
            .device
            .get_func("reduce", "reduce_mag_continuous")
            .unwrap();
        unsafe {
            f_mag.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared_mag,
                },
                (
                    &self.spins,
                    &mut self.reduce_partial_mx,
                    &mut self.reduce_partial_my,
                    &mut self.reduce_partial_mz,
                    n_sites as i32,
                    self.n_comp as i32,
                ),
            )?;
        }

        // --- Energy ---
        let shared_e = BLOCK_SIZE as u32 * 4;
        let f_energy = self
            .device
            .get_func("reduce", "reduce_energy_continuous")
            .unwrap();
        unsafe {
            f_energy.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared_e,
                },
                (
                    &self.spins,
                    &mut self.reduce_partial_e,
                    self.n as i32,
                    self.n_comp as i32,
                    j,
                ),
            )?;
        }

        // Transfer partial sums (small: ~n_blocks floats each, not N³)
        let mx_host = self.device.dtoh_sync_copy(&self.reduce_partial_mx)?;
        let my_host = self.device.dtoh_sync_copy(&self.reduce_partial_my)?;
        let mz_host = self.device.dtoh_sync_copy(&self.reduce_partial_mz)?;
        let e_host = self.device.dtoh_sync_copy(&self.reduce_partial_e)?;

        let mx: f64 = mx_host.iter().map(|&x| x as f64).sum();
        let my: f64 = my_host.iter().map(|&x| x as f64).sum();
        let mz: f64 = mz_host.iter().map(|&x| x as f64).sum();
        let energy: f64 = e_host.iter().map(|&x| x as f64).sum();

        Ok((energy, mx, my, mz))
    }

    /// Legacy measure_raw — now delegates to measure_gpu with pre-allocated buffers.
    pub fn measure_raw(&mut self) -> anyhow::Result<(f64, f64, f64, f64)> {
        self.measure_gpu(1.0)
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
