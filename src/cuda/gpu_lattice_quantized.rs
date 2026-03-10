use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use half::f16;
use rand::Rng;
use std::sync::Arc;

use crate::cuda::reduce_gpu;
use crate::cuda::wolff::{wolff_cluster_flip_angle, wolff_cluster_flip_continuous};

const CONTINUOUS_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/continuous_spin_kernel.ptx"));
const BLOCK_SIZE: u32 = 256;

// ============================================================
// FP16 Heisenberg lattice — 3 × f16 per spin (48 bits vs 96)
// ============================================================

pub struct Fp16HeisenbergLattice {
    pub n: usize,
    pub device: Arc<CudaDevice>,
    pub spins: CudaSlice<u16>, // __half stored as u16
    rng_states: CudaSlice<u8>,
    n_threads: u32,
    reduce_partial_mx: CudaSlice<f32>,
    reduce_partial_my: CudaSlice<f32>,
    reduce_partial_mz: CudaSlice<f32>,
    reduce_partial_e: CudaSlice<f32>,
}

impl Fp16HeisenbergLattice {
    pub fn new(n: usize, seed: u64, device: Arc<CudaDevice>) -> anyhow::Result<Self> {
        device.load_ptx(
            CONTINUOUS_PTX.into(),
            "continuous",
            &[
                "continuous_metropolis_fp16_kernel",
                "continuous_overrelax_fp16_kernel",
                "init_continuous_rng_kernel",
            ],
        )?;
        reduce_gpu::load_reduce_kernels(&device)?;

        let n_sites = n * n * n;
        let n_comp = 3;
        let n_threads = (n_sites / 2) as u32;

        // Random initial spins on host (unit vectors), stored as f16
        let mut host_spins = vec![f16::ZERO; n_sites * n_comp];
        let mut rng = rand::thread_rng();
        for i in 0..n_sites {
            let z: f32 = rng.gen_range(-1.0..1.0);
            let phi: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            let r = (1.0 - z * z).sqrt();
            host_spins[i * n_comp] = f16::from_f32(r * phi.cos());
            host_spins[i * n_comp + 1] = f16::from_f32(r * phi.sin());
            host_spins[i * n_comp + 2] = f16::from_f32(z);
        }

        // Upload as u16 (same bit representation as __half)
        let host_u16: Vec<u16> = host_spins.iter().map(|h| h.to_bits()).collect();
        let spins = device.htod_sync_copy(&host_u16)?;

        let rng_states = device.alloc_zeros::<u8>((n_threads as usize) * 64)?;

        let n_blocks = ((n_sites as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let reduce_partial_mx = device.alloc_zeros::<f32>(n_blocks as usize)?;
        let reduce_partial_my = device.alloc_zeros::<f32>(n_blocks as usize)?;
        let reduce_partial_mz = device.alloc_zeros::<f32>(n_blocks as usize)?;
        let reduce_partial_e = device.alloc_zeros::<f32>(n_blocks as usize)?;

        let mut lat = Self {
            n,
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

    pub fn sweep(
        &mut self,
        beta: f32,
        j: f32,
        delta: f32,
        n_overrelax: usize,
    ) -> anyhow::Result<()> {
        let grid = (self.n_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let n = self.n as i32;
        let nc = 3i32;

        // Overrelaxation sweeps
        for _ in 0..n_overrelax {
            for parity in [0i32, 1i32] {
                let f = self
                    .device
                    .get_func("continuous", "continuous_overrelax_fp16_kernel")
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
                .get_func("continuous", "continuous_metropolis_fp16_kernel")
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
        Ok(())
    }

    pub fn measure_gpu(&mut self, j: f32) -> anyhow::Result<(f64, f64, f64, f64)> {
        let n_sites = self.n * self.n * self.n;
        let n_blocks = ((n_sites as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let nc = 3i32;

        // Magnetisation
        let shared_mag = BLOCK_SIZE as u32 * 4 * 3;
        let f_mag = self.device.get_func("reduce", "reduce_mag_fp16").unwrap();
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
                    nc,
                ),
            )?;
        }

        // Energy
        let shared_e = BLOCK_SIZE as u32 * 4;
        let f_energy = self
            .device
            .get_func("reduce", "reduce_energy_fp16")
            .unwrap();
        unsafe {
            f_energy.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared_e,
                },
                (&self.spins, &mut self.reduce_partial_e, self.n as i32, nc, j),
            )?;
        }

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

    /// Wolff embedding cluster step — copy to host as f32, BFS, copy back as f16.
    pub fn wolff_step<R: Rng>(&mut self, beta: f32, j: f32, rng: &mut R) -> anyhow::Result<()> {
        let spins_u16 = self.device.dtoh_sync_copy(&self.spins)?;
        let mut spins_f32: Vec<f32> = spins_u16
            .iter()
            .map(|&bits| f16::from_bits(bits).to_f32())
            .collect();

        wolff_cluster_flip_continuous(&mut spins_f32, self.n, 3, beta, j, rng);

        let spins_u16_out: Vec<u16> = spins_f32
            .iter()
            .map(|&v| f16::from_f32(v).to_bits())
            .collect();
        self.device
            .htod_sync_copy_into(&spins_u16_out, &mut self.spins)?;
        Ok(())
    }
}

// ============================================================
// XY angle-only lattice — 1 × f16 per spin (16 bits vs 64)
// ============================================================

pub struct XyAngleLattice {
    pub n: usize,
    pub device: Arc<CudaDevice>,
    pub angles: CudaSlice<u16>, // __half angle per spin
    rng_states: CudaSlice<u8>,
    n_threads: u32,
    reduce_partial_mx: CudaSlice<f32>,
    reduce_partial_my: CudaSlice<f32>,
    reduce_partial_e: CudaSlice<f32>,
}

impl XyAngleLattice {
    pub fn new(n: usize, seed: u64, device: Arc<CudaDevice>) -> anyhow::Result<Self> {
        device.load_ptx(
            CONTINUOUS_PTX.into(),
            "continuous",
            &[
                "xy_angle_metropolis_kernel",
                "xy_angle_overrelax_kernel",
                "init_continuous_rng_kernel",
            ],
        )?;
        reduce_gpu::load_reduce_kernels(&device)?;

        let n_sites = n * n * n;
        let n_threads = (n_sites / 2) as u32;

        // Random initial angles
        let mut rng = rand::thread_rng();
        let host_angles: Vec<u16> = (0..n_sites)
            .map(|_| {
                let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
                f16::from_f32(angle).to_bits()
            })
            .collect();

        let angles = device.htod_sync_copy(&host_angles)?;
        let rng_states = device.alloc_zeros::<u8>((n_threads as usize) * 64)?;

        let n_blocks = ((n_sites as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let reduce_partial_mx = device.alloc_zeros::<f32>(n_blocks as usize)?;
        let reduce_partial_my = device.alloc_zeros::<f32>(n_blocks as usize)?;
        let reduce_partial_e = device.alloc_zeros::<f32>(n_blocks as usize)?;

        let mut lat = Self {
            n,
            device,
            angles,
            rng_states,
            n_threads,
            reduce_partial_mx,
            reduce_partial_my,
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

    pub fn sweep(
        &mut self,
        beta: f32,
        j: f32,
        delta: f32,
        n_overrelax: usize,
    ) -> anyhow::Result<()> {
        let grid = (self.n_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let n = self.n as i32;

        // Overrelaxation
        for _ in 0..n_overrelax {
            for parity in [0i32, 1i32] {
                let f = self
                    .device
                    .get_func("continuous", "xy_angle_overrelax_kernel")
                    .unwrap();
                unsafe {
                    f.launch(
                        LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&mut self.angles, n, j, parity),
                    )?;
                }
            }
        }

        // Metropolis
        for parity in [0i32, 1i32] {
            let f = self
                .device
                .get_func("continuous", "xy_angle_metropolis_kernel")
                .unwrap();
            unsafe {
                f.launch(
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (BLOCK_SIZE, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &mut self.angles,
                        &mut self.rng_states,
                        n,
                        beta,
                        j,
                        delta,
                        parity,
                    ),
                )?;
            }
        }
        Ok(())
    }

    /// Returns (energy, mx, my, 0.0) — mz is always 0 for XY.
    pub fn measure_gpu(&mut self, j: f32) -> anyhow::Result<(f64, f64, f64, f64)> {
        let n_sites = self.n * self.n * self.n;
        let n_blocks = ((n_sites as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Magnetisation
        let shared_mag = BLOCK_SIZE as u32 * 4 * 2;
        let f_mag = self
            .device
            .get_func("reduce", "reduce_mag_xy_angle")
            .unwrap();
        unsafe {
            f_mag.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared_mag,
                },
                (
                    &self.angles,
                    &mut self.reduce_partial_mx,
                    &mut self.reduce_partial_my,
                    n_sites as i32,
                ),
            )?;
        }

        // Energy
        let shared_e = BLOCK_SIZE as u32 * 4;
        let f_energy = self
            .device
            .get_func("reduce", "reduce_energy_xy_angle")
            .unwrap();
        unsafe {
            f_energy.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared_e,
                },
                (
                    &self.angles,
                    &mut self.reduce_partial_e,
                    self.n as i32,
                    j,
                ),
            )?;
        }

        let mx_host = self.device.dtoh_sync_copy(&self.reduce_partial_mx)?;
        let my_host = self.device.dtoh_sync_copy(&self.reduce_partial_my)?;
        let e_host = self.device.dtoh_sync_copy(&self.reduce_partial_e)?;

        let mx: f64 = mx_host.iter().map(|&x| x as f64).sum();
        let my: f64 = my_host.iter().map(|&x| x as f64).sum();
        let energy: f64 = e_host.iter().map(|&x| x as f64).sum();

        Ok((energy, mx, my, 0.0))
    }

    /// Wolff embedding for XY: project onto random axis, BFS, reflect angles.
    pub fn wolff_step<R: Rng>(&mut self, beta: f32, j: f32, rng: &mut R) -> anyhow::Result<()> {
        let angles_u16 = self.device.dtoh_sync_copy(&self.angles)?;
        let mut angles_f32: Vec<f32> = angles_u16
            .iter()
            .map(|&bits| f16::from_bits(bits).to_f32())
            .collect();

        wolff_cluster_flip_angle(&mut angles_f32, self.n, beta, j, rng);

        let angles_u16_out: Vec<u16> = angles_f32
            .iter()
            .map(|&v| f16::from_f32(v).to_bits())
            .collect();
        self.device
            .htod_sync_copy_into(&angles_u16_out, &mut self.angles)?;
        Ok(())
    }
}
