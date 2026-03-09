use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use crate::observables::Observables;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
const BLOCK_SIZE: u32 = 256;

pub struct LatticeGpu {
    pub n: usize,
    device: Arc<CudaDevice>,
    spins: CudaSlice<i8>,
    // Philox RNG state: 16 bytes per thread
    rng_states: CudaSlice<u8>,
    n_threads: u32,
    // Pre-allocated reduction buffers (avoid per-sample allocation)
    reduce_partial_mag: CudaSlice<f32>,
    reduce_partial_e: CudaSlice<f32>,
}

impl LatticeGpu {
    pub fn new(n: usize, seed: u64) -> anyhow::Result<Self> {
        let device = CudaDevice::new(0)?;
        device.load_ptx(
            PTX.into(),
            "ising",
            &[
                "metropolis_sweep_kernel",
                "init_rng_kernel",
                "sum_spins_kernel",
            ],
        )?;

        let size = n * n * n;
        let n_threads = (size / 2) as u32;

        // Random initial spins on host
        let host_spins: Vec<i8> = (0..size)
            .map(|_| if rand::random::<bool>() { 1i8 } else { -1i8 })
            .collect();

        let spins = device.htod_sync_copy(&host_spins)?;
        // sizeof(curandStatePhilox4_32_10) = 64 bytes per thread
        let rng_states = device.alloc_zeros::<u8>((n_threads as usize) * 64)?;

        // Pre-allocate reduction buffers
        let n_blocks = ((size as u32) + 256 - 1) / 256;
        let reduce_partial_mag = device.alloc_zeros::<f32>(n_blocks as usize)?;
        let reduce_partial_e = device.alloc_zeros::<f32>(n_blocks as usize)?;

        // Load reduction kernels once
        crate::cuda::reduce_gpu::load_reduce_kernels(&device)?;

        let mut gpu = Self {
            n,
            device,
            spins,
            rng_states,
            n_threads,
            reduce_partial_mag,
            reduce_partial_e,
        };
        gpu.init_rng(seed)?;
        Ok(gpu)
    }

    fn init_rng(&mut self, seed: u64) -> anyhow::Result<()> {
        let grid = (self.n_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let f = self.device.get_func("ising", "init_rng_kernel").unwrap();
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

    /// Run one full Metropolis sweep (black pass then white pass).
    pub fn step(&mut self, beta: f32, j: f32, h: f32) -> anyhow::Result<()> {
        let grid = (self.n_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let n = self.n as i32;
        for parity in [0i32, 1i32] {
            let f = self
                .device
                .get_func("ising", "metropolis_sweep_kernel")
                .unwrap();
            unsafe {
                f.launch(
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (BLOCK_SIZE, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut self.spins, &mut self.rng_states, n, beta, j, h, parity),
                )?;
            }
        }
        // No synchronize — next kernel launch or dtoh_sync_copy will
        // implicitly wait for these kernels on the same stream.
        Ok(())
    }

    /// Copy spins from device to host.
    pub fn get_spins(&self) -> anyhow::Result<Vec<i8>> {
        Ok(self.device.dtoh_sync_copy(&self.spins)?)
    }

    /// Measure E and |M| per spin using GPU reduction with pre-allocated buffers.
    /// No host↔device spin transfer.
    pub fn measure_gpu(&mut self, j: f32) -> anyhow::Result<(f64, f64)> {
        let n_sites = self.n * self.n * self.n;
        let n3 = n_sites as f64;
        let n_blocks = ((n_sites as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let shared = BLOCK_SIZE as u32 * 4;

        // Magnetisation
        let f_mag = self.device.get_func("reduce", "reduce_mag_ising").unwrap();
        unsafe {
            f_mag.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared,
                },
                (&self.spins, &mut self.reduce_partial_mag, n_sites as i32),
            )?;
        }
        let mag_host = self.device.dtoh_sync_copy(&self.reduce_partial_mag)?;
        let total_mag: f64 = mag_host.iter().map(|&x| x as f64).sum();

        // Energy
        let f_energy = self
            .device
            .get_func("reduce", "reduce_energy_ising")
            .unwrap();
        unsafe {
            f_energy.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared,
                },
                (&self.spins, &mut self.reduce_partial_e, self.n as i32, j),
            )?;
        }
        let energy_host = self.device.dtoh_sync_copy(&self.reduce_partial_e)?;
        let total_energy: f64 = energy_host.iter().map(|&x| x as f64).sum();

        Ok((total_energy / n3, total_mag.abs() / n3))
    }

    /// Measure total energy using GPU reduction (for replica exchange).
    pub fn energy_gpu(&mut self, j: f32) -> anyhow::Result<f64> {
        let n_sites = self.n * self.n * self.n;
        let n_blocks = ((n_sites as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let shared = BLOCK_SIZE as u32 * 4;

        let f_energy = self
            .device
            .get_func("reduce", "reduce_energy_ising")
            .unwrap();
        unsafe {
            f_energy.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared,
                },
                (&self.spins, &mut self.reduce_partial_e, self.n as i32, j),
            )?;
        }
        let energy_host = self.device.dtoh_sync_copy(&self.reduce_partial_e)?;
        let total_energy: f64 = energy_host.iter().map(|&x| x as f64).sum();
        Ok(total_energy)
    }

    /// Warm up: run `sweeps` Metropolis sweeps on GPU, discard results.
    pub fn warm_up(&mut self, beta: f32, j: f32, h: f32, sweeps: usize) -> anyhow::Result<()> {
        for _ in 0..sweeps {
            self.step(beta, j, h)?;
        }
        Ok(())
    }

    /// Measure observables by running `samples` GPU sweeps, computing E and M
    /// host-side after each sweep.
    pub fn measure(
        &mut self,
        temperature: f64,
        j_coupling: f64,
        h_field: f64,
        samples: usize,
    ) -> anyhow::Result<Observables> {
        let beta = 1.0 / temperature;
        let n3 = (self.n * self.n * self.n) as f64;

        let mut sum_e = 0.0_f64;
        let mut sum_e2 = 0.0_f64;
        let mut sum_m = 0.0_f64;
        let mut sum_m2 = 0.0_f64;
        let mut sum_m4 = 0.0_f64;
        let mut sum_m_signed = 0.0_f64;
        let mut sum_m_signed2 = 0.0_f64;

        for _ in 0..samples {
            self.step(beta as f32, j_coupling as f32, h_field as f32)?;

            let spins = self.get_spins()?;
            let (e, m) = energy_magnetisation_host(&spins, self.n, j_coupling, h_field);

            let e_per = e / n3;
            let m_per = (m / n3).abs();
            let m_signed = m / n3;

            sum_e += e_per;
            sum_e2 += e_per * e_per;
            sum_m += m_per;
            sum_m2 += m_per * m_per;
            sum_m4 += m_per * m_per * m_per * m_per;
            sum_m_signed += m_signed;
            sum_m_signed2 += m_signed * m_signed;
        }

        let s = samples as f64;
        let avg_e = sum_e / s;
        let avg_e2 = sum_e2 / s;
        let avg_m = sum_m / s;
        let avg_m2 = sum_m2 / s;
        let avg_m4 = sum_m4 / s;
        let avg_m_signed = sum_m_signed / s;
        let avg_m_signed2 = sum_m_signed2 / s;

        let cv = beta * beta * (avg_e2 - avg_e * avg_e) * n3;
        let chi = beta * (avg_m_signed2 - avg_m_signed * avg_m_signed) * n3;

        Ok(Observables {
            temperature,
            energy: avg_e,
            magnetisation: avg_m,
            heat_capacity: cv,
            susceptibility: chi,
            m2: avg_m2,
            m4: avg_m4,
        })
    }

    /// Domain wall density: fraction of neighbour pairs with opposite spins.
    pub fn domain_wall_density(&self) -> anyhow::Result<f64> {
        let spins = self.get_spins()?;
        let n = self.n;
        let mut walls = 0usize;
        let mut bonds = 0usize;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let idx = i * n * n + j * n + k;
                    let s = spins[idx];
                    // Only count forward neighbours to avoid double-counting
                    let fwd = [
                        ((i + 1) % n) * n * n + j * n + k,
                        i * n * n + ((j + 1) % n) * n + k,
                        i * n * n + j * n + (k + 1) % n,
                    ];
                    for &nb in &fwd {
                        bonds += 1;
                        if spins[nb] != s {
                            walls += 1;
                        }
                    }
                }
            }
        }
        Ok(walls as f64 / bonds as f64)
    }
}

/// Compute total energy and magnetisation from a host-side spin array.
/// 3D cubic lattice with periodic boundaries.
fn energy_magnetisation_host(spins: &[i8], n: usize, j: f64, h: f64) -> (f64, f64) {
    let mut e = 0.0_f64;
    let mut m = 0.0_f64;
    for i in 0..n {
        for jj in 0..n {
            for k in 0..n {
                let idx = i * n * n + jj * n + k;
                let s = spins[idx] as f64;
                // 6 neighbours, periodic
                let nb_sum = spins[((i + 1) % n) * n * n + jj * n + k] as f64
                    + spins[((i + n - 1) % n) * n * n + jj * n + k] as f64
                    + spins[i * n * n + ((jj + 1) % n) * n + k] as f64
                    + spins[i * n * n + ((jj + n - 1) % n) * n + k] as f64
                    + spins[i * n * n + jj * n + (k + 1) % n] as f64
                    + spins[i * n * n + jj * n + (k + n - 1) % n] as f64;
                e += -j * s * nb_sum / 2.0 - h * s;
                m += s;
            }
        }
    }
    (e, m)
}
