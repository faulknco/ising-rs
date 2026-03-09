use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

const MSC_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/msc_kernel.ptx"));
const BLOCK_SIZE: u32 = 256;

pub struct MscLattice {
    pub n: usize,
    device: Arc<CudaDevice>,
    spins_msc: CudaSlice<u32>,
    rng_states: CudaSlice<u8>,
    n_words: u32,
    partial_mag: CudaSlice<f32>,
    partial_energy: CudaSlice<f32>,
    boltz_probs: CudaSlice<f32>,
}

impl MscLattice {
    pub fn new(n: usize, seed: u64, device: Arc<CudaDevice>) -> anyhow::Result<Self> {
        assert!(n >= 32, "MSC lattice requires n >= 32");
        assert!(n % 32 == 0, "MSC lattice requires n divisible by 32");

        device.load_ptx(
            MSC_PTX.into(),
            "msc",
            &[
                "msc_init_rng_kernel",
                "msc_init_spins_kernel",
                "msc_randomise_kernel",
                "msc_metropolis_kernel",
                "msc_reduce_mag",
                "msc_reduce_energy",
            ],
        )?;

        let n_words = (n * n * (n / 32)) as u32;
        let n_blocks = (n_words + BLOCK_SIZE - 1) / BLOCK_SIZE;

        let spins_msc = device.alloc_zeros::<u32>(n_words as usize)?;
        // Philox RNG: 16 bytes per thread, one thread per word
        let rng_states = device.alloc_zeros::<u8>((n_words as usize) * 16)?;
        let partial_mag = device.alloc_zeros::<f32>(n_blocks as usize)?;
        let partial_energy = device.alloc_zeros::<f32>(n_blocks as usize)?;
        let boltz_probs = device.alloc_zeros::<f32>(7)?;

        let mut lattice = Self {
            n,
            device,
            spins_msc,
            rng_states,
            n_words,
            partial_mag,
            partial_energy,
            boltz_probs,
        };

        lattice.init_rng(seed)?;
        lattice.init_spins()?;

        Ok(lattice)
    }

    pub fn init_rng(&mut self, seed: u64) -> anyhow::Result<()> {
        let grid = (self.n_words + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let f = self.device.get_func("msc", "msc_init_rng_kernel").unwrap();
        unsafe {
            f.launch(
                LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut self.rng_states, seed, self.n_words as i32),
            )?;
        }
        self.device.synchronize()?;
        Ok(())
    }

    pub fn init_spins(&mut self) -> anyhow::Result<()> {
        let grid = (self.n_words + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let f = self
            .device
            .get_func("msc", "msc_init_spins_kernel")
            .unwrap();
        unsafe {
            f.launch(
                LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut self.spins_msc, self.n_words as i32),
            )?;
        }
        self.device.synchronize()?;
        Ok(())
    }

    pub fn randomise(&mut self) -> anyhow::Result<()> {
        let grid = (self.n_words + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let f = self
            .device
            .get_func("msc", "msc_randomise_kernel")
            .unwrap();
        unsafe {
            f.launch(
                LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &mut self.spins_msc,
                    &mut self.rng_states,
                    self.n_words as i32,
                ),
            )?;
        }
        Ok(())
    }

    /// Compute and upload the 7-element Boltzmann probability table for
    /// neighbour-sum values k in -3..=3 (3D cubic lattice).
    pub fn set_temperature(&mut self, beta: f32, j: f32) -> anyhow::Result<()> {
        let mut probs = [0.0f32; 7];
        for k in -3i32..=3 {
            let delta_e = 4.0 * j * k as f32;
            probs[(k + 3) as usize] = if delta_e <= 0.0 {
                1.0
            } else {
                (-beta * delta_e).exp().min(1.0)
            };
        }
        self.boltz_probs = self.device.htod_sync_copy(&probs)?;
        Ok(())
    }

    /// Run one full MSC Metropolis sweep (parity 0 then parity 1).
    pub fn step(&mut self, beta: f32, j: f32) -> anyhow::Result<()> {
        let grid = (self.n_words + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let n = self.n as i32;
        for parity in [0i32, 1i32] {
            let f = self
                .device
                .get_func("msc", "msc_metropolis_kernel")
                .unwrap();
            unsafe {
                f.launch(
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (BLOCK_SIZE, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &mut self.spins_msc,
                        &mut self.rng_states,
                        &self.boltz_probs,
                        n,
                        self.n_words as i32,
                        beta,
                        j,
                        parity,
                    ),
                )?;
            }
        }
        Ok(())
    }

    /// Measure energy per spin and |magnetisation| per spin via GPU reductions.
    pub fn measure_gpu(&mut self, j: f32) -> anyhow::Result<(f64, f64)> {
        let n_sites = (self.n * self.n * self.n) as f64;
        let n_blocks = (self.n_words + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let shared = BLOCK_SIZE * 4;

        // Magnetisation reduction
        let f_mag = self.device.get_func("msc", "msc_reduce_mag").unwrap();
        unsafe {
            f_mag.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared,
                },
                (
                    &self.spins_msc,
                    &mut self.partial_mag,
                    self.n_words as i32,
                ),
            )?;
        }
        let mag_host = self.device.dtoh_sync_copy(&self.partial_mag)?;
        let total_mag: f64 = mag_host.iter().map(|&x| x as f64).sum();

        // Energy reduction
        let f_energy = self.device.get_func("msc", "msc_reduce_energy").unwrap();
        unsafe {
            f_energy.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared,
                },
                (
                    &self.spins_msc,
                    &mut self.partial_energy,
                    self.n as i32,
                    self.n_words as i32,
                    j,
                ),
            )?;
        }
        let energy_host = self.device.dtoh_sync_copy(&self.partial_energy)?;
        let total_energy: f64 = energy_host.iter().map(|&x| x as f64).sum();

        Ok((total_energy / n_sites, total_mag.abs() / n_sites))
    }

    /// Measure total energy only (for replica exchange).
    pub fn energy_gpu(&mut self, j: f32) -> anyhow::Result<f64> {
        let n_blocks = (self.n_words + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let shared = BLOCK_SIZE * 4;

        let f_energy = self.device.get_func("msc", "msc_reduce_energy").unwrap();
        unsafe {
            f_energy.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared,
                },
                (
                    &self.spins_msc,
                    &mut self.partial_energy,
                    self.n as i32,
                    self.n_words as i32,
                    j,
                ),
            )?;
        }
        let energy_host = self.device.dtoh_sync_copy(&self.partial_energy)?;
        let total_energy: f64 = energy_host.iter().map(|&x| x as f64).sum();
        Ok(total_energy)
    }

    /// Warm up: set temperature then run `sweeps` MSC Metropolis sweeps.
    pub fn warm_up(&mut self, beta: f32, j: f32, sweeps: usize) -> anyhow::Result<()> {
        self.set_temperature(beta, j)?;
        for _ in 0..sweeps {
            self.step(beta, j)?;
        }
        Ok(())
    }
}
