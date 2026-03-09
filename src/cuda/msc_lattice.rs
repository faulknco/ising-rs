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
                "msc_batched_metropolis_kernel",
                "msc_batched_reduce_mag",
                "msc_batched_reduce_energy",
            ],
        )?;

        let n_words = (n * n * (n / 32)) as u32;
        let n_blocks = (n_words + BLOCK_SIZE - 1) / BLOCK_SIZE;

        let spins_msc = device.alloc_zeros::<u32>(n_words as usize)?;
        // sizeof(curandStatePhilox4_32_10) = 64 bytes per thread
        let rng_states = device.alloc_zeros::<u8>((n_words as usize) * 64)?;
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
        let f = self.device.get_func("msc", "msc_randomise_kernel").unwrap();
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

    /// Compute and upload the 7-element Boltzmann probability table.
    /// Index by n_anti (0..=6): delta_E = 4*J*(3 - n_anti).
    /// n_anti=0: all neighbours aligned, flipping costs energy (12J).
    /// n_anti=6: all neighbours anti-aligned, flipping lowers energy (-12J).
    pub fn set_temperature(&mut self, beta: f32, j: f32) -> anyhow::Result<()> {
        let mut probs = [0.0f32; 7];
        for n_anti in 0..=6i32 {
            let delta_e = 4.0 * j * (3 - n_anti) as f32;
            probs[n_anti as usize] = if delta_e <= 0.0 {
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
                        n,
                        beta,
                        j,
                        parity,
                        &self.boltz_probs,
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
                (&self.spins_msc, &mut self.partial_mag, self.n_words as i32),
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

// ============================================================
// Batched multi-replica MSC lattice
// ============================================================

/// Batched multi-replica MSC lattice.
/// All replicas share a single concatenated spin buffer and single kernel launch.
pub struct BatchedMscLattice {
    pub n: usize,
    pub n_replicas: usize,
    device: Arc<CudaDevice>,
    spins_all: CudaSlice<u32>, // concatenated: n_replicas * words_per_replica
    rng_states: CudaSlice<u8>, // Philox: 16 bytes * total_words
    boltz_probs_all: CudaSlice<f32>, // [7 * n_replicas]
    words_per_replica: u32,
    total_words: u32,
    partial_mag: CudaSlice<f32>, // for per-replica reduction
    partial_energy: CudaSlice<f32>,
}

impl BatchedMscLattice {
    /// Create a batched MSC lattice with `n_replicas` replicas of size `n^3`.
    /// All spins initialised to all-up; RNG seeded per thread.
    pub fn new(
        n: usize,
        n_replicas: usize,
        seed: u64,
        device: Arc<CudaDevice>,
    ) -> anyhow::Result<Self> {
        assert!(n >= 32, "MSC lattice requires n >= 32");
        assert!(n % 32 == 0, "MSC lattice requires n divisible by 32");
        assert!(n_replicas >= 1, "need at least 1 replica");

        // PTX may already be loaded by MscLattice::new(); load_ptx is idempotent
        // in cudarc so this is safe to call again.
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
                "msc_batched_metropolis_kernel",
                "msc_batched_reduce_mag",
                "msc_batched_reduce_energy",
            ],
        )?;

        let words_per_replica = (n * n * (n / 32)) as u32;
        let total_words = words_per_replica * n_replicas as u32;
        let n_blocks_per_replica = (words_per_replica + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Allocate concatenated buffers
        let spins_all = device.alloc_zeros::<u32>(total_words as usize)?;
        // sizeof(curandStatePhilox4_32_10) = 64 bytes per thread
        let rng_states = device.alloc_zeros::<u8>((total_words as usize) * 64)?;
        let boltz_probs_all = device.alloc_zeros::<f32>(7 * n_replicas)?;
        let partial_mag = device.alloc_zeros::<f32>(n_blocks_per_replica as usize)?;
        let partial_energy = device.alloc_zeros::<f32>(n_blocks_per_replica as usize)?;

        let mut lattice = Self {
            n,
            n_replicas,
            device,
            spins_all,
            rng_states,
            boltz_probs_all,
            words_per_replica,
            total_words,
            partial_mag,
            partial_energy,
        };

        // Init RNG for all threads across all replicas
        lattice.init_rng(seed)?;
        // Init all spins to all-up
        lattice.init_spins()?;

        Ok(lattice)
    }

    fn init_rng(&mut self, seed: u64) -> anyhow::Result<()> {
        let grid = (self.total_words + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let f = self.device.get_func("msc", "msc_init_rng_kernel").unwrap();
        unsafe {
            f.launch(
                LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut self.rng_states, seed, self.total_words as i32),
            )?;
        }
        self.device.synchronize()?;
        Ok(())
    }

    fn init_spins(&mut self) -> anyhow::Result<()> {
        let grid = (self.total_words + BLOCK_SIZE - 1) / BLOCK_SIZE;
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
                (&mut self.spins_all, self.total_words as i32),
            )?;
        }
        self.device.synchronize()?;
        Ok(())
    }

    /// Compute and upload the Boltzmann probability tables for all replicas.
    /// `betas` must have length `n_replicas`.
    /// Compute and upload Boltzmann probability tables for all replicas.
    /// Index by n_anti (0..=6): delta_E = 4*J*(3 - n_anti).
    pub fn set_temperatures(&mut self, betas: &[f32], j: f32) -> anyhow::Result<()> {
        assert_eq!(betas.len(), self.n_replicas);
        let mut probs = vec![0.0f32; 7 * self.n_replicas];
        for (r, &beta) in betas.iter().enumerate() {
            for n_anti in 0..=6i32 {
                let delta_e = 4.0 * j * (3 - n_anti) as f32;
                probs[r * 7 + n_anti as usize] = if delta_e <= 0.0 {
                    1.0
                } else {
                    (-beta * delta_e).exp().min(1.0)
                };
            }
        }
        self.boltz_probs_all = self.device.htod_sync_copy(&probs)?;
        Ok(())
    }

    /// Run one full batched MSC Metropolis sweep (parity 0 then parity 1)
    /// across all replicas in a single kernel launch per parity.
    pub fn step_all(&mut self) -> anyhow::Result<()> {
        let grid = (self.total_words + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let n = self.n as i32;
        let n_replicas = self.n_replicas as i32;

        for parity in [0i32, 1i32] {
            let f = self
                .device
                .get_func("msc", "msc_batched_metropolis_kernel")
                .unwrap();
            unsafe {
                f.launch(
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (BLOCK_SIZE, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &mut self.spins_all,
                        &mut self.rng_states,
                        n,
                        n_replicas,
                        parity,
                        &self.boltz_probs_all,
                    ),
                )?;
            }
        }
        Ok(())
    }

    /// Measure energy per spin and |magnetisation| per spin for a single replica.
    pub fn measure_replica(&mut self, replica: usize, j: f32) -> anyhow::Result<(f64, f64)> {
        assert!(replica < self.n_replicas);
        let n_sites = (self.n * self.n * self.n) as f64;
        let offset = (replica as u32 * self.words_per_replica) as i32;
        let wpr = self.words_per_replica as i32;
        let n_blocks = (self.words_per_replica + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let shared = BLOCK_SIZE * 4;

        // Magnetisation reduction
        let f_mag = self
            .device
            .get_func("msc", "msc_batched_reduce_mag")
            .unwrap();
        unsafe {
            f_mag.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared,
                },
                (&self.spins_all, &mut self.partial_mag, wpr, offset),
            )?;
        }
        let mag_host = self.device.dtoh_sync_copy(&self.partial_mag)?;
        let total_mag: f64 = mag_host.iter().map(|&x| x as f64).sum();

        // Energy reduction
        let f_energy = self
            .device
            .get_func("msc", "msc_batched_reduce_energy")
            .unwrap();
        unsafe {
            f_energy.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared,
                },
                (
                    &self.spins_all,
                    &mut self.partial_energy,
                    self.n as i32,
                    j,
                    offset,
                ),
            )?;
        }
        let energy_host = self.device.dtoh_sync_copy(&self.partial_energy)?;
        let total_energy: f64 = energy_host.iter().map(|&x| x as f64).sum();

        Ok((total_energy / n_sites, total_mag.abs() / n_sites))
    }

    /// Measure total energy for a single replica (for replica exchange).
    pub fn energy_replica(&mut self, replica: usize, j: f32) -> anyhow::Result<f64> {
        assert!(replica < self.n_replicas);
        let offset = (replica as u32 * self.words_per_replica) as i32;
        let n_blocks = (self.words_per_replica + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let shared = BLOCK_SIZE * 4;

        let f_energy = self
            .device
            .get_func("msc", "msc_batched_reduce_energy")
            .unwrap();
        unsafe {
            f_energy.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared,
                },
                (
                    &self.spins_all,
                    &mut self.partial_energy,
                    self.n as i32,
                    j,
                    offset,
                ),
            )?;
        }
        let energy_host = self.device.dtoh_sync_copy(&self.partial_energy)?;
        let total_energy: f64 = energy_host.iter().map(|&x| x as f64).sum();
        Ok(total_energy)
    }
}
