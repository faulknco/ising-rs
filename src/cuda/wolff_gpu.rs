use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use rand::Rng;
use std::sync::Arc;

const WOLFF_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/wolff_gpu_kernel.ptx"));
const BLOCK_SIZE: u32 = 256;
const MAX_PROPAGATION_ITERS: usize = 200;

pub struct WolffGpuLattice {
    pub n: usize,
    device: Arc<CudaDevice>,
    spins: CudaSlice<i8>,
    rng_states: CudaSlice<u8>,     // Philox: 16 bytes * N^3 threads
    bonds: CudaSlice<u8>,          // N^3 * 6 bytes (one byte per bond direction)
    labels: CudaSlice<u32>,        // N^3
    changed_flag: CudaSlice<i32>,  // single element
    seed_result: CudaSlice<u32>,   // single element
    partial_mag: CudaSlice<f32>,   // for reduction
    partial_energy: CudaSlice<f32>, // for reduction
    n_sites: u32,
}

impl WolffGpuLattice {
    pub fn new(n: usize, seed: u64, device: Arc<CudaDevice>) -> anyhow::Result<Self> {
        device.load_ptx(
            WOLFF_PTX.into(),
            "wolff",
            &[
                "wolff_init_rng_kernel",
                "wolff_bond_proposal_kernel",
                "wolff_propagate_kernel",
                "wolff_flatten_labels_kernel",
                "wolff_flip_cluster_kernel",
                "wolff_pick_seed_kernel",
            ],
        )?;

        // Load reduction kernels for measurement
        crate::cuda::reduce_gpu::load_reduce_kernels(&device)?;

        let n_sites = (n * n * n) as u32;
        let n_blocks = (n_sites + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Random initial spins on host
        let host_spins: Vec<i8> = (0..n_sites as usize)
            .map(|_| if rand::random::<bool>() { 1i8 } else { -1i8 })
            .collect();

        let spins = device.htod_sync_copy(&host_spins)?;
        // sizeof(curandStatePhilox4_32_10) = 64 bytes per thread
        let rng_states = device.alloc_zeros::<u8>((n_sites as usize) * 64)?;
        let bonds = device.alloc_zeros::<u8>((n_sites as usize) * 6)?;
        let labels = device.alloc_zeros::<u32>(n_sites as usize)?;
        let changed_flag = device.alloc_zeros::<i32>(1)?;
        let seed_result = device.alloc_zeros::<u32>(1)?;
        let partial_mag = device.alloc_zeros::<f32>(n_blocks as usize)?;
        let partial_energy = device.alloc_zeros::<f32>(n_blocks as usize)?;

        let mut gpu = Self {
            n,
            device,
            spins,
            rng_states,
            bonds,
            labels,
            changed_flag,
            seed_result,
            partial_mag,
            partial_energy,
            n_sites,
        };
        gpu.init_rng(seed)?;
        Ok(gpu)
    }

    fn init_rng(&mut self, seed: u64) -> anyhow::Result<()> {
        let grid = (self.n_sites + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let f = self
            .device
            .get_func("wolff", "wolff_init_rng_kernel")
            .unwrap();
        unsafe {
            f.launch(
                LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut self.rng_states, seed, self.n_sites as i32),
            )?;
        }
        self.device.synchronize()?;
        Ok(())
    }

    /// Run one Wolff cluster step. Returns the number of propagation iterations.
    pub fn step(
        &mut self,
        beta: f32,
        j: f32,
        rng: &mut impl Rng,
    ) -> anyhow::Result<usize> {
        // Wolff algorithm only works for ferromagnetic coupling
        if j <= 0.0 {
            return Ok(0);
        }

        // Each undirected bond (i,j) is proposed from both sides independently.
        // If each side activates with probability q, the effective activation
        // probability is 1-(1-q)^2. We want this to equal p = 1-exp(-2βJ).
        // Solving: (1-q)^2 = exp(-2βJ), so q = 1 - exp(-βJ).
        let p_add = 1.0_f32 - (-beta * j).exp();
        let n = self.n as i32;
        let grid = (self.n_sites + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Phase A: Bond proposal
        let f_bond = self
            .device
            .get_func("wolff", "wolff_bond_proposal_kernel")
            .unwrap();
        unsafe {
            f_bond.launch(
                LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &self.spins,
                    &mut self.bonds,
                    &mut self.labels,
                    &mut self.rng_states,
                    n,
                    p_add,
                ),
            )?;
        }

        // Phase B: Iterative label propagation
        let mut iters = 0usize;
        loop {
            // Reset changed_flag to 0
            self.device
                .htod_sync_copy_into(&[0i32], &mut self.changed_flag)?;

            let f_prop = self
                .device
                .get_func("wolff", "wolff_propagate_kernel")
                .unwrap();
            unsafe {
                f_prop.launch(
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (BLOCK_SIZE, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &self.bonds,
                        &mut self.labels,
                        &mut self.changed_flag,
                        self.n as i32,
                    ),
                )?;
            }

            let f_flatten = self
                .device
                .get_func("wolff", "wolff_flatten_labels_kernel")
                .unwrap();
            unsafe {
                f_flatten.launch(
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (BLOCK_SIZE, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut self.labels, self.n_sites as i32),
                )?;
            }

            let flag_host = self.device.dtoh_sync_copy(&self.changed_flag)?;
            iters += 1;

            if flag_host[0] == 0 || iters >= MAX_PROPAGATION_ITERS {
                break;
            }
        }

        // Phase C: Pick seed and flip cluster
        let seed_idx = rng.gen_range(0..self.n_sites) as u32;

        // Single-thread kernel to read the label of the seed site
        let f_pick = self
            .device
            .get_func("wolff", "wolff_pick_seed_kernel")
            .unwrap();
        unsafe {
            f_pick.launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&self.labels, &mut self.seed_result, seed_idx),
            )?;
        }

        // Copy the seed's cluster label back to host (scalar value needed by flip kernel)
        let seed_label_host = self.device.dtoh_sync_copy(&self.seed_result)?;
        let flip_label = seed_label_host[0];

        let f_flip = self
            .device
            .get_func("wolff", "wolff_flip_cluster_kernel")
            .unwrap();
        unsafe {
            f_flip.launch(
                LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &mut self.spins,
                    &self.labels,
                    flip_label,
                    self.n_sites as i32,
                ),
            )?;
        }

        Ok(iters)
    }

    /// Measure E and |M| per spin using GPU reduction (no host spin transfer).
    pub fn measure_gpu(&mut self, j: f32) -> anyhow::Result<(f64, f64)> {
        let n3 = self.n_sites as f64;
        let n_blocks = (self.n_sites + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let shared = BLOCK_SIZE as u32 * 4;

        // Magnetisation
        let f_mag = self
            .device
            .get_func("reduce", "reduce_mag_ising")
            .unwrap();
        unsafe {
            f_mag.launch(
                LaunchConfig {
                    grid_dim: (n_blocks, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: shared,
                },
                (&self.spins, &mut self.partial_mag, self.n_sites as i32),
            )?;
        }
        let mag_host = self.device.dtoh_sync_copy(&self.partial_mag)?;
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
                (&self.spins, &mut self.partial_energy, self.n as i32, j),
            )?;
        }
        let energy_host = self.device.dtoh_sync_copy(&self.partial_energy)?;
        let total_energy: f64 = energy_host.iter().map(|&x| x as f64).sum();

        Ok((total_energy / n3, total_mag.abs() / n3))
    }

    /// Measure total energy using GPU reduction (for replica exchange).
    pub fn energy_gpu(&mut self, j: f32) -> anyhow::Result<f64> {
        let n_blocks = (self.n_sites + BLOCK_SIZE - 1) / BLOCK_SIZE;
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
                (&self.spins, &mut self.partial_energy, self.n as i32, j),
            )?;
        }
        let energy_host = self.device.dtoh_sync_copy(&self.partial_energy)?;
        let total_energy: f64 = energy_host.iter().map(|&x| x as f64).sum();
        Ok(total_energy)
    }

    /// Warm up: run `steps` Wolff cluster steps, discard results.
    pub fn warm_up(
        &mut self,
        beta: f32,
        j: f32,
        steps: usize,
        rng: &mut impl Rng,
    ) -> anyhow::Result<()> {
        for _ in 0..steps {
            self.step(beta, j, rng)?;
        }
        Ok(())
    }

    /// Copy spins from device to host.
    pub fn get_spins(&self) -> anyhow::Result<Vec<i8>> {
        Ok(self.device.dtoh_sync_copy(&self.spins)?)
    }
}
