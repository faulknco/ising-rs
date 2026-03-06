use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
const BLOCK_SIZE: u32 = 256;

pub struct LatticeGpu {
    pub n: usize,
    device: Arc<CudaDevice>,
    spins: CudaSlice<i8>,
    // curandState is ~48 bytes per thread; store as raw u8 slice
    rng_states: CudaSlice<u8>,
    n_threads: u32,
}

impl LatticeGpu {
    pub fn new(n: usize, seed: u64) -> anyhow::Result<Self> {
        let device = CudaDevice::new(0)?;
        device.load_ptx(PTX.into(), "ising", &[
            "metropolis_sweep_kernel",
            "init_rng_kernel",
            "sum_spins_kernel",
        ])?;

        let size = n * n * n;
        let n_threads = (size / 2) as u32;

        // Random initial spins on host
        let host_spins: Vec<i8> = (0..size)
            .map(|_| if rand::random::<bool>() { 1i8 } else { -1i8 })
            .collect();

        let spins = device.htod_sync_copy(&host_spins)?;
        // 48 bytes per curandState
        let rng_states = device.alloc_zeros::<u8>((n_threads as usize) * 48)?;

        let mut gpu = Self { n, device, spins, rng_states, n_threads };
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
        let f = self.device.get_func("ising", "metropolis_sweep_kernel").unwrap();

        for parity in [0i32, 1i32] {
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
        self.device.synchronize()?;
        Ok(())
    }

    /// Copy spins from device to host.
    pub fn get_spins(&self) -> anyhow::Result<Vec<i8>> {
        Ok(self.device.dtoh_sync_copy(&self.spins)?)
    }

    /// Magnetisation |<M>| per spin (host-side reduction).
    pub fn magnetisation(&self) -> anyhow::Result<f64> {
        let spins = self.get_spins()?;
        let sum: i32 = spins.iter().map(|&s| s as i32).sum();
        Ok((sum as f64 / spins.len() as f64).abs())
    }
}
