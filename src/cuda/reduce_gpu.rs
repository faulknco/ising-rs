use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

const REDUCE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/reduce_kernel.ptx"));
const BLOCK_SIZE: u32 = 256;

/// Load reduction kernels into the device.
pub fn load_reduce_kernels(device: &Arc<CudaDevice>) -> anyhow::Result<()> {
    device.load_ptx(
        REDUCE_PTX.into(),
        "reduce",
        &[
            "reduce_mag_ising",
            "reduce_energy_ising",
            "reduce_mag_continuous",
            "reduce_energy_continuous",
        ],
    )?;
    Ok(())
}

/// GPU-side reduction: returns (total_energy, total_magnetisation) for Ising cubic.
pub fn reduce_ising(
    device: &Arc<CudaDevice>,
    spins: &CudaSlice<i8>,
    n: usize,
    j: f32,
) -> anyhow::Result<(f64, f64)> {
    let n_sites = n * n * n;
    let n_blocks = ((n_sites as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let shared = BLOCK_SIZE as u32 * 4; // sizeof(float)

    // --- Magnetisation ---
    let mut partial_mag = device.alloc_zeros::<f32>(n_blocks as usize)?;
    let f_mag = device.get_func("reduce", "reduce_mag_ising").unwrap();
    unsafe {
        f_mag.launch(
            LaunchConfig {
                grid_dim: (n_blocks, 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: shared,
            },
            (spins, &mut partial_mag, n_sites as i32),
        )?;
    }
    let mag_host = device.dtoh_sync_copy(&partial_mag)?;
    let total_mag: f64 = mag_host.iter().map(|&x| x as f64).sum();

    // --- Energy ---
    let mut partial_e = device.alloc_zeros::<f32>(n_blocks as usize)?;
    let f_energy = device.get_func("reduce", "reduce_energy_ising").unwrap();
    unsafe {
        f_energy.launch(
            LaunchConfig {
                grid_dim: (n_blocks, 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: shared,
            },
            (spins, &mut partial_e, n as i32, j),
        )?;
    }
    let energy_host = device.dtoh_sync_copy(&partial_e)?;
    let total_energy: f64 = energy_host.iter().map(|&x| x as f64).sum();

    Ok((total_energy, total_mag))
}

/// GPU-side measurement: returns (energy_per_spin, abs_mag_per_spin) for Ising cubic.
/// No host↔device spin transfer — only partial sums come back (~n_blocks floats).
pub fn measure_ising_gpu(
    device: &Arc<CudaDevice>,
    spins: &CudaSlice<i8>,
    n: usize,
    j: f32,
) -> anyhow::Result<(f64, f64)> {
    let (total_energy, total_mag) = reduce_ising(device, spins, n, j)?;
    let n3 = (n * n * n) as f64;
    Ok((total_energy / n3, total_mag.abs() / n3))
}

/// GPU-side reduction for continuous spins: returns (mx, my, mz).
/// For XY (n_comp=2), mz will be 0.
pub fn reduce_continuous_mag(
    device: &Arc<CudaDevice>,
    spins: &CudaSlice<f32>,
    n_sites: usize,
    n_comp: usize,
) -> anyhow::Result<(f64, f64, f64)> {
    let n_blocks = ((n_sites as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let shared = BLOCK_SIZE as u32 * 4 * 3; // 3 float arrays

    let mut partial_mx = device.alloc_zeros::<f32>(n_blocks as usize)?;
    let mut partial_my = device.alloc_zeros::<f32>(n_blocks as usize)?;
    let mut partial_mz = device.alloc_zeros::<f32>(n_blocks as usize)?;

    let f = device.get_func("reduce", "reduce_mag_continuous").unwrap();
    unsafe {
        f.launch(
            LaunchConfig {
                grid_dim: (n_blocks, 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: shared,
            },
            (
                spins,
                &mut partial_mx,
                &mut partial_my,
                &mut partial_mz,
                n_sites as i32,
                n_comp as i32,
            ),
        )?;
    }

    let mx: f64 = device.dtoh_sync_copy(&partial_mx)?.iter().map(|&x| x as f64).sum();
    let my: f64 = device.dtoh_sync_copy(&partial_my)?.iter().map(|&x| x as f64).sum();
    let mz: f64 = device.dtoh_sync_copy(&partial_mz)?.iter().map(|&x| x as f64).sum();

    Ok((mx, my, mz))
}

/// GPU-side energy reduction for continuous spins (XY/Heisenberg).
pub fn reduce_continuous_energy(
    device: &Arc<CudaDevice>,
    spins: &CudaSlice<f32>,
    n: usize,
    n_comp: usize,
    j: f32,
) -> anyhow::Result<f64> {
    let n_sites = n * n * n;
    let n_blocks = ((n_sites as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let shared = BLOCK_SIZE as u32 * 4; // sizeof(float)

    let mut partial_e = device.alloc_zeros::<f32>(n_blocks as usize)?;
    let f = device.get_func("reduce", "reduce_energy_continuous").unwrap();
    unsafe {
        f.launch(
            LaunchConfig {
                grid_dim: (n_blocks, 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: shared,
            },
            (spins, &mut partial_e, n as i32, n_comp as i32, j),
        )?;
    }
    let energy_host = device.dtoh_sync_copy(&partial_e)?;
    let total_energy: f64 = energy_host.iter().map(|&x| x as f64).sum();
    Ok(total_energy)
}

/// GPU-side measurement for continuous spins: returns (energy, mx, my, mz).
/// No host↔device spin transfer.
pub fn measure_continuous_gpu(
    device: &Arc<CudaDevice>,
    spins: &CudaSlice<f32>,
    n: usize,
    n_comp: usize,
    j: f32,
) -> anyhow::Result<(f64, f64, f64, f64)> {
    let n_sites = n * n * n;
    let (mx, my, mz) = reduce_continuous_mag(device, spins, n_sites, n_comp)?;
    let energy = reduce_continuous_energy(device, spins, n, n_comp, j)?;
    Ok((energy, mx, my, mz))
}
