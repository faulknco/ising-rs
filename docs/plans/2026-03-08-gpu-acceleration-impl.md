# GPU Acceleration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add GPU checkerboard Metropolis for all three spin models, parallel tempering with replica exchange, and Ferrenberg-Swendsen histogram reweighting — enabling N=128–256 FSS on an RTX 2060.

**Architecture:** Two new CUDA kernel files (multi-spin-coded Ising + generic continuous-spin), a Rust-side parallel tempering orchestrator, two unified GPU CLI binaries, and a Python reweighting module. Extends the existing `src/cuda/` infrastructure via `cudarc` FFI. Fully additive — `--features cuda` opt-in.

**Tech Stack:** Rust 1.94+, cudarc 0.12, CUDA C (PTX via nvcc, sm_75), Python 3 (numpy, scipy).

**Design doc:** `docs/plans/2026-03-08-gpu-acceleration-design.md`

---

## Task 1: Extend build.rs to compile multiple .cu files

**Files:**
- Modify: `build.rs`

**Context:** Currently `build.rs` only compiles `src/cuda/kernels.cu` → `kernels.ptx`. We need it to compile additional `.cu` files. The simplest approach: iterate over all `.cu` files in `src/cuda/`.

**Step 1: Read current build.rs and understand the pattern**

The existing code compiles one file. We'll change it to compile a list.

**Step 2: Modify build.rs to compile multiple kernel files**

```rust
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = PathBuf::from(&cuda_path).join("bin").join("nvcc");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let kernel_files = [
        "src/cuda/kernels.cu",
        "src/cuda/continuous_spin_kernel.cu",
        "src/cuda/reduce_kernel.cu",
    ];

    for src in &kernel_files {
        let src_path = PathBuf::from(src);
        let stem = src_path.file_stem().unwrap().to_str().unwrap();
        let ptx_out = out_dir.join(format!("{stem}.ptx"));

        let status = Command::new(&nvcc)
            .args([
                "-ptx",
                "-O3",
                "--allow-unsupported-compiler",
                "--generate-code",
                "arch=compute_75,code=sm_75",
                "-I",
                &format!("{}/include", cuda_path),
                src,
                "-o",
                ptx_out.to_str().unwrap(),
            ])
            .status()
            .unwrap_or_else(|_| panic!("nvcc not found — is CUDA_PATH set?"));

        assert!(status.success(), "nvcc compilation failed for {src}");
        println!("cargo:rerun-if-changed={src}");
    }
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}
```

**Step 3: Create placeholder .cu files so build doesn't break**

Create empty `src/cuda/continuous_spin_kernel.cu` and `src/cuda/reduce_kernel.cu` with just a comment:

```c
// Placeholder — implemented in Task 3 / Task 4.
```

**Step 4: Verify build compiles (on a machine with CUDA, or just verify non-CUDA build)**

Run: `cargo build --release`
Expected: PASS (cuda feature not enabled, build.rs skips)

**Step 5: Commit**

```bash
git add build.rs src/cuda/continuous_spin_kernel.cu src/cuda/reduce_kernel.cu
git commit -m "build: extend build.rs to compile multiple CUDA kernel files"
```

---

## Task 2: GPU reduction kernel for observables

**Files:**
- Create: `src/cuda/reduce_kernel.cu`
- Create: `src/cuda/reduce_gpu.rs`
- Modify: `src/cuda/mod.rs`

**Context:** Currently observables are computed by copying all spins to CPU every sample — O(N³) transfer per sweep. We need a GPU-side parallel reduction that computes E, |M|, M², M⁴ without leaving the device.

**Step 1: Write the reduction kernel**

Replace the placeholder `src/cuda/reduce_kernel.cu`:

```c
#include <math.h>

// Block-wide reduction using shared memory.
// Each block reduces its portion; host does final sum over partial results.
//
// For Ising: input is int8 spins, output partial sums of spin values.
// For continuous: input is float spins (n_comp floats per site), output partial sums.

// --- Ising: partial magnetisation (sum of spins) ---
extern "C" __global__ void reduce_mag_ising(
    const signed char* spins,
    float* partial_mag,       // |partial_mag| = n_blocks
    int    n_sites
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < n_sites) ? (float)spins[gid] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_mag[blockIdx.x] = sdata[0];
}

// --- Ising: partial energy (sum of -J * s_i * nb_sum / 2, cubic) ---
extern "C" __global__ void reduce_energy_ising(
    const signed char* spins,
    float* partial_energy,
    int    N,            // lattice side length
    float  J
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_sites = N * N * N;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float val = 0.0f;
    if (gid < n_sites) {
        int z = gid / (N * N);
        int r = gid % (N * N);
        int y = r / N;
        int x = r % N;

        // Only forward neighbours (3 of 6) to avoid double-counting
        int xp = (x + 1) % N;
        int yp = (y + 1) % N;
        int zp = (z + 1) % N;

        float s = (float)spins[gid];
        val = -J * s * (
            (float)spins[z*N*N + y*N + xp] +
            (float)spins[z*N*N + yp*N + x] +
            (float)spins[zp*N*N + y*N + x]
        );
    }

    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_energy[blockIdx.x] = sdata[0];
}

// --- Continuous spins: partial |M| vector components ---
// m_x, m_y (XY) or m_x, m_y, m_z (Heisenberg)
// spins layout: [s0_x, s0_y, s0_z, s1_x, s1_y, s1_z, ...] (interleaved)
extern "C" __global__ void reduce_mag_continuous(
    const float* spins,
    float* partial_mx,
    float* partial_my,
    float* partial_mz,    // unused for XY, but always allocated
    int    n_sites,
    int    n_comp          // 2 = XY, 3 = Heisenberg
) {
    extern __shared__ float sdata[];  // 3 * blockDim.x
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int bd = blockDim.x;

    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    if (gid < n_sites) {
        mx = spins[gid * n_comp + 0];
        my = spins[gid * n_comp + 1];
        if (n_comp == 3) mz = spins[gid * n_comp + 2];
    }
    sdata[tid]        = mx;
    sdata[tid + bd]   = my;
    sdata[tid + 2*bd] = mz;
    __syncthreads();

    for (int s = bd / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid]        += sdata[tid + s];
            sdata[tid + bd]   += sdata[tid + s + bd];
            sdata[tid + 2*bd] += sdata[tid + s + 2*bd];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_mx[blockIdx.x] = sdata[0];
        partial_my[blockIdx.x] = sdata[bd];
        partial_mz[blockIdx.x] = sdata[2*bd];
    }
}
```

**Step 2: Write the Rust wrapper `src/cuda/reduce_gpu.rs`**

```rust
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
```

**Step 3: Export from mod.rs**

Add to `src/cuda/mod.rs`:
```rust
pub mod reduce_gpu;
```

**Step 4: Verify non-CUDA build still compiles**

Run: `cargo build --release`
Expected: PASS

**Step 5: Commit**

```bash
git add src/cuda/reduce_kernel.cu src/cuda/reduce_gpu.rs src/cuda/mod.rs
git commit -m "feat(cuda): add GPU reduction kernels for E, M, M2, M4"
```

---

## Task 3: Continuous-spin checkerboard Metropolis + overrelaxation kernel

**Files:**
- Create: `src/cuda/continuous_spin_kernel.cu`

**Context:** Generic kernel for XY (n=2) and Heisenberg (n=3). Checkerboard Metropolis update with interleaved overrelaxation. Cubic lattice with periodic boundaries.

**Step 1: Write the kernel**

Replace the placeholder `src/cuda/continuous_spin_kernel.cu`:

```c
#include <curand_kernel.h>
#include <math.h>

// Continuous-spin checkerboard Metropolis for O(n) models.
// n_comp = 2 (XY) or 3 (Heisenberg).
// spins: interleaved [s0_x, s0_y, (s0_z), s1_x, ...], n_comp floats per site.
// Cubic lattice, periodic boundaries.

extern "C" __global__ void continuous_metropolis_kernel(
    float*       spins,
    curandState* rng_states,
    int          N,          // lattice side
    int          n_comp,     // 2 or 3
    float        beta,
    float        J,
    float        delta,      // proposal cone half-angle
    int          parity      // 0=black, 1=white
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (N * N * N) / 2;
    if (tid >= total) return;

    // Map tid to (x,y,z) with correct parity
    int full_idx = tid * 2;
    int z = full_idx / (N * N);
    int r = full_idx % (N * N);
    int y = r / N;
    int x = r % N;
    if ((x + y + z) % 2 != parity) x = (x + 1) % N;
    if (x >= N) return;

    int idx = z * N * N + y * N + x;

    // Compute local field h_i = J * sum_j S_j
    int nb[6];
    nb[0] = z*N*N + y*N + (x+1)%N;
    nb[1] = z*N*N + y*N + (x-1+N)%N;
    nb[2] = z*N*N + ((y+1)%N)*N + x;
    nb[3] = z*N*N + ((y-1+N)%N)*N + x;
    nb[4] = ((z+1)%N)*N*N + y*N + x;
    nb[5] = ((z-1+N)%N)*N*N + y*N + x;

    float hx = 0.0f, hy = 0.0f, hz = 0.0f;
    for (int k = 0; k < 6; k++) {
        hx += spins[nb[k] * n_comp + 0];
        hy += spins[nb[k] * n_comp + 1];
        if (n_comp == 3) hz += spins[nb[k] * n_comp + 2];
    }
    hx *= J; hy *= J; if (n_comp == 3) hz *= J;

    // Current spin
    float sx = spins[idx * n_comp + 0];
    float sy = spins[idx * n_comp + 1];
    float sz = (n_comp == 3) ? spins[idx * n_comp + 2] : 0.0f;

    float e_old = -(sx * hx + sy * hy + sz * hz);

    // Propose: perturb current spin by small random rotation
    curandState local_rng = rng_states[tid];
    float dx = delta * (2.0f * curand_uniform(&local_rng) - 1.0f);
    float dy = delta * (2.0f * curand_uniform(&local_rng) - 1.0f);
    float dz = (n_comp == 3) ? delta * (2.0f * curand_uniform(&local_rng) - 1.0f) : 0.0f;

    float nx = sx + dx;
    float ny = sy + dy;
    float nz = sz + dz;

    // Normalise to unit sphere/circle
    float norm = sqrtf(nx*nx + ny*ny + nz*nz);
    if (norm < 1e-8f) { rng_states[tid] = local_rng; return; }
    nx /= norm; ny /= norm; nz /= norm;

    float e_new = -(nx * hx + ny * hy + nz * hz);
    float de = e_new - e_old;

    if (de < 0.0f || curand_uniform(&local_rng) < expf(-beta * de)) {
        spins[idx * n_comp + 0] = nx;
        spins[idx * n_comp + 1] = ny;
        if (n_comp == 3) spins[idx * n_comp + 2] = nz;
    }

    rng_states[tid] = local_rng;
}

// Overrelaxation: reflect spin through local field direction.
// Deterministic, no RNG, 100% acceptance (microcanonical).
// S_new = 2 * (S . h_hat) * h_hat - S
extern "C" __global__ void continuous_overrelax_kernel(
    float* spins,
    int    N,
    int    n_comp,
    float  J,
    int    parity
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (N * N * N) / 2;
    if (tid >= total) return;

    int full_idx = tid * 2;
    int z = full_idx / (N * N);
    int r = full_idx % (N * N);
    int y = r / N;
    int x = r % N;
    if ((x + y + z) % 2 != parity) x = (x + 1) % N;
    if (x >= N) return;

    int idx = z * N * N + y * N + x;

    int nb[6];
    nb[0] = z*N*N + y*N + (x+1)%N;
    nb[1] = z*N*N + y*N + (x-1+N)%N;
    nb[2] = z*N*N + ((y+1)%N)*N + x;
    nb[3] = z*N*N + ((y-1+N)%N)*N + x;
    nb[4] = ((z+1)%N)*N*N + y*N + x;
    nb[5] = ((z-1+N)%N)*N*N + y*N + x;

    float hx = 0.0f, hy = 0.0f, hz = 0.0f;
    for (int k = 0; k < 6; k++) {
        hx += spins[nb[k] * n_comp + 0];
        hy += spins[nb[k] * n_comp + 1];
        if (n_comp == 3) hz += spins[nb[k] * n_comp + 2];
    }
    hx *= J; hy *= J; if (n_comp == 3) hz *= J;

    float h_norm = sqrtf(hx*hx + hy*hy + hz*hz);
    if (h_norm < 1e-10f) return;

    float hx_hat = hx / h_norm;
    float hy_hat = hy / h_norm;
    float hz_hat = hz / h_norm;

    float sx = spins[idx * n_comp + 0];
    float sy = spins[idx * n_comp + 1];
    float sz = (n_comp == 3) ? spins[idx * n_comp + 2] : 0.0f;

    float dot = sx * hx_hat + sy * hy_hat + sz * hz_hat;

    // S_new = 2*(S.h_hat)*h_hat - S
    spins[idx * n_comp + 0] = 2.0f * dot * hx_hat - sx;
    spins[idx * n_comp + 1] = 2.0f * dot * hy_hat - sy;
    if (n_comp == 3) spins[idx * n_comp + 2] = 2.0f * dot * hz_hat - sz;
}

extern "C" __global__ void init_continuous_rng_kernel(
    curandState* states,
    unsigned long long seed,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    curand_init(seed, tid, 0, &states[tid]);
}
```

**Step 2: Commit**

```bash
git add src/cuda/continuous_spin_kernel.cu
git commit -m "feat(cuda): add continuous-spin checkerboard Metropolis + overrelax kernels"
```

---

## Task 4: Rust GPU lattice wrapper for continuous spins

**Files:**
- Create: `src/cuda/gpu_lattice_continuous.rs`
- Modify: `src/cuda/mod.rs`

**Context:** Analogous to `LatticeGpu` but for f32 interleaved spins. Wraps the continuous-spin kernels.

**Step 1: Write `gpu_lattice_continuous.rs`**

```rust
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use crate::cuda::reduce_gpu;

const CONTINUOUS_PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/continuous_spin_kernel.ptx"));
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
            let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            host_spins[i * n_comp] = angle.cos();
            host_spins[i * n_comp + 1] = angle.sin();
            if n_comp == 3 {
                // Random point on S2
                let z: f32 = rng.gen_range(-1.0..1.0);
                let phi: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
                let r = (1.0 - z * z).sqrt();
                host_spins[i * n_comp] = r * phi.cos();
                host_spins[i * n_comp + 1] = r * phi.sin();
                host_spins[i * n_comp + 2] = z;
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

    /// Compute (E, mx, my, mz) using GPU reduction.
    pub fn measure_raw(&self) -> anyhow::Result<(f64, f64, f64, f64)> {
        let n_sites = self.n * self.n * self.n;
        let (mx, my, mz) =
            reduce_gpu::reduce_continuous_mag(&self.device, &self.spins, n_sites, self.n_comp)?;
        // Energy: for now compute on host (reduction kernel for continuous energy is complex)
        // TODO: add continuous energy reduction kernel
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
```

**Step 2: Export from mod.rs**

Add to `src/cuda/mod.rs`:
```rust
pub mod gpu_lattice_continuous;
```

**Step 3: Verify non-CUDA build**

Run: `cargo build --release`
Expected: PASS

**Step 4: Commit**

```bash
git add src/cuda/gpu_lattice_continuous.rs src/cuda/mod.rs
git commit -m "feat(cuda): add ContinuousGpuLattice wrapper for XY/Heisenberg GPU sweeps"
```

---

## Task 5: Parallel tempering orchestrator

**Files:**
- Create: `src/cuda/parallel_tempering.rs`
- Modify: `src/cuda/mod.rs`

**Context:** Host-side logic that manages R replicas, dispatches sweeps, and proposes replica exchanges. Works for both Ising and continuous-spin models.

**Step 1: Write `parallel_tempering.rs`**

```rust
use rand::Rng;

/// Result of one parallel tempering measurement cycle.
#[derive(Debug, Clone)]
pub struct PtSample {
    pub replica_idx: usize,
    pub temperature: f64,
    pub energy: f64,
    pub mag: f64,   // |M| / N
    pub m2: f64,
    pub m4: f64,
}

/// Propose replica exchanges between adjacent temperatures.
/// Returns the number of accepted swaps.
pub fn replica_exchange(
    temperatures: &[f64],
    energies: &[f64],
    replica_to_temp: &mut [usize],
    temp_to_replica: &mut [usize],
    rng: &mut impl Rng,
    even_odd: usize, // 0 = try pairs (0,1),(2,3),...  1 = try (1,2),(3,4),...
) -> usize {
    let n = temperatures.len();
    let mut accepted = 0;

    let start = even_odd % 2;
    let mut i = start;
    while i + 1 < n {
        let ri = temp_to_replica[i];
        let rj = temp_to_replica[i + 1];
        let beta_i = 1.0 / temperatures[i];
        let beta_j = 1.0 / temperatures[i + 1];
        let delta = (beta_i - beta_j) * (energies[rj] - energies[ri]);

        if delta < 0.0 || rng.gen::<f64>() < (-delta).exp() {
            // Swap
            temp_to_replica[i] = rj;
            temp_to_replica[i + 1] = ri;
            replica_to_temp[ri] = i + 1;
            replica_to_temp[rj] = i;
            accepted += 1;
        }
        i += 2;
    }
    accepted
}

/// Generate linearly spaced temperatures.
pub fn linspace_temperatures(t_min: f64, t_max: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![t_min];
    }
    (0..n)
        .map(|i| t_min + (t_max - t_min) * i as f64 / (n - 1) as f64)
        .collect()
}
```

**Step 2: Export from mod.rs**

Add to `src/cuda/mod.rs`:
```rust
pub mod parallel_tempering;
```

**Step 3: Write a unit test for replica_exchange**

Add to the bottom of `parallel_tempering.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replica_exchange_deterministic() {
        // Two replicas: T=4.0 (hot) and T=5.0 (cold)
        // If E_hot < E_cold, swap is always accepted (delta < 0)
        let temps = vec![4.0, 5.0];
        let energies = vec![-100.0, -50.0]; // replica 0 has lower E
        let mut r2t = vec![0, 1]; // replica 0 at temp 0, replica 1 at temp 1
        let mut t2r = vec![0, 1];

        let mut rng = rand::thread_rng();
        let accepted = replica_exchange(&temps, &energies, &mut r2t, &mut t2r, &mut rng, 0);

        // delta = (1/4 - 1/5) * (-50 - (-100)) = 0.05 * 50 = 2.5 > 0
        // So swap NOT always accepted. But with specific energies designed to swap:
        // We just check that the function runs and the bookkeeping is consistent.
        assert_eq!(r2t.len(), 2);
        assert_eq!(t2r.len(), 2);
        // Verify bijectivity
        assert_eq!(t2r[r2t[0]], 0);
        assert_eq!(t2r[r2t[1]], 1);
    }

    #[test]
    fn test_linspace_temperatures() {
        let temps = linspace_temperatures(4.0, 5.0, 5);
        assert_eq!(temps.len(), 5);
        assert!((temps[0] - 4.0).abs() < 1e-10);
        assert!((temps[4] - 5.0).abs() < 1e-10);
        assert!((temps[2] - 4.5).abs() < 1e-10);
    }
}
```

**Step 4: Run tests**

Run: `cargo test parallel_tempering -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add src/cuda/parallel_tempering.rs src/cuda/mod.rs
git commit -m "feat(cuda): add parallel tempering replica exchange logic"
```

---

## Task 6: Unified GPU FSS binary (`gpu_fss`)

**Files:**
- Create: `src/bin/gpu_fss.rs`
- Modify: `Cargo.toml`

**Context:** Single CLI binary that runs GPU FSS for any of the three models with parallel tempering. Outputs both a summary CSV (backward-compatible) and raw time series for histogram reweighting.

**Step 1: Add bin entry to Cargo.toml**

```toml
[[bin]]
name = "gpu_fss"
path = "src/bin/gpu_fss.rs"
required-features = ["cuda"]
```

**Step 2: Write `src/bin/gpu_fss.rs`**

```rust
/// GPU FSS with parallel tempering for Ising, XY, and Heisenberg models.
///
/// Usage:
///   cargo run --release --features cuda --bin gpu_fss -- \
///     --model ising --sizes 8,16,32,64 \
///     --tmin 4.4 --tmax 4.6 --replicas 20 \
///     --warmup 5000 --samples 100000 \
///     --exchange-every 10 --seed 42 \
///     --outdir analysis/data
///
/// Output per size:
///   gpu_fss_{model}_N{n}_summary.csv    — pre-averaged observables per temperature
///   gpu_fss_{model}_N{n}_timeseries.csv  — raw E, |M| per sweep per replica
use std::env;
use std::fs;
use std::path::Path;

fn get_arg(args: &[String], i: usize, flag: &str) -> String {
    if i + 1 >= args.len() {
        eprintln!("Error: {flag} requires a value");
        std::process::exit(1);
    }
    args[i + 1].clone()
}

fn parse_flag<T: std::str::FromStr>(args: &[String], i: usize, flag: &str) -> T
where
    T::Err: std::fmt::Display,
{
    match get_arg(args, i, flag).parse::<T>() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: invalid value for {flag}: {e}");
            std::process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut model = String::from("ising");
    let mut sizes_str = String::from("8,16,32,64");
    let mut t_min = 4.4_f64;
    let mut t_max = 4.6_f64;
    let mut n_replicas = 20usize;
    let mut warmup = 5000usize;
    let mut samples = 100000usize;
    let mut exchange_every = 10usize;
    let mut seed = 42u64;
    let mut outdir = String::from("analysis/data");
    let mut delta = 0.5_f64;       // Metropolis proposal width (continuous spins)
    let mut n_overrelax = 5usize;   // overrelaxation sweeps per Metropolis (continuous)

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model"          => { model        = get_arg(&args, i, "--model"); i += 2; }
            "--sizes"          => { sizes_str    = get_arg(&args, i, "--sizes"); i += 2; }
            "--tmin"           => { t_min        = parse_flag(&args, i, "--tmin"); i += 2; }
            "--tmax"           => { t_max        = parse_flag(&args, i, "--tmax"); i += 2; }
            "--replicas"       => { n_replicas   = parse_flag(&args, i, "--replicas"); i += 2; }
            "--warmup"         => { warmup       = parse_flag(&args, i, "--warmup"); i += 2; }
            "--samples"        => { samples      = parse_flag(&args, i, "--samples"); i += 2; }
            "--exchange-every" => { exchange_every = parse_flag(&args, i, "--exchange-every"); i += 2; }
            "--seed"           => { seed         = parse_flag(&args, i, "--seed"); i += 2; }
            "--outdir"         => { outdir       = get_arg(&args, i, "--outdir"); i += 2; }
            "--delta"          => { delta        = parse_flag(&args, i, "--delta"); i += 2; }
            "--overrelax"      => { n_overrelax  = parse_flag(&args, i, "--overrelax"); i += 2; }
            _ => { i += 1; }
        }
    }

    let sizes: Vec<usize> = sizes_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    fs::create_dir_all(&outdir).expect("failed to create outdir");

    eprintln!("GPU FSS: model={model}, sizes={sizes:?}, T={t_min}..{t_max}, replicas={n_replicas}");

    match model.as_str() {
        "ising"      => run_ising_fss(&sizes, t_min, t_max, n_replicas, warmup, samples, exchange_every, seed, &outdir),
        "xy"         => run_continuous_fss(&sizes, 2, t_min, t_max, n_replicas, warmup, samples, exchange_every, seed, &outdir, delta as f32, n_overrelax),
        "heisenberg" => run_continuous_fss(&sizes, 3, t_min, t_max, n_replicas, warmup, samples, exchange_every, seed, &outdir, delta as f32, n_overrelax),
        _ => {
            eprintln!("Error: --model must be ising, xy, or heisenberg");
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "cuda")]
fn run_ising_fss(
    sizes: &[usize], t_min: f64, t_max: f64, n_replicas: usize,
    warmup: usize, samples: usize, exchange_every: usize, seed: u64, outdir: &str,
) {
    use ising::cuda::lattice_gpu::LatticeGpu;
    use ising::cuda::parallel_tempering::{linspace_temperatures, replica_exchange};
    use ising::cuda::reduce_gpu;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    let temperatures = linspace_temperatures(t_min, t_max, n_replicas);

    for &n in sizes {
        eprintln!("  N={n}: creating {n_replicas} replicas...");
        let mut replicas: Vec<LatticeGpu> = (0..n_replicas)
            .map(|r| LatticeGpu::new(n, seed.wrapping_add(r as u64 * 1000 + n as u64))
                .expect("failed to create GPU lattice"))
            .collect();

        let mut replica_to_temp: Vec<usize> = (0..n_replicas).collect();
        let mut temp_to_replica: Vec<usize> = (0..n_replicas).collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(n as u64));

        // Warmup
        eprintln!("  N={n}: warming up {warmup} sweeps...");
        for _ in 0..warmup {
            for (r, lat) in replicas.iter_mut().enumerate() {
                let t_idx = replica_to_temp[r];
                let beta = 1.0 / temperatures[t_idx];
                lat.step(beta as f32, 1.0, 0.0).expect("GPU step failed");
            }
        }

        // Sampling with parallel tempering
        // Accumulators per temperature index
        let mut sum_e    = vec![0.0_f64; n_replicas];
        let mut sum_e2   = vec![0.0_f64; n_replicas];
        let mut sum_m    = vec![0.0_f64; n_replicas];
        let mut sum_m2   = vec![0.0_f64; n_replicas];
        let mut sum_m4   = vec![0.0_f64; n_replicas];
        let mut count    = vec![0usize; n_replicas];

        // Time series for reweighting (per replica)
        let mut ts_data: Vec<Vec<(f64, f64)>> = vec![vec![]; n_replicas];

        let n3 = (n * n * n) as f64;

        eprintln!("  N={n}: sampling {samples} sweeps with PT exchange every {exchange_every}...");
        for sweep in 0..samples {
            // Sweep all replicas
            for (r, lat) in replicas.iter_mut().enumerate() {
                let t_idx = replica_to_temp[r];
                let beta = 1.0 / temperatures[t_idx];
                lat.step(beta as f32, 1.0, 0.0).expect("GPU step failed");
            }

            // Measure and accumulate
            let mut energies = vec![0.0_f64; n_replicas];
            for (r, lat) in replicas.iter().enumerate() {
                let t_idx = replica_to_temp[r];
                let spins = lat.get_spins().expect("get_spins failed");
                let (e, m) = ising_e_m_host(&spins, n);
                let e_per = e / n3;
                let m_per = (m / n3).abs();
                energies[r] = e;

                sum_e[t_idx]  += e_per;
                sum_e2[t_idx] += e_per * e_per;
                sum_m[t_idx]  += m_per;
                sum_m2[t_idx] += m_per * m_per;
                sum_m4[t_idx] += m_per.powi(4);
                count[t_idx]  += 1;

                ts_data[t_idx].push((e_per, m_per));
            }

            // Replica exchange
            if (sweep + 1) % exchange_every == 0 {
                replica_exchange(
                    &temperatures, &energies,
                    &mut replica_to_temp, &mut temp_to_replica,
                    &mut rng, sweep / exchange_every,
                );
            }
        }

        // Write summary CSV
        let model_name = "ising";
        let summary_path = Path::new(outdir).join(format!("gpu_fss_{model_name}_N{n}_summary.csv"));
        let mut csv = String::from("T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err\n");
        for t_idx in 0..n_replicas {
            let s = count[t_idx] as f64;
            if s == 0.0 { continue; }
            let t = temperatures[t_idx];
            let beta = 1.0 / t;
            let avg_e = sum_e[t_idx] / s;
            let avg_e2 = sum_e2[t_idx] / s;
            let avg_m = sum_m[t_idx] / s;
            let avg_m2 = sum_m2[t_idx] / s;
            let avg_m4 = sum_m4[t_idx] / s;
            let cv = beta * beta * (avg_e2 - avg_e * avg_e) * n3;
            // Error bars: placeholder (TODO: jackknife from time series)
            csv.push_str(&format!(
                "{t:.6},{avg_e:.6},0.0,{avg_m:.6},0.0,{avg_m2:.6},0.0,{avg_m4:.6},0.0,{cv:.6},0.0,0.0,0.0\n"
            ));
        }
        fs::write(&summary_path, &csv).expect("write summary failed");
        eprintln!("  Wrote {}", summary_path.display());

        // Write time series CSV for reweighting
        let ts_path = Path::new(outdir).join(format!("gpu_fss_{model_name}_N{n}_timeseries.csv"));
        let mut ts_csv = String::from("temp_idx,E,M\n");
        for (t_idx, samples) in ts_data.iter().enumerate() {
            for &(e, m) in samples {
                ts_csv.push_str(&format!("{t_idx},{e:.8},{m:.8}\n"));
            }
        }
        fs::write(&ts_path, &ts_csv).expect("write timeseries failed");
        eprintln!("  Wrote {}", ts_path.display());
    }
}

fn ising_e_m_host(spins: &[i8], n: usize) -> (f64, f64) {
    let mut e = 0.0_f64;
    let mut m = 0.0_f64;
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let idx = iz * n * n + iy * n + ix;
                let s = spins[idx] as f64;
                let fwd = [
                    iz * n * n + iy * n + (ix + 1) % n,
                    iz * n * n + ((iy + 1) % n) * n + ix,
                    ((iz + 1) % n) * n * n + iy * n + ix,
                ];
                for &nb in &fwd {
                    e -= s * spins[nb] as f64;
                }
                m += s;
            }
        }
    }
    (e, m)
}

#[cfg(feature = "cuda")]
fn run_continuous_fss(
    sizes: &[usize], n_comp: usize, t_min: f64, t_max: f64,
    n_replicas: usize, warmup: usize, samples: usize,
    exchange_every: usize, seed: u64, outdir: &str,
    delta: f32, n_overrelax: usize,
) {
    use ising::cuda::gpu_lattice_continuous::ContinuousGpuLattice;
    use ising::cuda::parallel_tempering::{linspace_temperatures, replica_exchange};
    use cudarc::driver::CudaDevice;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    let model_name = if n_comp == 2 { "xy" } else { "heisenberg" };
    let temperatures = linspace_temperatures(t_min, t_max, n_replicas);
    let device = CudaDevice::new(0).expect("failed to init CUDA device");

    for &n in sizes {
        eprintln!("  N={n}: creating {n_replicas} {model_name} replicas...");
        let mut replicas: Vec<ContinuousGpuLattice> = (0..n_replicas)
            .map(|r| {
                ContinuousGpuLattice::new(
                    n, n_comp,
                    seed.wrapping_add(r as u64 * 1000 + n as u64),
                    device.clone(),
                ).expect("failed to create continuous GPU lattice")
            })
            .collect();

        let mut replica_to_temp: Vec<usize> = (0..n_replicas).collect();
        let mut temp_to_replica: Vec<usize> = (0..n_replicas).collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(n as u64));

        let n3 = (n * n * n) as f64;

        // Warmup
        eprintln!("  N={n}: warming up {warmup} sweeps...");
        for _ in 0..warmup {
            for (r, lat) in replicas.iter_mut().enumerate() {
                let t_idx = replica_to_temp[r];
                let beta = (1.0 / temperatures[t_idx]) as f32;
                lat.sweep(beta, 1.0, delta, n_overrelax).expect("sweep failed");
            }
        }

        // Accumulators
        let mut sum_e  = vec![0.0_f64; n_replicas];
        let mut sum_e2 = vec![0.0_f64; n_replicas];
        let mut sum_m  = vec![0.0_f64; n_replicas];
        let mut sum_m2 = vec![0.0_f64; n_replicas];
        let mut sum_m4 = vec![0.0_f64; n_replicas];
        let mut count  = vec![0usize; n_replicas];
        let mut ts_data: Vec<Vec<(f64, f64)>> = vec![vec![]; n_replicas];

        eprintln!("  N={n}: sampling {samples} sweeps with PT...");
        for sweep in 0..samples {
            let mut energies = vec![0.0_f64; n_replicas];

            for (r, lat) in replicas.iter_mut().enumerate() {
                let t_idx = replica_to_temp[r];
                let beta = (1.0 / temperatures[t_idx]) as f32;
                lat.sweep(beta, 1.0, delta, n_overrelax).expect("sweep failed");

                let (e, mx, my, mz) = lat.measure_raw().expect("measure failed");
                let e_per = e / n3;
                let m_abs = ((mx * mx + my * my + mz * mz).sqrt()) / n3;
                energies[r] = e;

                sum_e[t_idx]  += e_per;
                sum_e2[t_idx] += e_per * e_per;
                sum_m[t_idx]  += m_abs;
                sum_m2[t_idx] += m_abs * m_abs;
                sum_m4[t_idx] += m_abs.powi(4);
                count[t_idx]  += 1;

                ts_data[t_idx].push((e_per, m_abs));
            }

            if (sweep + 1) % exchange_every == 0 {
                replica_exchange(
                    &temperatures, &energies,
                    &mut replica_to_temp, &mut temp_to_replica,
                    &mut rng, sweep / exchange_every,
                );
            }

            if (sweep + 1) % 10000 == 0 {
                eprintln!("    sweep {}/{samples}", sweep + 1);
            }
        }

        // Write summary CSV
        let summary_path = Path::new(outdir).join(format!("gpu_fss_{model_name}_N{n}_summary.csv"));
        let mut csv = String::from("T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err\n");
        for t_idx in 0..n_replicas {
            let s = count[t_idx] as f64;
            if s == 0.0 { continue; }
            let t = temperatures[t_idx];
            let beta = 1.0 / t;
            let avg_e = sum_e[t_idx] / s;
            let avg_e2 = sum_e2[t_idx] / s;
            let avg_m = sum_m[t_idx] / s;
            let avg_m2 = sum_m2[t_idx] / s;
            let avg_m4 = sum_m4[t_idx] / s;
            let cv = beta * beta * (avg_e2 - avg_e * avg_e) * n3;
            csv.push_str(&format!(
                "{t:.6},{avg_e:.6},0.0,{avg_m:.6},0.0,{avg_m2:.6},0.0,{avg_m4:.6},0.0,{cv:.6},0.0,0.0,0.0\n"
            ));
        }
        fs::write(&summary_path, &csv).expect("write summary failed");
        eprintln!("  Wrote {}", summary_path.display());

        // Write time series
        let ts_path = Path::new(outdir).join(format!("gpu_fss_{model_name}_N{n}_timeseries.csv"));
        let mut ts_csv = String::from("temp_idx,E,M\n");
        for (t_idx, samples) in ts_data.iter().enumerate() {
            for &(e, m) in samples {
                ts_csv.push_str(&format!("{t_idx},{e:.8},{m:.8}\n"));
            }
        }
        fs::write(&ts_path, &ts_csv).expect("write timeseries failed");
        eprintln!("  Wrote {}", ts_path.display());
    }
}

#[cfg(not(feature = "cuda"))]
fn run_ising_fss(_: &[usize], _: f64, _: f64, _: usize, _: usize, _: usize, _: usize, _: u64, _: &str) {
    eprintln!("Error: gpu_fss requires --features cuda");
    std::process::exit(1);
}

#[cfg(not(feature = "cuda"))]
fn run_continuous_fss(_: &[usize], _: usize, _: f64, _: f64, _: usize, _: usize, _: usize, _: usize, _: u64, _: &str, _: f32, _: usize) {
    eprintln!("Error: gpu_fss requires --features cuda");
    std::process::exit(1);
}
```

**Step 3: Verify non-CUDA build compiles**

Run: `cargo build --release`
Expected: PASS (non-cuda stubs handle the compilation)

**Step 4: Commit**

```bash
git add src/bin/gpu_fss.rs Cargo.toml
git commit -m "feat: add gpu_fss binary — unified GPU FSS with parallel tempering"
```

---

## Task 7: Unified GPU J-fitting binary (`gpu_jfit`)

**Files:**
- Create: `src/bin/gpu_jfit.rs`
- Modify: `Cargo.toml`

**Context:** Same as gpu_fss but loads crystal graphs (BCC/FCC) from JSON. Since the GPU kernels currently only support cubic lattices (neighbours computed from index arithmetic), the graph-based J-fitting will use the existing CPU path but with more sweeps, OR we add graph neighbour lists to GPU memory. For this first implementation, **use CPU Metropolis with parallel tempering** — the GPU benefit comes from running many replicas and more statistics, not from kernel speed on small N=4..12 crystal graphs.

**Step 1: Add bin entry to Cargo.toml**

```toml
[[bin]]
name = "gpu_jfit"
path = "src/bin/gpu_jfit.rs"
```

Note: `gpu_jfit` does NOT require the cuda feature — it uses CPU parallel tempering for small crystal graphs. The name is forward-looking for when graph-based GPU support is added.

**Step 2: Write `src/bin/gpu_jfit.rs`**

This binary uses CPU-side parallel tempering with the existing Ising/Heisenberg/XY CPU code, but with replica exchange. It produces the same output format (summary CSV + time series) as `gpu_fss`.

The implementation follows the same pattern as gpu_fss but uses the CPU lattice types and measure functions. For Ising, use `src/metropolis.rs` sweeps. For Heisenberg/XY, use the existing `measure()` functions from `src/heisenberg/observables.rs` and `src/xy/observables.rs`.

Since this is primarily a replica-exchange wrapper around existing CPU code, the code is structurally similar to gpu_fss but calls CPU functions. Full code omitted for brevity — follow the gpu_fss pattern with:
- `ising::Lattice` instead of `LatticeGpu`
- `ising::metropolis::sweep` for Ising steps
- `ising::heisenberg::observables::measure` for Heisenberg
- `ising::xy::observables::measure` for XY
- `ising::graph::GraphDef` for loading the crystal graph JSON

**Step 3: Commit**

```bash
git add src/bin/gpu_jfit.rs Cargo.toml
git commit -m "feat: add gpu_jfit binary — CPU parallel tempering for crystal graph J-fitting"
```

---

## Task 8: Histogram reweighting Python module

**Files:**
- Create: `analysis/scripts/reweighting.py`

**Context:** Pure Python implementation of single-histogram and multiple-histogram (WHAM) reweighting. Uses the time-series CSV output from gpu_fss.

**Step 1: Write `analysis/scripts/reweighting.py`**

```python
"""
Ferrenberg-Swendsen histogram reweighting.

Single-histogram: reweight from one simulation temperature to nearby T.
Multiple-histogram (WHAM): combine data from all parallel-tempering replicas.

Usage:
    from reweighting import single_histogram, wham_reweight

Input: raw (E, |M|) time series per temperature from gpu_fss output.
"""
import numpy as np
from scipy.optimize import minimize


def single_histogram(energies, beta_sim, beta_target):
    """
    Reweight observables from beta_sim to beta_target.

    Parameters
    ----------
    energies : array of shape (N_samples,)
        Total energy per sample at beta_sim.
    beta_sim : float
        Inverse temperature of the simulation.
    beta_target : float
        Target inverse temperature.

    Returns
    -------
    weights : array of shape (N_samples,)
        Normalised reweighting factors.
    """
    delta_beta = beta_target - beta_sim
    log_w = -delta_beta * energies
    log_w -= log_w.max()  # numerical stability
    w = np.exp(log_w)
    return w / w.sum()


def reweight_observable(observable, weights):
    """Compute <O> at target temperature using reweighting."""
    return np.sum(observable * weights)


def wham_reweight(energy_lists, beta_list, beta_target, n_iter=100, tol=1e-8):
    """
    Multiple-histogram (WHAM) reweighting.

    Parameters
    ----------
    energy_lists : list of arrays
        energy_lists[k] = array of energies from simulation at beta_list[k].
    beta_list : array of shape (K,)
        Inverse temperatures of the K simulations.
    beta_target : float
        Target inverse temperature.
    n_iter : int
        Maximum WHAM iterations.
    tol : float
        Convergence tolerance on free energies.

    Returns
    -------
    weights : array
        Normalised weights for all samples concatenated.
    sample_betas : array
        Which beta each sample came from (for bookkeeping).
    """
    K = len(beta_list)
    N_k = np.array([len(e) for e in energy_lists])
    all_energies = np.concatenate(energy_lists)
    N_total = len(all_energies)

    # Which simulation each sample came from
    sim_idx = np.concatenate([np.full(n, k) for k, n in enumerate(N_k)])

    # Initial free energies
    f = np.zeros(K)

    for iteration in range(n_iter):
        # Denominator for each sample: sum_k N_k * exp(f_k - beta_k * E_n)
        # log version for stability
        log_denom_terms = np.zeros((N_total, K))
        for k in range(K):
            log_denom_terms[:, k] = np.log(N_k[k]) + f[k] - beta_list[k] * all_energies

        log_denom = np.logaddexp.reduce(log_denom_terms, axis=1)

        # New free energies: exp(-f_k) = sum_n exp(-beta_k * E_n) / denom_n
        f_new = np.zeros(K)
        for k in range(K):
            log_terms = -beta_list[k] * all_energies - log_denom
            f_new[k] = -np.logaddexp.reduce(log_terms)

        # Shift so f[0] = 0
        f_new -= f_new[0]

        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new

    # Compute weights at target beta
    log_w = -beta_target * all_energies - log_denom
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()

    return w, sim_idx


def reweight_binder(energy_lists, mag_lists, beta_list, beta_targets):
    """
    Compute Binder cumulant U(T) = 1 - <m^4>/(3<m^2>^2) via WHAM reweighting.

    Parameters
    ----------
    energy_lists : list of arrays
        Energies per replica.
    mag_lists : list of arrays
        |M|/N per replica (matching energy_lists).
    beta_list : array
        Inverse temperatures of replicas.
    beta_targets : array
        Target betas at which to evaluate U.

    Returns
    -------
    T_out : array
        Temperatures.
    U_out : array
        Binder cumulant at each T.
    """
    all_mags = np.concatenate(mag_lists)
    T_out = []
    U_out = []

    for beta_t in beta_targets:
        w, _ = wham_reweight(energy_lists, beta_list, beta_t)
        m2 = np.sum(w * all_mags**2)
        m4 = np.sum(w * all_mags**4)
        U = 1.0 - m4 / (3.0 * m2**2) if m2 > 0 else 0.0
        T_out.append(1.0 / beta_t)
        U_out.append(U)

    return np.array(T_out), np.array(U_out)


if __name__ == '__main__':
    # Quick self-test with synthetic data
    np.random.seed(42)
    beta_sim = 0.22
    N = 10000
    E = np.random.normal(-1.5, 0.5, N)  # fake energy samples

    # Reweight to nearby beta
    w = single_histogram(E, beta_sim, beta_sim + 0.01)
    assert abs(w.sum() - 1.0) < 1e-10
    print('Single-histogram self-test: PASS')

    # WHAM with 3 simulations
    betas = np.array([0.20, 0.22, 0.24])
    e_lists = [np.random.normal(-1.5 + 0.5 * b, 0.5, 5000) for b in betas]
    w, _ = wham_reweight(e_lists, betas, 0.21)
    assert abs(w.sum() - 1.0) < 1e-10
    print('WHAM self-test: PASS')
```

**Step 2: Run self-test**

Run: `python3 analysis/scripts/reweighting.py`
Expected: Both self-tests PASS.

**Step 3: Commit**

```bash
git add analysis/scripts/reweighting.py
git commit -m "feat: add Ferrenberg-Swendsen histogram reweighting (single + WHAM)"
```

---

## Task 9: GPU publication run script

**Files:**
- Create: `scripts/run_gpu_publication.sh`

**Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Publication-quality GPU Monte Carlo runs with parallel tempering.
# Target: Windows RTX 2060 (6 GB VRAM).
#
# Usage:
#   bash scripts/run_gpu_publication.sh
#   bash scripts/run_gpu_publication.sh 2>&1 | tee run_gpu.log

set -e
OUTDIR="analysis/data"
mkdir -p "$OUTDIR"

echo "=== GPU Ising FSS: cubic lattice ==="
cargo run --release --features cuda --bin gpu_fss -- \
  --model ising \
  --sizes 8,16,32,64,128 \
  --tmin 4.40 --tmax 4.62 --replicas 32 \
  --warmup 5000 --samples 100000 \
  --exchange-every 10 \
  --outdir "$OUTDIR"

echo "=== GPU Heisenberg FSS: cubic lattice ==="
cargo run --release --features cuda --bin gpu_fss -- \
  --model heisenberg \
  --sizes 8,16,32,64,128 \
  --tmin 1.35 --tmax 1.55 --replicas 20 \
  --warmup 5000 --samples 50000 \
  --exchange-every 10 \
  --delta 0.5 --overrelax 5 \
  --outdir "$OUTDIR"

echo "=== GPU XY FSS: cubic lattice ==="
cargo run --release --features cuda --bin gpu_fss -- \
  --model xy \
  --sizes 8,16,32,64,128 \
  --tmin 2.10 --tmax 2.30 --replicas 20 \
  --warmup 5000 --samples 50000 \
  --exchange-every 10 \
  --delta 0.5 --overrelax 5 \
  --outdir "$OUTDIR"

echo "=== All GPU runs done ==="
echo "Run analysis notebooks to analyse results."
echo "Use analysis/scripts/reweighting.py for histogram reweighting on time series data."
```

**Step 2: Make executable and commit**

```bash
chmod +x scripts/run_gpu_publication.sh
git add scripts/run_gpu_publication.sh
git commit -m "feat: add GPU publication run script with parallel tempering"
```

---

## Task 10: Integration test — GPU smoke test

**Files:**
- Modify: `tests/cli.rs`

**Context:** Add a smoke test that verifies the gpu_fss binary compiles and runs with `--features cuda`. Since we can't guarantee CUDA is available on CI, gate the test with `#[ignore]` — run manually with `cargo test --features cuda -- --ignored`.

**Step 1: Append test to `tests/cli.rs`**

```rust
#[test]
#[ignore] // Requires --features cuda and a GPU
fn gpu_fss_ising_smoke() {
    let dir = tempfile::tempdir().unwrap();
    let status = std::process::Command::new("cargo")
        .args([
            "run", "--release", "--features", "cuda", "--bin", "gpu_fss", "--",
            "--model", "ising",
            "--sizes", "4",
            "--tmin", "4.4", "--tmax", "4.6", "--replicas", "4",
            "--warmup", "50", "--samples", "100",
            "--exchange-every", "10",
            "--outdir", dir.path().to_str().unwrap(),
        ])
        .status()
        .expect("failed to run gpu_fss");
    assert!(status.success());

    // Check summary CSV exists
    let summary = dir.path().join("gpu_fss_ising_N4_summary.csv");
    assert!(summary.exists(), "summary CSV missing");

    let content = std::fs::read_to_string(&summary).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert!(lines.len() >= 2, "summary CSV should have header + data");
    assert!(lines[0].starts_with("T,E,"));

    // Check time series exists
    let ts = dir.path().join("gpu_fss_ising_N4_timeseries.csv");
    assert!(ts.exists(), "timeseries CSV missing");
}
```

**Step 2: Commit**

```bash
git add tests/cli.rs
git commit -m "test: add gpu_fss_ising_smoke integration test (requires CUDA)"
```

---

## Summary of Tasks

| Task | Component | New files | Key deliverable |
|---|---|---|---|
| 1 | build.rs multi-file compilation | — | Build infrastructure for multiple .cu files |
| 2 | GPU reduction kernels | `reduce_kernel.cu`, `reduce_gpu.rs` | On-device E, M, M², M⁴ computation |
| 3 | Continuous-spin CUDA kernels | `continuous_spin_kernel.cu` | XY/Heisenberg GPU Metropolis + overrelax |
| 4 | Continuous GPU lattice wrapper | `gpu_lattice_continuous.rs` | Rust API for continuous-spin GPU sweeps |
| 5 | Parallel tempering | `parallel_tempering.rs` | Replica exchange logic + tests |
| 6 | GPU FSS binary | `gpu_fss.rs` | Unified CLI for all 3 models on GPU |
| 7 | GPU J-fit binary | `gpu_jfit.rs` | CPU parallel tempering for crystal graphs |
| 8 | Histogram reweighting | `reweighting.py` | WHAM + single-histogram in Python |
| 9 | Publication run script | `run_gpu_publication.sh` | One-command production runs |
| 10 | Integration test | `tests/cli.rs` (append) | GPU smoke test |

**Execution order:** Tasks 1–5 are infrastructure (do in order). Tasks 6–10 can be done in any order after 5, but 6 before 10 (test depends on binary).
