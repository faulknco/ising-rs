#include <math.h>

// Block-wide reduction using shared memory.
// Each block reduces its portion; host does final sum over partial results.

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
