#include <math.h>
#include <cuda_fp16.h>

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

// --- Continuous spins: partial energy (sum of -J * S_i · S_j - D * sz^2, forward neighbours only) ---
extern "C" __global__ void reduce_energy_continuous(
    const float* spins,
    float* partial_energy,
    int    N,
    int    n_comp,
    float  J,
    float  D           // uniaxial anisotropy (Heisenberg only)
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

        int fwd[3];
        fwd[0] = z*N*N + y*N + (x+1)%N;
        fwd[1] = z*N*N + ((y+1)%N)*N + x;
        fwd[2] = ((z+1)%N)*N*N + y*N + x;

        for (int f = 0; f < 3; f++) {
            float dot = 0.0f;
            for (int c = 0; c < n_comp; c++) {
                dot += spins[gid * n_comp + c] * spins[fwd[f] * n_comp + c];
            }
            val -= J * dot;
        }

        // Anisotropy: -D * sz^2 (per site, not per bond)
        if (n_comp == 3) {
            float sz = spins[gid * n_comp + 2];
            val -= D * sz * sz;
        }
    }

    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_energy[blockIdx.x] = sdata[0];
}

// ============================================================
// FP16 continuous spin reductions
// ============================================================

extern "C" __global__ void reduce_mag_fp16(
    const __half* spins,
    float* partial_mx,
    float* partial_my,
    float* partial_mz,
    int    n_sites,
    int    n_comp
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int bd = blockDim.x;

    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    if (gid < n_sites) {
        mx = __half2float(spins[gid * n_comp + 0]);
        my = __half2float(spins[gid * n_comp + 1]);
        if (n_comp == 3) mz = __half2float(spins[gid * n_comp + 2]);
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

extern "C" __global__ void reduce_energy_fp16(
    const __half* spins,
    float* partial_energy,
    int    N,
    int    n_comp,
    float  J,
    float  D
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

        int fwd[3];
        fwd[0] = z*N*N + y*N + (x+1)%N;
        fwd[1] = z*N*N + ((y+1)%N)*N + x;
        fwd[2] = ((z+1)%N)*N*N + y*N + x;

        for (int f = 0; f < 3; f++) {
            float dot = 0.0f;
            for (int c = 0; c < n_comp; c++) {
                dot += __half2float(spins[gid * n_comp + c]) *
                       __half2float(spins[fwd[f] * n_comp + c]);
            }
            val -= J * dot;
        }

        if (n_comp == 3) {
            float sz = __half2float(spins[gid * n_comp + 2]);
            val -= D * sz * sz;
        }
    }

    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_energy[blockIdx.x] = sdata[0];
}

// ============================================================
// XY angle-only reductions (1 angle per spin as __half)
// ============================================================

extern "C" __global__ void reduce_mag_xy_angle(
    const __half* angles,
    float* partial_mx,
    float* partial_my,
    int    n_sites
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int bd = blockDim.x;

    float mx = 0.0f, my = 0.0f;
    if (gid < n_sites) {
        float theta = __half2float(angles[gid]);
        mx = cosf(theta);
        my = sinf(theta);
    }
    sdata[tid]      = mx;
    sdata[tid + bd] = my;
    __syncthreads();

    for (int s = bd / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid]      += sdata[tid + s];
            sdata[tid + bd] += sdata[tid + s + bd];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_mx[blockIdx.x] = sdata[0];
        partial_my[blockIdx.x] = sdata[bd];
    }
}

extern "C" __global__ void reduce_energy_xy_angle(
    const __half* angles,
    float* partial_energy,
    int    N,
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

        float theta_i = __half2float(angles[gid]);
        float ci = cosf(theta_i), si = sinf(theta_i);

        int fwd[3];
        fwd[0] = z*N*N + y*N + (x+1)%N;
        fwd[1] = z*N*N + ((y+1)%N)*N + x;
        fwd[2] = ((z+1)%N)*N*N + y*N + x;

        for (int f = 0; f < 3; f++) {
            float theta_j = __half2float(angles[fwd[f]]);
            val -= J * (ci * cosf(theta_j) + si * sinf(theta_j));
        }
    }

    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_energy[blockIdx.x] = sdata[0];
}
