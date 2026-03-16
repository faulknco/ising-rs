#include <curand_kernel.h>
#include <math.h>

typedef curandStatePhilox4_32_10 RngState;

// Checkerboard Metropolis kernel for 3D cubic Ising model.
// parity=0: update black sites (x+y+z even), parity=1: white sites (x+y+z odd).
// spins: device array of int8_t, size N*N*N.
// rng_states: per-thread Philox RNG state (16 bytes).

extern "C" __global__ void metropolis_sweep_kernel(
    signed char* spins,
    RngState*     rng_states,
    int           N,
    float         beta,
    float         J,
    float         h,
    int           parity
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (N * N * N) / 2;
    if (tid >= total) return;

    // Map thread id to a (x,y,z) with (x+y+z) % 2 == parity
    int full_idx = tid * 2;
    int z = full_idx / (N * N);
    int r = full_idx % (N * N);
    int y = r / N;
    int x = r % N;
    if ((x + y + z) % 2 != parity) x = (x + 1) % N;
    if (x >= N) { return; }

    int idx = z * N * N + y * N + x;

    int xp = (x + 1) % N, xm = (x - 1 + N) % N;
    int yp = (y + 1) % N, ym = (y - 1 + N) % N;
    int zp = (z + 1) % N, zm = (z - 1 + N) % N;

    float nb_sum = (float)(
        spins[z*N*N + y*N + xp] +
        spins[z*N*N + y*N + xm] +
        spins[z*N*N + yp*N + x] +
        spins[z*N*N + ym*N + x] +
        spins[zp*N*N + y*N + x] +
        spins[zm*N*N + y*N + x]
    );

    float spin_f = (float)spins[idx];
    float delta_e = 2.0f * spin_f * (J * nb_sum + h);

    RngState local_rng = rng_states[tid];
    float u = curand_uniform(&local_rng);
    rng_states[tid] = local_rng;

    if (delta_e < 0.0f || u < expf(-beta * delta_e)) {
        spins[idx] = -spins[idx];
    }
}

extern "C" __global__ void init_rng_kernel(RngState* states, unsigned long long seed, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    curand_init(seed, tid, 0, &states[tid]);
}

extern "C" __global__ void sum_spins_kernel(
    const signed char* spins,
    int* partial_sums,
    int n
) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < n) ? (int)spins[gid] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}
