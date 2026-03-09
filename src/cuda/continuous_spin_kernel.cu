#include <curand_kernel.h>
#include <math.h>

typedef curandStatePhilox4_32_10 RngState;

// Continuous-spin checkerboard Metropolis for O(n) models.
// n_comp = 2 (XY) or 3 (Heisenberg).
// spins: interleaved [s0_x, s0_y, (s0_z), s1_x, ...], n_comp floats per site.
// Cubic lattice, periodic boundaries.

extern "C" __global__ void continuous_metropolis_kernel(
    float*       spins,
    RngState*    rng_states,
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
    RngState local_rng = rng_states[tid];
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
    RngState*    states,
    unsigned long long seed,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    curand_init(seed, tid, 0, &states[tid]);
}
