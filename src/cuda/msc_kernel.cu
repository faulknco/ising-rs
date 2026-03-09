#include <curand_kernel.h>
#include <math.h>

typedef curandStatePhilox4_32_10 RngState;

extern "C" __global__ void msc_init_rng_kernel(
    RngState* states,
    unsigned long long seed,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    curand_init(seed, tid, 0, &states[tid]);
}

extern "C" __global__ void msc_init_spins_kernel(
    unsigned int* spins_msc,
    int n_words
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_words) return;
    spins_msc[tid] = 0xFFFFFFFF;
}

extern "C" __global__ void msc_randomise_kernel(
    unsigned int* spins_msc,
    RngState* rng_states,
    int n_words
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_words) return;
    RngState local_rng = rng_states[tid];
    spins_msc[tid] = curand(&local_rng);
    rng_states[tid] = local_rng;
}

extern "C" __global__ void msc_metropolis_kernel(
    unsigned int* spins_msc,
    RngState*     rng_states,
    int           N,
    float         beta,
    float         J,
    int           parity,
    const float*  boltz_probs
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int wpr = N / 32;
    int n_words = N * N * wpr;
    if (tid >= n_words) return;

    int wx = tid % wpr;
    int y  = (tid / wpr) % N;
    int z  = tid / (wpr * N);
    int x_base = wx * 32;

    unsigned int parity_mask;
    int base_parity = (x_base + y + z) % 2;
    if (base_parity == parity) {
        parity_mask = 0x55555555;
    } else {
        parity_mask = 0xAAAAAAAA;
    }

    unsigned int my_word = spins_msc[tid];

    int wx_right = (wx + 1) % wpr;
    int wx_left  = (wx + wpr - 1) % wpr;
    unsigned int word_xp_raw = spins_msc[z * N * wpr + y * wpr + wx_right];
    unsigned int word_xm_raw = spins_msc[z * N * wpr + y * wpr + wx_left];

    unsigned int nb_xp = (my_word >> 1) | ((word_xp_raw & 1u) << 31);
    unsigned int nb_xm = (my_word << 1) | ((word_xm_raw >> 31) & 1u);

    int yp = (y + 1) % N;
    int ym = (y + N - 1) % N;
    unsigned int nb_yp = spins_msc[z * N * wpr + yp * wpr + wx];
    unsigned int nb_ym = spins_msc[z * N * wpr + ym * wpr + wx];

    int zp = (z + 1) % N;
    int zm = (z + N - 1) % N;
    unsigned int nb_zp = spins_msc[zp * N * wpr + y * wpr + wx];
    unsigned int nb_zm = spins_msc[zm * N * wpr + y * wpr + wx];

    unsigned int anti_xp = my_word ^ nb_xp;
    unsigned int anti_xm = my_word ^ nb_xm;
    unsigned int anti_yp = my_word ^ nb_yp;
    unsigned int anti_ym = my_word ^ nb_ym;
    unsigned int anti_zp = my_word ^ nb_zp;
    unsigned int anti_zm = my_word ^ nb_zm;

    unsigned int s1 = anti_xp ^ anti_xm ^ anti_yp;
    unsigned int c1 = (anti_xp & anti_xm) | (anti_xp & anti_yp) | (anti_xm & anti_yp);
    unsigned int s2 = anti_ym ^ anti_zp ^ anti_zm;
    unsigned int c2 = (anti_ym & anti_zp) | (anti_ym & anti_zm) | (anti_zp & anti_zm);

    unsigned int t0 = s1 ^ s2;
    unsigned int t1 = s1 & s2;
    unsigned int u0 = t0 ^ c1;
    unsigned int u1a = t0 & c1;
    unsigned int v1 = t1 ^ u1a ^ c2;
    unsigned int v2 = (t1 & u1a) | (t1 & c2) | (u1a & c2);

    RngState local_rng = rng_states[tid];
    unsigned int flip_mask = 0;

    for (int k = -3; k <= 3; k++) {
        int n_anti = k + 3;
        unsigned int want_b2 = (n_anti >> 2) & 1 ? 0xFFFFFFFF : 0;
        unsigned int want_b1 = (n_anti >> 1) & 1 ? 0xFFFFFFFF : 0;
        unsigned int want_b0 = (n_anti >> 0) & 1 ? 0xFFFFFFFF : 0;

        unsigned int has_count = ~(v2 ^ want_b2) & ~(v1 ^ want_b1) & ~(u0 ^ want_b0);
        if (has_count == 0) continue;

        float prob = boltz_probs[k + 3];
        if (prob >= 1.0f) {
            flip_mask |= has_count;
        } else if (prob > 0.0f) {
            unsigned int rand_bits = curand(&local_rng);
            unsigned int threshold = (unsigned int)(prob * 4294967296.0f);
            if (rand_bits < threshold) {
                flip_mask |= has_count;
            }
        }
    }

    flip_mask &= parity_mask;
    spins_msc[tid] = my_word ^ flip_mask;
    rng_states[tid] = local_rng;
}

extern "C" __global__ void msc_reduce_mag(
    const unsigned int* spins_msc,
    float* partial_mag,
    int n_words
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (gid < n_words) {
        int up = __popc(spins_msc[gid]);
        val = (float)(2 * up - 32);
    }
    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_mag[blockIdx.x] = sdata[0];
}

extern "C" __global__ void msc_reduce_energy(
    const unsigned int* spins_msc,
    float* partial_energy,
    int N,
    float J
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int wpr = N / 32;
    int n_words = N * N * wpr;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float val = 0.0f;
    if (gid < n_words) {
        int wx = gid % wpr;
        int y  = (gid / wpr) % N;
        int z  = gid / (wpr * N);

        unsigned int my_word = spins_msc[gid];

        int wx_right = (wx + 1) % wpr;
        unsigned int word_xp = spins_msc[z * N * wpr + y * wpr + wx_right];
        unsigned int nb_xp = (my_word >> 1) | ((word_xp & 1u) << 31);

        int yp = (y + 1) % N;
        unsigned int nb_yp = spins_msc[z * N * wpr + yp * wpr + wx];

        int zp = (z + 1) % N;
        unsigned int nb_zp = spins_msc[zp * N * wpr + y * wpr + wx];

        int same_xp = __popc(~(my_word ^ nb_xp));
        int same_yp = __popc(~(my_word ^ nb_yp));
        int same_zp = __popc(~(my_word ^ nb_zp));

        val = -J * (float)((2*same_xp - 32) + (2*same_yp - 32) + (2*same_zp - 32));
    }

    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_energy[blockIdx.x] = sdata[0];
}
