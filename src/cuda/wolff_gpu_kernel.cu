#include <curand_kernel.h>
#include <math.h>

typedef curandStatePhilox4_32_10 RngState;

extern "C" __global__ void wolff_init_rng_kernel(
    RngState* states,
    unsigned long long seed,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    curand_init(seed, tid, 0, &states[tid]);
}

// Phase A: Propose bonds and initialise labels.
extern "C" __global__ void wolff_bond_proposal_kernel(
    const signed char* spins,
    unsigned char*     bonds,      // [N^3 * 6]
    unsigned int*      labels,     // [N^3]
    RngState*          rng_states, // [N^3]
    int                N,
    float              p_add
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_sites = N * N * N;
    if (gid >= n_sites) return;

    labels[gid] = (unsigned int)gid;

    int z = gid / (N * N);
    int r = gid % (N * N);
    int y = r / N;
    int x = r % N;

    signed char my_spin = spins[gid];

    int nb[6];
    nb[0] = z*N*N + y*N + (x+1)%N;
    nb[1] = z*N*N + y*N + (x-1+N)%N;
    nb[2] = z*N*N + ((y+1)%N)*N + x;
    nb[3] = z*N*N + ((y-1+N)%N)*N + x;
    nb[4] = ((z+1)%N)*N*N + y*N + x;
    nb[5] = ((z-1+N)%N)*N*N + y*N + x;

    RngState local_rng = rng_states[gid];

    for (int d = 0; d < 6; d++) {
        int bond_idx = gid * 6 + d;
        if (spins[nb[d]] == my_spin && curand_uniform(&local_rng) < p_add) {
            bonds[bond_idx] = 1;
        } else {
            bonds[bond_idx] = 0;
        }
    }

    rng_states[gid] = local_rng;
}

// Phase B: Label propagation iteration.
extern "C" __global__ void wolff_propagate_kernel(
    const unsigned char* bonds,
    unsigned int*        labels,
    int*                 changed,
    int                  N
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_sites = N * N * N;
    if (gid >= n_sites) return;

    unsigned int my_label = labels[gid];
    unsigned int min_label = my_label;

    int z = gid / (N * N);
    int r = gid % (N * N);
    int y = r / N;
    int x = r % N;

    int nb[6];
    nb[0] = z*N*N + y*N + (x+1)%N;
    nb[1] = z*N*N + y*N + (x-1+N)%N;
    nb[2] = z*N*N + ((y+1)%N)*N + x;
    nb[3] = z*N*N + ((y-1+N)%N)*N + x;
    nb[4] = ((z+1)%N)*N*N + y*N + x;
    nb[5] = ((z-1+N)%N)*N*N + y*N + x;

    for (int d = 0; d < 6; d++) {
        if (bonds[gid * 6 + d]) {
            unsigned int nb_label = labels[nb[d]];
            if (nb_label < min_label) {
                min_label = nb_label;
            }
        }
    }

    if (min_label < my_label) {
        atomicMin(&labels[my_label], min_label);
        atomicMin(&labels[gid], min_label);
        atomicMax(changed, 1);
    }
}

// Phase B helper: flatten labels (path compression).
extern "C" __global__ void wolff_flatten_labels_kernel(
    unsigned int* labels,
    int n_sites
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_sites) return;

    unsigned int lbl = labels[gid];
    while (labels[lbl] != lbl) {
        lbl = labels[lbl];
    }
    labels[gid] = lbl;
}

// Phase C: Flip a chosen cluster.
extern "C" __global__ void wolff_flip_cluster_kernel(
    signed char*        spins,
    const unsigned int* labels,
    unsigned int        flip_label,
    int                 n_sites
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_sites) return;

    if (labels[gid] == flip_label) {
        spins[gid] = -spins[gid];
    }
}

// Utility: read label of seed site.
extern "C" __global__ void wolff_pick_seed_kernel(
    const unsigned int* labels,
    unsigned int*       result,
    int                 seed_idx
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        result[0] = labels[seed_idx];
    }
}
