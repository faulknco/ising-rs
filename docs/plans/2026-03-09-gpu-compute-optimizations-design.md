# GPU Compute Optimizations: MSC + GPU Wolff + Philox RNG

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:writing-plans to create the implementation plan from this design.

**Goal:** Close the per-device efficiency gap with HPC Monte Carlo by implementing multi-spin coding, GPU-resident Wolff cluster, batched multi-replica kernels, and Philox RNG. Combined target: 60-100x speedup for Ising Metropolis path, 15-30x + eliminated critical slowing for Wolff path.

**Target hardware:** RTX 2060 (sm_75, 6 GB VRAM, 2560 CUDA cores).

**Prerequisite:** Phase 1 GPU implementation complete (checkerboard Metropolis, parallel tempering, GPU reductions).

---

## 1. Multi-Site Coded Ising (MSC)

### Encoding

Pack 32 lattice sites into one `u32` word. Bit k of `spins_msc[word_idx]` = spin at site `word_idx * 32 + k`. Bit 1 = up (+1), bit 0 = down (-1).

### Lattice Layout

The 3D cubic lattice of N^3 sites becomes `N^3 / 32` words. Sites are packed along the x-axis into 32-bit words, so each word holds a contiguous row of 32 x-coordinates. N must be a multiple of 32 (satisfied by target sizes: 32, 64, 128, 192).

Checkerboard parity still applies — black/white passes update disjoint sublattices. Each thread processes one 32-bit word containing 32 sites of the same parity.

### Neighbor Sum via Popcount

For each of the 6 cubic neighbors:

- **x+1, x-1:** Bit-rotate within the word (circular shift handles PBC along x).
- **y+1, y-1:** Load from adjacent words at offset +/- `N/32` in the word array.
- **z+1, z-1:** Load from words at offset +/- `N^2/32`.

For each neighbor direction, XOR the spin word with the neighbor word. `__popc(xor_word)` counts anti-aligned bits. From 6 neighbor XORs, each bit position gets a delta_E value.

### Acceptance

For each of the 5 possible delta_E values (0, +/-4J, +/-8J, +/-12J for z=6):

1. Pre-compute Boltzmann acceptance probability `min(1, exp(-beta * delta_E))` per replica temperature.
2. Generate one random `u32` per word via Philox RNG.
3. For each bit position: compare random bits against threshold to build an accept/reject mask.
4. Apply: `spins_msc[idx] ^= (flip_candidates & accept_mask)`.

### Expected Speedup

~20-30x over single-spin Metropolis. 32 sites per thread, fewer memory transactions per site, arithmetic replaced by bitwise ops + single-cycle `__popc()`.

---

## 2. Batched Multi-Replica Launch

### Memory Layout

Concatenate all replica spin buffers into one contiguous GPU allocation:

```
spins_all: [replica_0: N^3/32 words | replica_1: N^3/32 words | ... | replica_R-1]
```

### Thread Mapping

Total threads = `(N^3 / 32 / 2) * n_replicas` per checkerboard pass. Each thread computes:

- `replica = tid / words_per_replica_parity`
- `local_word = tid % words_per_replica_parity`
- Reads `beta` from `betas[replica]`

### Replica Exchange

After every `exchange_every` sweeps, a separate small kernel proposes swaps between adjacent-temperature replicas (even/odd scheduling). Reads per-replica energy from reduction buffers — no spin transfer, just swap beta assignments.

### RNG

One Philox state per thread across all replicas. Total RNG memory: `(N^3/32/2) * n_replicas * 16` bytes.

### Expected Speedup

~3x from eliminating per-replica kernel launch overhead and improving occupancy.

---

## 3. GPU Wolff Cluster (Label Propagation)

### Algorithm

Based on Komura & Okabe (2015). Three phases per cluster step:

**Phase A — Bond proposal (1 kernel launch):**
Each site proposes a bond to each neighbor with probability `p_add = 1 - exp(-2*beta*J)`, only if the neighbor has the same spin. Each site initializes `label[i] = i`.

**Phase B — Label propagation (iterative, 5-20 launches):**
Each site looks at bonded neighbors and adopts the minimum label via `atomicMin`. Repeats until convergence (no label changes), detected via a single-int atomic flag.

**Phase C — Cluster flip (1 kernel launch):**
Pick the seed site's label as the flip target. All sites with matching label get flipped.

### Memory Overhead

- Label array: N^3 x 4 bytes (u32)
- Bond-active array: N^3 x 6 bytes (one byte per neighbor direction)
- For N=192: ~28 MB labels + ~42 MB bonds = ~70 MB. Fits in 6 GB.

### Integration with MSC

Wolff operates on unpacked `i8` spins — cluster growth is irregular, no bitwise shortcut. GPU Wolff is an *alternative* to MSC Metropolis, not a composition:

- **Near Tc:** GPU Wolff (eliminates critical slowing, z ~= 0.33)
- **Away from Tc / KZ quench:** MSC Metropolis (maximum raw throughput)

Selection via CLI flag `--algorithm wolff|metropolis|auto`. Auto mode uses Wolff when `|T - Tc| / Tc < 0.1`, Metropolis otherwise.

### Expected Speedup

~5-10x effective near Tc by eliminating critical slowing down (z_Wolff ~= 0.33 vs z_Metropolis ~= 2).

---

## 4. Philox 4x32 RNG Replacement

### Change

Replace `curandState` (XORWOW, 48 bytes/thread) with `curandStatePhilox4_32_10` (16 bytes/thread) in all GPU kernels.

### API

Drop-in replacement — same `curand_uniform()`, `curand_init()` calls. Philox is counter-based so init is faster (no skip-ahead warmup).

### Memory Savings

For N=192, 32 replicas:
- Old: (192^3 / 2) * 32 * 48 bytes = ~5.4 GB (near VRAM limit)
- New: (192^3 / 2) * 32 * 16 bytes = ~1.8 GB (comfortable)

Unlocks larger lattices and more replicas within 6 GB.

### Scope

All kernel files updated:
- `kernels.cu` (Ising Metropolis)
- `continuous_spin_kernel.cu` (XY/Heisenberg)
- `msc_kernel.cu` (new MSC kernel)
- `wolff_gpu_kernel.cu` (new GPU Wolff kernel)

### Quality

Philox passes BigCrush. Standard RNG for GPU Monte Carlo (MILC, lattice QCD). No statistical quality concern.

### Expected Impact

~1.2x speedup (faster init, smaller state → better occupancy). Main win is VRAM savings enabling larger runs.

---

## 5. Testing & Correctness

Every optimization must pass a statistical parity test before merging. **Bitwise identity is NOT required** for MSC vs single-spin — different data layouts and RNG consumption patterns make exact match unrealistic and the wrong gate. The correct gate is: same detailed-balance target distribution → same equilibrium observables within statistical error.

| Test | Method | Tolerance |
|------|--------|-----------|
| MSC vs single-spin Metropolis | Same T/N, 10k+ sweeps, compare <E>, <\|M\|>, Cv, χ | Within 2σ jackknife error |
| Batched vs unbatched (single replica) | Same seed → compare spins | Bitwise identical (same kernel logic) |
| GPU Wolff vs CPU Wolff | 10k sweeps, compare <E>, <\|M\|> + report τ_int | Within 2σ; τ_int comparable |
| Philox vs curandState | Same kernel, different RNG → statistical comparison | <E>, <\|M\|> within 2σ |

Performance validation: benchmark matrix across three temperature regimes (low T, near Tc, high T) at N=32,64,128. Report ESS/sec (effective samples per second = N_samples / τ_int / wall_clock), not just raw sweeps/sec.

---

## 6. File Plan

### New Files

| File | Purpose |
|------|---------|
| `src/cuda/msc_kernel.cu` | MSC checkerboard Metropolis (32 sites/word) |
| `src/cuda/msc_lattice.rs` | Rust orchestration for MSC |
| `src/cuda/wolff_gpu_kernel.cu` | Label-propagation Wolff (bond, propagate, flip) |
| `src/cuda/wolff_gpu.rs` | Rust orchestration for GPU Wolff |

### Modified Files

| File | Change |
|------|--------|
| `src/cuda/kernels.cu` | curandState -> Philox |
| `src/cuda/continuous_spin_kernel.cu` | curandState -> Philox |
| `src/cuda/lattice_gpu.rs` | Philox RNG init, algorithm selection |
| `src/cuda/gpu_lattice_continuous.rs` | Philox RNG init |
| `src/cuda/mod.rs` | Export new modules |
| `build.rs` | Compile new .cu files |
| `src/bin/gpu_fss.rs` | --algorithm flag, MSC path |
| `tests/gpu_parity.rs` | MSC, Wolff, Philox, batched parity tests |

---

## 7. Expected Combined Impact

Speedups are not cleanly multiplicative — memory bandwidth, occupancy, reductions, and host-side I/O will limit real gains. Conservative and stretch targets:

| Optimization | Conservative | Stretch | VRAM Effect |
|---|---|---|---|
| MSC (32 sites/word) | ~10-15x | ~20-30x | 32x less spin memory |
| Batched replicas | ~1.5-2x | ~3x | Neutral |
| GPU Wolff (near Tc) | Uncertain | ~5-10x effective | +70 MB labels/bonds |
| Philox RNG | ~1.1x | ~1.2x | 3x less RNG memory |
| **Combined (Metropolis)** | **~15-30x** | **~30-60x** | Fits N=256+ in 6 GB |
| **Combined (Wolff near Tc)** | **Prototype first** | **~15-30x + no critical slowing** | |

GPU Wolff is a promising but unproven path. If it underperforms, keep CPU Wolff + GPU Metropolis split.

The correct performance metric is **effective samples per second** (ESS/sec = N_samples / τ_int / wall_clock), not raw sweeps/sec.
