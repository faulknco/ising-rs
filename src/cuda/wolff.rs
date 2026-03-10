use rand::Rng;

/// Optimised Wolff cluster flip for continuous Cartesian spins (XY or Heisenberg).
///
/// Operates entirely on host-side `&mut [f32]` data — no GPU calls.
/// Uses DFS (Vec stack) instead of BFS and a u64-packed bitset for cluster membership.
pub fn wolff_cluster_flip_continuous<R: Rng>(
    spins: &mut [f32],
    n: usize,
    n_comp: usize,
    beta: f32,
    j: f32,
    rng: &mut R,
) {
    let n_sites = n * n * n;
    let n2 = n * n;

    // Random reflection vector on unit sphere/circle
    let r: [f32; 3] = if n_comp == 3 {
        let z: f32 = rng.gen_range(-1.0..1.0);
        let phi: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
        let rho = (1.0 - z * z).sqrt();
        [rho * phi.cos(), rho * phi.sin(), z]
    } else {
        let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
        [angle.cos(), angle.sin(), 0.0]
    };

    // Compute projected spins: proj[i] = S_i · r
    let proj: Vec<f32> = (0..n_sites)
        .map(|i| {
            let mut dot = 0.0f32;
            for c in 0..n_comp {
                dot += spins[i * n_comp + c] * r[c];
            }
            dot
        })
        .collect();

    // Pick random seed site
    let seed_site: usize = rng.gen_range(0..n_sites);

    // Bitset for cluster membership (u64-packed, 8x smaller than Vec<bool>)
    let n_words = (n_sites + 63) / 64;
    let mut in_cluster = vec![0u64; n_words];

    #[inline(always)]
    fn set_bit(bits: &mut [u64], i: usize) {
        bits[i >> 6] |= 1u64 << (i & 63);
    }
    #[inline(always)]
    fn test_bit(bits: &[u64], i: usize) -> bool {
        (bits[i >> 6] >> (i & 63)) & 1 != 0
    }

    set_bit(&mut in_cluster, seed_site);

    // DFS stack (faster than BFS VecDeque — better cache locality, less overhead)
    let mut stack = Vec::with_capacity(1024);
    stack.push(seed_site);

    while let Some(site) = stack.pop() {
        let iz = site / n2;
        let iy = (site % n2) / n;
        let ix = site % n;

        let neighbors = [
            iz * n2 + iy * n + (ix + 1) % n,
            iz * n2 + iy * n + (ix + n - 1) % n,
            iz * n2 + ((iy + 1) % n) * n + ix,
            iz * n2 + ((iy + n - 1) % n) * n + ix,
            ((iz + 1) % n) * n2 + iy * n + ix,
            ((iz + n - 1) % n) * n2 + iy * n + ix,
        ];

        let proj_site = proj[site];
        for &nb in &neighbors {
            if test_bit(&in_cluster, nb) {
                continue;
            }
            let bond = proj_site * proj[nb];
            if bond <= 0.0 {
                continue;
            }
            let p_add = 1.0 - (-2.0 * beta * j * bond).exp();
            if rng.gen::<f32>() < p_add {
                set_bit(&mut in_cluster, nb);
                stack.push(nb);
            }
        }
    }

    // Flip cluster: S_i → S_i - 2(S_i · r)r
    // Iterate only set bits for efficiency
    for word_idx in 0..n_words {
        let mut word = in_cluster[word_idx];
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            let i = word_idx * 64 + bit;
            if i < n_sites {
                for c in 0..n_comp {
                    spins[i * n_comp + c] -= 2.0 * proj[i] * r[c];
                }
            }
            word &= word - 1; // clear lowest set bit
        }
    }
}

/// Optimised Wolff cluster flip for XY angle-only representation.
///
/// Operates on `&mut [f32]` angles. Uses DFS + bitset.
/// Reflection through plane ⊥ r: θ_new = π + 2φ_r − θ, wrapped to [0, 2π).
pub fn wolff_cluster_flip_angle<R: Rng>(
    angles: &mut [f32],
    n: usize,
    beta: f32,
    j: f32,
    rng: &mut R,
) {
    let n_sites = n * n * n;
    let n2 = n * n;

    // Random reflection axis angle
    let phi_r: f32 = rng.gen_range(0.0..std::f32::consts::TAU);

    // Projected spins: sigma_i = cos(theta_i - phi_r)
    let proj: Vec<f32> = angles.iter().map(|&theta| (theta - phi_r).cos()).collect();

    let seed_site: usize = rng.gen_range(0..n_sites);

    let n_words = (n_sites + 63) / 64;
    let mut in_cluster = vec![0u64; n_words];

    #[inline(always)]
    fn set_bit(bits: &mut [u64], i: usize) {
        bits[i >> 6] |= 1u64 << (i & 63);
    }
    #[inline(always)]
    fn test_bit(bits: &[u64], i: usize) -> bool {
        (bits[i >> 6] >> (i & 63)) & 1 != 0
    }

    set_bit(&mut in_cluster, seed_site);

    let mut stack = Vec::with_capacity(1024);
    stack.push(seed_site);

    while let Some(site) = stack.pop() {
        let iz = site / n2;
        let iy = (site % n2) / n;
        let ix = site % n;

        let neighbors = [
            iz * n2 + iy * n + (ix + 1) % n,
            iz * n2 + iy * n + (ix + n - 1) % n,
            iz * n2 + ((iy + 1) % n) * n + ix,
            iz * n2 + ((iy + n - 1) % n) * n + ix,
            ((iz + 1) % n) * n2 + iy * n + ix,
            ((iz + n - 1) % n) * n2 + iy * n + ix,
        ];

        let proj_site = proj[site];
        for &nb in &neighbors {
            if test_bit(&in_cluster, nb) {
                continue;
            }
            let bond = proj_site * proj[nb];
            if bond <= 0.0 {
                continue;
            }
            let p_add = 1.0 - (-2.0 * beta * j * bond).exp();
            if rng.gen::<f32>() < p_add {
                set_bit(&mut in_cluster, nb);
                stack.push(nb);
            }
        }
    }

    // Flip cluster: reflect through plane ⊥ r → θ_new = π + 2φ_r − θ
    let reflect_base = std::f32::consts::PI + 2.0 * phi_r;
    for word_idx in 0..n_words {
        let mut word = in_cluster[word_idx];
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            let i = word_idx * 64 + bit;
            if i < n_sites {
                let a = reflect_base - angles[i];
                angles[i] = a.rem_euclid(std::f32::consts::TAU);
            }
            word &= word - 1;
        }
    }
}
