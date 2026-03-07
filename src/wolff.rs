use crate::lattice::Lattice;
use crate::metropolis::sweep as metropolis_sweep;
/// Wolff single-cluster algorithm.
///
/// Metropolis suffers critical slowing down near Tc: the autocorrelation
/// time diverges as ξ^z (z ≈ 2.17 for 3D Ising with Metropolis).
/// The Wolff algorithm eliminates this by flipping entire correlated
/// clusters at once, giving z ≈ 0.33 — roughly 7× faster decorrelation
/// near Tc.
///
/// One Wolff step:
///   1. Pick a random seed spin σ_seed
///   2. Grow a cluster: for each same-spin neighbour σⱼ = σ_seed,
///      add it to the cluster with probability p = 1 - exp(-2βJ)
///   3. Flip every spin in the cluster
///
/// The add probability p comes from the detailed balance condition.
/// For J < 0 (antiferromagnet), p = 0 and the cluster never grows —
/// Wolff degenerates to single-spin flips. We fall back to Metropolis
/// for J ≤ 0.
///
/// Note: Wolff is only exact for h = 0. With h ≠ 0 we run a
/// Metropolis sweep afterward to handle the field term.
use rand::Rng;

/// One Wolff cluster flip. Returns the cluster size (useful for diagnostics).
pub fn step(lattice: &mut Lattice, beta: f64, j: f64, rng: &mut impl Rng) -> usize {
    if j <= 0.0 {
        return 0;
    } // Wolff only works for ferromagnetic J

    let p_add = 1.0 - (-2.0 * beta * j).exp();
    let size = lattice.size();

    // Pick a random seed
    let seed = rng.gen_range(0..size);
    let target_spin = lattice.spins[seed];

    // BFS cluster growth using a stack (faster than a queue for cache)
    let mut cluster = vec![false; size];
    let mut stack = Vec::with_capacity(size / 4);

    cluster[seed] = true;
    stack.push(seed);

    while let Some(idx) = stack.pop() {
        for &nb in &lattice.neighbours[idx] {
            if !cluster[nb] && lattice.spins[nb] == target_spin && rng.gen::<f64>() < p_add {
                cluster[nb] = true;
                stack.push(nb);
            }
        }
    }

    // Flip the whole cluster
    let mut count = 0;
    for (i, &in_cluster) in cluster.iter().enumerate() {
        if in_cluster {
            lattice.spins[i] = -lattice.spins[i];
            count += 1;
        }
    }

    count
}

/// Run `n` Wolff cluster flips.
/// If h ≠ 0, follows each Wolff step with a Metropolis sweep to handle
/// the field term correctly.
pub fn warm_up(lattice: &mut Lattice, beta: f64, j: f64, h: f64, n: usize, rng: &mut impl Rng) {
    for _ in 0..n {
        step(lattice, beta, j, rng);
        if h.abs() > 1e-9 {
            metropolis_sweep(lattice, beta, j, h, rng);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{Geometry, Lattice};
    use rand::SeedableRng;

    #[test]
    fn wolff_flips_at_least_one() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = Lattice::new(8, Geometry::Cubic3D);
        let count = step(&mut lat, 0.5, 1.0, &mut rng);
        assert!(count >= 1, "Wolff should flip at least the seed spin");
    }

    #[test]
    fn wolff_preserves_lattice_size() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = Lattice::new(6, Geometry::Cubic3D);
        let n = lat.size();
        for _ in 0..100 {
            step(&mut lat, 0.5, 1.0, &mut rng);
        }
        assert_eq!(lat.size(), n);
        assert!(
            lat.spins.iter().all(|&s| s == 1 || s == -1),
            "all spins should be ±1"
        );
    }

    #[test]
    fn wolff_returns_zero_for_antiferromagnet() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = Lattice::new(4, Geometry::Cubic3D);
        let count = step(&mut lat, 0.5, -1.0, &mut rng);
        assert_eq!(count, 0, "Wolff should not flip for J <= 0");
    }

    #[test]
    fn wolff_low_temp_large_cluster() {
        // At low T (high beta), p_add ≈ 1, so cluster should be large
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = Lattice::new(6, Geometry::Cubic3D);
        // All spins are +1, low T → p_add close to 1 → should flip nearly all
        let count = step(&mut lat, 10.0, 1.0, &mut rng);
        assert!(
            count > lat.size() / 2,
            "at low T, cluster should be > N/2, got {count}/{}",
            lat.size()
        );
    }

    #[test]
    fn wolff_high_temp_small_cluster() {
        // At very high T (small beta), p_add ≈ 0, clusters should be small
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = Lattice::new(8, Geometry::Cubic3D);
        let mut total = 0;
        let trials = 100;
        for _ in 0..trials {
            total += step(&mut lat, 0.01, 1.0, &mut rng);
        }
        let avg = total as f64 / trials as f64;
        assert!(
            avg < 10.0,
            "at high T, average cluster size should be small, got {avg}"
        );
    }

    #[test]
    fn warm_up_changes_state() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = Lattice::new(6, Geometry::Cubic3D);
        let before: Vec<i8> = lat.spins.clone();
        warm_up(&mut lat, 0.5, 1.0, 0.0, 50, &mut rng);
        assert_ne!(lat.spins, before, "warm_up should change spin state");
    }
}
