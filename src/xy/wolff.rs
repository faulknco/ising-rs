use crate::xy::XyLattice;
use rand::Rng;

/// One Wolff cluster flip on the XY lattice.
///
/// Algorithm (Wolff 1989, adapted to O(2)):
/// 1. Pick a random mirror vector r = (cos φ, sin φ) uniformly on S¹.
/// 2. Pick a random seed spin uniformly from the lattice.
/// 3. Grow a cluster via BFS: for each active spin i with positive projection
///    pᵢ = Sᵢ·r > 0, add neighbour j (with pⱼ > 0) with probability
///    p = 1 − exp(−2βJ · pᵢ · pⱼ).
/// 4. Flip all cluster spins: Sᵢ → Sᵢ − 2(Sᵢ·r)r, then renormalise.
///
/// One call decorrelates the system as efficiently as O(N) Metropolis sweeps
/// near Tc — hence measure() calls sweep() once per sample.
pub fn sweep(lat: &mut XyLattice, beta: f64, j: f64, rng: &mut impl Rng) {
    let n = lat.size();
    if n == 0 {
        return;
    }

    // 1. Random mirror vector r on S¹
    let phi: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
    let r = [phi.cos(), phi.sin()];

    // 2. Random seed spin
    let seed = rng.gen_range(0..n);

    // 3. BFS cluster growth
    let mut in_cluster = vec![false; n];
    let mut stack: Vec<usize> = Vec::with_capacity(n / 4 + 8);

    let proj_seed = lat.spins[seed][0] * r[0] + lat.spins[seed][1] * r[1];
    if proj_seed > 0.0 {
        in_cluster[seed] = true;
        stack.push(seed);
    }

    while let Some(idx) = stack.pop() {
        let proj_i = lat.spins[idx][0] * r[0] + lat.spins[idx][1] * r[1];

        let nb_indices: Vec<usize> = lat.neighbours[idx].clone();
        for nb in nb_indices {
            if in_cluster[nb] {
                continue;
            }
            let proj_j = lat.spins[nb][0] * r[0] + lat.spins[nb][1] * r[1];
            if proj_j <= 0.0 {
                continue;
            }
            let p_bond = 1.0 - (-2.0 * beta * j * proj_i * proj_j).exp();
            if rng.gen::<f64>() < p_bond {
                in_cluster[nb] = true;
                stack.push(nb);
            }
        }
    }

    // 4. Flip r-component of all cluster spins and renormalise
    for (i, &in_c) in in_cluster.iter().enumerate() {
        if !in_c {
            continue;
        }
        let s = &mut lat.spins[i];
        let dot = s[0] * r[0] + s[1] * r[1];
        s[0] -= 2.0 * dot * r[0];
        s[1] -= 2.0 * dot * r[1];
        let norm = (s[0] * s[0] + s[1] * s[1]).sqrt();
        s[0] /= norm;
        s[1] /= norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xy::{XyLattice, magnetisation_per_spin};
    use rand::SeedableRng;

    fn cubic3d_neighbours(n: usize) -> Vec<Vec<usize>> {
        (0..n * n * n)
            .map(|idx| {
                let z = idx / (n * n);
                let y = (idx / n) % n;
                let x = idx % n;
                vec![
                    ((x + 1) % n) + y * n + z * n * n,
                    ((x + n - 1) % n) + y * n + z * n * n,
                    x + ((y + 1) % n) * n + z * n * n,
                    x + ((y + n - 1) % n) * n + z * n * n,
                    x + y * n + ((z + 1) % n) * n * n,
                    x + y * n + ((z + n - 1) % n) * n * n,
                ]
            })
            .collect()
    }

    #[test]
    fn spins_remain_unit_after_sweep() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = cubic3d_neighbours(4);
        let mut lat = XyLattice::new(nb);
        lat.randomise(&mut rng);
        for _ in 0..100 {
            sweep(&mut lat, 0.5, 1.0, &mut rng);
        }
        for s in &lat.spins {
            let norm = (s[0] * s[0] + s[1] * s[1]).sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "spin norm = {norm}");
        }
    }

    #[test]
    fn low_temp_wolff_stays_ordered() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = cubic3d_neighbours(4);
        let mut lat = XyLattice::new(nb);
        for _ in 0..200 {
            sweep(&mut lat, 100.0, 1.0, &mut rng);
        }
        let m = magnetisation_per_spin(&lat);
        assert!(m > 0.90, "at T≈0 |m| should be >0.90 after Wolff, got {m}");
    }

    #[test]
    fn high_temp_wolff_disorders() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(99);
        let nb = cubic3d_neighbours(6);
        let mut lat = XyLattice::new(nb);
        lat.randomise(&mut rng);
        for _ in 0..500 {
            sweep(&mut lat, 0.01, 1.0, &mut rng);
        }
        let m = magnetisation_per_spin(&lat);
        assert!(m < 0.3, "at T→∞ |m| should be small, got {m}");
    }

    #[test]
    fn cluster_flip_conserves_energy_direction() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(7);
        let nb = cubic3d_neighbours(4);
        let mut lat = XyLattice::new(nb);
        let (e_before, _) = crate::xy::energy_magnetisation(&lat, 1.0);
        for _ in 0..50 {
            sweep(&mut lat, 10.0, 1.0, &mut rng);
        }
        let (e_after, _) = crate::xy::energy_magnetisation(&lat, 1.0);
        assert!(
            e_after < e_before * 0.5,
            "energy should stay near ground state at low T: before={e_before}, after={e_after}"
        );
    }
}
