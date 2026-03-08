use crate::heisenberg::HeisenbergLattice;

/// One full over-relaxation sweep.
///
/// For each spin Sᵢ, reflect through the local field hᵢ = J Σⱼ Sⱼ:
///   S'ᵢ = 2(Sᵢ·ĥ)ĥ/|ĥ|² − Sᵢ
///
/// This is deterministic and energy-conserving (microcanonical).
/// Reduces autocorrelation time without a Boltzmann acceptance step.
/// Reference: Peczak, Ferrenberg, Landau, PRB 1991.
///
/// Each spin is visited exactly once in index order (sequential sweep),
/// as required by the standard over-relaxation algorithm, as opposed to
/// the random-site visitation used in Metropolis sweeps.
pub fn sweep(lat: &mut HeisenbergLattice, j: f64) {
    let size = lat.size();
    for idx in 0..size {
        // Local field (unnormalised): h = J Σⱼ Sⱼ
        let mut hx = 0.0_f64;
        let mut hy = 0.0_f64;
        let mut hz = 0.0_f64;
        for &nb in &lat.neighbours[idx] {
            hx += j * lat.spins[nb][0];
            hy += j * lat.spins[nb][1];
            hz += j * lat.spins[nb][2];
        }

        let h2 = hx * hx + hy * hy + hz * hz;
        if h2 < 1e-30 {
            // |h|² < 1e-30: local field is effectively zero (well below f64 representable
            // spin magnitudes). Reflection is undefined for a zero field — skip this spin.
            continue;
        }

        let s = lat.spins[idx];
        // S'ᵢ = 2(Sᵢ·h)h/|h|² − Sᵢ
        let sdoth = s[0] * hx + s[1] * hy + s[2] * hz;
        let scale = 2.0 * sdoth / h2;

        let nx = scale * hx - s[0];
        let ny = scale * hy - s[1];
        let nz = scale * hz - s[2];

        // Renormalise to unit length: algebraically exact but f64 drift accumulates over many sweeps.
        let norm = (nx * nx + ny * ny + nz * nz).sqrt();
        lat.spins[idx] = [nx / norm, ny / norm, nz / norm];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heisenberg::{energy_magnetisation, HeisenbergLattice};

    fn ring(n: usize) -> HeisenbergLattice {
        let nb = (0..n).map(|i| vec![(i + n - 1) % n, (i + 1) % n]).collect();
        HeisenbergLattice::new(nb)
    }

    #[test]
    fn spins_remain_unit_after_overrelax() {
        use rand::SeedableRng;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = ring(50);
        lat.randomise(&mut rng);
        sweep(&mut lat, 1.0);
        for s in &lat.spins {
            let norm = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "spin norm = {norm}");
        }
    }

    #[test]
    fn overrelax_conserves_energy() {
        use rand::SeedableRng;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = ring(20);
        lat.randomise(&mut rng);
        let (e_before, _) = energy_magnetisation(&lat, 1.0);
        sweep(&mut lat, 1.0);
        let (e_after, _) = energy_magnetisation(&lat, 1.0);
        assert!(
            (e_after - e_before).abs() < 1e-8,
            "over-relaxation should conserve energy: before={e_before}, after={e_after}"
        );
    }

    fn cubic3d(n: usize) -> HeisenbergLattice {
        let nb = (0..n * n * n)
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
            .collect();
        HeisenbergLattice::new(nb)
    }

    #[test]
    fn overrelax_conserves_energy_cubic3d() {
        use rand::SeedableRng;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(99);
        let mut lat = cubic3d(4);
        lat.randomise(&mut rng);
        let (e_before, _) = energy_magnetisation(&lat, 1.0);
        sweep(&mut lat, 1.0);
        let (e_after, _) = energy_magnetisation(&lat, 1.0);
        assert!(
            (e_after - e_before).abs() < 1e-8,
            "3D over-relaxation should conserve energy: before={e_before}, after={e_after}"
        );
    }
}
