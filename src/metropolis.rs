use rand::Rng;
use crate::lattice::Lattice;

/// One full Metropolis sweep: visits every spin once in random order.
///
/// ΔE = 2 σᵢ ( J Σⱼ σⱼ + h )
/// Accept if ΔE < 0, else accept with probability exp(−βΔE).
pub fn sweep(lattice: &mut Lattice, beta: f64, j: f64, h: f64, rng: &mut impl Rng) {
    let size = lattice.size();
    for _ in 0..size {
        // Pick a random spin
        let idx = rng.gen_range(0..size);
        let spin = lattice.spins[idx] as f64;

        // Sum of neighbour spins
        let neighbour_sum: f64 = lattice.neighbours[idx]
            .iter()
            .map(|&k| lattice.spins[k] as f64)
            .sum();

        let delta_e = 2.0 * spin * (j * neighbour_sum + h);

        if delta_e < 0.0 || rng.gen::<f64>() < (-beta * delta_e).exp() {
            lattice.spins[idx] = -lattice.spins[idx];
        }
    }
}

/// Warm up the lattice for `steps` sweeps before taking measurements.
pub fn warm_up(lattice: &mut Lattice, beta: f64, j: f64, h: f64, steps: usize, rng: &mut impl Rng) {
    for _ in 0..steps {
        sweep(lattice, beta, j, h, rng);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{Lattice, Geometry};
    use crate::observables::energy_magnetisation;
    use rand::SeedableRng;

    #[test]
    fn sweep_preserves_spin_values() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = Lattice::new(6, Geometry::Cubic3D);
        lat.randomise(&mut rng);
        for _ in 0..50 {
            sweep(&mut lat, 0.5, 1.0, 0.0, &mut rng);
        }
        assert!(lat.spins.iter().all(|&s| s == 1 || s == -1),
            "all spins should remain ±1 after sweeps");
    }

    #[test]
    fn sweep_changes_state() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = Lattice::new(6, Geometry::Cubic3D);
        lat.randomise(&mut rng);
        let before: Vec<i8> = lat.spins.clone();
        for _ in 0..10 {
            sweep(&mut lat, 0.5, 1.0, 0.0, &mut rng);
        }
        assert_ne!(lat.spins, before, "sweeps should change spin state");
    }

    #[test]
    fn low_temp_preserves_ground_state() {
        // At very low T, Metropolis should almost never flip away from ground state
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = Lattice::new(4, Geometry::Cubic3D);
        // Start all +1 (ground state)
        let beta = 100.0;
        for _ in 0..100 {
            sweep(&mut lat, beta, 1.0, 0.0, &mut rng);
        }
        let (_, m) = energy_magnetisation(&lat, 1.0, 0.0);
        let m_per = m / lat.size() as f64;
        assert!((m_per.abs() - 1.0).abs() < 0.01,
            "at T≈0, ground state should be preserved, |m|={}", m_per.abs());
    }

    #[test]
    fn high_temp_disorders() {
        // At high T, starting from ordered state should disorder
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = Lattice::new(8, Geometry::Square2D);
        let beta = 0.01; // T = 100
        for _ in 0..1000 {
            sweep(&mut lat, beta, 1.0, 0.0, &mut rng);
        }
        let (_, m) = energy_magnetisation(&lat, 1.0, 0.0);
        let m_per = (m / lat.size() as f64).abs();
        assert!(m_per < 0.3,
            "at T→∞, |m| should be small, got {m_per}");
    }
}
