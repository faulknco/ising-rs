pub mod fss;
pub mod metropolis;
pub mod observables;
pub mod overrelax;
pub mod sweep;

use rand::Rng;

/// A 3D unit vector spin.
pub type Spin3 = [f64; 3];

/// Heisenberg lattice with vector spins and adjacency list.
#[derive(Debug, Clone)]
pub struct HeisenbergLattice {
    pub spins: Vec<Spin3>,
    pub neighbours: Vec<Vec<usize>>,
}

impl HeisenbergLattice {
    /// All spins initialised to (0, 0, 1) — ordered state.
    pub fn new(neighbours: Vec<Vec<usize>>) -> Self {
        let n = neighbours.len();
        Self {
            spins: vec![[0.0, 0.0, 1.0]; n],
            neighbours,
        }
    }

    pub fn size(&self) -> usize {
        self.spins.len()
    }

    /// Randomise all spins uniformly on the unit sphere.
    pub fn randomise(&mut self, rng: &mut impl Rng) {
        for s in self.spins.iter_mut() {
            *s = random_unit_vector(rng);
        }
    }
}

/// Sample a uniformly random unit vector on S².
/// Uses Marsaglia (1972): rejection sample in the unit disk.
pub fn random_unit_vector(rng: &mut impl Rng) -> Spin3 {
    loop {
        let x: f64 = rng.gen_range(-1.0..1.0);
        let y: f64 = rng.gen_range(-1.0..1.0);
        let r2 = x * x + y * y;
        if r2 >= 1.0 {
            continue;
        }
        let s = 2.0 * (1.0 - r2).sqrt();
        return [s * x, s * y, 1.0 - 2.0 * r2];
    }
}

/// Total energy and magnetisation vector of the current configuration.
/// E = −J Σ_{(i,j)} Sᵢ·Sⱼ  (each bond counted once via nb > idx)
/// M = Σᵢ Sᵢ
pub fn energy_magnetisation(lat: &HeisenbergLattice, j: f64) -> (f64, [f64; 3]) {
    let mut e = 0.0_f64;
    let mut m = [0.0_f64; 3];
    for (idx, s) in lat.spins.iter().enumerate() {
        m[0] += s[0];
        m[1] += s[1];
        m[2] += s[2];
        for &nb in &lat.neighbours[idx] {
            if nb > idx {
                let sn = &lat.spins[nb];
                e -= j * (s[0] * sn[0] + s[1] * sn[1] + s[2] * sn[2]);
            }
        }
    }
    (e, m)
}

/// |M| per spin.
pub fn magnetisation_per_spin(lat: &HeisenbergLattice) -> f64 {
    let n = lat.size() as f64;
    let mut mx = 0.0_f64;
    let mut my = 0.0_f64;
    let mut mz = 0.0_f64;
    for s in &lat.spins {
        mx += s[0];
        my += s[1];
        mz += s[2];
    }
    (mx * mx + my * my + mz * mz).sqrt() / n
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn cubic_neighbours(n: usize) -> Vec<Vec<usize>> {
        // Simple 1D ring for unit tests
        (0..n).map(|i| vec![(i + n - 1) % n, (i + 1) % n]).collect()
    }

    #[test]
    fn random_unit_vector_is_unit() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        for _ in 0..1000 {
            let v = random_unit_vector(&mut rng);
            let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((norm - 1.0).abs() < 1e-12, "unit vector norm = {norm}");
        }
    }

    #[test]
    fn ordered_state_energy() {
        // All spins (0,0,1), 1D ring of 4 sites, J=1: E = -4 bonds * 1.0 = -4
        let nb = cubic_neighbours(4);
        let lat = HeisenbergLattice::new(nb);
        let (e, _) = energy_magnetisation(&lat, 1.0);
        assert!((e - (-4.0)).abs() < 1e-12, "ordered 1D ring energy = {e}");
    }

    #[test]
    fn ordered_state_magnetisation() {
        let nb = cubic_neighbours(8);
        let lat = HeisenbergLattice::new(nb);
        let m = magnetisation_per_spin(&lat);
        assert!((m - 1.0).abs() < 1e-12, "|m| of ordered state = {m}");
    }

    #[test]
    fn randomise_produces_unit_spins() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = cubic_neighbours(100);
        let mut lat = HeisenbergLattice::new(nb);
        lat.randomise(&mut rng);
        for s in &lat.spins {
            let norm = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
            assert!((norm - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn random_unit_vector_is_uniform() {
        // For uniform distribution on S², E[z] = 0 and Var[z] = 1/3.
        // Over 10_000 samples, mean z should be within 0.05 of 0.
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(123);
        let n = 10_000;
        let mean_z: f64 = (0..n).map(|_| random_unit_vector(&mut rng)[2]).sum::<f64>() / n as f64;
        assert!(
            mean_z.abs() < 0.05,
            "mean z should be near 0 for uniform S², got {mean_z}"
        );
        let var_z: f64 = (0..n)
            .map(|_| {
                let z = random_unit_vector(&mut rng)[2];
                z * z
            })
            .sum::<f64>()
            / n as f64;
        assert!(
            (var_z - 1.0 / 3.0).abs() < 0.05,
            "Var[z] should be ≈1/3 for uniform S², got {var_z}"
        );
    }
}
