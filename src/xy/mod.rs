pub mod wolff;
pub mod observables;
pub mod fss;

use rand::Rng;

/// A 2D unit vector spin on S¹.
pub type Spin2 = [f64; 2];

/// XY lattice with 2D unit vector spins and adjacency list.
#[derive(Debug, Clone)]
pub struct XyLattice {
    pub spins: Vec<Spin2>,
    pub neighbours: Vec<Vec<usize>>,
}

impl XyLattice {
    /// All spins initialised to (1, 0) — ordered state.
    pub fn new(neighbours: Vec<Vec<usize>>) -> Self {
        let n = neighbours.len();
        Self {
            spins: vec![[1.0, 0.0]; n],
            neighbours,
        }
    }

    /// Number of sites in the lattice.
    pub fn size(&self) -> usize {
        self.spins.len()
    }

    /// Randomise all spins uniformly on S¹.
    pub fn randomise(&mut self, rng: &mut impl Rng) {
        for s in self.spins.iter_mut() {
            *s = random_unit_circle(rng);
        }
    }
}

/// Sample a uniformly random unit vector on S¹.
/// Samples a uniform angle φ ∈ [0, 2π) and returns (cos φ, sin φ).
pub fn random_unit_circle(rng: &mut impl Rng) -> Spin2 {
    let phi: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
    [phi.cos(), phi.sin()]
}

/// Total energy and magnetisation vector of the current configuration.
/// E = −J Σ_{(i,j)} Sᵢ·Sⱼ  (each bond counted once via nb > idx)
/// M = Σᵢ Sᵢ
pub fn energy_magnetisation(lat: &XyLattice, j: f64) -> (f64, [f64; 2]) {
    let mut e = 0.0_f64;
    let mut m = [0.0_f64; 2];
    for (idx, s) in lat.spins.iter().enumerate() {
        m[0] += s[0];
        m[1] += s[1];
        for &nb in &lat.neighbours[idx] {
            if nb > idx {
                let sn = &lat.spins[nb];
                e -= j * (s[0] * sn[0] + s[1] * sn[1]);
            }
        }
    }
    (e, m)
}

/// |M| per spin.
pub fn magnetisation_per_spin(lat: &XyLattice) -> f64 {
    let n = lat.size() as f64;
    let mut mx = 0.0_f64;
    let mut my = 0.0_f64;
    for s in &lat.spins {
        mx += s[0];
        my += s[1];
    }
    (mx * mx + my * my).sqrt() / n
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn ring(n: usize) -> Vec<Vec<usize>> {
        (0..n).map(|i| vec![(i + n - 1) % n, (i + 1) % n]).collect()
    }

    #[test]
    fn random_unit_circle_is_unit() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        for _ in 0..1000 {
            let v = random_unit_circle(&mut rng);
            let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
            assert!((norm - 1.0).abs() < 1e-12, "unit circle norm = {norm}");
        }
    }

    #[test]
    fn ordered_state_energy() {
        // All spins (1,0), 1D ring of 4 sites, J=1: E = -4 bonds * 1.0 = -4
        let nb = ring(4);
        let lat = XyLattice::new(nb);
        let (e, _) = energy_magnetisation(&lat, 1.0);
        assert!((e - (-4.0)).abs() < 1e-12, "ordered ring energy = {e}");
    }

    #[test]
    fn energy_scales_with_j() {
        // All spins (1,0), 4-site ring: E = -J * N_bonds = -J * 4
        let nb = ring(4);
        let lat = XyLattice::new(nb);
        let (e2, _) = energy_magnetisation(&lat, 2.0);
        assert!((e2 - (-8.0)).abs() < 1e-12, "j=2.0 ring energy = {e2}");
    }

    #[test]
    fn ordered_state_magnetisation() {
        let nb = ring(8);
        let lat = XyLattice::new(nb);
        let m = magnetisation_per_spin(&lat);
        assert!((m - 1.0).abs() < 1e-12, "|m| of ordered state = {m}");
    }

    #[test]
    fn randomise_produces_unit_spins() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = ring(100);
        let mut lat = XyLattice::new(nb);
        lat.randomise(&mut rng);
        for s in &lat.spins {
            let norm = (s[0] * s[0] + s[1] * s[1]).sqrt();
            assert!((norm - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn random_unit_circle_is_uniform() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(123);
        let n = 10_000;
        let mean_x: f64 =
            (0..n).map(|_| random_unit_circle(&mut rng)[0]).sum::<f64>() / n as f64;
        assert!(
            mean_x.abs() < 0.05,
            "mean cos should be near 0 for uniform S¹, got {mean_x}"
        );
        let var_x: f64 = (0..n)
            .map(|_| {
                let c = random_unit_circle(&mut rng)[0];
                c * c
            })
            .sum::<f64>()
            / n as f64;
        assert!(
            (var_x - 0.5).abs() < 0.05,
            "Var[cos] should be ≈0.5 for uniform S¹, got {var_x}"
        );
    }
}
