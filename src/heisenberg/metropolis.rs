use crate::heisenberg::{HeisenbergLattice, Spin3};
use rand::Rng;

/// N Metropolis update proposals drawn with replacement (random-site sweep).
/// On average each spin is proposed once per sweep, but individual sites may
/// be skipped or updated multiple times in a single call.
///
/// delta: cap angle for proposed rotation (radians). Tune to ~50% acceptance.
/// ΔE = −J [ (S'ᵢ − Sᵢ) · Σⱼ Sⱼ ]
pub fn sweep(lat: &mut HeisenbergLattice, beta: f64, j: f64, delta: f64, rng: &mut impl Rng) {
    let size = lat.size();
    for _ in 0..size {
        let idx = rng.gen_range(0..size);

        // Local field: sum of neighbour spins
        let mut hx = 0.0_f64;
        let mut hy = 0.0_f64;
        let mut hz = 0.0_f64;
        for &nb in &lat.neighbours[idx] {
            hx += lat.spins[nb][0];
            hy += lat.spins[nb][1];
            hz += lat.spins[nb][2];
        }

        let s = lat.spins[idx];
        let e_old = -j * (s[0] * hx + s[1] * hy + s[2] * hz);

        // Propose new spin: rotate by random angle <= delta from current
        let s_new = propose_rotation(&s, delta, rng);
        let e_new = -j * (s_new[0] * hx + s_new[1] * hy + s_new[2] * hz);

        let delta_e = e_new - e_old;
        if delta_e < 0.0 || rng.gen::<f64>() < (-beta * delta_e).exp() {
            lat.spins[idx] = s_new;
        }
    }
}

/// Warm up: run `steps` sweeps discarding results.
pub fn warm_up(
    lat: &mut HeisenbergLattice,
    beta: f64,
    j: f64,
    delta: f64,
    steps: usize,
    rng: &mut impl Rng,
) {
    for _ in 0..steps {
        sweep(lat, beta, j, delta, rng);
    }
}

/// Propose a new spin by rotating `s` by a random angle in [0, delta].
fn propose_rotation(s: &Spin3, delta: f64, rng: &mut impl Rng) -> Spin3 {
    let cos_delta = delta.cos();
    let cos_theta: f64 = rng.gen_range(cos_delta..1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    let phi: f64 = rng.gen_range(0.0..std::f64::consts::TAU);

    let (u, v) = perpendicular_frame(s);

    let (cphi, sphi) = (phi.cos(), phi.sin());
    let nx = sin_theta * cphi * u[0] + sin_theta * sphi * v[0] + cos_theta * s[0];
    let ny = sin_theta * cphi * u[1] + sin_theta * sphi * v[1] + cos_theta * s[1];
    let nz = sin_theta * cphi * u[2] + sin_theta * sphi * v[2] + cos_theta * s[2];

    let norm = (nx * nx + ny * ny + nz * nz).sqrt();
    [nx / norm, ny / norm, nz / norm]
}

/// Build an orthonormal frame {u, v} perpendicular to s.
fn perpendicular_frame(s: &Spin3) -> (Spin3, Spin3) {
    let t: Spin3 = if s[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };

    let ts = t[0] * s[0] + t[1] * s[1] + t[2] * s[2];
    let ux = t[0] - ts * s[0];
    let uy = t[1] - ts * s[1];
    let uz = t[2] - ts * s[2];
    let un = (ux * ux + uy * uy + uz * uz).sqrt();
    let u = [ux / un, uy / un, uz / un];

    let v = [
        s[1] * u[2] - s[2] * u[1],
        s[2] * u[0] - s[0] * u[2],
        s[0] * u[1] - s[1] * u[0],
    ];
    (u, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heisenberg::{magnetisation_per_spin, HeisenbergLattice};
    use rand::SeedableRng;

    fn ring(n: usize) -> HeisenbergLattice {
        let nb = (0..n).map(|i| vec![(i + n - 1) % n, (i + 1) % n]).collect();
        HeisenbergLattice::new(nb)
    }

    fn cubic3d(n: usize) -> HeisenbergLattice {
        let nb: Vec<Vec<usize>> = (0..n * n * n)
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
    fn spins_remain_unit_after_sweep() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = ring(20);
        lat.randomise(&mut rng);
        for _ in 0..100 {
            sweep(&mut lat, 0.5, 1.0, 0.3, &mut rng);
        }
        for s in &lat.spins {
            let norm = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "spin norm = {norm}");
        }
    }

    #[test]
    fn low_temp_preserves_order() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = cubic3d(4);
        // start ordered (all spins (0,0,1) from new())
        for _ in 0..500 {
            sweep(&mut lat, 100.0, 1.0, 0.05, &mut rng);
        }
        let m = magnetisation_per_spin(&lat);
        assert!(m > 0.95, "at T≈0 |m| should be >0.95, got {m}");
    }

    #[test]
    fn high_temp_disorders() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(99);
        let mut lat = cubic3d(6);
        lat.randomise(&mut rng);
        for _ in 0..1000 {
            sweep(&mut lat, 0.01, 1.0, 1.0, &mut rng);
        }
        let m = magnetisation_per_spin(&lat);
        assert!(m < 0.3, "at T→∞ |m| should be small, got {m}");
    }
}
