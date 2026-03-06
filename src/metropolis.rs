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
