use rand::Rng;

/// Propose replica exchanges between adjacent temperatures.
/// Returns the number of accepted swaps.
pub fn replica_exchange(
    temperatures: &[f64],
    energies: &[f64],
    replica_to_temp: &mut [usize],
    temp_to_replica: &mut [usize],
    rng: &mut impl Rng,
    even_odd: usize, // 0 = try pairs (0,1),(2,3),...  1 = try (1,2),(3,4),...
) -> usize {
    let n = temperatures.len();
    let mut accepted = 0;

    let start = even_odd % 2;
    let mut i = start;
    while i + 1 < n {
        let ri = temp_to_replica[i];
        let rj = temp_to_replica[i + 1];
        let beta_i = 1.0 / temperatures[i];
        let beta_j = 1.0 / temperatures[i + 1];
        let delta = (beta_i - beta_j) * (energies[rj] - energies[ri]);

        if delta < 0.0 || rng.gen::<f64>() < (-delta).exp() {
            // Swap
            temp_to_replica[i] = rj;
            temp_to_replica[i + 1] = ri;
            replica_to_temp[ri] = i + 1;
            replica_to_temp[rj] = i;
            accepted += 1;
        }
        i += 2;
    }
    accepted
}

/// Generate linearly spaced temperatures.
pub fn linspace_temperatures(t_min: f64, t_max: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![t_min];
    }
    (0..n)
        .map(|i| t_min + (t_max - t_min) * i as f64 / (n - 1) as f64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replica_exchange_deterministic() {
        let temps = vec![4.0, 5.0];
        let energies = vec![-100.0, -50.0];
        let mut r2t = vec![0, 1];
        let mut t2r = vec![0, 1];

        let mut rng = rand::thread_rng();
        let _accepted = replica_exchange(&temps, &energies, &mut r2t, &mut t2r, &mut rng, 0);

        // Verify bijectivity
        assert_eq!(r2t.len(), 2);
        assert_eq!(t2r.len(), 2);
        assert_eq!(t2r[r2t[0]], 0);
        assert_eq!(t2r[r2t[1]], 1);
    }

    #[test]
    fn test_linspace_temperatures() {
        let temps = linspace_temperatures(4.0, 5.0, 5);
        assert_eq!(temps.len(), 5);
        assert!((temps[0] - 4.0).abs() < 1e-10);
        assert!((temps[4] - 5.0).abs() < 1e-10);
        assert!((temps[2] - 4.5).abs() < 1e-10);
    }
}
