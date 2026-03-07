/// Critical exponent fitting near the Curie temperature.
///
/// Near Tc, the observables obey power laws:
///   |⟨M⟩| ~ (Tc - T)^β        β ≈ 0.326  (3D Ising universality class)
///   Cv     ~ |T - Tc|^{-α}     α ≈ 0.110
///   χ      ~ |T - Tc|^{-γ}     γ ≈ 1.237
///
/// Taking logs linearises each:
///   log|⟨M⟩| = β·log(Tc - T) + const   (T < Tc only)
///   log(Cv)  = -α·log|T - Tc| + const
///   log(χ)   = -γ·log|T - Tc| + const
///
/// We fit each using ordinary least-squares on log-log data in a window
/// [Tc - window, Tc + window], excluding points too close to Tc where
/// finite-size effects dominate.
use crate::observables::Observables;

/// Fitted critical exponents with errors vs 3D Ising theory.
#[derive(Debug, Clone)]
pub struct CriticalExponents {
    pub tc:    f64,   // estimated Curie temperature
    pub beta:  f64,   // magnetisation exponent
    pub alpha: f64,   // heat capacity exponent
    pub gamma: f64,   // susceptibility exponent
    /// Relative errors vs 3D Ising theory values
    pub beta_err:  f64,
    pub alpha_err: f64,
    pub gamma_err: f64,
}

// 3D Ising universality class (best known values)
const THEORY_BETA:  f64 = 0.3265;
const THEORY_ALPHA: f64 = 0.1096;
const THEORY_GAMMA: f64 = 1.2372;

impl CriticalExponents {
    /// Fit exponents from a completed temperature sweep.
    ///
    /// `window` controls how many points either side of Tc are used for fitting.
    /// Too wide → includes non-critical behaviour; too narrow → too few points.
    /// 0.8 J/kB works well for N=15-20 lattices.
    pub fn fit(data: &[Observables], window: f64) -> Option<Self> {
        if data.len() < 5 { return None; }

        let tc = estimate_tc(data)?;

        let beta  = fit_beta(data, tc, window)?;
        let alpha = fit_alpha(data, tc, window)?;
        let gamma = fit_gamma(data, tc, window)?;

        Some(Self {
            tc,
            beta,
            alpha,
            gamma,
            beta_err:  (beta  - THEORY_BETA).abs()  / THEORY_BETA,
            alpha_err: (alpha - THEORY_ALPHA).abs() / THEORY_ALPHA,
            gamma_err: (gamma - THEORY_GAMMA).abs() / THEORY_GAMMA,
        })
    }
}

/// Tc = temperature of maximum |dM/dT| (inflection point of magnetisation curve).
fn estimate_tc(data: &[Observables]) -> Option<f64> {
    let mut max_dm = 0.0_f64;
    let mut tc = None;
    for i in 1..data.len() - 1 {
        let dm = (data[i - 1].magnetisation - data[i + 1].magnetisation).abs();
        if dm > max_dm {
            max_dm = dm;
            tc = Some(data[i].temperature);
        }
    }
    tc
}

/// Fit β: log|M| = β·log(Tc - T) + c  for T < Tc
fn fit_beta(data: &[Observables], tc: f64, window: f64) -> Option<f64> {
    let min_dist = 0.05; // exclude points within this of Tc (finite-size effects)
    let pts: Vec<(f64, f64)> = data.iter()
        .filter(|o| {
            let dt = tc - o.temperature;
            dt > min_dist && dt < window && o.magnetisation > 1e-6
        })
        .map(|o| ((tc - o.temperature).ln(), o.magnetisation.ln()))
        .collect();

    ols_slope(&pts)
}

/// Fit α: log(Cv) = -α·log|T - Tc| + c
fn fit_alpha(data: &[Observables], tc: f64, window: f64) -> Option<f64> {
    let min_dist = 0.05;
    let pts: Vec<(f64, f64)> = data.iter()
        .filter(|o| {
            let dt = (o.temperature - tc).abs();
            dt > min_dist && dt < window && o.heat_capacity > 1e-6
        })
        .map(|o| (-(o.temperature - tc).abs().ln(), o.heat_capacity.ln()))
        .collect();

    // slope of log(Cv) vs -log|T-Tc| gives α
    ols_slope(&pts)
}

/// Fit γ: log(χ) = -γ·log|T - Tc| + c
fn fit_gamma(data: &[Observables], tc: f64, window: f64) -> Option<f64> {
    let min_dist = 0.05;
    let pts: Vec<(f64, f64)> = data.iter()
        .filter(|o| {
            let dt = (o.temperature - tc).abs();
            dt > min_dist && dt < window && o.susceptibility > 1e-6
        })
        .map(|o| (-(o.temperature - tc).abs().ln(), o.susceptibility.ln()))
        .collect();

    ols_slope(&pts)
}

/// Ordinary least squares: slope of y ~ slope·x + intercept.
/// Returns None if fewer than 3 points.
fn ols_slope(pts: &[(f64, f64)]) -> Option<f64> {
    let n = pts.len();
    if n < 3 { return None; }

    let n = n as f64;
    let sum_x:  f64 = pts.iter().map(|(x, _)| x).sum();
    let sum_y:  f64 = pts.iter().map(|(_, y)| y).sum();
    let sum_xx: f64 = pts.iter().map(|(x, _)| x * x).sum();
    let sum_xy: f64 = pts.iter().map(|(x, y)| x * y).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 { return None; }

    Some((n * sum_xy - sum_x * sum_y) / denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ols_slope_known() {
        // y = 2x + 1  →  slope = 2
        let pts: Vec<(f64, f64)> = (0..10)
            .map(|i| { let x = i as f64; (x, 2.0 * x + 1.0) })
            .collect();
        let slope = ols_slope(&pts).unwrap();
        assert!((slope - 2.0).abs() < 1e-10, "slope = {slope}");
    }
}
