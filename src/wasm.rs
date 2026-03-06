use wasm_bindgen::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::fitting::CriticalExponents;
use crate::lattice::{Geometry, Lattice};
use crate::metropolis::{sweep, warm_up};
use crate::observables::{measure, Observables};
use crate::wolff;

/// The live simulation state exposed to JavaScript.
///
/// JS creates one instance, then calls step() in a loop from a Web Worker.
/// The spin array is read directly from WASM memory via a typed array view.
#[wasm_bindgen]
pub struct IsingWasm {
    lattice: Lattice,
    rng: Xoshiro256PlusPlus,
    j: f64,
    h: f64,
}

#[wasm_bindgen]
impl IsingWasm {
    /// Create a new lattice.
    /// geometry: 0 = Square2D, 1 = Triangular2D, 2 = Cubic3D
    #[wasm_bindgen(constructor)]
    pub fn new(n: usize, geometry: u8, j: f64, h: f64, seed: u64) -> Self {
        let geom = match geometry {
            1 => Geometry::Triangular2D,
            2 => Geometry::Cubic3D,
            _ => Geometry::Square2D,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut lattice = Lattice::new(n, geom);
        lattice.randomise(&mut rng);
        Self { lattice, rng, j, h }
    }

    /// Run one full Metropolis sweep (N² spin-flip attempts).
    pub fn step(&mut self, temperature: f64) {
        let beta = 1.0 / f64::max(temperature, 0.01);
        sweep(&mut self.lattice, beta, self.j, self.h, &mut self.rng);
    }

    /// Run one Wolff cluster flip. Returns cluster size.
    /// Falls back to Metropolis for J ≤ 0 or h ≠ 0.
    pub fn step_wolff(&mut self, temperature: f64) -> usize {
        let beta = 1.0 / f64::max(temperature, 0.01);
        let cluster_size = wolff::step(&mut self.lattice, beta, self.j, &mut self.rng);
        // Handle magnetic field with a Metropolis sweep
        if self.h.abs() > 1e-9 {
            sweep(&mut self.lattice, beta, self.j, self.h, &mut self.rng);
        }
        cluster_size
    }

    /// Run `n` sweeps — useful for warm-up from JS without per-frame overhead.
    pub fn warm_up(&mut self, temperature: f64, n: usize) {
        let beta = 1.0 / f64::max(temperature, 0.01);
        warm_up(&mut self.lattice, beta, self.j, self.h, n, &mut self.rng);
    }

    /// Run `n` Wolff cluster flips for warm-up.
    pub fn warm_up_wolff(&mut self, temperature: f64, n: usize) {
        let beta = 1.0 / f64::max(temperature, 0.01);
        wolff::warm_up(&mut self.lattice, beta, self.j, self.h, n, &mut self.rng);
    }

    /// Pointer into WASM linear memory for the spin array.
    /// JS reads this as Int8Array for zero-copy access.
    pub fn spins_ptr(&self) -> *const i8 {
        self.lattice.spins.as_ptr()
    }

    /// Number of spins (N² or N³).
    pub fn size(&self) -> usize {
        self.lattice.size()
    }

    /// Lattice dimension N.
    pub fn n(&self) -> usize {
        self.lattice.n
    }

    /// Update J and h without recreating the lattice.
    pub fn set_params(&mut self, j: f64, h: f64) {
        self.j = j;
        self.h = h;
    }

    /// Randomise the lattice (reset).
    pub fn randomise(&mut self) {
        self.lattice.randomise(&mut self.rng);
    }

    /// Run a full temperature sweep and return CSV.
    /// use_wolff: true = Wolff cluster algorithm (faster near Tc),
    ///            false = Metropolis (default, works for any J/h)
    pub fn temperature_sweep(
        &mut self,
        t_min: f64,
        t_max: f64,
        steps: usize,
        warmup: usize,
        samples: usize,
        use_wolff: bool,
    ) -> String {
        let mut out = String::from("T,E,M,Cv,chi\n");

        // Anneal from high → low
        for step in (0..steps).rev() {
            let t = t_min + (t_max - t_min) * step as f64 / (steps - 1) as f64;
            let beta = 1.0 / f64::max(t, 0.01);

            if use_wolff && self.j > 0.0 {
                wolff::warm_up(&mut self.lattice, beta, self.j, self.h, warmup, &mut self.rng);
            } else {
                warm_up(&mut self.lattice, beta, self.j, self.h, warmup, &mut self.rng);
            }

            let obs = measure(&mut self.lattice, beta, self.j, self.h, samples, &mut self.rng);
            out.push_str(&format!(
                "{:.4},{:.6},{:.6},{:.6},{:.6}\n",
                obs.temperature, obs.energy, obs.magnetisation,
                obs.heat_capacity, obs.susceptibility,
            ));
        }

        let header = "T,E,M,Cv,chi\n".to_string();
        let mut rows: Vec<&str> = out.lines().skip(1).collect();
        rows.reverse();
        header + &rows.join("\n") + "\n"
    }

    /// Average magnetisation |<M>| of the current configuration.
    pub fn magnetisation(&self) -> f64 {
        let sum: i32 = self.lattice.spins.iter().map(|&s| s as i32).sum();
        (sum as f64 / self.lattice.size() as f64).abs()
    }

    /// Copy of spin array as Int8Array — safe to transfer to main thread.
    pub fn get_spins_copy(&self) -> Vec<i8> {
        self.lattice.spins.clone()
    }

    /// Parse sweep CSV and fit critical exponents. Returns JSON string:
    /// { tc, beta, alpha, gamma, beta_err, alpha_err, gamma_err,
    ///   theory_beta, theory_alpha, theory_gamma }
    /// Returns empty string if fitting fails (too few points, etc).
    pub fn fit_exponents(&self, csv: &str, window: f64) -> String {
        let data = parse_csv(csv);
        match CriticalExponents::fit(&data, window) {
            None => String::new(),
            Some(e) => format!(
                r#"{{"tc":{tc:.3},"beta":{beta:.4},"alpha":{alpha:.4},"gamma":{gamma:.4},"beta_err":{be:.4},"alpha_err":{ae:.4},"gamma_err":{ge:.4},"theory_beta":0.3265,"theory_alpha":0.1096,"theory_gamma":1.2372}}"#,
                tc    = e.tc,
                beta  = e.beta,
                alpha = e.alpha,
                gamma = e.gamma,
                be    = e.beta_err,
                ae    = e.alpha_err,
                ge    = e.gamma_err,
            ),
        }
    }
}

/// Parse the CSV produced by temperature_sweep() into Observables.
fn parse_csv(csv: &str) -> Vec<Observables> {
    csv.lines()
        .skip(1) // header
        .filter_map(|line| {
            let cols: Vec<f64> = line.split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if cols.len() < 5 { return None; }
            Some(Observables {
                temperature:   cols[0],
                energy:        cols[1],
                magnetisation: cols[2],
                heat_capacity: cols[3],
                susceptibility: cols[4],
                m2: 0.0,
                m4: 0.0,
            })
        })
        .collect()
}
