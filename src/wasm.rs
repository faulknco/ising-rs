use wasm_bindgen::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::lattice::{Geometry, Lattice};
use crate::metropolis::{sweep, warm_up};
use crate::observables::measure;

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

    /// Run `n` sweeps — useful for warm-up from JS without per-frame overhead.
    pub fn warm_up(&mut self, temperature: f64, n: usize) {
        let beta = 1.0 / f64::max(temperature, 0.01);
        warm_up(&mut self.lattice, beta, self.j, self.h, n, &mut self.rng);
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

    /// Run a full temperature sweep and return CSV bytes.
    /// t_min, t_max, steps, warmup, samples — same as CLI.
    pub fn temperature_sweep(
        &mut self,
        t_min: f64,
        t_max: f64,
        steps: usize,
        warmup: usize,
        samples: usize,
    ) -> String {
        let mut out = String::from("T,E,M,Cv,chi\n");

        // Anneal from high → low, same as CLI
        for step in (0..steps).rev() {
            let t = t_min + (t_max - t_min) * step as f64 / (steps - 1) as f64;
            let beta = 1.0 / f64::max(t, 0.01);
            warm_up(&mut self.lattice, beta, self.j, self.h, warmup, &mut self.rng);
            let obs = measure(&mut self.lattice, beta, self.j, self.h, samples, &mut self.rng);
            out.push_str(&format!(
                "{:.4},{:.6},{:.6},{:.6},{:.6}\n",
                obs.temperature, obs.energy, obs.magnetisation,
                obs.heat_capacity, obs.susceptibility,
            ));
        }

        // Rows are currently T_min..T_max reversed — sort by reversing lines
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
}
