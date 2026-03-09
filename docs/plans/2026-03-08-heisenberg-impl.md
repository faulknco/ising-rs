# Heisenberg Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a classical Heisenberg Monte Carlo engine to ising-rs for J-fitting on BCC/FCC crystal graphs, targeting Paper 2 (Physical Review B).

**Architecture:** New `src/heisenberg/` module, additive only. Metropolis + over-relaxation algorithm. CPU only. Jackknife error estimation on all observables. Two CLI binaries: `heisenberg_fss` (cubic validation) and `heisenberg_jfit` (crystal graph sweeps).

**Tech Stack:** Rust stable, rand/rand_xoshiro (already in Cargo.toml), existing `src/graph.rs` JSON loader, existing `analysis/graphs/bcc_*.json` and `fcc_*.json`.

**Physics reference:**
- Cubic Heisenberg Tc(J=1) = 1.4432 J/kB
- O(3) exponents: ν=0.7112, γ=1.3960, β=0.3646
- BCC Fe: Tc_exp = 1043 K, J_lit = 16.3 meV (Pajda 2001)
- FCC Ni: Tc_exp = 627 K, J_lit = 4.1 meV (Pajda 2001)

---

## Task 1: Scaffold `src/heisenberg/mod.rs` — spin type, lattice, energy

**Files:**
- Create: `src/heisenberg/mod.rs`
- Modify: `src/lib.rs`

**Step 1: Add module declaration to lib.rs**

In `src/lib.rs`, add after the existing `pub mod wolff;` line:

```rust
pub mod heisenberg;
```

**Step 2: Create `src/heisenberg/mod.rs`**

```rust
pub mod metropolis;
pub mod observables;
pub mod overrelax;
pub mod sweep;
pub mod fss;

use rand::Rng;

/// A 3D unit vector spin.
pub type Spin3 = [f64; 3];

/// Heisenberg lattice with vector spins and adjacency list.
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
        if r2 >= 1.0 { continue; }
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
        m[0] += s[0]; m[1] += s[1]; m[2] += s[2];
        for &nb in &lat.neighbours[idx] {
            if nb > idx {
                let sn = &lat.spins[nb];
                e -= j * (s[0]*sn[0] + s[1]*sn[1] + s[2]*sn[2]);
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
        mx += s[0]; my += s[1]; mz += s[2];
    }
    (mx*mx + my*my + mz*mz).sqrt() / n
}
```

**Step 3: Write tests in `src/heisenberg/mod.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn cubic_neighbours(n: usize) -> Vec<Vec<usize>> {
        // Simple 1D ring for unit tests — each site connected to left and right
        (0..n).map(|i| vec![(i + n - 1) % n, (i + 1) % n]).collect()
    }

    #[test]
    fn random_unit_vector_is_unit() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        for _ in 0..1000 {
            let v = random_unit_vector(&mut rng);
            let norm = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
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
            let norm = (s[0]*s[0] + s[1]*s[1] + s[2]*s[2]).sqrt();
            assert!((norm - 1.0).abs() < 1e-12);
        }
    }
}
```

**Step 4: Build to verify**

```bash
cargo build 2>&1 | head -20
```

Expected: compiles (missing submodule stubs will error — create empty files first):

```bash
touch src/heisenberg/metropolis.rs
touch src/heisenberg/observables.rs
touch src/heisenberg/overrelax.rs
touch src/heisenberg/sweep.rs
touch src/heisenberg/fss.rs
```

Then `cargo build` should compile.

**Step 5: Run tests**

```bash
cargo test heisenberg::tests 2>&1
```

Expected: 4 tests pass.

**Step 6: Commit**

```bash
git add src/heisenberg/mod.rs src/heisenberg/metropolis.rs src/heisenberg/observables.rs src/heisenberg/overrelax.rs src/heisenberg/sweep.rs src/heisenberg/fss.rs src/lib.rs
git commit -m "feat: scaffold heisenberg module with spin type, lattice, energy/magnetisation"
```

---

## Task 2: Metropolis sweep for Heisenberg

**Files:**
- Modify: `src/heisenberg/metropolis.rs`

**Step 1: Write tests first**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::heisenberg::{HeisenbergLattice, energy_magnetisation, magnetisation_per_spin};
    use rand::SeedableRng;

    fn ring(n: usize) -> HeisenbergLattice {
        let nb = (0..n).map(|i| vec![(i+n-1)%n, (i+1)%n]).collect();
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
            let norm = (s[0]*s[0] + s[1]*s[1] + s[2]*s[2]).sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "spin norm = {norm}");
        }
    }

    #[test]
    fn low_temp_preserves_order() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb: Vec<Vec<usize>> = {
            // 4x4x4 cubic
            let n = 4;
            (0..n*n*n).map(|idx| {
                let z = idx/(n*n); let y = (idx/n)%n; let x = idx%n;
                vec![
                    ((x+1)%n) + y*n + z*n*n, ((x+n-1)%n) + y*n + z*n*n,
                    x + ((y+1)%n)*n + z*n*n, x + ((y+n-1)%n)*n + z*n*n,
                    x + y*n + ((z+1)%n)*n*n, x + y*n + ((z+n-1)%n)*n*n,
                ]
            }).collect()
        };
        let mut lat = HeisenbergLattice::new(nb);
        // start ordered
        for _ in 0..200 {
            sweep(&mut lat, 100.0, 1.0, 0.1, &mut rng);
        }
        let m = magnetisation_per_spin(&lat);
        assert!(m > 0.95, "at T≈0 |m| should be >0.95, got {m}");
    }

    #[test]
    fn high_temp_disorders() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(99);
        let nb: Vec<Vec<usize>> = (0..512).map(|i| vec![(i+511)%512, (i+1)%512]).collect();
        let mut lat = HeisenbergLattice::new(nb);
        for _ in 0..500 {
            sweep(&mut lat, 0.01, 1.0, 0.5, &mut rng);
        }
        let m = magnetisation_per_spin(&lat);
        assert!(m < 0.3, "at T→∞ |m| should be small, got {m}");
    }
}
```

**Step 2: Run to confirm they fail**

```bash
cargo test heisenberg::metropolis 2>&1 | head -20
```

Expected: compile error (empty file).

**Step 3: Implement**

```rust
use crate::heisenberg::{HeisenbergLattice, Spin3, random_unit_vector};
use rand::Rng;

/// One full Metropolis sweep over all spins in random order.
///
/// delta: cap angle for proposed rotation (radians). Tune to ~50% acceptance.
/// ΔE = −J [ (S'ᵢ − Sᵢ) · Σⱼ Sⱼ ]
pub fn sweep(
    lat: &mut HeisenbergLattice,
    beta: f64,
    j: f64,
    delta: f64,
    rng: &mut impl Rng,
) {
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
        let e_old = -j * (s[0]*hx + s[1]*hy + s[2]*hz);

        // Propose new spin: rotate by random angle <= delta from current
        let s_new = propose_rotation(&s, delta, rng);
        let e_new = -j * (s_new[0]*hx + s_new[1]*hy + s_new[2]*hz);

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
/// Uses the cap-rotation method: sample uniformly on spherical cap.
fn propose_rotation(s: &Spin3, delta: f64, rng: &mut impl Rng) -> Spin3 {
    // Sample a random unit vector on the spherical cap of half-angle delta
    let cos_delta = delta.cos();
    let cos_theta: f64 = rng.gen_range(cos_delta..1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    let phi: f64 = rng.gen_range(0.0..std::f64::consts::TAU);

    // Build in frame where s is the z-axis
    // Find two perpendicular vectors to s
    let (u, v) = perpendicular_frame(s);

    let nx = sin_theta * phi.cos() * u[0] + sin_theta * phi.sin() * v[0] + cos_theta * s[0];
    let ny = sin_theta * phi.cos() * u[1] + sin_theta * phi.sin() * v[1] + cos_theta * s[1];
    let nz = sin_theta * phi.cos() * u[2] + sin_theta * phi.sin() * v[2] + cos_theta * s[2];

    // Normalise to correct floating point drift
    let norm = (nx*nx + ny*ny + nz*nz).sqrt();
    [nx/norm, ny/norm, nz/norm]
}

/// Build an orthonormal frame {u, v} perpendicular to s.
fn perpendicular_frame(s: &Spin3) -> (Spin3, Spin3) {
    // Pick a vector not parallel to s
    let t: Spin3 = if s[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };

    // u = t - (t·s)s  (Gram-Schmidt), then normalise
    let ts = t[0]*s[0] + t[1]*s[1] + t[2]*s[2];
    let ux = t[0] - ts*s[0];
    let uy = t[1] - ts*s[1];
    let uz = t[2] - ts*s[2];
    let un = (ux*ux + uy*uy + uz*uz).sqrt();
    let u = [ux/un, uy/un, uz/un];

    // v = s × u
    let v = [
        s[1]*u[2] - s[2]*u[1],
        s[2]*u[0] - s[0]*u[2],
        s[0]*u[1] - s[1]*u[0],
    ];
    (u, v)
}
```

**Step 4: Run tests**

```bash
cargo test heisenberg::metropolis 2>&1
```

Expected: 3 tests pass.

**Step 5: Commit**

```bash
git add src/heisenberg/metropolis.rs
git commit -m "feat: add Heisenberg Metropolis sweep with spherical cap rotation"
```

---

## Task 3: Over-relaxation sweep

**Files:**
- Modify: `src/heisenberg/overrelax.rs`

**Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::heisenberg::{HeisenbergLattice, energy_magnetisation};

    fn ring(n: usize) -> HeisenbergLattice {
        let nb = (0..n).map(|i| vec![(i+n-1)%n, (i+1)%n]).collect();
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
            let norm = (s[0]*s[0] + s[1]*s[1] + s[2]*s[2]).sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "spin norm = {norm}");
        }
    }

    #[test]
    fn overrelax_conserves_energy() {
        // Over-relaxation is energy-conserving (microcanonical move)
        use rand::SeedableRng;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let mut lat = ring(20);
        lat.randomise(&mut rng);
        let (e_before, _) = energy_magnetisation(&lat, 1.0);
        sweep(&mut lat, 1.0);
        let (e_after, _) = energy_magnetisation(&lat, 1.0);
        assert!((e_after - e_before).abs() < 1e-8,
            "over-relaxation should conserve energy: before={e_before}, after={e_after}");
    }
}
```

**Step 2: Run to confirm fail**

```bash
cargo test heisenberg::overrelax 2>&1 | head -10
```

**Step 3: Implement**

```rust
use crate::heisenberg::HeisenbergLattice;

/// One full over-relaxation sweep.
///
/// For each spin Sᵢ, reflect through the local field hᵢ = J Σⱼ Sⱼ:
///   S'ᵢ = 2(Sᵢ·ĥ)ĥ − Sᵢ
///
/// This is deterministic and energy-conserving (microcanonical).
/// Reduces autocorrelation time without a Boltzmann acceptance step.
/// j is passed for generality but over-relaxation is independent of beta.
pub fn sweep(lat: &mut HeisenbergLattice, j: f64) {
    let size = lat.size();
    for idx in 0..size {
        // Local field (unnormalised)
        let mut hx = 0.0_f64;
        let mut hy = 0.0_f64;
        let mut hz = 0.0_f64;
        for &nb in &lat.neighbours[idx] {
            hx += j * lat.spins[nb][0];
            hy += j * lat.spins[nb][1];
            hz += j * lat.spins[nb][2];
        }

        let h2 = hx*hx + hy*hy + hz*hz;
        if h2 < 1e-30 {
            // Vanishing local field — skip (spin is free, reflection undefined)
            continue;
        }

        let s = lat.spins[idx];
        // S'ᵢ = 2(Sᵢ·h)h/|h|² − Sᵢ
        let sdoth = s[0]*hx + s[1]*hy + s[2]*hz;
        let scale = 2.0 * sdoth / h2;

        let nx = scale*hx - s[0];
        let ny = scale*hy - s[1];
        let nz = scale*hz - s[2];

        // Renormalise to correct floating point drift
        let norm = (nx*nx + ny*ny + nz*nz).sqrt();
        lat.spins[idx] = [nx/norm, ny/norm, nz/norm];
    }
}
```

**Step 4: Run tests**

```bash
cargo test heisenberg::overrelax 2>&1
```

Expected: 2 tests pass.

**Step 5: Commit**

```bash
git add src/heisenberg/overrelax.rs
git commit -m "feat: add Heisenberg over-relaxation sweep (energy-conserving microcanonical move)"
```

---

## Task 4: Combined sweep driver

**Files:**
- Modify: `src/heisenberg/sweep.rs`

**Step 1: Implement**

No separate tests needed — integration tested via observables. Write directly:

```rust
use crate::heisenberg::HeisenbergLattice;
use crate::heisenberg::metropolis;
use crate::heisenberg::overrelax;
use rand::Rng;

/// Configuration for a Heisenberg temperature sweep.
pub struct HeisSweepConfig {
    pub n_overrelax: usize,   // over-relaxation sweeps per Metropolis sweep (typically 5)
    pub delta: f64,           // Metropolis cap angle — tune to ~50% acceptance (~0.5 rad for cubic near Tc)
    pub j: f64,
    pub t_min: f64,
    pub t_max: f64,
    pub t_steps: usize,
    pub warmup_sweeps: usize,
    pub sample_sweeps: usize,
    pub seed: u64,
}

impl Default for HeisSweepConfig {
    fn default() -> Self {
        Self {
            n_overrelax: 5,
            delta: 0.5,
            j: 1.0,
            t_min: 0.8,
            t_max: 2.0,
            t_steps: 41,
            warmup_sweeps: 500,
            sample_sweeps: 500,
            seed: 42,
        }
    }
}

/// One combined sweep: 1 Metropolis + n_overrelax over-relaxation sweeps.
pub fn combined_sweep(
    lat: &mut HeisenbergLattice,
    beta: f64,
    j: f64,
    delta: f64,
    n_overrelax: usize,
    rng: &mut impl Rng,
) {
    metropolis::sweep(lat, beta, j, delta, rng);
    for _ in 0..n_overrelax {
        overrelax::sweep(lat, j);
    }
}
```

**Step 2: Build**

```bash
cargo build 2>&1 | grep -E "error|warning" | grep -v "warning: unused" | head -20
```

**Step 3: Commit**

```bash
git add src/heisenberg/sweep.rs
git commit -m "feat: add Heisenberg combined sweep driver (Metropolis + over-relaxation)"
```

---

## Task 5: Observables with jackknife errors

**Files:**
- Modify: `src/heisenberg/observables.rs`

**Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::heisenberg::HeisenbergLattice;
    use rand::SeedableRng;

    fn cubic3d_neighbours(n: usize) -> Vec<Vec<usize>> {
        (0..n*n*n).map(|idx| {
            let z = idx/(n*n); let y = (idx/n)%n; let x = idx%n;
            vec![
                ((x+1)%n) + y*n + z*n*n, ((x+n-1)%n) + y*n + z*n*n,
                x + ((y+1)%n)*n + z*n*n, x + ((y+n-1)%n)*n + z*n*n,
                x + y*n + ((z+1)%n)*n*n, x + y*n + ((z+n-1)%n)*n*n,
            ]
        }).collect()
    }

    #[test]
    fn measure_low_temp_ordered() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = cubic3d_neighbours(4);
        let mut lat = HeisenbergLattice::new(nb);
        let obs = measure(&mut lat, 0.5_f64.recip(), 1.0, 0.3, 5, 200, 200, &mut rng);
        // At very low T, |m| ≈ 1
        assert!(obs.magnetisation > 0.9,
            "at T=0.5 |m| should be >0.9, got {}", obs.magnetisation);
    }

    #[test]
    fn measure_high_temp_disordered() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = cubic3d_neighbours(6);
        let mut lat = HeisenbergLattice::new(nb);
        lat.randomise(&mut rng);
        let obs = measure(&mut lat, (10.0_f64).recip(), 1.0, 0.5, 5, 200, 500, &mut rng);
        assert!(obs.magnetisation < 0.3,
            "at T=10 |m| should be <0.3, got {}", obs.magnetisation);
    }

    #[test]
    fn binder_ground_state() {
        // At T→0, all m samples ≈ 1, so U ≈ 2/3
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = cubic3d_neighbours(4);
        let mut lat = HeisenbergLattice::new(nb);
        let obs = measure(&mut lat, (0.1_f64).recip(), 1.0, 0.1, 5, 500, 500, &mut rng);
        let u = 1.0 - obs.m4 / (3.0 * obs.m2 * obs.m2);
        assert!((u - 2.0/3.0).abs() < 0.05,
            "Binder cumulant at T→0 should be ≈2/3, got {u}");
    }

    #[test]
    fn jackknife_errors_are_positive() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = cubic3d_neighbours(4);
        let mut lat = HeisenbergLattice::new(nb);
        lat.randomise(&mut rng);
        let obs = measure(&mut lat, (2.0_f64).recip(), 1.0, 0.5, 5, 100, 200, &mut rng);
        assert!(obs.energy_err >= 0.0);
        assert!(obs.magnetisation_err >= 0.0);
        assert!(obs.heat_capacity_err >= 0.0);
        assert!(obs.susceptibility_err >= 0.0);
    }
}
```

**Step 2: Run to confirm fail**

```bash
cargo test heisenberg::observables 2>&1 | head -10
```

**Step 3: Implement**

```rust
use crate::heisenberg::{HeisenbergLattice, energy_magnetisation};
use crate::heisenberg::sweep::combined_sweep;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct HeisenbergObservables {
    pub temperature: f64,
    pub energy: f64,         pub energy_err: f64,
    pub magnetisation: f64,  pub magnetisation_err: f64,
    pub heat_capacity: f64,  pub heat_capacity_err: f64,
    pub susceptibility: f64, pub susceptibility_err: f64,
    pub m2: f64,             pub m2_err: f64,
    pub m4: f64,             pub m4_err: f64,
}

/// Measure observables with jackknife error estimation.
///
/// n_overrelax: over-relaxation sweeps per Metropolis sweep
/// delta: Metropolis cap angle
/// warmup: number of combined sweeps before sampling
/// samples: number of combined sweeps for measurements (must be >= 20 for jackknife)
#[allow(clippy::too_many_arguments)]
pub fn measure(
    lat: &mut HeisenbergLattice,
    beta: f64,
    j: f64,
    delta: f64,
    n_overrelax: usize,
    warmup: usize,
    samples: usize,
    rng: &mut impl Rng,
) -> HeisenbergObservables {
    let n = lat.size() as f64;

    // Warmup
    for _ in 0..warmup {
        combined_sweep(lat, beta, j, delta, n_overrelax, rng);
    }

    // Collect raw time series
    let mut e_series = Vec::with_capacity(samples);
    let mut m_series = Vec::with_capacity(samples);

    for _ in 0..samples {
        combined_sweep(lat, beta, j, delta, n_overrelax, rng);
        let (e, mv) = energy_magnetisation(lat, j);
        let m_abs = (mv[0]*mv[0] + mv[1]*mv[1] + mv[2]*mv[2]).sqrt() / n;
        e_series.push(e / n);
        m_series.push(m_abs);
    }

    // Full-sample averages
    let avg_e  = mean(&e_series);
    let avg_m  = mean(&m_series);
    let avg_e2 = mean_sq(&e_series);
    let avg_m2 = mean_of_sq(&m_series);
    let avg_m4 = mean_of_pow4(&m_series);

    let cv  = beta * beta * n * (avg_e2 - avg_e * avg_e);
    let chi = beta * n * (avg_m2 - avg_m * avg_m);

    // Jackknife with n_blocks = 20
    let n_blocks = 20;
    let block_size = samples / n_blocks;

    let (e_err, m_err, cv_err, chi_err, m2_err, m4_err) =
        jackknife_errors(&e_series, &m_series, beta, n, n_blocks, block_size);

    HeisenbergObservables {
        temperature: 1.0 / beta,
        energy: avg_e,         energy_err: e_err,
        magnetisation: avg_m,  magnetisation_err: m_err,
        heat_capacity: cv,     heat_capacity_err: cv_err,
        susceptibility: chi,   susceptibility_err: chi_err,
        m2: avg_m2,            m2_err,
        m4: avg_m4,            m4_err,
    }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn mean_sq(v: &[f64]) -> f64 {
    v.iter().map(|x| x*x).sum::<f64>() / v.len() as f64
}

fn mean_of_sq(v: &[f64]) -> f64 {
    mean_sq(v)
}

fn mean_of_pow4(v: &[f64]) -> f64 {
    v.iter().map(|x| x*x*x*x).sum::<f64>() / v.len() as f64
}

/// Jackknife error estimation. Returns (e_err, m_err, cv_err, chi_err, m2_err, m4_err).
fn jackknife_errors(
    e: &[f64],
    m: &[f64],
    beta: f64,
    n: f64,
    n_blocks: usize,
    block_size: usize,
) -> (f64, f64, f64, f64, f64, f64) {
    let mut jk_e   = Vec::with_capacity(n_blocks);
    let mut jk_m   = Vec::with_capacity(n_blocks);
    let mut jk_cv  = Vec::with_capacity(n_blocks);
    let mut jk_chi = Vec::with_capacity(n_blocks);
    let mut jk_m2  = Vec::with_capacity(n_blocks);
    let mut jk_m4  = Vec::with_capacity(n_blocks);

    let total = n_blocks * block_size;

    for b in 0..n_blocks {
        let lo = b * block_size;
        let hi = lo + block_size;

        // Leave-one-block-out averages
        let e_jk: Vec<f64>  = e[..lo].iter().chain(&e[hi..total]).copied().collect();
        let m_jk: Vec<f64>  = m[..lo].iter().chain(&m[hi..total]).copied().collect();

        let ae  = mean(&e_jk);
        let am  = mean(&m_jk);
        let ae2 = mean_sq(&e_jk);
        let am2 = mean_of_sq(&m_jk);
        let am4 = mean_of_pow4(&m_jk);

        jk_e.push(ae);
        jk_m.push(am);
        jk_cv.push(beta * beta * n * (ae2 - ae * ae));
        jk_chi.push(beta * n * (am2 - am * am));
        jk_m2.push(am2);
        jk_m4.push(am4);
    }

    (
        jackknife_std(&jk_e),
        jackknife_std(&jk_m),
        jackknife_std(&jk_cv),
        jackknife_std(&jk_chi),
        jackknife_std(&jk_m2),
        jackknife_std(&jk_m4),
    )
}

/// Jackknife standard error from leave-one-out estimates.
fn jackknife_std(jk: &[f64]) -> f64 {
    let nb = jk.len() as f64;
    let mean = jk.iter().sum::<f64>() / nb;
    let var = jk.iter().map(|x| (x - mean).powi(2)).sum::<f64>() * (nb - 1.0) / nb;
    var.sqrt()
}
```

**Step 4: Run tests**

```bash
cargo test heisenberg::observables 2>&1
```

Expected: 4 tests pass.

**Step 5: Commit**

```bash
git add src/heisenberg/observables.rs
git commit -m "feat: add Heisenberg observables with 20-block jackknife error estimation"
```

---

## Task 6: FSS runner

**Files:**
- Modify: `src/heisenberg/fss.rs`

**Step 1: Implement**

```rust
use crate::heisenberg::{HeisenbergLattice, observables::{measure, HeisenbergObservables}};
use crate::heisenberg::sweep::HeisSweepConfig;
use crate::lattice::{Geometry, Lattice};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub struct HeisFssConfig {
    pub sizes: Vec<usize>,
    pub geometry: Geometry,       // Cubic3D for validation
    pub j: f64,
    pub t_min: f64,
    pub t_max: f64,
    pub t_steps: usize,
    pub warmup_sweeps: usize,
    pub sample_sweeps: usize,
    pub n_overrelax: usize,
    pub delta: f64,
    pub seed: u64,
}

impl Default for HeisFssConfig {
    fn default() -> Self {
        Self {
            sizes: vec![8, 12, 16, 20, 24],
            geometry: Geometry::Cubic3D,
            j: 1.0,
            t_min: 0.8,
            t_max: 2.0,
            t_steps: 41,
            warmup_sweeps: 500,
            sample_sweeps: 500,
            n_overrelax: 5,
            delta: 0.5,
            seed: 42,
        }
    }
}

/// Run FSS temperature sweeps for each lattice size.
/// Returns Vec<(n, Vec<HeisenbergObservables>)>.
pub fn run_heisenberg_fss(config: &HeisFssConfig) -> Vec<(usize, Vec<HeisenbergObservables>)> {
    config.sizes.iter().map(|&n| {
        eprintln!("Heisenberg FSS: N={n}");
        let seed = config.seed.wrapping_add(n as u64);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

        // Build cubic lattice neighbours using existing Lattice infrastructure
        let ising_lat = Lattice::new(n, config.geometry);
        let mut lat = HeisenbergLattice::new(ising_lat.neighbours.clone());
        lat.randomise(&mut rng);

        let temps: Vec<f64> = (0..config.t_steps).map(|i| {
            config.t_min + (config.t_max - config.t_min) * i as f64 / (config.t_steps - 1) as f64
        }).collect();

        let results: Vec<HeisenbergObservables> = temps.iter().map(|&t| {
            let beta = 1.0 / t;
            measure(
                &mut lat, beta, config.j, config.delta,
                config.n_overrelax, config.warmup_sweeps, config.sample_sweeps,
                &mut rng,
            )
        }).collect();

        (n, results)
    }).collect()
}
```

**Step 2: Build**

```bash
cargo build 2>&1 | grep "error" | head -20
```

**Step 3: Commit**

```bash
git add src/heisenberg/fss.rs
git commit -m "feat: add Heisenberg FSS runner over multiple lattice sizes"
```

---

## Task 7: CLI binary `heisenberg_fss`

**Files:**
- Create: `src/bin/heisenberg_fss.rs`
- Modify: `Cargo.toml`

**Step 1: Add binary to Cargo.toml**

In `Cargo.toml`, after the existing `[[bin]]` entries:

```toml
[[bin]]
name = "heisenberg_fss"
path = "src/bin/heisenberg_fss.rs"
```

**Step 2: Implement**

```rust
/// CLI: run Heisenberg FSS for multiple lattice sizes.
///
/// Usage:
///   cargo run --release --bin heisenberg_fss
///   cargo run --release --bin heisenberg_fss -- --sizes 8,12,16,20 --outdir analysis/data
///
/// Output: one CSV per size at <outdir>/heisenberg_fss_N<n>.csv
/// Columns: T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err
use std::env;
use std::fs;
use std::path::Path;
use ising::heisenberg::fss::{HeisFssConfig, run_heisenberg_fss};
use ising::lattice::Geometry;

fn get_arg(args: &[String], i: usize, flag: &str) -> String {
    if i + 1 >= args.len() {
        eprintln!("Error: {flag} requires a value");
        std::process::exit(1);
    }
    args[i + 1].clone()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config = HeisFssConfig::default();
    let mut outdir = String::from("analysis/data");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sizes"    => {
                config.sizes = get_arg(&args, i, "--sizes").split(',')
                    .filter_map(|s| s.parse().ok()).collect();
                i += 2;
            }
            "--tmin"     => { config.t_min = get_arg(&args, i, "--tmin").parse().unwrap(); i += 2; }
            "--tmax"     => { config.t_max = get_arg(&args, i, "--tmax").parse().unwrap(); i += 2; }
            "--steps"    => { config.t_steps = get_arg(&args, i, "--steps").parse().unwrap(); i += 2; }
            "--warmup"   => { config.warmup_sweeps = get_arg(&args, i, "--warmup").parse().unwrap(); i += 2; }
            "--samples"  => { config.sample_sweeps = get_arg(&args, i, "--samples").parse().unwrap(); i += 2; }
            "--overrelax"=> { config.n_overrelax = get_arg(&args, i, "--overrelax").parse().unwrap(); i += 2; }
            "--delta"    => { config.delta = get_arg(&args, i, "--delta").parse().unwrap(); i += 2; }
            "--seed"     => { config.seed = get_arg(&args, i, "--seed").parse().unwrap(); i += 2; }
            "--j"        => { config.j = get_arg(&args, i, "--j").parse().unwrap(); i += 2; }
            "--outdir"   => { outdir = get_arg(&args, i, "--outdir"); i += 2; }
            _            => { i += 1; }
        }
    }

    fs::create_dir_all(&outdir).expect("failed to create outdir");

    let results = run_heisenberg_fss(&config);

    for (n, obs_list) in &results {
        let path = Path::new(&outdir).join(format!("heisenberg_fss_N{n}.csv"));
        let mut csv = String::from("T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err\n");
        for o in obs_list {
            csv.push_str(&format!(
                "{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                o.temperature,
                o.energy, o.energy_err,
                o.magnetisation, o.magnetisation_err,
                o.m2, o.m2_err,
                o.m4, o.m4_err,
                o.heat_capacity, o.heat_capacity_err,
                o.susceptibility, o.susceptibility_err,
            ));
        }
        fs::write(&path, &csv).expect("failed to write CSV");
        eprintln!("Wrote {}", path.display());
    }
}
```

**Step 3: Build and smoke test**

```bash
cargo build --release --bin heisenberg_fss 2>&1 | grep "error" | head -10
cargo run --release --bin heisenberg_fss -- --sizes 4 --tmin 1.0 --tmax 2.0 --steps 5 --warmup 50 --samples 60 --outdir /tmp/heis_test
head -3 /tmp/heis_test/heisenberg_fss_N4.csv
```

Expected: CSV with 5 rows and 13 columns, no NaN values.

**Step 4: Commit**

```bash
git add src/bin/heisenberg_fss.rs Cargo.toml
git commit -m "feat: add heisenberg_fss CLI binary with full jackknife CSV output"
```

---

## Task 8: CLI binary `heisenberg_jfit` (crystal graph sweeps)

**Files:**
- Create: `src/bin/heisenberg_jfit.rs`
- Modify: `Cargo.toml`

**Step 1: Add binary to Cargo.toml**

```toml
[[bin]]
name = "heisenberg_jfit"
path = "src/bin/heisenberg_jfit.rs"
```

**Step 2: Implement**

```rust
/// CLI: run Heisenberg temperature sweep on a graph loaded from JSON.
///
/// Usage:
///   cargo run --release --bin heisenberg_jfit -- \
///     --graph analysis/graphs/bcc_N8.json \
///     --tmin 4.0 --tmax 9.0 --steps 41 \
///     --warmup 500 --samples 500 \
///     --outdir analysis/data
///
/// Output: <outdir>/heisenberg_jfit_<graphname>.csv
use std::env;
use std::fs;
use std::path::Path;
use ising::heisenberg::{HeisenbergLattice, observables::measure};
use ising::graph::GraphDef;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn get_arg(args: &[String], i: usize, flag: &str) -> String {
    if i + 1 >= args.len() {
        eprintln!("Error: {flag} requires a value");
        std::process::exit(1);
    }
    args[i + 1].clone()
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut graph_path = String::new();
    let mut outdir = String::from("analysis/data");
    let mut t_min = 4.0_f64;
    let mut t_max = 9.0_f64;
    let mut t_steps = 41usize;
    let mut warmup = 500usize;
    let mut samples = 500usize;
    let mut n_overrelax = 5usize;
    let mut delta = 0.5_f64;
    let mut j = 1.0_f64;
    let mut seed = 42u64;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--graph"     => { graph_path = get_arg(&args, i, "--graph"); i += 2; }
            "--outdir"    => { outdir = get_arg(&args, i, "--outdir"); i += 2; }
            "--tmin"      => { t_min = get_arg(&args, i, "--tmin").parse().unwrap(); i += 2; }
            "--tmax"      => { t_max = get_arg(&args, i, "--tmax").parse().unwrap(); i += 2; }
            "--steps"     => { t_steps = get_arg(&args, i, "--steps").parse().unwrap(); i += 2; }
            "--warmup"    => { warmup = get_arg(&args, i, "--warmup").parse().unwrap(); i += 2; }
            "--samples"   => { samples = get_arg(&args, i, "--samples").parse().unwrap(); i += 2; }
            "--overrelax" => { n_overrelax = get_arg(&args, i, "--overrelax").parse().unwrap(); i += 2; }
            "--delta"     => { delta = get_arg(&args, i, "--delta").parse().unwrap(); i += 2; }
            "--j"         => { j = get_arg(&args, i, "--j").parse().unwrap(); i += 2; }
            "--seed"      => { seed = get_arg(&args, i, "--seed").parse().unwrap(); i += 2; }
            _             => { i += 1; }
        }
    }

    if graph_path.is_empty() {
        eprintln!("Error: --graph <path.json> is required");
        std::process::exit(1);
    }

    let content = fs::read_to_string(&graph_path).expect("failed to read graph file");
    let graph = GraphDef::from_json(&content).expect("failed to parse graph JSON");

    let graph_name = Path::new(&graph_path)
        .file_stem().unwrap().to_str().unwrap().to_string();

    eprintln!("Heisenberg jfit: graph={graph_name}, N={}, T={t_min}..{t_max}",
        graph.n_nodes);

    // Build adjacency list from edge list
    let mut neighbours: Vec<Vec<usize>> = vec![vec![]; graph.n_nodes];
    for (a, b) in &graph.edges {
        neighbours[*a].push(*b);
        neighbours[*b].push(*a);
    }

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut lat = HeisenbergLattice::new(neighbours);
    lat.randomise(&mut rng);

    fs::create_dir_all(&outdir).expect("failed to create outdir");

    let path = Path::new(&outdir).join(format!("heisenberg_jfit_{graph_name}.csv"));
    let mut csv = String::from("T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err\n");

    for step in 0..t_steps {
        let t = t_min + (t_max - t_min) * step as f64 / (t_steps - 1) as f64;
        let beta = 1.0 / t;
        let obs = measure(&mut lat, beta, j, delta, n_overrelax, warmup, samples, &mut rng);
        csv.push_str(&format!(
            "{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            obs.temperature,
            obs.energy, obs.energy_err,
            obs.magnetisation, obs.magnetisation_err,
            obs.m2, obs.m2_err,
            obs.m4, obs.m4_err,
            obs.heat_capacity, obs.heat_capacity_err,
            obs.susceptibility, obs.susceptibility_err,
        ));
        eprintln!("  T={t:.3} M={:.4}±{:.4}", obs.magnetisation, obs.magnetisation_err);
    }

    fs::write(&path, &csv).expect("failed to write CSV");
    eprintln!("Wrote {}", path.display());
}
```

**Step 3: Build and smoke test**

```bash
cargo build --release --bin heisenberg_jfit 2>&1 | grep "error" | head -10
cargo run --release --bin heisenberg_jfit -- \
  --graph analysis/graphs/bcc_N4.json \
  --tmin 5.0 --tmax 8.0 --steps 3 --warmup 50 --samples 60
```

Expected: 3 rows of CSV output, no errors.

**Step 4: Commit**

```bash
git add src/bin/heisenberg_jfit.rs Cargo.toml
git commit -m "feat: add heisenberg_jfit CLI binary for crystal graph J-fitting sweeps"
```

---

## Task 9: Integration tests

**Files:**
- Modify: `tests/cli.rs` (add to existing file)

**Step 1: Add tests**

```rust
// In tests/cli.rs, add:

#[test]
fn heisenberg_fss_smoke() {
    let dir = tempdir().unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_heisenberg_fss"))
        .args([
            "--sizes", "4",
            "--tmin", "1.0", "--tmax", "2.0", "--steps", "3",
            "--warmup", "20", "--samples", "40",
            "--outdir", dir.path().to_str().unwrap(),
        ])
        .status()
        .expect("failed to run heisenberg_fss");
    assert!(status.success());

    let csv = std::fs::read_to_string(dir.path().join("heisenberg_fss_N4.csv")).unwrap();
    let rows: Vec<&str> = csv.lines().collect();
    assert_eq!(rows.len(), 4); // header + 3 data rows
    assert!(rows[0].contains("E_err"), "CSV should have error columns");
}

#[test]
fn heisenberg_jfit_smoke() {
    let dir = tempdir().unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_heisenberg_jfit"))
        .args([
            "--graph", "analysis/graphs/bcc_N4.json",
            "--tmin", "5.0", "--tmax", "8.0", "--steps", "3",
            "--warmup", "20", "--samples", "40",
            "--outdir", dir.path().to_str().unwrap(),
        ])
        .status()
        .expect("failed to run heisenberg_jfit");
    assert!(status.success());

    let csv_path = dir.path().join("heisenberg_jfit_bcc_N4.csv");
    let csv = std::fs::read_to_string(csv_path).unwrap();
    let rows: Vec<&str> = csv.lines().collect();
    assert_eq!(rows.len(), 4);
}
```

**Step 2: Run**

```bash
cargo test --test cli heisenberg 2>&1
```

Expected: 2 tests pass.

**Step 3: Commit**

```bash
git add tests/cli.rs
git commit -m "test: add integration tests for heisenberg_fss and heisenberg_jfit binaries"
```

---

## Task 10: Publication run scripts

**Files:**
- Create: `scripts/run_heisenberg_publication.sh`

**Step 1: Create**

```bash
#!/usr/bin/env bash
# Publication-quality Heisenberg runs.
# Run on Windows machine (RTX 2060) — CPU is sufficient for these sizes.
# Estimated time: ~2-4 hours on modern CPU.

set -e
OUTDIR="analysis/data"
mkdir -p "$OUTDIR"

echo "=== Heisenberg FSS: cubic lattice validation ==="
cargo run --release --bin heisenberg_fss -- \
  --sizes 8,12,16,20,24,32 \
  --tmin 1.0 --tmax 2.0 --steps 41 \
  --warmup 2000 --samples 2000 \
  --n-overrelax 5 --delta 0.5 \
  --outdir "$OUTDIR"

echo "=== Heisenberg J-fitting: BCC iron ==="
for N in 4 6 8 10 12; do
  cargo run --release --bin heisenberg_jfit -- \
    --graph "analysis/graphs/bcc_N${N}.json" \
    --tmin 4.0 --tmax 9.0 --steps 41 \
    --warmup 2000 --samples 2000 \
    --outdir "$OUTDIR"
done

echo "=== Heisenberg J-fitting: FCC nickel ==="
for N in 4 6 8 10 12; do
  cargo run --release --bin heisenberg_jfit -- \
    --graph "analysis/graphs/fcc_N${N}.json" \
    --tmin 6.0 --tmax 14.0 --steps 41 \
    --warmup 2000 --samples 2000 \
    --outdir "$OUTDIR"
done

echo "All done. Run analysis/heisenberg_fss.ipynb and analysis/heisenberg_jfit.ipynb"
```

**Step 2: Commit**

```bash
chmod +x scripts/run_heisenberg_publication.sh
git add scripts/run_heisenberg_publication.sh
git commit -m "feat: add publication run script for Heisenberg FSS and J-fitting"
```

---

## Task 11: Notebook stubs

**Files:**
- Create: `analysis/heisenberg_fss.ipynb` (stub)
- Create: `analysis/heisenberg_jfit.ipynb` (stub)

These are stub notebooks — full analysis cells to be written after data is available.
Each notebook should have sections matching the design:

**heisenberg_fss.ipynb sections:**
1. Load all `heisenberg_fss_N*.csv`, plot E, |M|, Cv, χ with error bars
2. Binder cumulant U(T) crossings → Tc with uncertainty
3. Peak scaling: χ_max ~ L^{γ/ν}, M(Tc) ~ L^{-β/ν}
4. Scaling collapse → ν
5. Summary table: measured vs O(3) theory (ν=0.7112, γ=1.3960, β=0.3646)

**heisenberg_jfit.ipynb sections:**
1. Load BCC sweep CSVs, Binder crossings → Tc_BCC(J=1)
2. J_fit(Fe) = kB * 1043K / Tc_BCC — with uncertainty propagation
3. Load FCC sweep CSVs, Binder crossings → Tc_FCC(J=1)
4. J_fit(Ni) = kB * 627K / Tc_FCC
5. Comparison table: Ising vs Heisenberg vs Pajda 2001

**Step 1: Create stub notebooks via Jupyter (or copy structure from existing fss.ipynb)**

```bash
cp analysis/fss.ipynb analysis/heisenberg_fss.ipynb
cp analysis/fit_j.ipynb analysis/heisenberg_jfit.ipynb
```

Then clear all outputs and update section headers.

**Step 2: Commit**

```bash
git add analysis/heisenberg_fss.ipynb analysis/heisenberg_jfit.ipynb
git commit -m "feat: add stub notebooks for Heisenberg FSS and J-fitting analysis"
```

---

## Verification

After all tasks complete, run the full validation:

```bash
# All tests pass
cargo test 2>&1 | tail -5

# Smoke run produces valid CSVs
cargo run --release --bin heisenberg_fss -- \
  --sizes 6,8 --tmin 1.2 --tmax 1.8 --steps 5 \
  --warmup 200 --samples 200 --outdir /tmp/heis_verify

# Check output is sane
python3 -c "
import pandas as pd
df = pd.read_csv('/tmp/heis_verify/heisenberg_fss_N8.csv')
print(df.head())
assert df['E_err'].notna().all()
assert (df['M'] > 0).all()
print('OK')
"
```

Expected: 5-row DataFrame with no NaN errors and positive magnetisation.
