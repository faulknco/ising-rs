# XY Spin Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a classical XY spin model (O(2) universality) to ising-rs with the Wolff cluster algorithm, jackknife error estimation, two CLI binaries (FSS validation + J-fitting), and full analysis notebooks — zero modifications to any existing file except appending `pub mod xy;` to `src/lib.rs`.

**Architecture:** New `src/xy/` module (mod.rs, wolff.rs, observables.rs, fss.rs) mirroring `src/heisenberg/` exactly except for spin dimensionality and algorithm. Two new binaries in `src/bin/` registered in `Cargo.toml`. Two integration tests appended to `tests/cli.rs`. Three new Jupyter notebooks in `analysis/`. One new shell script in `scripts/`.

**Tech Stack:** Rust stable (edition 2021, rust-version 1.94), `rand` 0.8 + `rand_xoshiro` 0.6 (already in `Cargo.toml`), existing `src/graph.rs` JSON loader, existing `analysis/graphs/bcc_*.json` / `fcc_*.json`, Python/matplotlib/scipy for notebooks.

**Physics reference:**
- 3D cubic XY Tc(J=1) = 2.2016 J/kB
- O(2) exponents: ν=0.6717, γ=1.3177, β=0.3486
- Wolff Binder: U = 1 − ⟨m⁴⟩/(3⟨m²⟩²)
- BCC Fe XY Tc estimate: ~2.835 J/kB (z=8)
- FCC Ni XY Tc estimate: ~4.35 J/kB (z=12)
- BCC Fe: Tc_exp=1043 K, J_lit=16.3 meV (Pajda 2001)
- FCC Ni: Tc_exp=627 K, J_lit=4.1 meV (Pajda 2001)

---

## Task 1: `src/xy/mod.rs` — Spin2, XyLattice, energy/magnetisation, random_unit_circle

**Files:**
- Create: `/Users/faulknco/Projects/ising-rs/src/xy/mod.rs`

**Complete code:**

```rust
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
        // For uniform distribution on S¹, E[cos φ] = 0 and Var[cos φ] = 0.5.
        // Over 10_000 samples, mean cos should be within 0.05 of 0.
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
```

**Verification:**
```
cargo test xy::tests
```

**Commit message:** `feat(xy): add XyLattice, Spin2, energy/magnetisation, random_unit_circle`

---

## Task 2: `src/xy/wolff.rs` — Wolff cluster sweep

**Files:**
- Create: `/Users/faulknco/Projects/ising-rs/src/xy/wolff.rs`

**Physics notes:**
- One call to `sweep()` = one cluster flip attempt. Near Tc the cluster covers O(N) spins, so one call decorrelates the system as well as O(N) Metropolis sweeps.
- The mirror vector r is drawn fresh each call. Spins project onto r: σᵢ = Sᵢ·r. The embedded Ising model has bonds formed only between same-sign projections.
- Bond probability: p = 1 − exp(−2βJ · σᵢ · σⱼ) when σᵢ > 0 and σⱼ > 0 (i.e. same side); 0 otherwise.
- Flip: Sᵢ → Sᵢ − 2(Sᵢ·r)r, then renormalise to correct floating-point drift.
- BFS via a `Vec<usize>` stack (no allocator per call — reuse via clear).

**Complete code:**

```rust
use crate::xy::XyLattice;
use rand::Rng;

/// One Wolff cluster flip on the XY lattice.
///
/// Algorithm (Wolff 1989, adapted to O(2)):
/// 1. Pick a random mirror vector r = (cos φ, sin φ) uniformly on S¹.
/// 2. Pick a random seed spin s uniformly from the lattice.
/// 3. Grow a cluster via BFS: for each active spin i with positive projection
///    pᵢ = Sᵢ·r > 0, add neighbour j (with pⱼ > 0) with probability
///    p = 1 − exp(−2βJ · pᵢ · pⱼ).
/// 4. Flip all cluster spins: Sᵢ → Sᵢ − 2(Sᵢ·r)r, then renormalise.
///
/// One call decorrelates the system as efficiently as O(N) Metropolis sweeps
/// near Tc — hence measure() calls sweep() once per sample.
pub fn sweep(lat: &mut XyLattice, beta: f64, j: f64, rng: &mut impl Rng) {
    let n = lat.size();
    if n == 0 {
        return;
    }

    // 1. Random mirror vector r on S¹
    let phi: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
    let r = [phi.cos(), phi.sin()];

    // 2. Random seed spin
    let seed = rng.gen_range(0..n);

    // 3. BFS cluster growth
    // in_cluster[i] = true if spin i has been added to the cluster
    let mut in_cluster = vec![false; n];
    let mut stack: Vec<usize> = Vec::with_capacity(n / 4 + 8);

    // Check seed projection; only start cluster if seed projects positively
    let proj_seed = lat.spins[seed][0] * r[0] + lat.spins[seed][1] * r[1];
    if proj_seed > 0.0 {
        in_cluster[seed] = true;
        stack.push(seed);
    }

    while let Some(idx) = stack.pop() {
        let proj_i = lat.spins[idx][0] * r[0] + lat.spins[idx][1] * r[1];
        // proj_i > 0 guaranteed by construction (only positive-projection spins enter)

        // Iterate over neighbours (clone indices to avoid borrow conflict)
        let nb_indices: Vec<usize> = lat.neighbours[idx].clone();
        for nb in nb_indices {
            if in_cluster[nb] {
                continue;
            }
            let proj_j = lat.spins[nb][0] * r[0] + lat.spins[nb][1] * r[1];
            if proj_j <= 0.0 {
                continue; // wrong side — bond probability is 0
            }
            // Bond probability p = 1 − exp(−2βJ · pᵢ · pⱼ)
            let p_bond = 1.0 - (-2.0 * beta * j * proj_i * proj_j).exp();
            if rng.gen::<f64>() < p_bond {
                in_cluster[nb] = true;
                stack.push(nb);
            }
        }
    }

    // 4. Flip r-component of all cluster spins and renormalise
    for (i, &in_c) in in_cluster.iter().enumerate() {
        if !in_c {
            continue;
        }
        let s = &mut lat.spins[i];
        let dot = s[0] * r[0] + s[1] * r[1];
        s[0] -= 2.0 * dot * r[0];
        s[1] -= 2.0 * dot * r[1];
        // Renormalise to correct floating-point drift
        let norm = (s[0] * s[0] + s[1] * s[1]).sqrt();
        s[0] /= norm;
        s[1] /= norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xy::{XyLattice, magnetisation_per_spin};
    use rand::SeedableRng;

    fn cubic3d_neighbours(n: usize) -> Vec<Vec<usize>> {
        (0..n * n * n)
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
            .collect()
    }

    #[test]
    fn spins_remain_unit_after_sweep() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = cubic3d_neighbours(4);
        let mut lat = XyLattice::new(nb);
        lat.randomise(&mut rng);
        for _ in 0..100 {
            sweep(&mut lat, 0.5, 1.0, &mut rng);
        }
        for s in &lat.spins {
            let norm = (s[0] * s[0] + s[1] * s[1]).sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "spin norm = {norm}");
        }
    }

    #[test]
    fn low_temp_wolff_stays_ordered() {
        // Ordered start at very low T: cluster flips should not disorder the system
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = cubic3d_neighbours(4);
        let mut lat = XyLattice::new(nb);
        // At T=0.01 the system stays near |m|=1
        for _ in 0..200 {
            sweep(&mut lat, 100.0, 1.0, &mut rng);
        }
        let m = magnetisation_per_spin(&lat);
        assert!(m > 0.90, "at T≈0 |m| should be >0.90 after Wolff, got {m}");
    }

    #[test]
    fn high_temp_wolff_disorders() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(99);
        let nb = cubic3d_neighbours(6);
        let mut lat = XyLattice::new(nb);
        lat.randomise(&mut rng);
        for _ in 0..500 {
            sweep(&mut lat, 0.01, 1.0, &mut rng);
        }
        let m = magnetisation_per_spin(&lat);
        assert!(m < 0.3, "at T→∞ |m| should be small, got {m}");
    }

    #[test]
    fn cluster_flip_conserves_energy_direction() {
        // After a global flip (entire lattice in cluster), energy is unchanged.
        // We cannot guarantee the whole lattice is chosen, but at very low T
        // with an ordered start the cluster will be large.
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(7);
        let nb = cubic3d_neighbours(4);
        let mut lat = XyLattice::new(nb);
        // Ordered: all spins (1,0). Energy = -J * N_bonds
        let (e_before, _) = crate::xy::energy_magnetisation(&lat, 1.0);
        // After any cluster flip, |Sᵢ·Sⱼ| is unchanged for aligned neighbours
        for _ in 0..50 {
            sweep(&mut lat, 10.0, 1.0, &mut rng);
        }
        let (e_after, _) = crate::xy::energy_magnetisation(&lat, 1.0);
        // Energy at very low T should remain very close to ground state
        assert!(
            e_after < e_before * 0.5,
            "energy should stay near ground state at low T: before={e_before}, after={e_after}"
        );
    }
}
```

**Verification:**
```
cargo test xy::wolff::tests
```

**Commit message:** `feat(xy): add Wolff cluster sweep algorithm`

---

## Task 3: `src/xy/observables.rs` — XyObservables + measure() with jackknife

**Files:**
- Create: `/Users/faulknco/Projects/ising-rs/src/xy/observables.rs`

**Key differences from heisenberg/observables.rs:**
- Import `crate::xy::{XyLattice, energy_magnetisation}` and `crate::xy::wolff::sweep`
- `measure()` signature drops `delta` and `n_overrelax` (Wolff has no tuning parameters)
- Magnetisation: `(mv[0]*mv[0] + mv[1]*mv[1]).sqrt() / n` (2D vector, not 3D)
- All jackknife logic is byte-for-byte identical to the heisenberg version

**Complete code:**

```rust
use crate::xy::{XyLattice, energy_magnetisation};
use crate::xy::wolff::sweep;
use rand::Rng;

/// All measured observables for one temperature point, with jackknife error bars.
#[derive(Debug, Clone)]
pub struct XyObservables {
    pub temperature: f64,
    /// Mean energy per spin E/N.
    pub energy: f64,         pub energy_err: f64,
    /// Mean |M|/N (scalar magnetisation per spin).
    pub magnetisation: f64,  pub magnetisation_err: f64,
    /// Total heat capacity C = β²N(⟨(E/N)²⟩ − ⟨E/N⟩²).
    pub heat_capacity: f64,  pub heat_capacity_err: f64,
    /// Magnetic susceptibility χ = βN(⟨m²⟩ − ⟨m⟩²).
    pub susceptibility: f64, pub susceptibility_err: f64,
    /// ⟨m²⟩ (needed for Binder cumulant).
    pub m2: f64,             pub m2_err: f64,
    /// ⟨m⁴⟩ (needed for Binder cumulant).
    pub m4: f64,             pub m4_err: f64,
}

/// Equilibrate a lattice and measure observables with 20-block jackknife error estimation.
///
/// One call to `wolff::sweep()` = one cluster flip attempt, which near Tc decorrelates
/// the system as efficiently as O(N) Metropolis sweeps. The measure() function therefore
/// collects one observable sample per sweep call (no thinning needed).
///
/// # Arguments
/// - `lat`: lattice (mutated in place; callers may re-use across temperatures)
/// - `beta`: inverse temperature 1/T
/// - `j`: exchange coupling
/// - `warmup`: equilibration Wolff sweeps (discarded)
/// - `samples`: measurement Wolff sweeps (must be ≥ 20 for jackknife; divisible by 20)
pub fn measure(
    lat: &mut XyLattice,
    beta: f64,
    j: f64,
    warmup: usize,
    samples: usize,
    rng: &mut impl Rng,
) -> XyObservables {
    let n = lat.size() as f64;

    // Equilibration: discard these sweeps
    for _ in 0..warmup {
        sweep(lat, beta, j, rng);
    }

    // Measurement: collect per-sweep time series
    let mut e_series = Vec::with_capacity(samples);
    let mut m_series = Vec::with_capacity(samples);

    for _ in 0..samples {
        sweep(lat, beta, j, rng);
        let (e, mv) = energy_magnetisation(lat, j);
        let m_abs = (mv[0] * mv[0] + mv[1] * mv[1]).sqrt() / n;
        e_series.push(e / n);
        m_series.push(m_abs);
    }

    // Full-sample averages
    let avg_e  = mean(&e_series);
    let avg_m  = mean(&m_series);
    let avg_e2 = mean_of_sq(&e_series);
    let avg_m2 = mean_of_sq(&m_series);
    let avg_m4 = mean_of_pow4(&m_series);

    // Derived quantities (extensive in system size)
    let cv  = beta * beta * n * (avg_e2 - avg_e * avg_e);
    let chi = beta * n * (avg_m2 - avg_m * avg_m);

    // Jackknife error estimation with 20 blocks
    let n_blocks = 20;
    let block_size = samples / n_blocks;
    debug_assert!(
        samples.is_multiple_of(n_blocks),
        "samples ({samples}) is not divisible by n_blocks ({n_blocks}); \
         {} samples will be silently discarded",
        samples % n_blocks
    );

    let (e_err, m_err, cv_err, chi_err, m2_err, m4_err) =
        jackknife_errors(&e_series, &m_series, beta, n, n_blocks, block_size);

    XyObservables {
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

fn mean_of_sq(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64
}

fn mean_of_pow4(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x * x * x).sum::<f64>() / v.len() as f64
}

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

        let e_jk: Vec<f64> = e[..lo].iter().chain(&e[hi..total]).copied().collect();
        let m_jk: Vec<f64> = m[..lo].iter().chain(&m[hi..total]).copied().collect();

        let ae  = mean(&e_jk);
        let am  = mean(&m_jk);
        let ae2 = mean_of_sq(&e_jk);
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

fn jackknife_std(jk: &[f64]) -> f64 {
    let nb = jk.len() as f64;
    let mean = jk.iter().sum::<f64>() / nb;
    let var = jk.iter().map(|x| (x - mean).powi(2)).sum::<f64>() * (nb - 1.0) / nb;
    var.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xy::XyLattice;
    use rand::SeedableRng;

    fn cubic3d_neighbours(n: usize) -> Vec<Vec<usize>> {
        (0..n * n * n)
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
            .collect()
    }

    #[test]
    fn measure_low_temp_ordered() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = cubic3d_neighbours(4);
        let mut lat = XyLattice::new(nb);
        let obs = measure(&mut lat, 0.5_f64.recip(), 1.0, 200, 200, &mut rng);
        assert!(
            obs.magnetisation > 0.85,
            "at T=0.5 |m| should be >0.85, got {}",
            obs.magnetisation
        );
    }

    #[test]
    fn measure_high_temp_disordered() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = cubic3d_neighbours(6);
        let mut lat = XyLattice::new(nb);
        lat.randomise(&mut rng);
        let obs = measure(&mut lat, (10.0_f64).recip(), 1.0, 200, 500, &mut rng);
        assert!(
            obs.magnetisation < 0.3,
            "at T=10 |m| should be <0.3, got {}",
            obs.magnetisation
        );
    }

    #[test]
    fn binder_ground_state() {
        // At T→0, U = 1 − ⟨m⁴⟩/(3⟨m²⟩²) → 2/3 (same formula as Heisenberg)
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = cubic3d_neighbours(4);
        let mut lat = XyLattice::new(nb);
        let obs = measure(&mut lat, (0.1_f64).recip(), 1.0, 500, 500, &mut rng);
        let u = 1.0 - obs.m4 / (3.0 * obs.m2 * obs.m2);
        assert!(
            (u - 2.0 / 3.0).abs() < 0.1,
            "Binder cumulant at T→0 should be ≈2/3, got {u}"
        );
    }

    #[test]
    fn jackknife_errors_are_positive() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);
        let nb = cubic3d_neighbours(4);
        let mut lat = XyLattice::new(nb);
        lat.randomise(&mut rng);
        let obs = measure(&mut lat, (2.2_f64).recip(), 1.0, 100, 200, &mut rng);
        assert!(obs.energy_err >= 0.0);
        assert!(obs.magnetisation_err >= 0.0);
        assert!(obs.heat_capacity_err >= 0.0);
        assert!(obs.susceptibility_err >= 0.0);
    }
}
```

**Verification:**
```
cargo test xy::observables::tests
```

**Commit message:** `feat(xy): add XyObservables + jackknife measure()`

---

## Task 4: `src/xy/fss.rs` — XyFssConfig + run_xy_fss()

**Files:**
- Create: `/Users/faulknco/Projects/ising-rs/src/xy/fss.rs`

**Key differences from heisenberg/fss.rs:**
- `XyFssConfig` replaces `HeisFssConfig`; fields `n_overrelax` and `delta` are absent (Wolff has no such parameters)
- Default temperature range: tmin=1.8, tmax=2.7 (straddles XY Tc=2.2016)
- Default sizes: [8, 12, 16, 20, 24, 32]
- Calls `xy::observables::measure()` with 4 args (no delta/overrelax)
- Returns `Vec<(usize, Vec<XyObservables>)>`

**Complete code:**

```rust
use crate::xy::{XyLattice, observables::{measure, XyObservables}};
use crate::lattice::{Geometry, Lattice};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Configuration for an XY FSS run over multiple lattice sizes.
pub struct XyFssConfig {
    /// Linear lattice sizes to simulate.
    pub sizes: Vec<usize>,
    /// Lattice geometry (Cubic3D for O(2) validation).
    pub geometry: Geometry,
    /// Exchange coupling J.
    pub j: f64,
    /// Minimum temperature (units J/k_B).
    pub t_min: f64,
    /// Maximum temperature (units J/k_B).
    pub t_max: f64,
    /// Number of temperature steps (uniformly spaced).
    pub t_steps: usize,
    /// Wolff cluster sweeps to discard before measuring.
    pub warmup_sweeps: usize,
    /// Wolff cluster sweeps to measure (must be divisible by 20).
    pub sample_sweeps: usize,
    /// Base RNG seed (each size uses seed.wrapping_add(n)).
    pub seed: u64,
}

impl Default for XyFssConfig {
    fn default() -> Self {
        Self {
            sizes: vec![8, 12, 16, 20, 24, 32],
            geometry: Geometry::Cubic3D,
            j: 1.0,
            t_min: 1.8,
            t_max: 2.7,
            t_steps: 41,
            warmup_sweeps: 2000,
            sample_sweeps: 2000,
            seed: 42,
        }
    }
}

/// Run XY FSS temperature sweeps for each lattice size.
///
/// For each size N, builds a cubic lattice of N³ spins, randomises it,
/// and sweeps through t_steps temperatures from t_min to t_max.
/// Each size uses a distinct RNG seed (base_seed + N) for independence.
///
/// Returns `Vec<(N, Vec<XyObservables>)>` ordered from t_min to t_max.
pub fn run_xy_fss(config: &XyFssConfig) -> Vec<(usize, Vec<XyObservables>)> {
    config.sizes.iter().map(|&n| {
        eprintln!("XY FSS: N={n}");
        let seed = config.seed.wrapping_add(n as u64);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

        let ising_lat = Lattice::new(n, config.geometry);
        let mut lat = XyLattice::new(ising_lat.neighbours.clone());
        lat.randomise(&mut rng);

        let temps: Vec<f64> = (0..config.t_steps).map(|i| {
            config.t_min
                + (config.t_max - config.t_min) * i as f64 / (config.t_steps - 1) as f64
        }).collect();

        let results: Vec<XyObservables> = temps.iter().map(|&t| {
            let beta = 1.0 / t;
            measure(
                &mut lat, beta, config.j,
                config.warmup_sweeps, config.sample_sweeps,
                &mut rng,
            )
        }).collect();

        (n, results)
    }).collect()
}
```

**Verification:**
```
cargo build 2>&1 | grep "xy"
```
(no unit tests needed here — logic is covered by observables tests and CLI smoke tests)

**Commit message:** `feat(xy): add XyFssConfig + run_xy_fss()`

---

## Task 5: `src/bin/xy_fss.rs` + `src/lib.rs` update + `Cargo.toml` update

**Files:**
- Create: `/Users/faulknco/Projects/ising-rs/src/bin/xy_fss.rs`
- Modify: `/Users/faulknco/Projects/ising-rs/src/lib.rs` — append `pub mod xy;`
- Modify: `/Users/faulknco/Projects/ising-rs/Cargo.toml` — add `[[bin]]` entry

**lib.rs change** (one line, after the existing `pub mod wolff;` line):
```rust
pub mod xy;
```

**Cargo.toml addition** (after the `heisenberg_jfit` `[[bin]]` entry):
```toml
[[bin]]
name = "xy_fss"
path = "src/bin/xy_fss.rs"
```

**Complete code for `src/bin/xy_fss.rs`:**

```rust
/// CLI: run XY model FSS for multiple lattice sizes using the Wolff cluster algorithm.
///
/// Usage:
///   cargo run --release --bin xy_fss
///   cargo run --release --bin xy_fss -- --sizes 8,12,16,20,24,32 --outdir analysis/data
///
/// Output: one CSV per size at <outdir>/xy_fss_N<n>.csv
/// Columns: T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err
use std::env;
use std::fs;
use std::path::Path;
use ising::xy::fss::{XyFssConfig, run_xy_fss};

fn get_arg(args: &[String], i: usize, flag: &str) -> String {
    if i + 1 >= args.len() {
        eprintln!("Error: {flag} requires a value");
        std::process::exit(1);
    }
    args[i + 1].clone()
}

fn parse_flag<T: std::str::FromStr>(args: &[String], i: usize, flag: &str) -> T
where
    T::Err: std::fmt::Display,
{
    match get_arg(args, i, flag).parse::<T>() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: invalid value for {flag}: {e}");
            std::process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config = XyFssConfig::default();
    let mut outdir = String::from("analysis/data");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sizes" => {
                config.sizes = get_arg(&args, i, "--sizes")
                    .split(',')
                    .map(|s| s.parse::<usize>().unwrap_or_else(|_| {
                        eprintln!("Error: invalid size value: {s}");
                        std::process::exit(1);
                    }))
                    .collect();
                i += 2;
            }
            "--tmin"    => { config.t_min = parse_flag::<f64>(&args, i, "--tmin"); i += 2; }
            "--tmax"    => { config.t_max = parse_flag::<f64>(&args, i, "--tmax"); i += 2; }
            "--steps"   => { config.t_steps = parse_flag::<usize>(&args, i, "--steps"); i += 2; }
            "--warmup"  => { config.warmup_sweeps = parse_flag::<usize>(&args, i, "--warmup"); i += 2; }
            "--samples" => { config.sample_sweeps = parse_flag::<usize>(&args, i, "--samples"); i += 2; }
            "--seed"    => { config.seed = parse_flag::<u64>(&args, i, "--seed"); i += 2; }
            "--j"       => { config.j = parse_flag::<f64>(&args, i, "--j"); i += 2; }
            "--outdir"  => { outdir = get_arg(&args, i, "--outdir"); i += 2; }
            _           => { i += 1; }
        }
    }

    fs::create_dir_all(&outdir).expect("failed to create outdir");

    let results = run_xy_fss(&config);

    for (n, obs_list) in &results {
        let path = Path::new(&outdir).join(format!("xy_fss_N{n}.csv"));
        let mut csv =
            String::from("T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err\n");
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

**Verification:**
```
cargo build --release --bin xy_fss
cargo run --release --bin xy_fss -- --sizes 4 --tmin 1.8 --tmax 2.7 --steps 3 --warmup 20 --samples 20 --outdir /tmp/xy_test
```

**Commit message:** `feat(xy): add xy_fss CLI binary + register module in lib.rs`

---

## Task 6: `src/bin/xy_jfit.rs` + `Cargo.toml` update

**Files:**
- Create: `/Users/faulknco/Projects/ising-rs/src/bin/xy_jfit.rs`
- Modify: `/Users/faulknco/Projects/ising-rs/Cargo.toml` — add `[[bin]]` entry

**Cargo.toml addition** (after xy_fss entry):
```toml
[[bin]]
name = "xy_jfit"
path = "src/bin/xy_jfit.rs"
```

**Complete code for `src/bin/xy_jfit.rs`:**

```rust
/// CLI: run XY temperature sweep on a graph loaded from JSON (J-fitting).
///
/// Usage:
///   cargo run --release --bin xy_jfit -- \
///     --graph analysis/graphs/bcc_N8.json \
///     --tmin 2.3 --tmax 3.5 --steps 41 \
///     --warmup 2000 --samples 2000 \
///     --outdir analysis/data
///
/// Output: <outdir>/xy_jfit_<graphname>.csv
/// Columns: T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err
use std::env;
use std::fs;
use std::path::Path;
use ising::xy::{XyLattice, observables::measure};
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

fn parse_flag<T: std::str::FromStr>(args: &[String], i: usize, flag: &str) -> T
where
    T::Err: std::fmt::Display,
{
    match get_arg(args, i, flag).parse::<T>() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: invalid value for {flag}: {e}");
            std::process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut graph_path = String::new();
    let mut outdir     = String::from("analysis/data");
    let mut t_min      = 2.3_f64;
    let mut t_max      = 3.5_f64;
    let mut t_steps    = 41usize;
    let mut warmup     = 2000usize;
    let mut samples    = 2000usize;
    let mut j          = 1.0_f64;
    let mut seed       = 42u64;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--graph"   => { graph_path = get_arg(&args, i, "--graph"); i += 2; }
            "--outdir"  => { outdir     = get_arg(&args, i, "--outdir"); i += 2; }
            "--tmin"    => { t_min      = parse_flag::<f64>(&args, i, "--tmin"); i += 2; }
            "--tmax"    => { t_max      = parse_flag::<f64>(&args, i, "--tmax"); i += 2; }
            "--steps"   => { t_steps    = parse_flag::<usize>(&args, i, "--steps"); i += 2; }
            "--warmup"  => { warmup     = parse_flag::<usize>(&args, i, "--warmup"); i += 2; }
            "--samples" => { samples    = parse_flag::<usize>(&args, i, "--samples"); i += 2; }
            "--j"       => { j          = parse_flag::<f64>(&args, i, "--j"); i += 2; }
            "--seed"    => { seed       = parse_flag::<u64>(&args, i, "--seed"); i += 2; }
            _           => { i += 1; }
        }
    }

    if graph_path.is_empty() {
        eprintln!("Error: --graph <path.json> is required");
        std::process::exit(1);
    }

    let content = fs::read_to_string(&graph_path).unwrap_or_else(|e| {
        eprintln!("Error: failed to read graph file {graph_path}: {e}");
        std::process::exit(1);
    });
    let graph = GraphDef::from_json(&content).expect("failed to parse graph JSON");

    let graph_name = Path::new(&graph_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_else(|| {
            eprintln!("Error: could not determine graph name: {graph_path}");
            std::process::exit(1);
        })
        .to_string();

    eprintln!(
        "XY jfit: graph={graph_name}, N={}, T={t_min}..{t_max}",
        graph.n_nodes
    );

    // Build adjacency list from edge list
    let mut neighbours: Vec<Vec<usize>> = vec![vec![]; graph.n_nodes];
    for (a, b) in &graph.edges {
        neighbours[*a].push(*b);
        neighbours[*b].push(*a);
    }

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut lat = XyLattice::new(neighbours);
    lat.randomise(&mut rng);

    fs::create_dir_all(&outdir).expect("failed to create outdir");

    let path = Path::new(&outdir).join(format!("xy_jfit_{graph_name}.csv"));
    let mut csv =
        String::from("T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err\n");

    for step in 0..t_steps {
        let t = t_min + (t_max - t_min) * step as f64 / (t_steps - 1) as f64;
        let beta = 1.0 / t;
        let obs = measure(&mut lat, beta, j, warmup, samples, &mut rng);
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

**Verification:**
```
cargo build --release --bin xy_jfit
cargo run --release --bin xy_jfit -- \
  --graph analysis/graphs/bcc_N4.json \
  --tmin 2.3 --tmax 3.5 --steps 3 --warmup 20 --samples 20 \
  --outdir /tmp/xy_jtest
```

**Commit message:** `feat(xy): add xy_jfit CLI binary for crystal-graph J-fitting`

---

## Task 7: CLI integration tests — append to `tests/cli.rs`

**Files:**
- Modify: `/Users/faulknco/Projects/ising-rs/tests/cli.rs` — append two test functions

**Code to append** (after the closing `}` of the `heisenberg_jfit_smoke` test block, with a blank line separator):

```rust
// ---------------------------------------------------------------------------
// xy_fss binary
// ---------------------------------------------------------------------------

#[test]
fn xy_fss_smoke() {
    let outdir = "/tmp/xy_test_fss";
    let status = cargo_bin("xy_fss")
        .args([
            "--sizes", "4",
            "--tmin", "1.8", "--tmax", "2.7", "--steps", "3",
            "--warmup", "20", "--samples", "20",
            "--outdir", outdir,
        ])
        .status()
        .expect("failed to run xy_fss");
    assert!(status.success(), "xy_fss exited with non-zero status");

    let csv = std::fs::read_to_string(format!("{outdir}/xy_fss_N4.csv"))
        .expect("CSV not written");
    let rows: Vec<&str> = csv.lines().collect();
    assert_eq!(rows.len(), 4, "expected header + 3 data rows, got {}", rows.len());
    assert!(rows[0].contains("E_err"), "CSV header missing error columns");
    assert!(!csv.contains("NaN"), "CSV contains NaN values");
}

// ---------------------------------------------------------------------------
// xy_jfit binary
// ---------------------------------------------------------------------------

#[test]
fn xy_jfit_smoke() {
    let outdir = "/tmp/xy_test_jfit";
    let status = cargo_bin("xy_jfit")
        .args([
            "--graph", "analysis/graphs/bcc_N4.json",
            "--tmin", "2.3", "--tmax", "3.5", "--steps", "3",
            "--warmup", "20", "--samples", "20",
            "--outdir", outdir,
        ])
        .status()
        .expect("failed to run xy_jfit");
    assert!(status.success(), "xy_jfit exited with non-zero status");

    let csv = std::fs::read_to_string(format!("{outdir}/xy_jfit_bcc_N4.csv"))
        .expect("CSV not written");
    let rows: Vec<&str> = csv.lines().collect();
    assert_eq!(rows.len(), 4, "expected header + 3 data rows, got {}", rows.len());
    assert!(rows[0].starts_with("T,E,"), "unexpected CSV header: {}", rows[0]);
    assert!(!csv.contains("NaN"), "CSV contains NaN values");
}
```

**Verification:**
```
cargo test --test cli xy_fss_smoke -- --nocapture
cargo test --test cli xy_jfit_smoke -- --nocapture
```

**Commit message:** `test(xy): add xy_fss_smoke and xy_jfit_smoke integration tests`

---

## Task 8: `analysis/xy_fss.ipynb`

**Files:**
- Create: `/Users/faulknco/Projects/ising-rs/analysis/xy_fss.ipynb`

This notebook mirrors `analysis/heisenberg_fss.ipynb` exactly, substituting XY physics constants and file names. Below is the full cell-by-cell content.

**Cell 0 (markdown):**
```markdown
# XY Model FSS — O(2) Universality Validation

Finite-size scaling analysis of the classical XY model on cubic lattices using the Wolff cluster algorithm.
Target: recover O(2) exponents ν=0.6717, γ=1.3177, β=0.3486 and Tc=2.2016 J/kB.

## Sections
1. Load `xy_fss_N*.csv` — plot E, |M|, Cv, χ vs T with error bars
2. Binder cumulant U(T) crossings → Tc with uncertainty
3. Peak scaling: χ_max ~ L^{γ/ν}, M(Tc) ~ L^{-β/ν}
4. Scaling collapse → ν
5. Summary table: measured vs O(2) theory
```

**Cell 1 (code):**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path('data')
SIZES = [8, 12, 16, 20, 24, 32]

dfs = {}
for n in SIZES:
    path = DATA_DIR / f'xy_fss_N{n}.csv'
    if path.exists():
        dfs[n] = pd.read_csv(path)
    else:
        print(f'Missing: {path}')

print(f'Loaded {len(dfs)} size(s):', list(dfs.keys()))
if dfs:
    display(next(iter(dfs.values())).head())
```

**Cell 2 (code):**
```python
## 1. Raw observables — E, |M|, Cv, χ vs T with error bars

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(SIZES)))

for ax, (col, label) in zip(axes.flat, [
    ('E',   'Energy per spin  E/N'),
    ('M',   'Magnetisation |m|'),
    ('Cv',  'Heat capacity Cv'),
    ('chi', 'Susceptibility χ'),
]):
    err_col = col + '_err'
    for (n, df), c in zip(dfs.items(), colors):
        ax.errorbar(df['T'], df[col], yerr=df[err_col],
                    label=f'L={n}', color=c, capsize=2, linewidth=1.2)
    ax.set_xlabel('T  (J/kB)')
    ax.set_ylabel(label)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Classical XY model — cubic lattice (Wolff algorithm)', fontsize=13)
plt.tight_layout()
plt.savefig('figures/xy_observables.png', dpi=150)
plt.show()
print('Saved figures/xy_observables.png')
```

**Cell 3 (code):**
```python
## 2. Binder cumulant crossings → Tc

def binder(df):
    """U = 1 - <m4> / (3 <m2>^2)"""
    return 1.0 - df['M4'] / (3.0 * df['M2']**2)

fig, ax = plt.subplots(figsize=(7, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(dfs)))
for (n, df), c in zip(dfs.items(), colors):
    u = binder(df)
    ax.plot(df['T'], u, label=f'L={n}', color=c, linewidth=1.5)
ax.axvline(2.2016, color='k', linestyle='--', alpha=0.5, label='Tc theory=2.2016')
ax.set_xlabel('T  (J/kB)')
ax.set_ylabel('Binder cumulant U')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_title('Binder cumulant — crossings give Tc  (XY)')
plt.tight_layout()
plt.savefig('figures/xy_binder.png', dpi=150)
plt.show()

from itertools import combinations

def crossing_temp(df1, df2):
    """Linear-interpolation crossing of two Binder curves."""
    u1, u2 = binder(df1).values, binder(df2).values
    t = df1['T'].values
    diff = u1 - u2
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0:
            return t[i] - diff[i] * (t[i+1] - t[i]) / (diff[i+1] - diff[i])
    return np.nan

size_list = sorted(dfs.keys())
crossings = []
for na, nb in combinations(size_list, 2):
    tc = crossing_temp(dfs[na], dfs[nb])
    crossings.append(tc)
    print(f'  L={na} × L={nb}: Tc = {tc:.4f}')

crossings = np.array([c for c in crossings if not np.isnan(c)])
Tc_mean = np.mean(crossings)
Tc_err  = np.std(crossings)
print(f'\nTc = {Tc_mean:.4f} ± {Tc_err:.4f}  (theory: 2.2016)')
```

**Cell 4 (code):**
```python
## 3. Peak scaling: χ_max ~ L^{γ/ν},  M(Tc) ~ L^{-β/ν}

from scipy.optimize import curve_fit

Tc = Tc_mean

Ls, chi_max, m_at_tc = [], [], []
for n, df in sorted(dfs.items()):
    Ls.append(n)
    chi_max.append(df['chi'].max())
    idx = (df['T'] - Tc).abs().idxmin()
    m_at_tc.append(df.loc[idx, 'M'])

Ls      = np.array(Ls, dtype=float)
chi_max = np.array(chi_max)
m_at_tc = np.array(m_at_tc)

def powerlaw(L, a, exp):
    return a * L**exp

popt_chi, pcov_chi = curve_fit(powerlaw, Ls, chi_max, p0=[1.0, 1.96])
popt_m,   pcov_m   = curve_fit(powerlaw, Ls, m_at_tc,  p0=[1.0, -0.52])

gamma_over_nu     = popt_chi[1]
beta_over_nu      = -popt_m[1]
gamma_over_nu_err = np.sqrt(pcov_chi[1, 1])
beta_over_nu_err  = np.sqrt(pcov_m[1, 1])

print(f'γ/ν = {gamma_over_nu:.4f} ± {gamma_over_nu_err:.4f}  (O(2) theory: {1.3177/0.6717:.4f})')
print(f'β/ν = {beta_over_nu:.4f}  ± {beta_over_nu_err:.4f}   (O(2) theory: {0.3486/0.6717:.4f})')

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
L_fine = np.linspace(Ls.min(), Ls.max(), 100)

ax = axes[0]
ax.loglog(Ls, chi_max, 'o', label='data')
ax.loglog(L_fine, powerlaw(L_fine, *popt_chi), '--',
          label=f'fit: γ/ν={gamma_over_nu:.3f}')
ax.set_xlabel('L'); ax.set_ylabel('χ_max'); ax.legend(); ax.grid(alpha=0.3)
ax.set_title('Susceptibility peak scaling (XY)')

ax = axes[1]
ax.loglog(Ls, m_at_tc, 'o', label='data')
ax.loglog(L_fine, powerlaw(L_fine, *popt_m), '--',
          label=f'fit: β/ν={beta_over_nu:.3f}')
ax.set_xlabel('L'); ax.set_ylabel('M(Tc)'); ax.legend(); ax.grid(alpha=0.3)
ax.set_title('Magnetisation at Tc scaling (XY)')

plt.tight_layout()
plt.savefig('figures/xy_peak_scaling.png', dpi=150)
plt.show()
```

**Cell 5 (code):**
```python
## 4. Scaling collapse → ν

from scipy.optimize import minimize_scalar

NU_THEORY   = 0.6717
BETA_THEORY = 0.3486

def collapse_spread(nu, dfs=dfs, Tc=Tc, beta_over_nu=BETA_THEORY/NU_THEORY):
    xs, ys = [], []
    for n, df in dfs.items():
        x = (df['T'].values - Tc) * n**(1.0/nu)
        y = df['M'].values * n**beta_over_nu
        xs.append(x); ys.append(y)
    x_common = np.linspace(-5, 5, 200)
    curves = []
    for x, y in zip(xs, ys):
        idx = np.argsort(x)
        yy = np.interp(x_common, x[idx], y[idx], left=np.nan, right=np.nan)
        curves.append(yy)
    stack = np.array(curves)
    spread = np.nanstd(stack, axis=0)
    return np.nanmean(spread)

result = minimize_scalar(collapse_spread, bounds=(0.4, 1.0), method='bounded')
nu_fit = result.x
print(f'ν_fit = {nu_fit:.4f}  (O(2) theory: {NU_THEORY})')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(dfs)))

for (title, nu), ax in zip([('Theory ν=0.6717', NU_THEORY),
                              (f'Fitted ν={nu_fit:.4f}', nu_fit)], axes):
    for (n, df), c in zip(sorted(dfs.items()), colors):
        x = (df['T'].values - Tc) * n**(1.0/nu)
        y = df['M'].values * n**(BETA_THEORY/nu)
        ax.plot(x, y, label=f'L={n}', color=c, linewidth=1.2)
    ax.set_xlim(-6, 6); ax.set_ylim(0, None)
    ax.set_xlabel('(T − Tc) · L^{1/ν}')
    ax.set_ylabel('M · L^{β/ν}')
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/xy_collapse.png', dpi=150)
plt.show()
```

**Cell 6 (code):**
```python
## 5. Summary table: measured vs O(2) theory

gamma_fit = gamma_over_nu * nu_fit
beta_fit  = beta_over_nu  * nu_fit
gamma_err = gamma_over_nu_err * nu_fit
beta_err  = beta_over_nu_err  * nu_fit

print('=' * 55)
print(f'{"Quantity":<12} {"Measured":>14} {"O(2) theory":>14}')
print('-' * 55)
print(f'{"Tc (J/kB)":<12} {Tc_mean:>10.4f}±{Tc_err:.4f} {"2.2016":>14}')
print(f'{"ν":<12} {nu_fit:>14.4f} {NU_THEORY:>14.4f}')
print(f'{"γ/ν":<12} {gamma_over_nu:>10.4f}±{gamma_over_nu_err:.4f} {1.3177/0.6717:>14.4f}')
print(f'{"β/ν":<12} {beta_over_nu:>10.4f}±{beta_over_nu_err:.4f}  {0.3486/0.6717:>14.4f}')
print(f'{"γ":<12} {gamma_fit:>10.4f}±{gamma_err:.4f} {1.3177:>14.4f}')
print(f'{"β":<12} {beta_fit:>10.4f}±{beta_err:.4f}  {0.3486:>14.4f}')
print('=' * 55)
print(f'\nTc error:  {100*abs(Tc_mean-2.2016)/2.2016:.2f}%')
print(f'ν  error:  {100*abs(nu_fit-NU_THEORY)/NU_THEORY:.2f}%')
print(f'γ  error:  {100*abs(gamma_fit-1.3177)/1.3177:.2f}%')
print(f'β  error:  {100*abs(beta_fit-0.3486)/0.3486:.2f}%')
```

**Commit message:** `feat(xy): add xy_fss.ipynb FSS analysis notebook`

---

## Task 9: `analysis/xy_jfit.ipynb`

**Files:**
- Create: `/Users/faulknco/Projects/ising-rs/analysis/xy_jfit.ipynb`

This mirrors `analysis/heisenberg_jfit.ipynb` with XY file names and temperature ranges.

**Cell 0 (markdown):**
```markdown
# XY Model J-fitting — BCC Fe and FCC Ni

Extract J_fit from XY simulated Tc and compare with Heisenberg, Ising, and Pajda 2001.

**Physics:**
- J_fit = kB · Tc_exp / Tc_sim(J=1)
- XY Tc on BCC(z=8) ≈ 2.835 J/kB, FCC(z=12) ≈ 4.35 J/kB
- BCC Fe: Tc_exp = 1043 K, J_lit = 16.3 meV (Pajda 2001)
- FCC Ni: Tc_exp = 627 K, J_lit = 4.1 meV (Pajda 2001)

## Sections
1. Load BCC XY sweeps → Binder crossings → Tc_XY_BCC → J_fit(Fe)
2. Load FCC XY sweeps → Binder crossings → Tc_XY_FCC → J_fit(Ni)
3. Comparison table: XY vs Heisenberg vs Ising vs Pajda 2001
```

**Cell 1 (code):**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path('data')
BCC_SIZES = [4, 6, 8, 10, 12]
FCC_SIZES = [4, 6, 8, 10, 12]

KB_EV  = 8.617333e-5   # eV/K
KB_MEV = KB_EV * 1000  # meV/K
TC_FE  = 1043.0        # K  (experimental)
TC_NI  = 627.0         # K  (experimental)
J_LIT_FE = 16.3        # meV (Pajda 2001)
J_LIT_NI = 4.1         # meV (Pajda 2001)

bcc_dfs = {}
for n in BCC_SIZES:
    path = DATA_DIR / f'xy_jfit_bcc_N{n}.csv'
    if path.exists():
        bcc_dfs[n] = pd.read_csv(path)
    else:
        print(f'Missing: {path}')

fcc_dfs = {}
for n in FCC_SIZES:
    path = DATA_DIR / f'xy_jfit_fcc_N{n}.csv'
    if path.exists():
        fcc_dfs[n] = pd.read_csv(path)
    else:
        print(f'Missing: {path}')

print(f'Loaded BCC: {list(bcc_dfs.keys())}, FCC: {list(fcc_dfs.keys())}')
```

**Cell 2 (code):**
```python
## 1. BCC iron — Binder crossings → Tc_BCC → J_fit(Fe)

from itertools import combinations

def binder(df):
    return 1.0 - df['M4'] / (3.0 * df['M2']**2)

def crossing_temp(df1, df2):
    u1, u2 = binder(df1).values, binder(df2).values
    t = df1['T'].values
    diff = u1 - u2
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0:
            return t[i] - diff[i] * (t[i+1] - t[i]) / (diff[i+1] - diff[i])
    return np.nan

fig, ax = plt.subplots(figsize=(7, 5))
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(bcc_dfs)))
for (n, df), c in zip(sorted(bcc_dfs.items()), colors):
    ax.plot(df['T'], binder(df), label=f'N={n}', color=c, linewidth=1.5)
ax.set_xlabel('T  (J/kB)'); ax.set_ylabel('Binder cumulant U')
ax.set_title('BCC iron — XY Binder cumulant crossings')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/xy_bcc_binder.png', dpi=150)
plt.show()

bcc_sizes = sorted(bcc_dfs.keys())
bcc_crossings = []
for na, nb in combinations(bcc_sizes, 2):
    tc = crossing_temp(bcc_dfs[na], bcc_dfs[nb])
    bcc_crossings.append(tc)
    print(f'  BCC N={na} × N={nb}: Tc = {tc:.4f}')

bcc_crossings = np.array([c for c in bcc_crossings if not np.isnan(c)])
Tc_BCC      = np.mean(bcc_crossings)
Tc_BCC_err  = np.std(bcc_crossings)

J_fit_Fe     = KB_MEV * TC_FE / Tc_BCC
J_fit_Fe_err = J_fit_Fe * (Tc_BCC_err / Tc_BCC)

print(f'\nTc_XY_BCC(J=1) = {Tc_BCC:.4f} ± {Tc_BCC_err:.4f}')
print(f'J_fit_XY(Fe)   = {J_fit_Fe:.2f} ± {J_fit_Fe_err:.2f} meV  (Pajda 2001: {J_LIT_FE} meV)')
print(f'Error vs literature: {100*abs(J_fit_Fe - J_LIT_FE)/J_LIT_FE:.1f}%')
```

**Cell 3 (code):**
```python
## 2. FCC nickel — Binder crossings → Tc_FCC → J_fit(Ni)

fig, ax = plt.subplots(figsize=(7, 5))
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(fcc_dfs)))
for (n, df), c in zip(sorted(fcc_dfs.items()), colors):
    ax.plot(df['T'], binder(df), label=f'N={n}', color=c, linewidth=1.5)
ax.set_xlabel('T  (J/kB)'); ax.set_ylabel('Binder cumulant U')
ax.set_title('FCC nickel — XY Binder cumulant crossings')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/xy_fcc_binder.png', dpi=150)
plt.show()

fcc_sizes = sorted(fcc_dfs.keys())
fcc_crossings = []
for na, nb in combinations(fcc_sizes, 2):
    tc = crossing_temp(fcc_dfs[na], fcc_dfs[nb])
    fcc_crossings.append(tc)
    print(f'  FCC N={na} × N={nb}: Tc = {tc:.4f}')

fcc_crossings = np.array([c for c in fcc_crossings if not np.isnan(c)])
Tc_FCC      = np.mean(fcc_crossings)
Tc_FCC_err  = np.std(fcc_crossings)

J_fit_Ni     = KB_MEV * TC_NI / Tc_FCC
J_fit_Ni_err = J_fit_Ni * (Tc_FCC_err / Tc_FCC)

print(f'\nTc_XY_FCC(J=1) = {Tc_FCC:.4f} ± {Tc_FCC_err:.4f}')
print(f'J_fit_XY(Ni)   = {J_fit_Ni:.2f} ± {J_fit_Ni_err:.2f} meV  (Pajda 2001: {J_LIT_NI} meV)')
print(f'Error vs literature: {100*abs(J_fit_Ni - J_LIT_NI)/J_LIT_NI:.1f}%')
```

**Cell 4 (code):**
```python
## 3. Comparison table: XY vs Heisenberg vs Ising vs Pajda 2001

# Literature / previously computed values:
# Ising (from fit_j.ipynb): Tc_BCC=6.3548, Tc_FCC=9.8924
# Heisenberg (from heisenberg_jfit.ipynb): fill in from that notebook output
J_ising_Fe     = KB_MEV * TC_FE / 6.3548
J_ising_Fe_err = J_ising_Fe * (0.1250 / 6.3548)
J_ising_Ni     = KB_MEV * TC_NI / 9.8924
J_ising_Ni_err = J_ising_Ni * (0.2010 / 9.8924)

# Heisenberg values — update from heisenberg_jfit.ipynb output
J_heis_Fe     = None   # placeholder
J_heis_Fe_err = None
J_heis_Ni     = None
J_heis_Ni_err = None

def fmt(val, err=None):
    if val is None: return '—'
    if err is None: return f'{val:.2f}'
    return f'{val:.2f} ± {err:.2f}'

def pct_err(val, ref):
    if val is None: return '—'
    return f'{100*abs(val-ref)/ref:.1f}%'

print('=' * 80)
print(f'{"Material":<10} {"Model":<14} {"J_fit (meV)":>18} {"J_lit (meV)":>12} {"Error":>8}')
print('-' * 80)
for material, j_ising, j_ising_err, j_heis, j_heis_err, j_xy, j_xy_err, j_lit in [
    ('Fe (BCC)', J_ising_Fe, J_ising_Fe_err, J_heis_Fe, J_heis_Fe_err, J_fit_Fe, J_fit_Fe_err, J_LIT_FE),
    ('Ni (FCC)', J_ising_Ni, J_ising_Ni_err, J_heis_Ni, J_heis_Ni_err, J_fit_Ni, J_fit_Ni_err, J_LIT_NI),
]:
    print(f'{material:<10} {"Ising":<14} {fmt(j_ising, j_ising_err):>18} {j_lit:>12.1f} {pct_err(j_ising, j_lit):>8}')
    print(f'{material:<10} {"Heisenberg":<14} {fmt(j_heis, j_heis_err):>18} {j_lit:>12.1f} {pct_err(j_heis, j_lit):>8}')
    print(f'{material:<10} {"XY":<14} {fmt(j_xy, j_xy_err):>18} {j_lit:>12.1f} {pct_err(j_xy, j_lit):>8}')
    print()
print('=' * 80)
print('\nLiterature: Pajda et al., Phys. Rev. B 64, 174402 (2001)')
```

**Commit message:** `feat(xy): add xy_jfit.ipynb J-fitting analysis notebook`

---

## Task 10: `analysis/three_way_comparison.ipynb`

**Files:**
- Create: `/Users/faulknco/Projects/ising-rs/analysis/three_way_comparison.ipynb`

**Cell 0 (markdown):**
```markdown
# Three-Way Comparison: Ising vs XY vs Heisenberg on BCC/FCC Crystal Graphs

Side-by-side comparison of all three classical spin models for J-fitting on Fe (BCC) and Ni (FCC).

## Data sources
- **Ising**: `bcc_iron_N{n}_J1.00_sweep.csv`, `fcc_nickel_N{n}_J1.00_sweep.csv` (columns: T,E,M,Cv,chi — no error columns)
- **Heisenberg**: `heisenberg_jfit_bcc_N{n}.csv`, `heisenberg_jfit_fcc_N{n}.csv`
- **XY**: `xy_jfit_bcc_N{n}.csv`, `xy_jfit_fcc_N{n}.csv`

## Sections
1. Binder cumulant crossings — all 3 models on BCC and FCC
2. Tc ratio table (XY/Ising, Heisenberg/Ising, XY/Heisenberg)
3. J_fit table — all 3 models vs Pajda 2001, with % errors
4. Bar chart — 3 models + literature for Fe and Ni
5. Physical interpretation
```

**Cell 1 (code):**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations

DATA_DIR = Path('data')
SIZES    = [4, 6, 8, 10, 12]

KB_EV  = 8.617333e-5
KB_MEV = KB_EV * 1000
TC_FE  = 1043.0
TC_NI  = 627.0
J_LIT_FE = 16.3
J_LIT_NI = 4.1

def load_model(prefix_fn, sizes, has_errors=True):
    dfs = {}
    for n in sizes:
        path = DATA_DIR / prefix_fn(n)
        if path.exists():
            dfs[n] = pd.read_csv(path)
        else:
            print(f'Missing: {path}')
    return dfs

ising_bcc  = load_model(lambda n: f'bcc_iron_N{n}_J1.00_sweep.csv',    SIZES, has_errors=False)
ising_fcc  = load_model(lambda n: f'fcc_nickel_N{n}_J1.00_sweep.csv',  SIZES, has_errors=False)
heis_bcc   = load_model(lambda n: f'heisenberg_jfit_bcc_N{n}.csv',     SIZES)
heis_fcc   = load_model(lambda n: f'heisenberg_jfit_fcc_N{n}.csv',     SIZES)
xy_bcc     = load_model(lambda n: f'xy_jfit_bcc_N{n}.csv',             SIZES)
xy_fcc     = load_model(lambda n: f'xy_jfit_fcc_N{n}.csv',             SIZES)

print('BCC loaded  — Ising:', list(ising_bcc.keys()),
      ' Heisenberg:', list(heis_bcc.keys()),
      ' XY:', list(xy_bcc.keys()))
print('FCC loaded  — Ising:', list(ising_fcc.keys()),
      ' Heisenberg:', list(heis_fcc.keys()),
      ' XY:', list(xy_fcc.keys()))
```

**Cell 2 (code):**
```python
## Section 1: Binder cumulant crossings — all 3 models, BCC and FCC

def binder(df):
    return 1.0 - df['M4'] / (3.0 * df['M2']**2)

def crossing_temp(df1, df2):
    u1, u2 = binder(df1).values, binder(df2).values
    t = df1['T'].values
    diff = u1 - u2
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0:
            return t[i] - diff[i] * (t[i+1] - t[i]) / (diff[i+1] - diff[i])
    return np.nan

def mean_crossing(dfs):
    sizes = sorted(dfs.keys())
    cs = [crossing_temp(dfs[a], dfs[b]) for a, b in combinations(sizes, 2)]
    cs = [c for c in cs if not np.isnan(c)]
    return np.mean(cs), np.std(cs)

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey='row')
models_bcc = [('Ising', ising_bcc, 'steelblue'),
              ('Heisenberg', heis_bcc, 'darkorange'),
              ('XY', xy_bcc, 'forestgreen')]
models_fcc = [('Ising', ising_fcc, 'steelblue'),
              ('Heisenberg', heis_fcc, 'darkorange'),
              ('XY', xy_fcc, 'forestgreen')]

for col, (name, dfs, color) in enumerate(models_bcc):
    ax = axes[0, col]
    colors_n = plt.cm.Blues(np.linspace(0.4, 0.9, len(dfs))) if name == 'Ising' else \
               plt.cm.Oranges(np.linspace(0.4, 0.9, len(dfs))) if name == 'Heisenberg' else \
               plt.cm.Greens(np.linspace(0.4, 0.9, len(dfs)))
    for (n, df), c in zip(sorted(dfs.items()), colors_n):
        ax.plot(df['T'], binder(df), label=f'N={n}', color=c, linewidth=1.4)
    ax.set_title(f'BCC Fe — {name}')
    ax.set_xlabel('T (J/kB)'); ax.set_ylabel('Binder U' if col == 0 else '')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

for col, (name, dfs, color) in enumerate(models_fcc):
    ax = axes[1, col]
    colors_n = plt.cm.Blues(np.linspace(0.4, 0.9, len(dfs))) if name == 'Ising' else \
               plt.cm.Oranges(np.linspace(0.4, 0.9, len(dfs))) if name == 'Heisenberg' else \
               plt.cm.Greens(np.linspace(0.4, 0.9, len(dfs)))
    for (n, df), c in zip(sorted(dfs.items()), colors_n):
        ax.plot(df['T'], binder(df), label=f'N={n}', color=c, linewidth=1.4)
    ax.set_title(f'FCC Ni — {name}')
    ax.set_xlabel('T (J/kB)'); ax.set_ylabel('Binder U' if col == 0 else '')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.suptitle('Binder cumulant crossings — all three models', fontsize=13)
plt.tight_layout()
plt.savefig('figures/three_way_binder.png', dpi=150)
plt.show()

# Compute all Tc values
Tc_ising_bcc,  Tc_ising_bcc_err  = mean_crossing(ising_bcc)
Tc_heis_bcc,   Tc_heis_bcc_err   = mean_crossing(heis_bcc)
Tc_xy_bcc,     Tc_xy_bcc_err     = mean_crossing(xy_bcc)
Tc_ising_fcc,  Tc_ising_fcc_err  = mean_crossing(ising_fcc)
Tc_heis_fcc,   Tc_heis_fcc_err   = mean_crossing(heis_fcc)
Tc_xy_fcc,     Tc_xy_fcc_err     = mean_crossing(xy_fcc)

print(f'BCC Ising  Tc = {Tc_ising_bcc:.4f} ± {Tc_ising_bcc_err:.4f}')
print(f'BCC Heis   Tc = {Tc_heis_bcc:.4f} ± {Tc_heis_bcc_err:.4f}')
print(f'BCC XY     Tc = {Tc_xy_bcc:.4f} ± {Tc_xy_bcc_err:.4f}')
print(f'FCC Ising  Tc = {Tc_ising_fcc:.4f} ± {Tc_ising_fcc_err:.4f}')
print(f'FCC Heis   Tc = {Tc_heis_fcc:.4f} ± {Tc_heis_fcc_err:.4f}')
print(f'FCC XY     Tc = {Tc_xy_fcc:.4f} ± {Tc_xy_fcc_err:.4f}')
```

**Cell 3 (code):**
```python
## Section 2: Tc ratio table

print('=' * 60)
print(f'{"Lattice":<10} {"XY/Ising":>12} {"Heis/Ising":>12} {"XY/Heis":>12}')
print('-' * 60)
for lat, tc_i, tc_h, tc_x in [
    ('BCC (Fe)', Tc_ising_bcc, Tc_heis_bcc, Tc_xy_bcc),
    ('FCC (Ni)', Tc_ising_fcc, Tc_heis_fcc, Tc_xy_fcc),
]:
    print(f'{lat:<10} {tc_x/tc_i:>12.4f} {tc_h/tc_i:>12.4f} {tc_x/tc_h:>12.4f}')
print('=' * 60)
print('\nFor infinite 3D systems: XY/Ising ≈ 0.346, Heis/Ising ≈ 0.227')
```

**Cell 4 (code):**
```python
## Section 3: J_fit table — all 3 models vs Pajda 2001

J_ising_Fe  = KB_MEV * TC_FE / Tc_ising_bcc
J_heis_Fe   = KB_MEV * TC_FE / Tc_heis_bcc
J_xy_Fe     = KB_MEV * TC_FE / Tc_xy_bcc
J_ising_Ni  = KB_MEV * TC_NI / Tc_ising_fcc
J_heis_Ni   = KB_MEV * TC_NI / Tc_heis_fcc
J_xy_Ni     = KB_MEV * TC_NI / Tc_xy_fcc

def pct(v, ref): return f'{100*abs(v-ref)/ref:.1f}%'

print('=' * 72)
print(f'{"Material":<10} {"Model":<14} {"J_fit (meV)":>14} {"J_lit":>8} {"Error":>8}')
print('-' * 72)
for material, rows, j_lit in [
    ('Fe (BCC)', [('Ising', J_ising_Fe), ('Heisenberg', J_heis_Fe), ('XY', J_xy_Fe)], J_LIT_FE),
    ('Ni (FCC)', [('Ising', J_ising_Ni), ('Heisenberg', J_heis_Ni), ('XY', J_xy_Ni)], J_LIT_NI),
]:
    for model, j in rows:
        print(f'{material:<10} {model:<14} {j:>14.2f} {j_lit:>8.1f} {pct(j,j_lit):>8}')
    print()
print('=' * 72)
```

**Cell 5 (code):**
```python
## Section 4: Bar chart — 3 models + literature for Fe and Ni

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (material, j_vals, j_lit) in zip(axes, [
    ('Fe (BCC)',
     [('Literature', J_LIT_FE, 0),
      ('Ising',      J_ising_Fe, 0),
      ('Heisenberg', J_heis_Fe, 0),
      ('XY',         J_xy_Fe, 0)],
     J_LIT_FE),
    ('Ni (FCC)',
     [('Literature', J_LIT_NI, 0),
      ('Ising',      J_ising_Ni, 0),
      ('Heisenberg', J_heis_Ni, 0),
      ('XY',         J_xy_Ni, 0)],
     J_LIT_NI),
]):
    labels = [r[0] for r in j_vals]
    vals   = [r[1] for r in j_vals]
    errs   = [r[2] for r in j_vals]
    colors = ['steelblue', 'orange', 'darkorange', 'forestgreen']
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=errs, capsize=5, color=colors[:len(labels)], alpha=0.85)
    ax.axhline(j_lit, color='k', linestyle='--', alpha=0.5, linewidth=1.0)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('J  (meV)'); ax.set_title(material)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Exchange coupling J_fit: three-model comparison vs Pajda 2001', fontsize=12)
plt.tight_layout()
plt.savefig('figures/three_way_jfit.png', dpi=150)
plt.show()
```

**Cell 6 (markdown):**
```markdown
## Section 5: Physical Interpretation

### Why do the three models give different J_fit values?

The three classical spin models differ in the **number of spin degrees of freedom**:
- **Ising** (Z₂): spins are discrete ±1 — maximal order at low T, highest simulated Tc
- **XY** (O(2)): continuous 2D unit vectors — intermediate Tc
- **Heisenberg** (O(3)): continuous 3D unit vectors — most spin entropy, lowest Tc

Since J_fit ∝ 1/Tc_sim, models with lower simulated Tc predict *larger* J values.
The Heisenberg model, with the most transverse fluctuations, gives the largest J_fit and
typically the best agreement with ab initio values that account for quantum spin-wave
corrections.

### Relation to experiment

Real Fe and Ni are best described by the **Heisenberg model** (quantum spin S=2 and S=1
respectively), but the classical Heisenberg model captures the correct universality class.
The XY model is an intermediate step useful for understanding symmetry breaking.

### Agreement with Pajda 2001

| Model | Fe error | Ni error |
|-------|----------|----------|
| Ising | ~14% | ~33% |
| XY | ~X% | ~X% |
| Heisenberg | ~X% | ~X% |

*(Fill in from simulation output)*
```

**Commit message:** `feat(xy): add three_way_comparison.ipynb notebook`

---

## Task 11: `scripts/run_xy_publication.sh`

**Files:**
- Create: `/Users/faulknco/Projects/ising-rs/scripts/run_xy_publication.sh`

**Complete code:**

```bash
#!/usr/bin/env bash
# Publication-quality XY Monte Carlo runs (Wolff cluster algorithm).
# Estimated wall time: 3–6 hours on a modern CPU (Wolff is faster than Metropolis near Tc).
#
# Usage:
#   bash scripts/run_xy_publication.sh
#   bash scripts/run_xy_publication.sh 2>&1 | tee run_xy.log

set -e
OUTDIR="analysis/data"
mkdir -p "$OUTDIR"

echo "=== XY FSS: cubic lattice validation ==="
# Target: O(2) universality, Tc=2.2016, straddle with tmin=1.8 tmax=2.7
cargo run --release --bin xy_fss -- \
  --sizes 8,12,16,20,24,32 \
  --tmin 1.8 --tmax 2.7 --steps 41 \
  --warmup 2000 --samples 2000 \
  --seed 42 \
  --outdir "$OUTDIR"

echo "=== XY J-fitting: BCC iron ==="
# Tc_XY_BCC(J=1) ~ 2.835 on BCC(z=8) — scan 2.3 to 3.5
for N in 4 6 8 10 12; do
  cargo run --release --bin xy_jfit -- \
    --graph "analysis/graphs/bcc_N${N}.json" \
    --tmin 2.3 --tmax 3.5 --steps 41 \
    --warmup 2000 --samples 2000 \
    --seed 42 \
    --outdir "$OUTDIR"
done

echo "=== XY J-fitting: FCC nickel ==="
# Tc_XY_FCC(J=1) ~ 4.35 on FCC(z=12) — scan 3.6 to 5.2
for N in 4 6 8 10 12; do
  cargo run --release --bin xy_jfit -- \
    --graph "analysis/graphs/fcc_N${N}.json" \
    --tmin 3.6 --tmax 5.2 --steps 41 \
    --warmup 2000 --samples 2000 \
    --seed 42 \
    --outdir "$OUTDIR"
done

echo "=== All done ==="
echo "Run the following notebooks to analyse results:"
echo "  jupyter notebook analysis/xy_fss.ipynb"
echo "  jupyter notebook analysis/xy_jfit.ipynb"
echo "  jupyter notebook analysis/three_way_comparison.ipynb"
```

**Commit message:** `feat(xy): add run_xy_publication.sh production script`

---

## Dependency and Sequencing

Tasks must be executed in this order:

1. Task 1 (`mod.rs`) — foundation types used by all other xy files
2. Task 2 (`wolff.rs`) — algorithm called by observables
3. Task 3 (`observables.rs`) — measure() called by fss.rs and binaries
4. Task 4 (`fss.rs`) — used by xy_fss binary
5. Task 5 (`xy_fss` binary + lib.rs + Cargo.toml) — requires all 4 above; lib.rs and Cargo.toml changes needed before cargo build succeeds
6. Task 6 (`xy_jfit` binary + Cargo.toml) — requires Tasks 1–3
7. Task 7 (integration tests) — requires Tasks 5 and 6 to be built
8. Tasks 8–11 (notebooks, script) — independent of each other, depend only on binary output

**Full test command after all tasks:**
```
cargo test
cargo test --test cli xy_fss_smoke
cargo test --test cli xy_jfit_smoke
```

**Full build verification:**
```
cargo build --release
cargo clippy -- -D warnings
```

---

## Potential Pitfalls

**1. Borrow conflict in wolff.rs BFS loop**

The line `let nb_indices: Vec<usize> = lat.neighbours[idx].clone();` is intentional. Attempting to iterate `&lat.neighbours[idx]` while also writing to `lat.spins` inside the loop body (step 4) would cause a borrow conflict. The clone avoids this cleanly without unsafe code.

**2. `is_multiple_of` requires nightly or specific Rust version**

The heisenberg code uses `samples.is_multiple_of(n_blocks)`. This method was stabilised in Rust 1.86 (April 2025). The project's `rust-version = "1.94"` in Cargo.toml means it is safe to use. If the method is unavailable, use `samples % n_blocks == 0` instead.

**3. Wolff efficiency at extreme temperatures**

At very high T (T >> Tc), clusters are small (single spins). At very low T (T << Tc), clusters cover the entire lattice. Both extremes are safe for the implementation, but the wall time per sweep varies from O(1) to O(N) in cluster size.

**4. `debug_assert!` divisibility check**

The `samples` argument to `measure()` must be divisible by 20. The smoke tests use `--samples 20` which satisfies this. If a user passes `--samples 40`, `40 % 20 == 0` holds. Production runs use `--samples 2000`, also divisible by 20. The `debug_assert!` catches mistakes in debug builds only.

**5. Graph file naming convention**

`xy_jfit` derives `graph_name` from the JSON file stem (`bcc_N8` from `bcc_N8.json`), producing output `xy_jfit_bcc_N8.csv`. The smoke test expects `xy_jfit_bcc_N4.csv` — this matches when `--graph analysis/graphs/bcc_N4.json` is passed.

**6. Notebook figure directory**

All notebooks call `plt.savefig('figures/...')`. The `figures/` subdirectory inside `analysis/` already exists (confirmed by the listing). No `mkdir` needed in notebooks.

---

### Critical Files for Implementation

- `/Users/faulknco/Projects/ising-rs/src/xy/mod.rs` - Core spin type, lattice struct, energy/magnetisation functions; foundation for all other xy files
- `/Users/faulknco/Projects/ising-rs/src/xy/wolff.rs` - Wolff cluster sweep algorithm; the only algorithm difference vs the Heisenberg module
- `/Users/faulknco/Projects/ising-rs/src/xy/observables.rs` - measure() with jackknife; called by both binaries and validated by unit tests
- `/Users/faulknco/Projects/ising-rs/src/lib.rs` - Needs `pub mod xy;` appended; single-line change that unlocks the entire module
- `/Users/faulknco/Projects/ising-rs/Cargo.toml` - Must register both `[[bin]]` entries (`xy_fss`, `xy_jfit`) for `cargo build` to compile the binaries