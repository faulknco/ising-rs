# Research Phase 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend ising-rs with publication-quality FSS, coarsening verification, Kibble-Zurek mechanism, ML phase detection, and mesh geometry support.

**Architecture:** Each research capability is a self-contained Rust module + CLI binary + Jupyter notebook. The existing infrastructure (lattice.rs, metropolis.rs, wolff.rs, observables.rs) is untouched. New capabilities are additive. GPU path wiring happens first so all subsequent experiments can use it.

**Tech Stack:** Rust (stable), cudarc 0.12, CUDA 12.x (sm_75 RTX 2060), Python 3.11+, numpy, pandas, matplotlib, scipy, torch (ML task only), Jupyter

---

## Context: Current State

The following already exist and work:
- `src/fss.rs` + `src/bin/fss.rs` -- multi-N FSS sweep, CSV output
- `src/coarsening.rs` + `src/bin/coarsening.rs` -- quench experiment, domain wall density
- `src/cuda/kernels.cu` + `src/cuda/lattice_gpu.rs` -- CUDA scaffold (compiles, not yet wired to CLI)
- `analysis/fss.ipynb` + `analysis/coarsening.ipynb` -- notebooks (not yet run with publication data)
- `src/lattice.rs` -- `neighbours: Vec<Vec<usize>>` already a generic adjacency list

**What is stubbed:** The `--gpu` flag on both CLI binaries prints a warning and falls back to CPU. Tasks 1-3 wire the actual GPU path.

**Physics reference:**
- 3D Ising Tc = 4.5115 J/kB, nu=0.6301, beta=0.3265, alpha=0.1096, gamma=1.2372
- Kibble-Zurek defect scaling: rho ~ (dT/dt)^(2*nu*z/(1+nu*z)) where z=2 (Metropolis dynamic exponent)
- Allen-Cahn: rho(t) ~ t^(-1/3) in 3D
- Binder cumulant: U = 1 - M4/(3*M2^2), crossing at Tc is finite-size-independent
- chi MUST use signed M variance (already fixed in observables.rs in commit 9242c80)

---

## Phase A: Wire GPU Path (Tasks 1-3)

These must be done on the Windows machine with RTX 2060 + CUDA 12.x installed.
CPU-only machines: skip to Phase B and use `--wolff` flag instead of `--gpu`.

---

### Task 1: Wire GPU sweep into src/bin/fss.rs

**Files:**
- Modify: `src/fss.rs`
- Modify: `src/bin/fss.rs`

The `--gpu` flag currently prints a warning. Wire it to `LatticeGpu`.

**Context:** `LatticeGpu` in `src/cuda/lattice_gpu.rs` has:
- `LatticeGpu::new(n, seed)` -- allocates GPU lattice
- `gpu.step(beta, j, h)` -- one full sweep (black + white kernel)
- `gpu.magnetisation()` -- host-side |<M>| per spin
- `gpu.get_spins()` -- host transfer

**Step 1: Add gpu_sweep_fss function to src/fss.rs**

Add after the existing `run_fss` function:

```rust
#[cfg(feature = "cuda")]
pub fn run_fss_gpu(config: &FssConfig) -> Vec<(usize, Vec<crate::observables::Observables>)> {
    use crate::cuda::lattice_gpu::LatticeGpu;

    config.sizes.iter().map(|&n| {
        eprintln!("FSS GPU: N={n}");
        let beta_values: Vec<f64> = (0..config.t_steps).map(|i| {
            let t = config.t_min + (config.t_max - config.t_min) * i as f64 / (config.t_steps - 1) as f64;
            1.0 / t
        }).collect();

        let mut obs_list = Vec::new();

        for &beta in &beta_values {
            let seed = config.seed.wrapping_add(n as u64).wrapping_add((beta * 1000.0) as u64);
            let mut gpu = LatticeGpu::new(n, seed).expect("GPU init failed");
            let t = 1.0 / beta;

            // Warmup
            for _ in 0..config.warmup_sweeps {
                gpu.step(beta as f32, config.j as f32, config.h as f32).unwrap();
            }

            // Accumulate observables (host-side: transfer spins each sample)
            let mut sum_m = 0.0f64;
            let mut sum_m2 = 0.0f64;
            let mut sum_m4 = 0.0f64;

            for _ in 0..config.sample_sweeps {
                gpu.step(beta as f32, config.j as f32, config.h as f32).unwrap();
                let spins = gpu.get_spins().unwrap();
                let size = spins.len() as f64;
                let m = spins.iter().map(|&s| s as f64).sum::<f64>() / size;
                let m_abs = m.abs();
                sum_m += m_abs;
                sum_m2 += m_abs * m_abs;
                sum_m4 += m_abs * m_abs * m_abs * m_abs;
            }

            let s = config.sample_sweeps as f64;
            let avg_m = sum_m / s;
            let avg_m2 = sum_m2 / s;
            let avg_m4 = sum_m4 / s;

            // chi and Cv not available without energy on GPU path -- set to 0.0
            // Use CPU path for chi/Cv; GPU path is for large-N M-based observables
            obs_list.push(crate::observables::Observables {
                temperature: t,
                energy: 0.0,
                magnetisation: avg_m,
                heat_capacity: 0.0,
                susceptibility: 0.0,
                m2: avg_m2,
                m4: avg_m4,
            });
        }

        (n, obs_list)
    }).collect()
}
```

**Step 2: Wire --gpu in src/bin/fss.rs**

Find the section that reads:
```rust
let results = run_fss(&config);
```

Replace with:
```rust
#[cfg(feature = "cuda")]
let results = if use_gpu {
    eprintln!("Using GPU path (RTX 2060, sm_75)");
    ising::fss::run_fss_gpu(&config)
} else {
    run_fss(&config)
};
#[cfg(not(feature = "cuda"))]
let results = {
    if use_gpu {
        eprintln!("Warning: --gpu requires --features cuda. Using CPU.");
    }
    run_fss(&config)
};
```

Remove the old `#[cfg(feature = "cuda")] if use_gpu { eprintln!(...) }` stub block.

**Step 3: Build and test**

```bash
cd /Users/faulknco/Projects/ising-rs
cargo build --release --features cuda --bin fss 2>&1 | tail -20
```

Expected: compiles without errors. If CUDA not installed, skip this task.

**Step 4: Quick GPU smoke test (N=8, 5 temp steps)**

```bash
cargo run --release --features cuda --bin fss -- \
  --sizes 8 --gpu --warmup 100 --samples 50 \
  --tmin 4.0 --tmax 5.0 --steps 5 \
  --outdir /tmp/gpu_test
cat /tmp/gpu_test/fss_N8.csv
```

Expected: 5 rows, M decreasing from ~0.7 to ~0.1 across T=4..5.

**Step 5: Commit**

```bash
git add src/fss.rs src/bin/fss.rs
git commit -m "feat: wire GPU path into fss binary"
```

---

### Task 2: Wire GPU sweep into src/bin/coarsening.rs

**Files:**
- Modify: `src/coarsening.rs`
- Modify: `src/bin/coarsening.rs`

**Step 1: Add run_coarsening_gpu to src/coarsening.rs**

Add after `run_coarsening`:

```rust
#[cfg(feature = "cuda")]
pub fn run_coarsening_gpu(config: &CoarseningConfig) -> Vec<CoarseningPoint> {
    use crate::cuda::lattice_gpu::LatticeGpu;
    use crate::lattice::Lattice;

    let mut gpu = LatticeGpu::new(config.n, config.seed).expect("GPU init failed");

    // Warmup at high T
    let beta_high = 1.0 / config.t_high;
    for _ in 0..config.warmup_sweeps {
        gpu.step(beta_high as f32, config.j as f32, 0.0).unwrap();
    }

    let beta_quench = 1.0 / f64::max(config.t_quench, 0.01);
    let mut results = Vec::new();

    for step in 0..config.total_steps {
        gpu.step(beta_quench as f32, config.j as f32, 0.0).unwrap();
        if step % config.sample_every == 0 {
            // Transfer spins, build CPU lattice for domain wall calculation
            let spins_host = gpu.get_spins().unwrap();
            let mut lattice = Lattice::new(config.n, config.geometry);
            lattice.spins = spins_host;
            results.push(CoarseningPoint {
                step,
                rho: domain_wall_density(&lattice),
            });
        }
    }

    results
}
```

**Step 2: Wire --gpu in src/bin/coarsening.rs**

Find:
```rust
let results = run_coarsening(&config);
```

Replace with:
```rust
#[cfg(feature = "cuda")]
let results = if use_gpu {
    eprintln!("Using GPU path");
    ising::coarsening::run_coarsening_gpu(&config)
} else {
    run_coarsening(&config)
};
#[cfg(not(feature = "cuda"))]
let results = {
    if use_gpu { eprintln!("Warning: --gpu requires --features cuda."); }
    run_coarsening(&config)
};
```

**Step 3: Build and test**

```bash
cargo build --release --features cuda --bin coarsening 2>&1 | tail -20
cargo run --release --features cuda --bin coarsening -- \
  --n 10 --gpu --steps 1000 --sample-every 100 --t-quench 2.5 \
  --outdir /tmp/gpu_test
cat /tmp/gpu_test/coarsening_N10_T2.50.csv
```

Expected: 10 rows, rho decreasing from ~0.35.

**Step 4: Commit**

```bash
git add src/coarsening.rs src/bin/coarsening.rs
git commit -m "feat: wire GPU path into coarsening binary"
```

---

### Task 3: Publication-quality FSS run on GPU

This task generates the data. Run on Windows with RTX 2060.

**Step 1: Run publication FSS**

```bash
cargo run --release --features cuda --bin fss -- \
  --sizes 8,12,16,20,24,28,32,40 \
  --gpu \
  --warmup 5000 --samples 2000 \
  --tmin 3.5 --tmax 5.5 --steps 61 \
  --outdir analysis/data
```

Expected wall time: ~5-10 min GPU vs ~45 min CPU.
Expected output: 8 CSV files, 61 rows each.

**Step 2: Verify N=40 data quality**

```bash
grep "4\." analysis/data/fss_N40.csv | head -5
```

Expected: M transitions sharply from ~0.85 (T=3.5) to ~0.02 (T=5.5), crossing near T=4.51.

**Step 3: CPU fallback (if GPU not available)**

```bash
cargo run --release --bin fss -- \
  --sizes 8,12,16,20,24,28 --wolff \
  --warmup 5000 --samples 2000 \
  --tmin 3.5 --tmax 5.5 --steps 61 \
  --outdir analysis/data
```

No commit needed -- data files are gitignored.

---

## Phase B: Coarsening Verification (Tasks 4-5)

---

### Task 4: Multi-parameter coarsening runs

**Goal:** Confirm z=1/3 is independent of T_quench and N.

**Step 1: Run coarsening at 4 quench temperatures**

```bash
for T in 1.5 2.0 2.5 3.0; do
  cargo run --release --bin coarsening -- \
    --n 30 --t-quench $T \
    --warmup 200 --steps 200000 --sample-every 10 \
    --outdir analysis/data
done
```

Expected: 4 CSV files. T=1.5 is cold so coarsening is slower but should still show z~1/3.

**Step 2: Run coarsening at 4 system sizes**

```bash
for N in 20 30 40 50; do
  cargo run --release --bin coarsening -- \
    --n $N --t-quench 2.5 \
    --warmup 200 --steps 200000 --sample-every 10 \
    --outdir analysis/data
done
```

Expected: larger N shows less finite-size noise at late time.

---

### Task 5: Update coarsening.ipynb for multi-parameter analysis

**Files:**
- Modify: `analysis/coarsening.ipynb`

**Step 1: Replace single-file load with multi-file load**

Replace the data loading cell with:

```python
import pandas as pd
import glob

files = sorted(glob.glob('data/coarsening_*.csv'))
datasets = {}
for f in files:
    key = f.split('/')[-1].replace('.csv','')
    datasets[key] = pd.read_csv(f)
print(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")
```

**Step 2: Add T_quench comparison plot**

```python
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

t_quenches = [1.5, 2.0, 2.5, 3.0]
colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
t_ref = np.logspace(1, 5, 100)

for T, c in zip(t_quenches, colors):
    key = f'coarsening_N30_T{T:.2f}'
    if key in datasets:
        df = datasets[key]
        mask = df['t'] > 0
        ax2.loglog(df.loc[mask,'t'], df.loc[mask,'rho'], alpha=0.6, color=c, label=f'T={T}')

ax2.loglog(t_ref, 0.5 * t_ref**(-1/3), 'k--', label='z=1/3 theory', linewidth=2)
ax2.set_xlabel('t (sweeps)')
ax2.set_ylabel('rho (domain wall density)')
ax2.set_title('Coarsening: T_quench dependence (N=30)')
ax2.legend()

sizes = [20, 30, 40, 50]
for N, c in zip(sizes, colors):
    key = f'coarsening_N{N}_T2.50'
    if key in datasets:
        df = datasets[key]
        mask = df['t'] > 0
        ax1.loglog(df.loc[mask,'t'], df.loc[mask,'rho'], alpha=0.6, color=c, label=f'N={N}')

ax1.loglog(t_ref, 0.5 * t_ref**(-1/3), 'k--', label='z=1/3 theory', linewidth=2)
ax1.set_xlabel('t (sweeps)')
ax1.set_ylabel('rho')
ax1.set_title('Coarsening: system size dependence (T=2.5)')
ax1.legend()

plt.tight_layout()
plt.savefig('data/coarsening_verification.png', dpi=150)
plt.show()
```

**Step 3: Add exponent extraction table**

```python
from scipy import stats

rows = []
skip_frac = 0.1

for key, df in datasets.items():
    mask = (df['t'] > 0) & (df['rho'] > 0)
    df_fit = df[mask].copy()
    skip = int(len(df_fit) * skip_frac)
    df_fit = df_fit.iloc[skip:]
    if len(df_fit) < 10:
        continue
    log_t = np.log(df_fit['t'].values)
    log_r = np.log(df_fit['rho'].values)
    slope, intercept, r, p, se = stats.linregress(log_t, log_r)
    rows.append({'dataset': key, 'z_measured': -slope, 'R2': r**2, 'se': se})

df_results = pd.DataFrame(rows)
df_results['z_theory'] = 1/3
df_results['pct_error'] = (df_results['z_measured'] - df_results['z_theory']).abs() / df_results['z_theory'] * 100
print(df_results[['dataset','z_measured','z_theory','pct_error','R2']].to_string(index=False))
```

**Step 4: Run notebook**

```bash
cd /Users/faulknco/Projects/ising-rs/analysis
jupyter nbconvert --to notebook --execute coarsening.ipynb --output coarsening.ipynb
```

**Step 5: Commit**

```bash
git add analysis/coarsening.ipynb
git commit -m "feat: multi-parameter coarsening analysis"
```

---

## Phase C: Kibble-Zurek Mechanism (Tasks 6-9)

The Kibble-Zurek mechanism (KZM) predicts that when a system is cooled through Tc at finite rate tau_Q (quench time), the density of topological defects scales as:

  rho_KZ ~ tau_Q^(-2*nu*z/(1+nu*z))

For 3D Ising with nu=0.6301 and Metropolis dynamic exponent z=2:
  exponent = 2*0.6301*2 / (1 + 0.6301*2) = 2.5204 / 2.2602 = 1.115

Protocol: start at T=6.0 (disordered), cool linearly to T=1.0 over tau_Q sweeps, measure final domain wall density.

---

### Task 6: Add Kibble-Zurek module src/kibble_zurek.rs

**Files:**
- Create: `src/kibble_zurek.rs`
- Modify: `src/lib.rs`

**Step 1: Write src/kibble_zurek.rs**

```rust
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::coarsening::domain_wall_density;
use crate::lattice::{Geometry, Lattice};
use crate::metropolis::{sweep, warm_up};

pub struct KzConfig {
    pub n: usize,
    pub geometry: Geometry,
    pub j: f64,
    pub t_start: f64,   // start temperature (disordered, > Tc)
    pub t_end: f64,     // end temperature (ordered, < Tc)
    pub tau_q: usize,   // quench time in sweeps
    pub seed: u64,
}

impl Default for KzConfig {
    fn default() -> Self {
        Self {
            n: 20,
            geometry: Geometry::Cubic3D,
            j: 1.0,
            t_start: 6.0,
            t_end: 1.0,
            tau_q: 1000,
            seed: 42,
        }
    }
}

pub struct KzResult {
    pub tau_q: usize,
    pub rho_final: f64,
}

/// Run one KZ quench: cool linearly from t_start to t_end over tau_q sweeps.
/// Returns domain wall density at end of quench.
pub fn run_kz(config: &KzConfig) -> KzResult {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(config.seed);
    let mut lattice = Lattice::new(config.n, config.geometry);
    lattice.randomise(&mut rng);

    // Warmup at t_start to get into disordered phase
    let beta_start = 1.0 / config.t_start;
    warm_up(&mut lattice, beta_start, config.j, 0.0, 200, &mut rng);

    // Linear ramp T from t_start to t_end over tau_q sweeps
    for step in 0..config.tau_q {
        let frac = step as f64 / config.tau_q as f64;
        let t = config.t_start + (config.t_end - config.t_start) * frac;
        let beta = 1.0 / t.max(0.01);
        sweep(&mut lattice, beta, config.j, 0.0, &mut rng);
    }

    KzResult {
        tau_q: config.tau_q,
        rho_final: domain_wall_density(&lattice),
    }
}

/// Run KZ experiment over a range of quench times.
/// Returns Vec<(tau_q, rho_final)> averaged over n_trials trials.
pub fn run_kz_sweep(
    n: usize,
    geometry: Geometry,
    j: f64,
    t_start: f64,
    t_end: f64,
    tau_q_values: &[usize],
    n_trials: usize,
    base_seed: u64,
) -> Vec<(usize, f64)> {
    tau_q_values.iter().map(|&tau_q| {
        let rho_avg: f64 = (0..n_trials).map(|trial| {
            let config = KzConfig {
                n,
                geometry,
                j,
                t_start,
                t_end,
                tau_q,
                seed: base_seed.wrapping_add(tau_q as u64).wrapping_add(trial as u64),
            };
            run_kz(&config).rho_final
        }).sum::<f64>() / n_trials as f64;
        (tau_q, rho_avg)
    }).collect()
}
```

**Step 2: Add module to src/lib.rs**

Open `src/lib.rs` and add:
```rust
pub mod kibble_zurek;
```

**Step 3: Verify it compiles**

```bash
cargo build --release --lib 2>&1 | tail -10
```

Expected: no errors.

**Step 4: Commit**

```bash
git add src/kibble_zurek.rs src/lib.rs
git commit -m "feat: add Kibble-Zurek module"
```

---

### Task 7: Add src/bin/kz.rs CLI binary

**Files:**
- Create: `src/bin/kz.rs`
- Modify: `Cargo.toml`

**Step 1: Write src/bin/kz.rs**

```rust
/// CLI: run Kibble-Zurek quench experiment.
///
/// Usage:
///   cargo run --release --bin kz -- --n 20 --trials 10 \
///     --tau-min 100 --tau-max 100000 --tau-steps 20
///
/// Output columns: tau_q,rho
use std::env;
use std::fs;
use std::path::Path;

use ising::kibble_zurek::run_kz_sweep;
use ising::lattice::Geometry;

fn get_arg(args: &[String], i: usize, flag: &str) -> String {
    if i + 1 >= args.len() {
        eprintln!("Error: {} requires a value", flag);
        std::process::exit(1);
    }
    args[i + 1].clone()
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut n: usize = 20;
    let mut geometry = Geometry::Cubic3D;
    let mut j: f64 = 1.0;
    let mut t_start: f64 = 6.0;
    let mut t_end: f64 = 1.0;
    let mut tau_min: usize = 100;
    let mut tau_max: usize = 100_000;
    let mut tau_steps: usize = 20;
    let mut n_trials: usize = 5;
    let mut seed: u64 = 42;
    let mut outdir = String::from("analysis/data");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n"         => { n = get_arg(&args, i, "--n").parse().unwrap(); i += 2; }
            "--geometry"  => {
                geometry = match get_arg(&args, i, "--geometry").as_str() {
                    "cubic"      => Geometry::Cubic3D,
                    "triangular" => Geometry::Triangular2D,
                    _            => Geometry::Square2D,
                };
                i += 2;
            }
            "--j"         => { j = get_arg(&args, i, "--j").parse().unwrap(); i += 2; }
            "--t-start"   => { t_start = get_arg(&args, i, "--t-start").parse().unwrap(); i += 2; }
            "--t-end"     => { t_end = get_arg(&args, i, "--t-end").parse().unwrap(); i += 2; }
            "--tau-min"   => { tau_min = get_arg(&args, i, "--tau-min").parse().unwrap(); i += 2; }
            "--tau-max"   => { tau_max = get_arg(&args, i, "--tau-max").parse().unwrap(); i += 2; }
            "--tau-steps" => { tau_steps = get_arg(&args, i, "--tau-steps").parse().unwrap(); i += 2; }
            "--trials"    => { n_trials = get_arg(&args, i, "--trials").parse().unwrap(); i += 2; }
            "--seed"      => { seed = get_arg(&args, i, "--seed").parse().unwrap(); i += 2; }
            "--outdir"    => { outdir = get_arg(&args, i, "--outdir"); i += 2; }
            _             => { i += 1; }
        }
    }

    // Build log-spaced tau_q values
    let tau_q_values: Vec<usize> = (0..tau_steps).map(|k| {
        let log_min = (tau_min as f64).ln();
        let log_max = (tau_max as f64).ln();
        let log_t = log_min + (log_max - log_min) * k as f64 / (tau_steps - 1) as f64;
        log_t.exp().round() as usize
    }).collect();

    eprintln!(
        "KZ sweep: N={n}, tau_q={tau_min}..{tau_max} ({tau_steps} steps, log-spaced), {n_trials} trials each"
    );

    let results = run_kz_sweep(n, geometry, j, t_start, t_end, &tau_q_values, n_trials, seed);

    fs::create_dir_all(&outdir).expect("failed to create outdir");
    let fname = format!("kz_N{n}.csv");
    let path = Path::new(&outdir).join(&fname);
    let mut csv = String::from("tau_q,rho\n");
    for (tau_q, rho) in &results {
        csv.push_str(&format!("{tau_q},{rho:.8}\n"));
    }
    fs::write(&path, &csv).expect("failed to write CSV");
    eprintln!("Wrote {}", path.display());
}
```

**Step 2: Register binary in Cargo.toml**

Add after the existing `[[bin]]` entries:

```toml
[[bin]]
name = "kz"
path = "src/bin/kz.rs"
```

**Step 3: Build and smoke test**

```bash
cargo build --release --bin kz 2>&1 | tail -10
cargo run --release --bin kz -- --n 10 --tau-min 100 --tau-max 10000 --tau-steps 5 --trials 2 --outdir /tmp
cat /tmp/kz_N10.csv
```

Expected: 5 rows, tau_q increasing, rho decreasing (slower quench = fewer defects).

**Step 4: Commit**

```bash
git add src/bin/kz.rs Cargo.toml
git commit -m "feat: add KZ CLI binary"
```

---

### Task 8: Run KZ experiments and generate data

**Step 1: KZ sweep, N=20**

```bash
cargo run --release --bin kz -- \
  --n 20 --trials 10 \
  --tau-min 100 --tau-max 500000 --tau-steps 25 \
  --outdir analysis/data
```

Expected wall time: 10-20 min CPU.

**Step 2: KZ sweep, N=30**

```bash
cargo run --release --bin kz -- \
  --n 30 --trials 5 \
  --tau-min 100 --tau-max 500000 --tau-steps 25 \
  --outdir analysis/data
```

Data files are gitignored. No commit needed.

---

### Task 9: Create analysis/kz.ipynb

**Files:**
- Create: `analysis/kz.ipynb`

Write a Jupyter notebook with these sections:

**Section 1: Load and plot raw data**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df20 = pd.read_csv('data/kz_N20.csv')
df30 = pd.read_csv('data/kz_N30.csv')

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(df20['tau_q'], df20['rho'], 'o-', label='N=20', color='#3b82f6')
ax.loglog(df30['tau_q'], df30['rho'], 's-', label='N=30', color='#10b981')
ax.set_xlabel('tau_Q (quench time, sweeps)')
ax.set_ylabel('rho (domain wall density)')
ax.set_title('Kibble-Zurek: defect density vs quench rate')
ax.legend()
plt.tight_layout()
plt.savefig('data/kz_raw.png', dpi=150)
```

**Section 2: Power-law fit and compare to KZM prediction**

```python
# Theoretical KZM exponent for 3D Ising Metropolis dynamics
nu = 0.6301   # correlation length exponent
z = 2.0       # Metropolis dynamic exponent (model A)
kzm_exponent = 2 * nu * z / (1 + nu * z)
print(f"KZM theory exponent: {kzm_exponent:.4f}")
# Expected: 1.1150

for label, df in [('N=20', df20), ('N=30', df30)]:
    skip = int(len(df) * 0.2)
    df_fit = df.iloc[skip:]
    log_tau = np.log(df_fit['tau_q'].values)
    log_rho = np.log(df_fit['rho'].values)
    slope, intercept, r, p, se = stats.linregress(log_tau, log_rho)
    pct_err = abs(-slope - kzm_exponent) / kzm_exponent * 100
    print(f"{label}: exponent = {-slope:.4f} +/- {se:.4f}, theory = {kzm_exponent:.4f}, error = {pct_err:.1f}%")
```

**Section 3: Overlay theory line**

```python
tau_ref = np.logspace(np.log10(df20['tau_q'].min()), np.log10(df20['tau_q'].max()), 100)
rho_mid = df20['rho'].median()
tau_mid = df20['tau_q'].median()
A = rho_mid * tau_mid**kzm_exponent
rho_theory = A * tau_ref**(-kzm_exponent)

ax.loglog(tau_ref, rho_theory, 'k--', label=f'KZM theory (exp={kzm_exponent:.3f})', linewidth=2)
ax.legend()
fig.savefig('data/kz_fit.png', dpi=150)
```

**Step 4: Commit**

```bash
git add analysis/kz.ipynb
git commit -m "feat: add Kibble-Zurek analysis notebook"
```

---

## Phase D: ML Phase Detection (Tasks 10-12)

Train a CNN on spin configurations. Input: N x N spin snapshot (2D slice of 3D lattice). Label: ordered (T < Tc) or disordered (T > Tc). The network learns the phase boundary without being told Tc explicitly.

Reference: Carrasquilla & Melko, Nature Physics 13, 431 (2017).

---

### Task 10: Add spin snapshot export to sweep binary

**Files:**
- Modify: `src/bin/sweep.rs`

Add `--save-snapshots` flag. When set, after equilibration at each temperature, save N x N spin slice as rows in a CSV (one row per snapshot, N*N comma-separated values, last column is temperature). 10 snapshots per temperature.

**Step 1: Add flag to arg parsing**

```rust
let mut save_snapshots = false;
// in match block:
"--save-snapshots" => { save_snapshots = true; i += 1; }
```

**Step 2: Add snapshot generation block after the main sweep**

```rust
if save_snapshots {
    use ising::lattice::Lattice;
    use ising::metropolis::warm_up;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    let snap_fname = format!("snapshots_N{}.csv", config.n);
    let snap_path = Path::new(&outdir).join(&snap_fname);
    let n2 = config.n * config.n;

    let mut snap_csv = String::new();
    let header: Vec<String> = (0..n2).map(|idx| format!("s{idx}")).collect();
    snap_csv.push_str(&header.join(","));
    snap_csv.push_str(",temperature\n");

    let t_values: Vec<f64> = (0..config.t_steps).map(|k| {
        config.t_min + (config.t_max - config.t_min) * k as f64 / (config.t_steps - 1) as f64
    }).collect();

    for &t in &t_values {
        let beta = 1.0 / t;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(config.seed.wrapping_add((t * 1000.0) as u64));
        let mut lattice = Lattice::new(config.n, config.geometry);
        lattice.randomise(&mut rng);
        warm_up(&mut lattice, beta, config.j, 0.0, config.warmup_sweeps, &mut rng);

        // Take 10 snapshots per temperature, spaced 100 sweeps apart
        for _ in 0..10 {
            warm_up(&mut lattice, beta, config.j, 0.0, 100, &mut rng);
            // Take z=0 slice (first N*N spins in cubic lattice)
            let slice: Vec<String> = lattice.spins[..n2].iter().map(|&s| s.to_string()).collect();
            snap_csv.push_str(&slice.join(","));
            snap_csv.push_str(&format!(",{t:.4}\n"));
        }
    }

    fs::write(&snap_path, &snap_csv).expect("failed to write snapshots");
    eprintln!("Wrote snapshots to {}", snap_path.display());
}
```

**Step 3: Generate training data**

```bash
cargo run --release --bin sweep -- \
  --n 20 --geometry cubic \
  --tmin 2.0 --tmax 7.0 --steps 50 \
  --warmup 2000 --samples 500 \
  --save-snapshots \
  --outdir analysis/data
```

Expected: `analysis/data/snapshots_N20.csv` -- 501 lines (header + 500 rows).

**Step 4: Verify**

```bash
head -1 analysis/data/snapshots_N20.csv | tr ',' '\n' | tail -3
wc -l analysis/data/snapshots_N20.csv
```

Expected: last 3 fields are s398,s399,temperature. Line count: 501.

**Step 5: Commit**

```bash
git add src/bin/sweep.rs
git commit -m "feat: add --save-snapshots flag to sweep binary"
```

---

### Task 11: Create analysis/ml_phase.ipynb

**Files:**
- Create: `analysis/ml_phase.ipynb`
- Modify: `analysis/requirements.txt`

**Step 1: Update requirements.txt**

Add:
```
torch>=2.0
torchvision>=0.15
scikit-learn>=1.3
```

**Step 2: Write the notebook cells**

**Cell 1 -- Imports and data loading:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/snapshots_N20.csv')
T_c = 4.5115
N = 20

X = df.iloc[:, :-1].values.astype(np.float32)
X = (X + 1) / 2  # map {-1,+1} to {0,1}
X = X.reshape(-1, 1, N, N)

T_vals = df['temperature'].values
y = (T_vals > T_c).astype(np.int64)  # 0=ordered, 1=disordered

print(f"Samples: {len(X)}, ordered: {(y==0).sum()}, disordered: {(y==1).sum()}")
```

**Cell 2 -- Define CNN:**
```python
class IsingCNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flat = 32 * (n // 4) * (n // 4)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.fc(self.conv(x))

model = IsingCNN(N)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Cell 3 -- Training:**
```python
X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(
    X, y, T_vals, test_size=0.2, random_state=42, stratify=y
)

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=32)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(30):
    model.train()
    for xb, yb in train_dl:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

model.train(False)
correct = sum((model(xb).argmax(1) == yb).sum().item() for xb, yb in test_dl)
print(f"Test accuracy: {correct/len(y_test)*100:.1f}%")
```

**Cell 4 -- P(ordered) vs T plot:**
```python
model.train(False)
X_all = torch.tensor(X)
with torch.no_grad():
    logits = model(X_all)
    probs = torch.softmax(logits, dim=1)[:, 0].numpy()  # P(ordered)

df_res = pd.DataFrame({'T': T_vals, 'P_ordered': probs})
df_mean = df_res.groupby('T').mean().reset_index()

plt.figure(figsize=(8, 5))
plt.plot(df_mean['T'], df_mean['P_ordered'], 'o-', color='#3b82f6')
plt.axvline(T_c, color='red', linestyle='--', label=f'Tc = {T_c}')
plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('Temperature T (J/kB)')
plt.ylabel('P(ordered)')
plt.title('CNN learned phase boundary (3D Ising)')
plt.legend()
plt.savefig('data/ml_phase_boundary.png', dpi=150)
plt.show()

crossing = df_mean[df_mean['P_ordered'] <= 0.5].iloc[0]['T']
print(f"Network Tc estimate: {crossing:.3f}  (theory: {T_c})")
```

**Step 3: Run notebook**

```bash
cd /Users/faulknco/Projects/ising-rs/analysis
jupyter nbconvert --to notebook --execute ml_phase.ipynb --output ml_phase.ipynb
```

Expected: CNN converges to >90% test accuracy, P(ordered) curve crosses 0.5 near T=4.51.

**Step 4: Commit**

```bash
git add analysis/ml_phase.ipynb analysis/requirements.txt
git commit -m "feat: add ML phase detection notebook"
```

---

### Task 12: Add PCA baseline to ml_phase.ipynb

**Files:**
- Modify: `analysis/ml_phase.ipynb`

Add after the CNN training cells:

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_flat = df.iloc[:, :-1].values.astype(np.float32)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=T_vals, cmap='coolwarm', alpha=0.6, s=10)
plt.colorbar(scatter, label='Temperature T')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of spin configurations (coloured by T)')
plt.savefig('data/ml_pca.png', dpi=150)
plt.show()

print(f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")

from scipy.stats import pearsonr
M_vals = np.abs(X_flat.mean(axis=1))
r, p = pearsonr(X_pca[:, 0], M_vals)
print(f"Pearson r(PC1, |M|) = {r:.3f}, p = {p:.2e}")
# Expected: |r| > 0.9, confirming PC1 captures magnetisation
```

**Commit:**

```bash
git add analysis/ml_phase.ipynb
git commit -m "feat: add PCA baseline to ML phase detection"
```

---

## Phase E: Mesh Geometry (Tasks 13-16)

Full design doc at `~/ising-mesh-design.md`. This phase implements ~100 lines of new Rust in two files.

---

### Task 13: Add Geometry::Mesh and Lattice::from_edges

**Files:**
- Modify: `src/lattice.rs`

**Step 1: Add Mesh variant to Geometry enum**

Change:
```rust
pub enum Geometry {
    Square2D,
    Triangular2D,
    Cubic3D,
}
```

To:
```rust
pub enum Geometry {
    Square2D,
    Triangular2D,
    Cubic3D,
    Mesh,
}
```

**Step 2: Update match arm in Lattice::new()**

Replace the `neighbours = match geometry { ... }` block with:

```rust
let neighbours = match geometry {
    Geometry::Square2D => Self::build_neighbours_2d_square(n),
    Geometry::Triangular2D => Self::build_neighbours_2d_triangular(n),
    Geometry::Cubic3D => Self::build_neighbours_3d_cubic(n),
    Geometry::Mesh => panic!("Use Lattice::from_edges() for Mesh geometry"),
};
```

**Step 3: Add from_edges constructor**

Add after the `new()` method:

```rust
/// Load a lattice from an arbitrary undirected edge list.
/// n_nodes: total number of spins.
/// edges: list of (i, j) pairs, 0-indexed.
pub fn from_edges(n_nodes: usize, edges: &[(usize, usize)]) -> Self {
    let spins = vec![1i8; n_nodes];
    let mut neighbours = vec![Vec::new(); n_nodes];
    for &(i, j) in edges {
        assert!(i < n_nodes && j < n_nodes, "edge ({i},{j}) out of range for n_nodes={n_nodes}");
        neighbours[i].push(j);
        neighbours[j].push(i);
    }
    Self {
        n: n_nodes,
        spins,
        neighbours,
        geometry: Geometry::Mesh,
    }
}
```

Note: for Mesh, `n` stores total node count (not side length). `size()` returns `spins.len()` which is correct.

**Step 4: Fix exhaustiveness in other files**

```bash
cargo build --release --lib 2>&1 | grep "non-exhaustive\|error"
```

If wasm.rs or other files match on Geometry, add `Geometry::Mesh => { /* not applicable */ }` arm.

**Step 5: Build**

```bash
cargo build --release --lib 2>&1 | tail -10
```

**Step 6: Commit**

```bash
git add src/lattice.rs
git commit -m "feat: add Geometry::Mesh variant and Lattice::from_edges"
```

---

### Task 14: Create src/graph.rs -- CSV and JSON graph loaders

**Files:**
- Create: `src/graph.rs`
- Modify: `src/lib.rs`

**Step 1: Write src/graph.rs**

```rust
use crate::lattice::Lattice;

/// A graph definition as a node count + edge list.
pub struct GraphDef {
    pub n_nodes: usize,
    pub edges: Vec<(usize, usize)>,
}

impl GraphDef {
    /// Load from edge list CSV.
    ///
    /// Format (lines starting with # are ignored):
    ///   0,1
    ///   0,2
    ///   1,3
    pub fn from_edge_csv(content: &str) -> anyhow::Result<Self> {
        let mut edges = Vec::new();
        let mut max_node = 0usize;

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = trimmed.split(',').collect();
            if parts.len() < 2 {
                anyhow::bail!("bad edge line: {trimmed}");
            }
            let i: usize = parts[0].trim().parse()?;
            let j: usize = parts[1].trim().parse()?;
            max_node = max_node.max(i).max(j);
            edges.push((i, j));
        }

        Ok(Self { n_nodes: max_node + 1, edges })
    }

    /// Load from JSON adjacency list.
    ///
    /// Format: {"n_nodes": 1000, "edges": [[0,1],[0,2],[1,3],...]}
    ///
    /// Uses minimal hand-rolled parsing to avoid adding serde as a dependency.
    pub fn from_json(content: &str) -> anyhow::Result<Self> {
        let n_nodes: usize = {
            let key = "\"n_nodes\"";
            let pos = content.find(key).ok_or_else(|| anyhow::anyhow!("missing n_nodes"))?;
            let after = &content[pos + key.len()..];
            let colon = after.find(':').ok_or_else(|| anyhow::anyhow!("missing : after n_nodes"))?;
            let num_str = after[colon+1..].trim_start();
            let end = num_str.find(|c: char| !c.is_ascii_digit()).unwrap_or(num_str.len());
            num_str[..end].parse()?
        };

        let edges_key = "\"edges\"";
        let pos = content.find(edges_key).ok_or_else(|| anyhow::anyhow!("missing edges"))?;
        let after = &content[pos + edges_key.len()..];
        let bracket = after.find('[').ok_or_else(|| anyhow::anyhow!("missing [ after edges"))?;
        let array_str = &after[bracket..];
        let close = Self::find_matching_bracket(array_str)?;
        let inner = &array_str[1..close];

        let mut edges = Vec::new();
        let mut chars = inner.chars().peekable();
        loop {
            while chars.peek().map_or(false, |&c| c != '[') { chars.next(); }
            if chars.next().is_none() { break; }
            let i_str: String = chars.by_ref().take_while(|c| c.is_ascii_digit()).collect();
            if i_str.is_empty() { break; }
            while chars.peek().map_or(false, |&c| c != ',' && !c.is_ascii_digit()) { chars.next(); }
            if chars.peek() == Some(&',') { chars.next(); }
            let j_str: String = chars.by_ref()
                .skip_while(|c| !c.is_ascii_digit())
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if j_str.is_empty() { break; }
            edges.push((i_str.parse::<usize>()?, j_str.parse::<usize>()?));
        }

        Ok(Self { n_nodes, edges })
    }

    fn find_matching_bracket(s: &str) -> anyhow::Result<usize> {
        let mut depth = 0i32;
        for (i, c) in s.chars().enumerate() {
            match c {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 { return Ok(i); }
                }
                _ => {}
            }
        }
        anyhow::bail!("unmatched bracket")
    }

    /// Convert to Lattice for simulation.
    pub fn into_lattice(self) -> Lattice {
        Lattice::from_edges(self.n_nodes, &self.edges)
    }
}
```

**Step 2: Add to src/lib.rs**

```rust
pub mod graph;
```

**Step 3: Build**

```bash
cargo build --release --lib 2>&1 | tail -10
```

**Step 4: Commit**

```bash
git add src/graph.rs src/lib.rs
git commit -m "feat: add graph.rs with CSV and JSON edge list loaders"
```

---

### Task 15: Create graph generation scripts

**Files:**
- Create: `analysis/graphs/gen_diluted.py`
- Create: `analysis/graphs/gen_bcc.py`

**Step 1: Write analysis/graphs/gen_diluted.py**

```python
#!/usr/bin/env python3
"""
Generate a diluted cubic lattice by randomly removing fraction p of bonds.

Usage:
  python gen_diluted.py --n 20 --p 0.1 --seed 42 --out diluted_N20_p10.json
"""
import argparse
import json
import random

def cubic_edges(n):
    edges = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                idx = i*n*n + j*n + k
                edges.append((idx, ((i+1)%n)*n*n + j*n + k))
                edges.append((idx, i*n*n + ((j+1)%n)*n + k))
                edges.append((idx, i*n*n + j*n + (k+1)%n))
    return edges

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--p', type=float, default=0.1, help='fraction of bonds to remove')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='diluted.json')
    args = p.parse_args()

    random.seed(args.seed)
    edges = cubic_edges(args.n)
    n_remove = int(len(edges) * args.p)
    remove_idx = set(random.sample(range(len(edges)), n_remove))
    kept = [e for i, e in enumerate(edges) if i not in remove_idx]

    data = {'n_nodes': args.n ** 3, 'edges': kept}
    with open(args.out, 'w') as f:
        json.dump(data, f)

    print(f"N={args.n}^3={args.n**3} nodes, {len(edges)} original bonds, "
          f"{len(kept)} kept ({args.p*100:.0f}% removed) -> {args.out}")

if __name__ == '__main__':
    main()
```

**Step 2: Write analysis/graphs/gen_bcc.py**

```python
#!/usr/bin/env python3
"""
Generate BCC crystal structure (8 nearest neighbours per node).
2*n^3 nodes total (corner + body-centre atoms per unit cell).

Usage:
  python gen_bcc.py --n 10 --out bcc_N10.json
"""
import argparse
import json

def bcc_edges(n):
    n3 = n ** 3

    def corner_idx(i, j, k):
        return ((i % n) * n + (j % n)) * n + (k % n)

    def body_idx(i, j, k):
        return n3 + ((i % n) * n + (j % n)) * n + (k % n)

    edges = set()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                b = body_idx(i, j, k)
                for di in [0, 1]:
                    for dj in [0, 1]:
                        for dk in [0, 1]:
                            c = corner_idx(i + di - 1, j + dj - 1, k + dk - 1)
                            edges.add((min(b, c), max(b, c)))
    return list(edges)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--out', default='bcc.json')
    args = p.parse_args()

    edges = bcc_edges(args.n)
    n_nodes = 2 * args.n ** 3
    data = {'n_nodes': n_nodes, 'edges': edges}
    with open(args.out, 'w') as f:
        json.dump(data, f)
    print(f"BCC: n={args.n}, {n_nodes} nodes, {len(edges)} edges -> {args.out}")

if __name__ == '__main__':
    main()
```

**Step 3: Commit scripts**

```bash
git add analysis/graphs/
git commit -m "feat: add graph generation scripts for diluted cubic and BCC"
```

---

### Task 16: Create mesh_sweep binary and run diluted cubic test

**Files:**
- Create: `src/bin/mesh_sweep.rs`
- Modify: `Cargo.toml`

**Step 1: Write src/bin/mesh_sweep.rs**

```rust
/// CLI: temperature sweep on an arbitrary graph loaded from file.
///
/// Usage:
///   cargo run --release --bin mesh_sweep -- --graph graphs/diluted.json --j 1.0
///
/// Output columns: T,E,M,M2,M4,Cv,chi
use std::env;
use std::fs;
use std::path::Path;

use ising::graph::GraphDef;
use ising::lattice::Geometry;
use ising::metropolis::warm_up;
use ising::observables::measure;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn get_arg(args: &[String], i: usize, flag: &str) -> String {
    if i + 1 >= args.len() {
        eprintln!("Error: {} requires a value", flag);
        std::process::exit(1);
    }
    args[i + 1].clone()
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut graph_path = String::new();
    let mut j: f64 = 1.0;
    let mut t_min: f64 = 3.5;
    let mut t_max: f64 = 5.5;
    let mut t_steps: usize = 41;
    let mut warmup: usize = 2000;
    let mut samples: usize = 1000;
    let mut seed: u64 = 42;
    let mut outdir = String::from("analysis/data");
    let mut out_prefix = String::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--graph"   => { graph_path = get_arg(&args, i, "--graph"); i += 2; }
            "--j"       => { j = get_arg(&args, i, "--j").parse().unwrap(); i += 2; }
            "--tmin"    => { t_min = get_arg(&args, i, "--tmin").parse().unwrap(); i += 2; }
            "--tmax"    => { t_max = get_arg(&args, i, "--tmax").parse().unwrap(); i += 2; }
            "--steps"   => { t_steps = get_arg(&args, i, "--steps").parse().unwrap(); i += 2; }
            "--warmup"  => { warmup = get_arg(&args, i, "--warmup").parse().unwrap(); i += 2; }
            "--samples" => { samples = get_arg(&args, i, "--samples").parse().unwrap(); i += 2; }
            "--seed"    => { seed = get_arg(&args, i, "--seed").parse().unwrap(); i += 2; }
            "--outdir"  => { outdir = get_arg(&args, i, "--outdir"); i += 2; }
            "--prefix"  => { out_prefix = get_arg(&args, i, "--prefix"); i += 2; }
            _           => { i += 1; }
        }
    }

    if graph_path.is_empty() {
        eprintln!("Error: --graph <path> is required");
        std::process::exit(1);
    }

    let content = fs::read_to_string(&graph_path)
        .unwrap_or_else(|e| { eprintln!("Cannot read {graph_path}: {e}"); std::process::exit(1); });
    let gdef = if graph_path.ends_with(".json") {
        GraphDef::from_json(&content)
    } else {
        GraphDef::from_edge_csv(&content)
    }.unwrap_or_else(|e| { eprintln!("Parse error: {e}"); std::process::exit(1); });

    let n_nodes = gdef.n_nodes;
    eprintln!("Graph: {n_nodes} nodes, {} edges", gdef.edges.len());

    let mut lattice = gdef.into_lattice();

    fs::create_dir_all(&outdir).expect("failed to create outdir");
    let prefix = if out_prefix.is_empty() {
        Path::new(&graph_path).file_stem().unwrap().to_str().unwrap().to_string()
    } else {
        out_prefix
    };
    let path = Path::new(&outdir).join(format!("{prefix}_sweep.csv"));

    let mut csv = String::from("T,E,M,M2,M4,Cv,chi\n");

    for step in 0..t_steps {
        let t = t_min + (t_max - t_min) * step as f64 / (t_steps - 1) as f64;
        let beta = 1.0 / t;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add((t * 1000.0) as u64));
        lattice.randomise(&mut rng);
        warm_up(&mut lattice, beta, j, 0.0, warmup, &mut rng);
        let obs = measure(&mut lattice, beta, j, 0.0, samples, &mut rng);
        csv.push_str(&format!(
            "{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            t, obs.energy, obs.magnetisation, obs.m2, obs.m4, obs.heat_capacity, obs.susceptibility
        ));
        eprintln!("T={t:.3}: M={:.4}", obs.magnetisation);
    }

    fs::write(&path, &csv).expect("failed to write CSV");
    eprintln!("Wrote {}", path.display());
}
```

**Step 2: Register in Cargo.toml**

```toml
[[bin]]
name = "mesh_sweep"
path = "src/bin/mesh_sweep.rs"
```

**Step 3: Build**

```bash
cargo build --release --bin mesh_sweep 2>&1 | tail -10
```

**Step 4: Generate diluted graphs and run sweep**

```bash
mkdir -p analysis/graphs
cd analysis/graphs

python gen_diluted.py --n 20 --p 0.0 --out diluted_N20_p00.json
python gen_diluted.py --n 20 --p 0.1 --out diluted_N20_p10.json
python gen_diluted.py --n 20 --p 0.3 --out diluted_N20_p30.json
python gen_diluted.py --n 20 --p 0.5 --out diluted_N20_p50.json

cd /Users/faulknco/Projects/ising-rs

for p in 00 10 30 50; do
  cargo run --release --bin mesh_sweep -- \
    --graph analysis/graphs/diluted_N20_p${p}.json \
    --tmin 3.0 --tmax 6.0 --steps 31 \
    --warmup 1000 --samples 500 \
    --prefix diluted_N20_p${p} \
    --outdir analysis/data
done
```

Expected: Tc decreases with dilution. p=0 gives Tc~4.51 (same as undiluted cubic). p=0.5 gives Tc~3.0-4.0.

**Step 5: BCC sweep**

```bash
cd analysis/graphs
python gen_bcc.py --n 10 --out bcc_N10.json
cd /Users/faulknco/Projects/ising-rs

cargo run --release --bin mesh_sweep -- \
  --graph analysis/graphs/bcc_N10.json \
  --tmin 4.0 --tmax 8.0 --steps 41 \
  --warmup 1000 --samples 500 \
  --prefix bcc_N10 \
  --outdir analysis/data
```

Expected: Tc > 4.51 (BCC has 8 neighbours vs 6 for cubic, so stronger ordering).

**Step 6: Commit**

```bash
git add src/bin/mesh_sweep.rs Cargo.toml
git commit -m "feat: mesh sweep binary and diluted cubic + BCC experiments"
```

---

## Summary of Expected Outputs

| Phase | Output file(s) | Key result to check |
|---|---|---|
| GPU wiring | fss_N*.csv with N=40 | M sharp transition at Tc=4.51 |
| Coarsening verification | coarsening_N*_T*.csv | z=1/3 across all T and N |
| Kibble-Zurek | kz_N20.csv, kz_N30.csv | rho ~ tau_Q^(-1.115) |
| ML phase detection | ml_phase_boundary.png | P(ordered) crosses 0.5 near T=4.51 |
| Diluted mesh | diluted_N20_p*_sweep.csv | Tc decreases with dilution fraction p |
| BCC crystal | bcc_N10_sweep.csv | Tc > 4.51 (8 neighbours vs 6) |

---

## Key Commands Reference

```bash
# Build everything
cargo build --release

# KZ sweep (CPU, ~20 min)
cargo run --release --bin kz -- --n 20 --trials 10 --tau-min 100 --tau-max 500000 --tau-steps 25

# Mesh sweep on diluted graph
cargo run --release --bin mesh_sweep -- --graph analysis/graphs/diluted_N20_p10.json --tmin 3.0 --tmax 6.0

# GPU FSS (Windows RTX 2060)
cargo run --release --features cuda --bin fss -- --sizes 8,12,16,20,24,28,32,40 --gpu --warmup 5000 --samples 2000

# Generate training data for ML
cargo run --release --bin sweep -- --n 20 --tmin 2.0 --tmax 7.0 --steps 50 --warmup 2000 --samples 500 --save-snapshots

# Run all notebooks
cd analysis && jupyter notebook
```

## Implementation Order

Tasks 1-3 (GPU wiring): Windows machine only, can be skipped if no GPU available.
Tasks 4-5 (coarsening): CPU, can run now.
Tasks 6-9 (Kibble-Zurek): CPU, depends on no previous tasks.
Tasks 10-12 (ML): CPU + Python, depends on no Rust changes.
Tasks 13-16 (Mesh): CPU, self-contained.

All phases are independent except where noted. Can start Phase B, C, D, E immediately on CPU.
