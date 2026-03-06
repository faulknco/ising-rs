# FSS + Domain Wall Dynamics + CUDA Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add finite-size scaling analysis, domain wall coarsening, and CUDA GPU backend to `ising-rs`.

**Architecture:** Rust CLI binaries generate CSV data; Python Jupyter notebooks consume it for analysis and plotting. GPU backend uses `cudarc` with a checkerboard Metropolis CUDA kernel targeting NVIDIA RTX 2060.

**Tech Stack:** Rust (existing), cudarc 0.12, CUDA 12.x, Python 3.x, numpy, matplotlib, scipy, pandas, jupyter

---

## Task 1: Add M2/M4 moments to Observables

**Files:**
- Modify: `src/observables.rs`

**Step 1: Add fields to the struct**

In `src/observables.rs`, add two fields to `Observables` after `susceptibility`:

```rust
pub struct Observables {
    pub temperature: f64,
    pub energy: f64,
    pub magnetisation: f64,
    pub heat_capacity: f64,
    pub susceptibility: f64,
    pub m2: f64,   // ⟨M²⟩ per spin² — needed for Binder cumulant
    pub m4: f64,   // ⟨M⁴⟩ per spin⁴ — needed for Binder cumulant
}
```

**Step 2: Track sum_m4 in measure()**

In `measure()`, add tracking variables alongside the existing ones:

```rust
let mut sum_m4 = 0.0_f64;
```

Inside the sample loop, after `sum_m2 += m_per * m_per;`, add:

```rust
sum_m4 += m_per * m_per * m_per * m_per;
```

After the loop, compute averages and add to the returned struct:

```rust
let avg_m4 = sum_m4 / s;

Observables {
    temperature: t,
    energy: avg_e,
    magnetisation: avg_m,
    heat_capacity: cv,
    susceptibility: chi,
    m2: avg_m2,
    m4: avg_m4,
}
```

**Step 3: Fix compile errors**

`wasm.rs` calls `parse_csv` which constructs `Observables` manually — add `m2: 0.0, m4: 0.0` to that struct literal (line ~178 in `src/wasm.rs`).

**Step 4: Build to verify**

```bash
cd /Users/faulknco/Projects/ising-rs
cargo build --release 2>&1 | head -30
```

Expected: compiles cleanly.

**Step 5: Commit**

```bash
git add src/observables.rs src/wasm.rs
git commit -m "feat: add M2/M4 moments to Observables for Binder cumulant"
```

---

## Task 2: Create src/fss.rs

**Files:**
- Create: `src/fss.rs`
- Modify: `src/lib.rs`

**Step 1: Write the module**

Create `src/fss.rs`:

```rust
use crate::lattice::Geometry;
use crate::observables::Observables;
use crate::sweep::{run, Algorithm, SweepConfig};

/// Configuration for a finite-size scaling run.
pub struct FssConfig {
    pub sizes: Vec<usize>,
    pub geometry: Geometry,
    pub j: f64,
    pub h: f64,
    pub t_min: f64,
    pub t_max: f64,
    pub t_steps: usize,
    pub warmup_sweeps: usize,
    pub sample_sweeps: usize,
    pub seed: u64,
    pub algorithm: Algorithm,
}

impl Default for FssConfig {
    fn default() -> Self {
        Self {
            sizes: vec![8, 12, 16, 20, 24, 28],
            geometry: Geometry::Cubic3D,
            j: 1.0,
            h: 0.0,
            t_min: 3.5,
            t_max: 5.5,
            t_steps: 41,
            warmup_sweeps: 500,
            sample_sweeps: 200,
            seed: 42,
            algorithm: Algorithm::Wolff,
        }
    }
}

/// Run a temperature sweep for each lattice size in config.sizes.
/// Returns Vec of (n, observables_per_temperature).
pub fn run_fss(config: &FssConfig) -> Vec<(usize, Vec<Observables>)> {
    config.sizes.iter().map(|&n| {
        eprintln!("FSS: N={n}");
        let sweep_cfg = SweepConfig {
            n,
            geometry: config.geometry,
            j: config.j,
            h: config.h,
            t_min: config.t_min,
            t_max: config.t_max,
            t_steps: config.t_steps,
            warmup_sweeps: config.warmup_sweeps,
            sample_sweeps: config.sample_sweeps,
            seed: config.seed,
            algorithm: config.algorithm,
        };
        let obs = run(&sweep_cfg);
        (n, obs)
    }).collect()
}
```

**Step 2: Export from lib.rs**

In `src/lib.rs`, add:

```rust
pub mod fss;
```

**Step 3: Build**

```bash
cargo build --release 2>&1 | head -20
```

Expected: clean compile.

**Step 4: Commit**

```bash
git add src/fss.rs src/lib.rs
git commit -m "feat: add fss module for multi-N sweep runner"
```

---

## Task 3: Create src/bin/fss.rs CLI

**Files:**
- Create: `src/bin/fss.rs`

**Step 1: Create the binary**

```rust
/// CLI: run finite-size scaling sweeps for multiple lattice sizes.
///
/// Usage:
///   cargo run --release --bin fss
///   cargo run --release --bin fss -- --sizes 8,12,16,20 --wolff --outdir analysis/data
///
/// Output: one CSV per size at <outdir>/fss_N<n>.csv
/// Columns: T,E,M,M2,M4,Cv,chi

use std::env;
use std::fs;
use std::path::Path;
use ising::fss::{FssConfig, run_fss};
use ising::lattice::Geometry;
use ising::sweep::Algorithm;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config = FssConfig::default();
    let mut outdir = String::from("analysis/data");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sizes" => {
                config.sizes = args[i+1].split(',')
                    .filter_map(|s| s.parse().ok())
                    .collect();
                i += 2;
            }
            "--geometry" => {
                config.geometry = match args[i+1].as_str() {
                    "cubic"      => Geometry::Cubic3D,
                    "triangular" => Geometry::Triangular2D,
                    _            => Geometry::Square2D,
                };
                i += 2;
            }
            "--j"       => { config.j = args[i+1].parse().unwrap(); i += 2; }
            "--warmup"  => { config.warmup_sweeps = args[i+1].parse().unwrap(); i += 2; }
            "--samples" => { config.sample_sweeps = args[i+1].parse().unwrap(); i += 2; }
            "--tmin"    => { config.t_min = args[i+1].parse().unwrap(); i += 2; }
            "--tmax"    => { config.t_max = args[i+1].parse().unwrap(); i += 2; }
            "--steps"   => { config.t_steps = args[i+1].parse().unwrap(); i += 2; }
            "--seed"    => { config.seed = args[i+1].parse().unwrap(); i += 2; }
            "--wolff"   => { config.algorithm = Algorithm::Wolff; i += 1; }
            "--outdir"  => { outdir = args[i+1].clone(); i += 2; }
            _           => { i += 1; }
        }
    }

    fs::create_dir_all(&outdir).expect("failed to create outdir");

    let results = run_fss(&config);

    for (n, obs_list) in &results {
        let path = Path::new(&outdir).join(format!("fss_N{n}.csv"));
        let mut csv = String::from("T,E,M,M2,M4,Cv,chi\n");
        for o in obs_list {
            csv.push_str(&format!(
                "{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                o.temperature, o.energy, o.magnetisation,
                o.m2, o.m4, o.heat_capacity, o.susceptibility
            ));
        }
        fs::write(&path, &csv).expect("failed to write CSV");
        eprintln!("Wrote {}", path.display());
    }
}
```

**Step 2: Create the output directory**

```bash
mkdir -p /Users/faulknco/Projects/ising-rs/analysis/data
```

**Step 3: Build and smoke-test with small sizes**

```bash
cd /Users/faulknco/Projects/ising-rs
cargo run --release --bin fss -- --sizes 8,10 --steps 10 --warmup 20 --samples 10 --outdir analysis/data
```

Expected: creates `analysis/data/fss_N8.csv` and `fss_N10.csv` with 10 rows each.

```bash
head analysis/data/fss_N8.csv
```

Expected first line: `T,E,M,M2,M4,Cv,chi`

**Step 4: Commit**

```bash
git add src/bin/fss.rs analysis/data/.gitkeep
git commit -m "feat: add fss CLI binary — multi-N sweep to CSV"
```

---

## Task 4: Create src/coarsening.rs

**Files:**
- Create: `src/coarsening.rs`
- Modify: `src/lib.rs`

**Step 1: Write the module**

Create `src/coarsening.rs`:

```rust
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::lattice::{Geometry, Lattice};
use crate::metropolis::{sweep, warm_up};

pub struct CoarseningConfig {
    pub n: usize,
    pub geometry: Geometry,
    pub j: f64,
    pub t_high: f64,       // initial disordered temperature
    pub t_quench: f64,     // target temperature after quench
    pub warmup_sweeps: usize, // sweeps at t_high before quench
    pub total_steps: usize,   // sweeps at t_quench
    pub sample_every: usize,  // record domain wall density every N sweeps
    pub seed: u64,
}

impl Default for CoarseningConfig {
    fn default() -> Self {
        Self {
            n: 30,
            geometry: Geometry::Cubic3D,
            j: 1.0,
            t_high: 10.0,
            t_quench: 0.5,
            warmup_sweeps: 200,
            total_steps: 50_000,
            sample_every: 100,
            seed: 42,
        }
    }
}

pub struct CoarseningPoint {
    pub step: usize,
    pub rho: f64,   // domain wall density
}

/// Fraction of nearest-neighbour pairs with opposite spins.
/// Each pair σᵢ,σⱼ: if σᵢ ≠ σⱼ → contributes 1 to count.
/// Normalised by total number of bonds.
pub fn domain_wall_density(lattice: &Lattice) -> f64 {
    let mut walls = 0usize;
    let mut bonds = 0usize;
    for (idx, &spin) in lattice.spins.iter().enumerate() {
        for &nb in &lattice.neighbours[idx] {
            if nb > idx {   // count each bond once
                bonds += 1;
                if lattice.spins[nb] != spin {
                    walls += 1;
                }
            }
        }
    }
    if bonds == 0 { return 0.0; }
    walls as f64 / bonds as f64
}

/// Run a quench experiment. Returns time series of domain wall density.
pub fn run_coarsening(config: &CoarseningConfig) -> Vec<CoarseningPoint> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(config.seed);
    let mut lattice = Lattice::new(config.n, config.geometry);
    lattice.randomise(&mut rng);

    // Equilibrate at high temperature
    let beta_high = 1.0 / config.t_high;
    warm_up(&mut lattice, beta_high, config.j, 0.0, config.warmup_sweeps, &mut rng);

    // Quench: simulate at low temperature, record domain wall density
    let beta_quench = 1.0 / f64::max(config.t_quench, 0.01);
    let mut results = Vec::new();

    for step in 0..config.total_steps {
        sweep(&mut lattice, beta_quench, config.j, 0.0, &mut rng);
        if step % config.sample_every == 0 {
            results.push(CoarseningPoint {
                step,
                rho: domain_wall_density(&lattice),
            });
        }
    }

    results
}
```

**Step 2: Export from lib.rs**

Add to `src/lib.rs`:

```rust
pub mod coarsening;
```

**Step 3: Build**

```bash
cargo build --release 2>&1 | head -20
```

Expected: clean compile.

**Step 4: Commit**

```bash
git add src/coarsening.rs src/lib.rs
git commit -m "feat: add coarsening module — domain wall density quench experiment"
```

---

## Task 5: Create src/bin/coarsening.rs CLI

**Files:**
- Create: `src/bin/coarsening.rs`

**Step 1: Create the binary**

```rust
/// CLI: run a quench experiment and output domain wall density vs time.
///
/// Usage:
///   cargo run --release --bin coarsening
///   cargo run --release --bin coarsening -- --n 30 --t-quench 0.5 --steps 50000
///
/// Output columns: t,rho

use std::env;
use std::fs;
use std::path::Path;
use ising::coarsening::{CoarseningConfig, run_coarsening};
use ising::lattice::Geometry;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config = CoarseningConfig::default();
    let mut outdir = String::from("analysis/data");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n"            => { config.n = args[i+1].parse().unwrap(); i += 2; }
            "--geometry"     => {
                config.geometry = match args[i+1].as_str() {
                    "cubic"      => Geometry::Cubic3D,
                    "triangular" => Geometry::Triangular2D,
                    _            => Geometry::Square2D,
                };
                i += 2;
            }
            "--j"            => { config.j = args[i+1].parse().unwrap(); i += 2; }
            "--t-high"       => { config.t_high = args[i+1].parse().unwrap(); i += 2; }
            "--t-quench"     => { config.t_quench = args[i+1].parse().unwrap(); i += 2; }
            "--warmup"       => { config.warmup_sweeps = args[i+1].parse().unwrap(); i += 2; }
            "--steps"        => { config.total_steps = args[i+1].parse().unwrap(); i += 2; }
            "--sample-every" => { config.sample_every = args[i+1].parse().unwrap(); i += 2; }
            "--seed"         => { config.seed = args[i+1].parse().unwrap(); i += 2; }
            "--outdir"       => { outdir = args[i+1].clone(); i += 2; }
            _                => { i += 1; }
        }
    }

    fs::create_dir_all(&outdir).expect("failed to create outdir");

    eprintln!(
        "Coarsening: N={}, geometry={:?}, T_quench={}, steps={}",
        config.n, config.geometry, config.t_quench, config.total_steps
    );

    let results = run_coarsening(&config);

    let fname = format!("coarsening_N{}_T{:.2}.csv", config.n, config.t_quench);
    let path = Path::new(&outdir).join(&fname);
    let mut csv = String::from("t,rho\n");
    for pt in &results {
        csv.push_str(&format!("{},{:.8}\n", pt.step, pt.rho));
    }
    fs::write(&path, &csv).expect("failed to write CSV");
    eprintln!("Wrote {}", path.display());
}
```

**Step 2: Smoke test**

```bash
cd /Users/faulknco/Projects/ising-rs
cargo run --release --bin coarsening -- --n 10 --steps 500 --sample-every 10 --outdir analysis/data
```

Expected: creates `analysis/data/coarsening_N10_T0.50.csv` with ~50 rows.

```bash
head analysis/data/coarsening_N10_T0.50.csv
```

Expected: `t,rho` header, then rows like `0,0.47823...`

**Step 3: Commit**

```bash
git add src/bin/coarsening.rs
git commit -m "feat: add coarsening CLI binary — quench experiment to CSV"
```

---

## Task 6: Add .gitignore for analysis/data

**Files:**
- Modify: `.gitignore`

**Step 1: Ignore generated CSV data but keep the directory**

```bash
echo "analysis/data/*.csv" >> /Users/faulknco/Projects/ising-rs/.gitignore
touch /Users/faulknco/Projects/ising-rs/analysis/data/.gitkeep
```

**Step 2: Commit**

```bash
cd /Users/faulknco/Projects/ising-rs
git add .gitignore analysis/data/.gitkeep
git commit -m "chore: gitignore generated CSV data, keep analysis/data dir"
```

---

## Task 7: Create analysis/fss.ipynb

**Files:**
- Create: `analysis/fss.ipynb`

**Step 1: Create the notebook**

Create `analysis/fss.ipynb` with the following cell structure. This is a JSON file — write it exactly:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Finite-Size Scaling Analysis — 3D Ising Model\n\nThis notebook performs finite-size scaling (FSS) analysis on Monte Carlo sweep data\ngenerated by the `fss` Rust binary. It extracts the critical temperature Tc and\ncritical exponents β, γ, α, ν by analysing how observables scale with lattice size N.\n\n**Generate data first:**\n```bash\ncargo run --release --bin fss -- --sizes 8,12,16,20,24,28 --wolff --outdir analysis/data\n```"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.cm as cm\n",
    "import pandas as pd\n",
    "from scipy.optimize import minimize_scalar, curve_fit\n",
    "from pathlib import Path\n",
    "\n",
    "plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})\n",
    "\n",
    "DATA_DIR = Path('data')\n",
    "SIZES = [8, 12, 16, 20, 24, 28]\n",
    "COLORS = cm.viridis(np.linspace(0.15, 0.85, len(SIZES)))\n",
    "\n",
    "# 3D Ising universality class — theory values\n",
    "NU    = 0.6301\n",
    "BETA  = 0.3265\n",
    "ALPHA = 0.1096\n",
    "GAMMA = 1.2372\n",
    "TC_THEORY = 4.5115  # 3D cubic Ising\n",
    "\n",
    "dfs = {}\n",
    "for n in SIZES:\n",
    "    p = DATA_DIR / f'fss_N{n}.csv'\n",
    "    if p.exists():\n",
    "        dfs[n] = pd.read_csv(p)\n",
    "        print(f'N={n}: {len(dfs[n])} points, T=[{dfs[n].T.min():.2f}, {dfs[n].T.max():.2f}]')\n",
    "    else:\n",
    "        print(f'MISSING: {p}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 1. Raw observables for all N"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n",
    "labels = ['E/spin', '|<M>|/spin', 'Cv', 'chi']\n",
    "cols   = ['E',      'M',          'Cv', 'chi']\n",
    "\n",
    "for ax, col, label in zip(axes.flat, cols, labels):\n",
    "    for (n, df), c in zip(dfs.items(), COLORS):\n",
    "        ax.plot(df['T'], df[col], color=c, label=f'N={n}')\n",
    "    ax.set_xlabel('T [J/kB]')\n",
    "    ax.set_ylabel(label)\n",
    "    ax.legend(fontsize=8)\n",
    "    ax.axvline(TC_THEORY, color='red', lw=0.8, ls='--', label='Tc theory')\n",
    "\n",
    "fig.suptitle('3D Ising: observables vs T for multiple N')\n",
    "plt.tight_layout()\n",
    "plt.savefig('fss_observables.png', bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 2. Binder Cumulant — Precise Tc\n\nThe Binder cumulant U = 1 - <M^4> / (3 <M^2>^2) crosses at the same Tc\nfor all N, giving a precise, finite-size-independent estimate of Tc."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(figsize=(8, 5))\n",
    "\n",
    "for (n, df), c in zip(dfs.items(), COLORS):\n",
    "    U = 1.0 - df['M4'] / (3.0 * df['M2']**2)\n",
    "    ax.plot(df['T'], U, color=c, label=f'N={n}')\n",
    "\n",
    "ax.axvline(TC_THEORY, color='red', lw=1, ls='--', label=f'Tc theory = {TC_THEORY}')\n",
    "ax.set_xlabel('T [J/kB]')\n",
    "ax.set_ylabel('Binder cumulant U')\n",
    "ax.set_title('Binder cumulant crossing → Tc')\n",
    "ax.legend(fontsize=9)\n",
    "plt.tight_layout()\n",
    "plt.savefig('fss_binder.png', bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "print('Curves should cross near Tc ~', TC_THEORY)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 3. Peak Scaling — Extract α/ν and γ/ν\n\nAt criticality: Cv_max ~ N^(alpha/nu), chi_max ~ N^(gamma/nu)\nLog-log slope gives the ratio of exponents."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sizes = sorted(dfs.keys())\n",
    "cv_peaks  = [dfs[n]['Cv'].max()  for n in sizes]\n",
    "chi_peaks = [dfs[n]['chi'].max() for n in sizes]\n",
    "logN = np.log(sizes)\n",
    "\n",
    "def fit_slope(logN, logY):\n",
    "    coeffs = np.polyfit(logN, logY, 1)\n",
    "    return coeffs[0], coeffs[1]\n",
    "\n",
    "slope_cv,  _ = fit_slope(logN, np.log(cv_peaks))\n",
    "slope_chi, _ = fit_slope(logN, np.log(chi_peaks))\n",
    "\n",
    "print(f'alpha/nu measured = {slope_cv:.4f}  (theory = {ALPHA/NU:.4f})')\n",
    "print(f'gamma/nu measured = {slope_chi:.4f}  (theory = {GAMMA/NU:.4f})')\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n",
    "for ax, peaks, slope, title, theory in zip(\n",
    "    axes,\n",
    "    [cv_peaks, chi_peaks],\n",
    "    [slope_cv, slope_chi],\n",
    "    ['Cv_max ~ N^(alpha/nu)', 'chi_max ~ N^(gamma/nu)'],\n",
    "    [ALPHA/NU, GAMMA/NU]\n",
    "):\n",
    "    ax.scatter(sizes, peaks, zorder=5)\n",
    "    Nfit = np.linspace(min(sizes)*0.9, max(sizes)*1.1, 100)\n",
    "    ax.plot(Nfit, np.exp(np.polyval([slope, np.log(peaks[0]) - slope*np.log(sizes[0])], np.log(Nfit))),\n",
    "            'r--', label=f'slope={slope:.3f} (theory={theory:.3f})')\n",
    "    ax.set_xscale('log'); ax.set_yscale('log')\n",
    "    ax.set_xlabel('N'); ax.legend()\n",
    "    ax.set_title(title)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('fss_peak_scaling.png', bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 4. Scaling Collapse — Universal Curve\n\nPlot N^(-gamma/nu) * chi vs (T - Tc) * N^(1/nu).\nAll curves should collapse onto one universal function when nu is correct."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def collapse_quality(nu, tc=TC_THEORY):\n",
    "    \"\"\"Measure quality of scaling collapse for chi. Lower = better.\"\"\"\n",
    "    all_x, all_y = [], []\n",
    "    for n, df in dfs.items():\n",
    "        x = (df['T'] - tc) * n**(1/nu)\n",
    "        y = df['chi'] / n**(GAMMA/nu)\n",
    "        all_x.extend(x.values)\n",
    "        all_y.extend(y.values)\n",
    "    # Sort by x, compute variance of y in sliding bins\n",
    "    order = np.argsort(all_x)\n",
    "    all_x = np.array(all_x)[order]\n",
    "    all_y = np.array(all_y)[order]\n",
    "    # Use sum of squared differences between adjacent points as proxy for scatter\n",
    "    diffs = np.diff(all_y)\n",
    "    return np.sum(diffs**2)\n",
    "\n",
    "# Optimise nu\n",
    "result = minimize_scalar(collapse_quality, bounds=(0.4, 0.9), method='bounded')\n",
    "nu_fit = result.x\n",
    "print(f'Optimised nu = {nu_fit:.4f}  (theory = {NU:.4f})')\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(9, 5))\n",
    "for (n, df), c in zip(dfs.items(), COLORS):\n",
    "    x = (df['T'] - TC_THEORY) * n**(1/nu_fit)\n",
    "    y = df['chi'] / n**(GAMMA/nu_fit)\n",
    "    ax.plot(x, y, color=c, label=f'N={n}')\n",
    "\n",
    "ax.set_xlabel(r'$(T - T_c) N^{1/\\nu}$')\n",
    "ax.set_ylabel(r'$N^{-\\gamma/\\nu} \\chi$')\n",
    "ax.set_title(f'Scaling collapse: nu_fit={nu_fit:.4f}, theory={NU:.4f}')\n",
    "ax.legend(fontsize=9)\n",
    "ax.set_xlim(-15, 15)\n",
    "plt.tight_layout()\n",
    "plt.savefig('fss_collapse.png', bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 5. Summary Table"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "results = {\n",
    "    'nu':    (nu_fit,               NU,    abs(nu_fit - NU)/NU * 100),\n",
    "    'alpha/nu': (slope_cv,          ALPHA/NU, abs(slope_cv - ALPHA/NU)/(ALPHA/NU) * 100),\n",
    "    'gamma/nu': (slope_chi,         GAMMA/NU, abs(slope_chi - GAMMA/NU)/(GAMMA/NU) * 100),\n",
    "}\n",
    "\n",
    "print(f'{\"exponent\":<12} {\"measured\":>10} {\"theory\":>10} {\"error %\":>10}')\n",
    "print('-' * 45)\n",
    "for name, (meas, theory, err) in results.items():\n",
    "    print(f'{name:<12} {meas:>10.4f} {theory:>10.4f} {err:>9.1f}%')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

**Step 2: Verify it opens**

```bash
cd /Users/faulknco/Projects/ising-rs/analysis
jupyter nbconvert --to notebook --execute fss.ipynb --ExecutePreprocessor.timeout=10 2>&1 | tail -5
```

Expected: either succeeds (if data exists) or fails cleanly with "MISSING: data/fss_N8.csv" — not a parse error.

**Step 3: Commit**

```bash
cd /Users/faulknco/Projects/ising-rs
git add analysis/fss.ipynb
git commit -m "feat: add FSS Jupyter notebook — Binder cumulant, peak scaling, collapse"
```

---

## Task 8: Create analysis/coarsening.ipynb

**Files:**
- Create: `analysis/coarsening.ipynb`

**Step 1: Create the notebook**

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Domain Wall Coarsening — 3D Ising Model\n\nThis notebook analyses the decay of domain wall density after an instantaneous quench\nfrom high T (disordered) to low T (deep ordered phase).\n\nTheory predicts Allen-Cahn coarsening: rho(t) ~ t^(-z)\n- 2D: z = 1/2\n- 3D: z = 1/3\n\n**Generate data first:**\n```bash\ncargo run --release --bin coarsening -- --n 30 --steps 50000 --sample-every 100 --outdir analysis/data\n```"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "from pathlib import Path\n",
    "from scipy import stats\n",
    "\n",
    "plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})\n",
    "\n",
    "DATA_DIR = Path('data')\n",
    "\n",
    "# Load coarsening data — try multiple files if present\n",
    "files = sorted(DATA_DIR.glob('coarsening_*.csv'))\n",
    "print('Found:', [f.name for f in files])\n",
    "\n",
    "dfs = {f.stem: pd.read_csv(f) for f in files}\n",
    "for name, df in dfs.items():\n",
    "    print(f'{name}: {len(df)} points, t=[{df.t.min()}, {df.t.max()}], rho=[{df.rho.min():.4f}, {df.rho.max():.4f}]')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 1. Domain wall density vs time"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
    "\n",
    "for name, df in dfs.items():\n",
    "    axes[0].plot(df['t'], df['rho'], label=name)\n",
    "    axes[1].loglog(df['t'].replace(0, np.nan), df['rho'], label=name)\n",
    "\n",
    "axes[0].set_xlabel('t (sweeps)')\n",
    "axes[0].set_ylabel('Domain wall density rho')\n",
    "axes[0].set_title('Linear scale')\n",
    "axes[0].legend(fontsize=8)\n",
    "\n",
    "axes[1].set_xlabel('t (sweeps)')\n",
    "axes[1].set_ylabel('rho')\n",
    "axes[1].set_title('Log-log scale')\n",
    "axes[1].legend(fontsize=8)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('coarsening_raw.png', bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 2. Fit coarsening exponent z\n\nFit log(rho) = -z * log(t) + const in the power-law regime.\nSkip first 10% of steps (transient)."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(figsize=(8, 5))\n",
    "\n",
    "for name, df in dfs.items():\n",
    "    # Skip t=0 and initial transient\n",
    "    mask = df['t'] > df['t'].max() * 0.1\n",
    "    sub = df[mask & (df['t'] > 0)]\n",
    "    \n",
    "    log_t   = np.log(sub['t'])\n",
    "    log_rho = np.log(sub['rho'])\n",
    "    \n",
    "    slope, intercept, r, p, se = stats.linregress(log_t, log_rho)\n",
    "    z = -slope\n",
    "    print(f'{name}: z = {z:.4f} (R2={r**2:.4f})')\n",
    "    print(f'  Theory: 3D Allen-Cahn z=1/3={1/3:.4f},  2D z=1/2={1/2:.4f}')\n",
    "    \n",
    "    ax.loglog(sub['t'], sub['rho'], label=name)\n",
    "    t_fit = np.array([sub['t'].min(), sub['t'].max()])\n",
    "    ax.loglog(t_fit, np.exp(intercept + slope*np.log(t_fit)), 'k--',\n",
    "              label=f'fit: z={z:.3f}')\n",
    "\n",
    "# Overlay theory lines\n",
    "t_ref = np.logspace(np.log10(sub['t'].min()), np.log10(sub['t'].max()), 100)\n",
    "ax.loglog(t_ref, t_ref[len(t_ref)//2]**(1/3) * t_ref**(-1/3), 'r:', label='3D theory z=1/3')\n",
    "ax.loglog(t_ref, t_ref[len(t_ref)//2]**(1/2) * t_ref**(-1/2), 'b:', label='2D theory z=1/2')\n",
    "\n",
    "ax.set_xlabel('t (sweeps)')\n",
    "ax.set_ylabel('Domain wall density rho')\n",
    "ax.set_title('Coarsening exponent fit')\n",
    "ax.legend(fontsize=9)\n",
    "plt.tight_layout()\n",
    "plt.savefig('coarsening_fit.png', bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 3. Size dependence — does z change with N?\n\nRun coarsening for multiple N values to check for finite-size effects:\n```bash\nfor N in 20 30 40; do\n  cargo run --release --bin coarsening -- --n $N --steps 50000 --sample-every 100 --outdir analysis/data\ndone\n```"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Summary table\n",
    "print(f'{\"dataset\":<30} {\"z_measured\":>12} {\"3D theory\":>12} {\"error %\":>10}')\n",
    "print('-' * 68)\n",
    "\n",
    "Z_3D = 1/3\n",
    "for name, df in dfs.items():\n",
    "    mask = df['t'] > df['t'].max() * 0.1\n",
    "    sub = df[mask & (df['t'] > 0)]\n",
    "    slope, *_ = stats.linregress(np.log(sub['t']), np.log(sub['rho']))\n",
    "    z = -slope\n",
    "    err = abs(z - Z_3D) / Z_3D * 100\n",
    "    print(f'{name:<30} {z:>12.4f} {Z_3D:>12.4f} {err:>9.1f}%')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

**Step 2: Commit**

```bash
cd /Users/faulknco/Projects/ising-rs
git add analysis/coarsening.ipynb
git commit -m "feat: add domain wall coarsening Jupyter notebook — Allen-Cahn exponent fit"
```

---

## Task 9: CUDA kernel — kernels.cu

**Files:**
- Create: `src/cuda/kernels.cu`

**Step 1: Create the CUDA directory**

```bash
mkdir -p /Users/faulknco/Projects/ising-rs/src/cuda
```

**Step 2: Write the kernel**

Create `src/cuda/kernels.cu`:

```cuda
#include <curand_kernel.h>
#include <math.h>

// Checkerboard Metropolis kernel for 3D cubic Ising model.
// parity=0: update black sites (x+y+z even), parity=1: white sites (x+y+z odd).
// spins: device array of int8_t, size N*N*N.
// rng_states: per-thread curandState.

extern "C" __global__ void metropolis_sweep_kernel(
    signed char* spins,
    curandState*  rng_states,
    int           N,
    float         beta,
    float         J,
    float         h,
    int           parity   // 0 = black, 1 = white
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (N * N * N) / 2;
    if (tid >= total) return;

    // Map thread id to a (x,y,z) with (x+y+z) % 2 == parity
    // Enumerate all sites with correct parity:
    // tid indexes into the half-lattice
    int full_idx = tid * 2;   // approximate; adjust below
    int z = full_idx / (N * N);
    int r = full_idx % (N * N);
    int y = r / N;
    int x = r % N;
    // Adjust x so that (x+y+z) % 2 == parity
    if ((x + y + z) % 2 != parity) x = (x + 1) % N;
    if (x >= N) { return; }  // safety

    int idx = z * N * N + y * N + x;

    // Compute sum of 6 neighbours (periodic boundary)
    int xp = (x + 1) % N, xm = (x - 1 + N) % N;
    int yp = (y + 1) % N, ym = (y - 1 + N) % N;
    int zp = (z + 1) % N, zm = (z - 1 + N) % N;

    float nb_sum = (float)(
        spins[z*N*N + y*N + xp] +
        spins[z*N*N + y*N + xm] +
        spins[z*N*N + yp*N + x] +
        spins[z*N*N + ym*N + x] +
        spins[zp*N*N + y*N + x] +
        spins[zm*N*N + y*N + x]
    );

    float spin_f = (float)spins[idx];
    float delta_e = 2.0f * spin_f * (J * nb_sum + h);

    curandState local_rng = rng_states[tid];
    float u = curand_uniform(&local_rng);
    rng_states[tid] = local_rng;

    if (delta_e < 0.0f || u < expf(-beta * delta_e)) {
        spins[idx] = -spins[idx];
    }
}

// Initialise per-thread curandState.
extern "C" __global__ void init_rng_kernel(curandState* states, unsigned long long seed, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    curand_init(seed, tid, 0, &states[tid]);
}

// Reduction: sum all spins → magnetisation.
extern "C" __global__ void sum_spins_kernel(
    const signed char* spins,
    int* partial_sums,
    int n
) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < n) ? (int)spins[gid] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}
```

**Step 3: Commit**

```bash
cd /Users/faulknco/Projects/ising-rs
git add src/cuda/kernels.cu
git commit -m "feat: add CUDA checkerboard Metropolis + RNG init kernels"
```

---

## Task 10: CUDA Rust wrapper — lattice_gpu.rs + mod.rs

**Files:**
- Create: `src/cuda/mod.rs`
- Create: `src/cuda/lattice_gpu.rs`

**Step 1: Create mod.rs**

```rust
pub mod lattice_gpu;
```

**Step 2: Create lattice_gpu.rs**

```rust
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::driver::sys::CUdeviceptr;
use std::sync::Arc;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
const BLOCK_SIZE: u32 = 256;

pub struct LatticeGpu {
    pub n: usize,
    device: Arc<CudaDevice>,
    spins: CudaSlice<i8>,
    rng_states: CudaSlice<u8>,  // curandState is opaque; size = n_threads * 48 bytes
    n_threads: u32,
}

impl LatticeGpu {
    pub fn new(n: usize, seed: u64) -> anyhow::Result<Self> {
        let device = CudaDevice::new(0)?;
        device.load_ptx(PTX.into(), "ising", &[
            "metropolis_sweep_kernel",
            "init_rng_kernel",
            "sum_spins_kernel",
        ])?;

        let size = n * n * n;
        let n_threads = (size / 2) as u32;

        // Random initial spins on host
        let host_spins: Vec<i8> = (0..size)
            .map(|_| if rand::random::<bool>() { 1i8 } else { -1i8 })
            .collect();

        let spins = device.htod_sync_copy(&host_spins)?;
        // curandState: 48 bytes per thread
        let rng_states = device.alloc_zeros::<u8>((n_threads as usize) * 48)?;

        let gpu = Self { n, device: device.clone(), spins, rng_states, n_threads };

        // Initialise RNG
        gpu.init_rng(seed)?;

        Ok(gpu)
    }

    fn init_rng(&self, seed: u64) -> anyhow::Result<()> {
        let grid = (self.n_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let f = self.device.get_func("ising", "init_rng_kernel").unwrap();
        unsafe {
            f.launch(
                LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (BLOCK_SIZE, 1, 1), shared_mem_bytes: 0 },
                (&mut self.rng_states, seed, self.n_threads as i32),
            )?;
        }
        self.device.synchronize()?;
        Ok(())
    }

    /// Run one full Metropolis sweep (black pass + white pass).
    pub fn step(&mut self, beta: f32, j: f32, h: f32) -> anyhow::Result<()> {
        let grid = (self.n_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let n = self.n as i32;
        let f = self.device.get_func("ising", "metropolis_sweep_kernel").unwrap();

        for parity in [0i32, 1i32] {
            unsafe {
                f.launch(
                    LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (BLOCK_SIZE, 1, 1), shared_mem_bytes: 0 },
                    (&mut self.spins, &mut self.rng_states, n, beta, j, h, parity),
                )?;
            }
        }
        self.device.synchronize()?;
        Ok(())
    }

    /// Copy spins back to host.
    pub fn get_spins(&self) -> anyhow::Result<Vec<i8>> {
        Ok(self.device.dtoh_sync_copy(&self.spins)?)
    }

    /// Magnetisation |<M>| per spin (via host reduction for now).
    pub fn magnetisation(&self) -> anyhow::Result<f64> {
        let spins = self.get_spins()?;
        let sum: i32 = spins.iter().map(|&s| s as i32).sum();
        Ok((sum as f64 / spins.len() as f64).abs())
    }
}
```

**Step 3: Add cudarc to Cargo.toml**

In `Cargo.toml`, add under `[dependencies]`:

```toml
[dependencies]
# ... existing deps ...

[dependencies.cudarc]
version = "0.12"
optional = true

[features]
cuda = ["dep:cudarc"]
```

Add to `src/lib.rs` behind the feature flag:

```rust
#[cfg(feature = "cuda")]
pub mod cuda;
```

**Step 4: Commit**

```bash
cd /Users/faulknco/Projects/ising-rs
git add src/cuda/mod.rs src/cuda/lattice_gpu.rs src/lib.rs Cargo.toml
git commit -m "feat: add CUDA LatticeGpu wrapper (cudarc, checkerboard sweep)"
```

---

## Task 11: build.rs — nvcc PTX compilation

**Files:**
- Create: `build.rs`

**Step 1: Write build.rs**

```rust
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only compile CUDA if the feature is enabled
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let cuda_path = env::var("CUDA_PATH")
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let nvcc = PathBuf::from(&cuda_path).join("bin").join("nvcc");
    let kernel_src = PathBuf::from("src/cuda/kernels.cu");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_out = out_dir.join("kernels.ptx");

    let status = Command::new(&nvcc)
        .args([
            "-ptx",
            "-O3",
            "--generate-code", "arch=compute_75,code=sm_75",  // RTX 2060 = Turing sm_75
            "-I", &format!("{}/include", cuda_path),
            kernel_src.to_str().unwrap(),
            "-o", ptx_out.to_str().unwrap(),
        ])
        .status()
        .expect("nvcc not found — is CUDA_PATH set and CUDA Toolkit installed?");

    assert!(status.success(), "nvcc compilation failed");

    println!("cargo:rerun-if-changed=src/cuda/kernels.cu");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}
```

**Step 2: Verify build.rs compiles on CPU-only (no CUDA)**

```bash
cd /Users/faulknco/Projects/ising-rs
cargo build --release 2>&1 | head -20
```

Expected: clean — build.rs exits early when `CARGO_FEATURE_CUDA` not set.

**Step 3: Commit**

```bash
git add build.rs
git commit -m "feat: add build.rs — nvcc PTX compilation for CUDA feature"
```

---

## Task 12: Wire --gpu flag into CLI binaries

**Files:**
- Modify: `src/bin/fss.rs`
- Modify: `src/bin/coarsening.rs`

**Step 1: Add --gpu flag to fss.rs**

In `src/bin/fss.rs`, add GPU branch behind feature flag. Add at top:

```rust
#[cfg(feature = "cuda")]
use ising::cuda::lattice_gpu::LatticeGpu;
```

In the arg parser, add:

```rust
"--gpu" => { use_gpu = true; i += 1; }
```

Add a note comment after the existing CPU `run_fss` call:

```rust
// GPU path: same sweeps, but uses LatticeGpu for each N
#[cfg(feature = "cuda")]
if use_gpu {
    eprintln!("GPU mode: using CUDA checkerboard Metropolis");
    // GPU FSS loop: construct LatticeGpu per N, run sweeps manually
    // (full implementation in a follow-up once LatticeGpu is validated)
    eprintln!("GPU FSS not yet implemented — falling back to CPU");
}
```

This ensures the binary compiles with `--features cuda` and has the extension point ready.

**Step 2: Build with cuda feature (on a machine with CUDA)**

```bash
# On Windows with CUDA installed:
cargo build --release --features cuda
```

Expected on non-CUDA machine: fails gracefully with nvcc error (not a Rust error).

**Step 3: Commit**

```bash
cd /Users/faulknco/Projects/ising-rs
git add src/bin/fss.rs src/bin/coarsening.rs
git commit -m "feat: add --gpu flag scaffold to fss and coarsening CLIs"
```

---

## Task 13: Add requirements.txt for Python notebooks

**Files:**
- Create: `analysis/requirements.txt`

**Step 1: Write requirements**

```
numpy>=1.24
matplotlib>=3.7
pandas>=2.0
scipy>=1.10
jupyter>=1.0
notebook>=7.0
```

**Step 2: Commit**

```bash
cd /Users/faulknco/Projects/ising-rs
git add analysis/requirements.txt
git commit -m "chore: add Python requirements for analysis notebooks"
```

---

## Task 14: End-to-end validation

**Step 1: Run FSS data generation (fast smoke test)**

```bash
cd /Users/faulknco/Projects/ising-rs
cargo run --release --bin fss -- \
  --sizes 8,12,16 \
  --wolff \
  --warmup 200 --samples 100 \
  --tmin 3.5 --tmax 5.5 --steps 21 \
  --outdir analysis/data
```

Expected: creates 3 CSVs, each with 21 rows.

**Step 2: Run coarsening data generation**

```bash
cargo run --release --bin coarsening -- \
  --n 20 --steps 5000 --sample-every 50 --outdir analysis/data
```

Expected: creates `coarsening_N20_T0.50.csv` with ~100 rows.

**Step 3: Verify notebook loads data**

```bash
cd /Users/faulknco/Projects/ising-rs/analysis
pip install -r requirements.txt -q
jupyter nbconvert --to notebook --execute fss.ipynb --output fss_executed.ipynb 2>&1 | tail -10
jupyter nbconvert --to notebook --execute coarsening.ipynb --output coarsening_executed.ipynb 2>&1 | tail -10
```

Expected: both execute without errors. PNG plots appear in the analysis directory.

**Step 4: Final commit**

```bash
cd /Users/faulknco/Projects/ising-rs
git add analysis/
git commit -m "chore: add executed notebook outputs to verify end-to-end pipeline"
```

---

## Windows 10 CUDA Setup (for GPU tasks 9-12)

Run once on the Windows machine before building with `--features cuda`:

```powershell
# 1. Download and install CUDA Toolkit 12.x
#    https://developer.nvidia.com/cuda-downloads
#    Select: Windows > 10 > x86_64 > exe (local)

# 2. Set CUDA_PATH
[Environment]::SetEnvironmentVariable(
    "CUDA_PATH",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
    "User"
)

# 3. Install Rust (if not present)
#    https://rustup.rs — download rustup-init.exe

# 4. Build
cargo build --release --features cuda --bin fss
cargo build --release --features cuda --bin coarsening

# 5. Run FSS on GPU (after --gpu flag is fully wired)
.\target\release\fss.exe --sizes 8,12,16,20,24,28 --wolff --gpu --outdir analysis\data
```

RTX 2060 compute capability: **sm_75** (Turing). The `nvcc` flag `arch=compute_75,code=sm_75` in `build.rs` targets this exactly.
