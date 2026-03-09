# Design: Classical Heisenberg Model on Crystal Graphs

Date: 2026-03-08

## Overview

Extend `ising-rs` with a classical Heisenberg simulator targeting Paper 2 (Physical Review B).
The primary physics goal is improved J-fitting for BCC iron and FCC nickel by using
continuous vector spins instead of discrete ±1 Ising spins.

**Publication arc:**
- Validate on cubic lattice → O(3) universality class exponents
- Apply to BCC Fe, FCC Ni crystal graphs → J_fit comparison vs Ising vs literature
- Key result: "Heisenberg reduces J_fit error from 12% (Ising) to X% for Fe"

---

## Physics

**Hamiltonian:**
H = −J Σ_{(i,j)∈E} Sᵢ · Sⱼ

where Sᵢ is a unit 3-vector. Same graph infrastructure as Ising.

**Algorithm per sweep:**
1. Metropolis pass — propose random spin rotation on unit sphere, cap angle δ tuned to ~50% acceptance
2. Over-relaxation pass — rotate each spin to its mirror image through the local field hᵢ = J Σⱼ Sⱼ: S'ᵢ = 2(Sᵢ·ĥ)ĥ − Sᵢ (deterministic, free, reduces autocorrelation)

Over-relaxation is the community standard (Peczak, Ferrenberg, Landau PRB 1991).

**Reference values (cubic lattice, J=1):**
- Tc = 1.4432 J/kB (O(3) universality)
- ν = 0.7112, γ = 1.3960, β = 0.3646

---

## Architecture

Fully additive — nothing in existing codebase changes.

```
src/
└── heisenberg/
    ├── mod.rs          — spin type, energy/magnetisation, unit sphere sampling
    ├── metropolis.rs   — single-spin Metropolis with adaptive δ
    ├── overrelax.rs    — over-relaxation sweep
    ├── sweep.rs        — combined driver: 1 Metropolis + N overrelax per measurement
    ├── observables.rs  — HeisenbergObservables with jackknife errors
    └── fss.rs          — FSS runner over sizes
src/bin/
    ├── heisenberg_fss.rs    — CLI: cubic FSS validation
    └── heisenberg_jfit.rs   — CLI: BCC/FCC crystal graph sweeps
analysis/
    ├── heisenberg_fss.ipynb    — O(3) exponents, validation table
    ├── heisenberg_jfit.ipynb   — J_fit comparison: Ising vs Heisenberg vs literature
    └── heisenberg_coarsening.ipynb  — optional, domain wall dynamics
```

---

## Key Data Structures

```rust
// src/heisenberg/mod.rs
pub type Spin3 = [f64; 3];  // unit vector

pub struct HeisenbergLattice {
    pub spins: Vec<Spin3>,
    pub neighbours: Vec<Vec<usize>>,  // reuses same adjacency list structure as Lattice
}

// src/heisenberg/observables.rs
pub struct HeisenbergObservables {
    pub temperature: f64,
    pub energy: f64,         pub energy_err: f64,
    pub magnetisation: f64,  pub magnetisation_err: f64,  // |⟨M⟩|/N
    pub heat_capacity: f64,  pub heat_capacity_err: f64,
    pub susceptibility: f64, pub susceptibility_err: f64,
    pub m2: f64,             pub m2_err: f64,
    pub m4: f64,             pub m4_err: f64,
}
```

---

## Error Estimation

Jackknife with 20 blocks — same scheme as Ising Paper 1:
- All observables: energy, |M|, Cv, χ, M², M⁴
- Tc uncertainty: std dev of pairwise Binder crossing temperatures
- Exponent uncertainties: spread across methods (peak scaling, collapse, Binder slope)
- J_fit uncertainty: propagated from Tc → δJ = J · δTc/Tc

CSV output includes all `_err` columns so notebooks get full jackknife data.

---

## CSV Output Format

```
# heisenberg_fss_N16.csv
T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err

# heisenberg_jfit_bcc_N8.csv
T,E,E_err,M,M_err,M2,M2_err,M4,M4_err,Cv,Cv_err,chi,chi_err
```

---

## GPU

CPU only for this phase. BCC/FCC graphs for J-fitting are small (up to ~7000 spins).
GPU extension (adjacency list in device memory) is a natural future step if large-N
runs are needed.

---

## Non-goals

- No GPU Heisenberg kernels (this phase)
- No Wolff cluster for Heisenberg (over-relaxation sufficient)
- No external field h (h=0 throughout, same as Ising Paper 1 FSS)
- No coarsening binary (notebook only, optional)
