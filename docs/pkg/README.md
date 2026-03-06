# Ising Model — 3D Metropolis Monte Carlo

Interactive 3D visualisation of the Ising model, built with Rust + WebAssembly + Three.js.

**[Live Demo →](https://faulknco.github.io/ising-rs)**

## What it is

A computational physics simulation of the [Ising model](https://en.wikipedia.org/wiki/Ising_model) — a discrete model of ferromagnetism where each lattice site holds a spin σ ∈ {−1, +1}. The energy of a configuration is:

```
H = −J Σ_{⟨i,j⟩} σᵢσⱼ − h Σᵢ σᵢ
```

The simulation uses the **Metropolis algorithm** (Monte Carlo) to sample spin configurations at thermal equilibrium. As temperature drops below the Curie temperature Tc, the lattice spontaneously orders into a ferromagnetic state — all spins aligned.

## Features

- **3D cubic lattice** rendered as instanced spheres (Three.js)
- **Real-time simulation** running in a Web Worker via WebAssembly
- **Temperature sweep** — plots ⟨E⟩, |⟨M⟩|, Cv, χ vs T with Tc estimation
- **Interactive controls** — T, J, h, lattice size N
- Verified against known results: ground state E = −3J, Tc ≈ 4.51 J/k_B

## Physics Results

| Observable | Result | Theory |
|---|---|---|
| Ground state energy (3D cubic) | −3.000 J | −3J |
| Curie temperature (3D cubic) | 4.40 J/k_B | 4.51 J/k_B |
| Curie temperature (2D square) | 2.30 J/k_B | 2.269 J/k_B (Onsager exact) |
| Curie temperature (2D triangular) | 3.70 J/k_B | 3.641 J/k_B |

## Stack

- **Rust** — physics engine (Metropolis algorithm, lattice, observables)
- **wasm-bindgen + wasm-pack** — compiles Rust to WebAssembly
- **Web Worker** — physics runs off the main thread, no UI blocking
- **Three.js** — instanced 3D sphere rendering
- **Vanilla JS** — no framework

## Build

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install wasm-pack
cargo install wasm-pack

# Build WASM
wasm-pack build --target web --out-dir www/pkg

# Serve
cd www && python3 -m http.server 8080
```

## Background

Originally a college assignment from JS TP (Theoretical Physics) at Trinity College Dublin, 2018 — implemented in Python. This version reimplements the same physics in Rust compiled to WASM for real-time interactive 3D visualisation in the browser.
