# Paper Literature Notes
Date: 2026-03-07
Status: Pre-writing — collect simulation results before drafting

---

## Proposed Paper Title (working)

"Ising model on arbitrary crystal and network topologies: GPU-accelerated Monte Carlo
with exchange coupling fitting and disorder universality"

---

## Publication Target

- Primary: Physical Review E (simulation + statistical mechanics)
- Backup: Computer Physics Communications (methodology focus)
- Stretch: Physical Review Research (if network/disorder angle is strong)

---

## What Makes This Novel

No single paper combines all four of the following:
1. Arbitrary graph topology (BCC, FCC, diluted, small-world) in a single simulator
2. J-fitting workflow connecting classical MC directly to experimental Tc (Fe, Ni)
3. Disorder universality study on real crystal graphs (not just cubic)
4. ML phase detection on non-square/non-cubic topologies

The closest competitor is Berche et al. 2004 (dilution in 3D cubic) — 20 years old,
no graph generality, no ML, no crystal fitting.

---

## Literature by Section

### 0. Direct Predecessor — Burke 2020 (TCD MSc Thesis)

**"A performance study of a template C++ class for parallel Monte Carlo simulations
of local statistical field theories on a three dimensional lattice"**
- Author: Liam Burke, supervised by Mike Peardon, School of Mathematics, TCD
- URL: https://www.tara.tcd.ie/handle/2262/98428
- Year: 2019/20 MSc High Performance Computing

**What it does:**
- Template C++ class for parallel 3D Ising Monte Carlo using MPI
- Same observables: magnetisation, susceptibility, energy
- Also implements φ⁴ quantum field theory on same lattice infrastructure
- Performance study: memory layout, MPI decomposition, parallel scaling

**How your project extends it:**

| Burke 2020                  | This project                                  |
|-----------------------------|-----------------------------------------------|
| C++ template class          | Rust (memory-safe, WASM-compatible)           |
| MPI CPU cluster             | CUDA GPU (RTX 2060, single machine)           |
| 3D cubic lattice only       | Arbitrary graph topology (BCC, FCC, diluted, network) |
| Performance study focus     | Physics study (FSS, KZM, J-fitting, ML detection) |
| φ⁴ field theory extension   | Heisenberg model extension (future work)      |

**How to cite in the paper:**
Cite in the Methods/Related Work section as the direct HPC predecessor.
Narrative: "We build on the MPI-parallel 3D Ising framework of Burke (2020),
replacing MPI+C++ with a GPU+Rust implementation and extending from fixed cubic
lattices to arbitrary graph topologies."

---

### 1. GPU Ising Simulation (Related Work / Methods)

**NVIDIA ising-gpu**
- URL: https://github.com/NVIDIA/ising-gpu
- What: 2D checkerboard GPU Ising in CUDA/C++
- Gap vs ours: 2D only, no graph support, no physics analysis

**GPU Ising for combinatorial optimization**
- URL: https://www.sciencedirect.com/science/article/am/pii/S0167926019301348
- What: Arbitrary graph GPU Ising, but for optimization (Max-Cut), not stat mech
- Gap vs ours: no thermodynamics, no phase transitions, no FSS

**Performance study of 2D Ising on GPUs (arXiv 1906.06297)**
- URL: https://arxiv.org/abs/1906.06297
- What: Benchmark of GPU implementations including Tensor Core approach
- Gap vs ours: benchmarks only, 2D, no physics

**GPU accelerated MC simulation of 2D and 3D Ising**
- URL: https://www.researchgate.net/publication/222660523_GPU_accelerated_Monte_Carlo_simulation_of_the_2D_and_3D_Ising_model
- What: Early GPU Ising (Preis et al.) — standard reference for GPU MC methods
- Use: Cite in methods section for GPU speedup comparison

---

### 2. BCC/FCC Crystal Structures and J-Fitting

**Ising model for BCC, FCC and diamond lattices: a comparison**
- URL: https://www.researchgate.net/publication/248928297_The_Ising_model_for_the_bcc_fcc_and_diamond_lattices_A_comparison
- What: Critical Kc = J/kBTc for BCC, FCC, diamond with high precision MC
- Key values: Kc(FCC) = 0.1020707(2) — use to validate our Tc(J=1) measurement
- Gap vs ours: no J-fitting to experiment, no disorder

**Interatomic exchange coupling of BCC iron**
- URL: https://arxiv.org/pdf/1005.2931
- What: DFT-computed J_ij for BCC Fe, first NN dominant
- Use: The literature J values to compare our fitted J against

**Pajda et al. PRB 64 174402 (2001)**
- What: Spin-wave J for Fe, Co, Ni from first principles
- Values: J_NN(Fe) = 16.3 meV, J_NN(Ni) = 4.1 meV
- Use: Primary comparison target for our J_fit values
- Note: Already hardcoded in fit_j.ipynb as J_LITERATURE

**First-principles MC with phonons — Tc(Fe) = 1060.9 K**
- What: Heavy DFT+phonon+magnon calculation reproducing Tc(Fe) = 1043 K
- Use: Shows that pure NN Ising model needs phonon corrections for exact agreement
  Explains why our simpler model will be ~20% off — cite as context, not as competition

---

### 3. Diluted Magnets / Harris Criterion

**Bond dilution in the 3D Ising model: a Monte Carlo study — Berche et al. 2004**
- URL: https://arxiv.org/abs/cond-mat/0402596
- What: THE key reference for 3D bond dilution. Phase diagram, crossover phenomena,
  effective exponents vs concentration
- Key result: disorder universality class survives for p < p_c (Harris criterion confirmed)
- Gap vs ours: cubic lattice only, no crystal graphs, no GPU, 2004 statistics

**Harris criterion: A.B. Harris, J. Phys. C 7, 1671 (1974)**
- What: Original theory — disorder irrelevant if nu > 2/d
  For 3D Ising: nu = 0.6301 > 2/3 = 0.667 — marginally stable
- Use: Theoretical motivation for the disorder study

**Quantum Monte Carlo study of bond/site diluted transverse-field Ising — 2025**
- URL: https://arxiv.org/html/2505.07627
- What: Very recent (2025) quantum version of the same problem
- Use: Shows the problem is still active; cite as contemporary context

---

### 4. ML Phase Detection

**Carrasquilla & Melko 2017 — Machine learning phases of matter**
- URL: https://www.nature.com/articles/nphys4035
- arXiv: https://arxiv.org/abs/1605.01735
- What: THE original paper — CNN on spin snapshots learns phase transition unsupervised
- Use: Primary citation for our ML section

**Machine-Learning Studies on Spin Models — Scientific Reports 2020**
- URL: https://www.nature.com/articles/s41598-020-58263-5
- What: Review of ML methods on spin models, correlation function approach
- Use: Background/related work

**ML of nonequilibrium phase transitions in Ising — MDPI 2023**
- URL: https://www.mdpi.com/2410-3896/8/3/83
- What: ML applied to nonequilibrium Ising (close to our KZ work)
- Gap vs ours: square lattice only, no arbitrary graph topology

**Novel angle (ours):** Applying CNN phase detection to BCC, FCC, diluted, and
small-world graph topologies. This has not been done — all existing work uses
square/cubic lattices. The question "does the CNN generalize across graph topology?"
is genuinely open.

---

### 5. Kibble-Zurek Mechanism

**KZM Wikipedia overview**
- URL: https://en.wikipedia.org/wiki/Kibble%E2%80%93Zurek_mechanism
- Use: Background only

**KZM in 3D structural Ising domains — Scientific Reports 2023**
- URL: https://www.nature.com/articles/s41598-023-30840-4
- What: Experimental KZM in NiTiO3 and BiTeI (real materials, not simulation)
  NiTiO3 agrees with 3D Ising KZM prediction; BiTeI deviates
- Use: Shows our simulation is relevant to real experimental systems

**Universal power-law KZM scaling in fast quenches — Mapping Ignorance 2023**
- URL: https://mappingignorance.org/2023/05/18/universal-power-law-kibble-zurek-scaling/
- What: Review of recent KZM theory developments

**KZM for nonequilibrium phase transitions with quenched disorder — Nature Comms Physics 2022**
- URL: https://www.nature.com/articles/s42005-022-00952-w
- What: KZM + disorder combined — very close to our diluted KZ angle
- Gap vs ours: theoretical/field theory, not GPU simulation on crystal graphs

**Novel angle (ours):** KZM on diluted/BCC/FCC graphs — how does bond disorder
modify the defect density scaling exponent kappa? This is theoretically predicted
(Harris criterion applies to KZM too) but not numerically verified on crystal graphs.

---

### 6. Ising on Complex Networks

**Critical phenomena in complex networks — EPJ B 2023**
- URL: https://link.springer.com/article/10.1140/epjb/s10051-023-00612-0
- What: Recent review of phase transitions on scale-free, random, small-world networks
- Key result: Scale-free networks (Barabasi-Albert) have Tc → infinity as N → infinity
  Small-world: finite-T transition exists for any nonzero rewiring probability

**Dorogovtsev, Goltsev & Mendes, Rev. Mod. Phys. 80, 1275 (2008)**
- What: Comprehensive review — Ising on complex networks
- Use: Standard reference for network Ising background

---

## Key Physical Values to Reproduce

| Quantity                     | Theory/Experiment       | Source                      |
|------------------------------|-------------------------|-----------------------------|
| Tc (3D cubic Ising, J=1)     | 4.5115 J/kB             | Hasenbusch 2010             |
| Kc = J/kBTc (FCC)            | 0.1020707(2)            | Berche comparison paper     |
| J_NN (BCC Fe)                | 16.3 meV                | Pajda et al. PRB 64 (2001)  |
| J_NN (FCC Ni)                | 4.1 meV                 | Pajda et al. PRB 64 (2001)  |
| Tc (Fe, experiment)          | 1043 K                  | —                           |
| Tc (Ni, experiment)          | 627 K                   | —                           |
| KZM exponent kappa (3D Ising)| 1.115                   | nu=0.6301, z=2              |
| Bond percolation pc (cubic)  | ~0.249                  | Stauffer & Aharony          |

---

## What Data You Need Before Writing

- [ ] FSS with N=8..40 on GPU — publication-quality exponents (Tasks 1-3, Windows)
- [ ] KZ sweep at N=40, N=50 — visible power-law, kappa fit
- [ ] BCC/FCC J-fitting results from fit_j.ipynb
- [ ] Dilution sweep Tc(p) on cubic lattice
- [ ] ML phase detection results on BCC/FCC graphs (snapshot data needed first)
- [ ] Coarsening z exponent across multiple N and T_quench (coarsening.ipynb)

---

## Is This Still the Ising Model?

Yes, completely. The Ising model is defined by its Hamiltonian:

    H = -J Σ σᵢσⱼ

Changing the graph (BCC, FCC, small-world, diluted) only changes *which pairs (i,j)
are summed over* — the physics is identical. "Ising model on complex networks" is a
well-established research field. The graph is the substrate; the model is still Ising.

---

## Related Models (for paper's Related Work section)

| Model            | Spins        | Key difference                             | When to use                          |
|------------------|--------------|--------------------------------------------|--------------------------------------|
| Heisenberg       | 3D vector    | More realistic for real magnets (Fe, Ni have continuous spin) | Better J-fitting accuracy |
| Potts (q-state)  | q discrete   | Different universality class               | Structural phase transitions         |
| XY               | 2D vector    | BKT transition in 2D (no true long-range order) | Superfluids, liquid crystals    |
| Edwards-Anderson | ±1, random J_ij | Spin glass, no ferromagnetism           | Disordered/frustrated systems        |

---

## Other Ways to Confirm the Same Physics

1. **Heisenberg model** — same graphs, vector spins (S_x, S_y, S_z). If J-fitting to
   Fe/Ni improves over Ising, that strengthens the paper. Architecture already supports
   this — just change the energy function. Fe and Ni are itinerant ferromagnets with
   continuous spin, so Heisenberg is more physically realistic.

2. **Renormalisation group (RG)** — analytical confirmation of critical exponents without
   simulation. Provides theoretical grounding for FSS results.

3. **Transfer matrix** — exact Tc for small systems, useful as a numerical sanity check
   against simulated Binder crossings.

4. **Series expansions** — high-T expansion gives Tc analytically for any graph structure.
   Good cross-check for BCC/FCC Tc values.

5. **Experiment** — KZM has been observed in NiTiO3 (agrees with 3D Ising prediction)
   and BiTeI (deviates). Our simulation of KZM on real crystal graphs is directly
   comparable to these neutron scattering experiments.

---

## Highest-Impact Extension for the Paper

Run the **Heisenberg model on the same BCC/FCC graphs** and compare J-fits to Ising.
Heisenberg is more physically realistic for iron and nickel — getting closer to
experimental Tc with Heisenberg would be a strong, concrete result.

The Ising vs Heisenberg comparison on identical graph structures is publishable on its own
as a methodology paper demonstrating the graph-agnostic architecture.

---

## Next Steps After Publication — Research Arc

### Track 1: Deeper Physics (extend the same simulator)

**Heisenberg model on the same graphs**
- Vector spins (S_x, S_y, S_z) instead of ±1 — more realistic for Fe, Ni, Co
- Better J-fitting accuracy, closer to experimental Tc
- Directly comparable to neutron scattering spin-wave measurements
- Standalone paper: "Heisenberg vs Ising on real crystal topologies"
- Target: Physical Review B

**Spin glass on network graphs**
- Edwards-Anderson model: random J_ij per bond (some ferromagnetic, some antiferromagnetic)
- Real-world relevance: disordered magnets, Hopfield neural networks, protein folding
- Connects directly to cybersecurity — spin glasses map onto constraint satisfaction
  problems that appear in network intrusion detection

**Quantum phase transitions**
- Transverse-field Ising model: add quantum tunneling term Γ Σ σˣ
- Quantum KZM — active research area, very few numerical studies on arbitrary graphs
- Requires quantum MC algorithm (path integral / worldline formulation) — significant
  extension but the graph infrastructure is already in place

---

### Track 2: The Cybersecurity Bridge (the unique long-term angle)

**Ising model on real network topology data**
- Use anonymised network traffic graphs from cybersecurity work
- Model information/anomaly propagation as an Ising-like spin system
- Phase transition = tipping point where a local anomaly cascades network-wide
- Directly relevant to threat detection and lateral movement modelling

This is Paper 2, and it is genuinely novel because:
- Nobody at the intersection of statistical physics + cybersecurity network topology
  has done this with a validated, published simulator
- Paper 1 provides the validated physics foundation
- Application framing opens different journals: Nature Communications,
  Physical Review Research, IEEE security journals

**Influence propagation / epidemic models**
- SIR epidemic model and Ising model are mathematically related (both are spin systems
  on graphs with a phase transition)
- Malware propagation on enterprise networks has the same topology as diluted BCC/FCC
  graphs studied in Paper 1
- Percolation threshold p_c corresponds to the point where a local vulnerability
  becomes a network-wide breach — has direct operational security meaning

---

### The Long-Term Publication Arc

  Paper 1 (now — build the simulation results first)
    GPU Ising on crystal graphs + disorder + ML phase detection
    Target: Physical Review E or Computer Physics Communications

  Paper 2 (after network data is available)
    Ising/spin-glass on real network topologies — physics of anomaly propagation
    Target: Nature Communications or Physical Review Research

  Paper 3 (if Heisenberg J-fitting is strong)
    Heisenberg model on crystal graphs, DFT comparison for Fe/Ni/Co
    Target: Physical Review B

The strongest long-term position: Paper 1 establishes credibility as a computational
physicist. Paper 2 establishes a unique bridge between statistical physics and
cybersecurity. That combination is essentially unique and would be visible in both
communities simultaneously.
