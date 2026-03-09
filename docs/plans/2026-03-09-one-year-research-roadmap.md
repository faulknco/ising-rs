# One-Year Research Roadmap

## Purpose

This roadmap is the long-horizon plan for turning `ising-rs` from a validated
classical benchmark engine into a platform for more interesting spin-physics
research and, eventually, quantum-critical model work.

The ordering matters:

1. lock down benchmark correctness
2. publish focused classical physics
3. extend toward more realistic spin models
4. use that validated infrastructure as the bridge to quantum work

## Phase 1: Benchmark Platform (0-2 months)

### Goal

Turn the repo into a defensible benchmark platform with reproducible outputs and
credible finite-size-scaling analysis.

### Main tasks

- finish the `L <= 192` GPU analysis for Ising, XY, and Heisenberg
- lock the primary analysis path for:
  - Binder `Tc`
  - `gamma/nu`
  - `beta/nu`
  - `nu` from Binder slope
- stabilize or replace the current WHAM path
- freeze a GPU benchmark result pack
- keep the CPU validation pack and dilution pack current and reproducible
- update README and manuscript claims to match what is actually validated

### Success criteria

- one strong CPU validation pack
- one strong dilution pack
- one GPU benchmark pack
- all of them reproducible from committed code, manifests, and scripts

### Notes

This phase is about credibility, not novelty. It is the foundation for
everything that follows.

## Phase 2: Focused Classical Physics (2-5 months)

### Goal

Use the validated engine to produce one genuinely interesting classical-physics
 study that is more than a benchmark exercise.

### Recommended direction

Universality and crossover in 3D continuous-spin systems, especially:

- anisotropy-driven Heisenberg-to-Ising crossover
- disorder-driven crossover in XY or Heisenberg
- controlled perturbations of standard universality classes

### Why this direction

- it builds directly on the GPU FSS work
- it is more interesting than repeating pure Ising benchmarks
- it is still close enough to the current engine that the development cost is reasonable
- it forms a natural bridge to future many-body / quantum work

### Main tasks

- define one narrow physics question
- run a multi-size campaign with a primary fit window on larger sizes
- measure crossover-sensitive observables and exponents
- separate benchmark validation from new-physics claims
- freeze a focused result pack for the study

### Success criteria

- one narrow, defensible physics claim
- one publishable paper direction
- one stable analysis workflow for that study

## Phase 3: Materials-Inspired Spin Modeling (5-8 months)

### Goal

Move from universality-class benchmarks into more realistic or
materials-inspired spin Hamiltonians.

### Candidate directions

- BCC / FCC Heisenberg with fitted exchange constants
- anisotropic spin Hamiltonians
- next-nearest-neighbor or competing couplings
- controlled crystal-derived graph studies, if the analysis layer is mature enough

### Why this phase matters

It pushes the repo from “benchmark code with papers” toward “research engine
that can support model-building.”

### Main tasks

- choose one applied direction rather than several
- define what must stay benchmarked versus what is exploratory
- keep all new workflows script-first and manifest-backed
- avoid adding model complexity faster than the analysis can support

### Success criteria

- one applied or materials-style case study
- validated benchmark section plus applied-physics section
- no backsliding into notebook-only workflows

## Phase 4: Quantum Bridge (8-12 months)

### Goal

Start the first serious quantum-facing extension of the engine.

### Recommended first target

Transverse-field Ising model.

### Why this is the right first step

- it is conceptually clean
- it connects naturally to the current Ising validation base
- it has a strong critical-scaling story
- it is a better first quantum target than trying to jump directly into a very broad “quantum theory engine”

### Practical direction

The likely path is:

- start with a small, validated quantum-critical benchmark
- reuse the existing scaling, uncertainty, and reproducibility infrastructure
- keep the quantum layer narrow and benchmark-driven at first

### Success criteria

- one prototype quantum workflow
- small validated benchmarks
- a clear boundary between classical, semiclassical, and quantum capability

## Recommended Priorities

In strict order:

1. finish the large-size GPU benchmark analysis
2. publish one focused crossover or disorder study
3. extend to one materials-inspired continuous-spin problem
4. begin the first quantum-critical model

## What To Avoid

- broad cleanup projects without research payoff
- too many model branches at once
- overclaiming GPU or quantum capability before the analysis layer is mature
- jumping into full quantum simulation before the benchmark and scaling workflows are trusted

## Decision Rule

When choosing between cleanup, analysis, and new research:

- prefer analysis if the data already exists but the claim is weak
- prefer new research if the benchmark foundation is already strong
- prefer cleanup only when it removes a concrete blocker for reproducibility or physics

## Short Version

Near term:

- finish GPU benchmark analysis
- lock the validated classical story

Middle term:

- do one interesting crossover or disorder paper
- move into materials-inspired continuous-spin work

Long term:

- build toward a transverse-field Ising style quantum-critical workflow
