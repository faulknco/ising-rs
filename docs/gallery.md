# Figure Gallery

This is a visual gallery of all figures produced by the ising-rs project, a Rust + CUDA Monte Carlo simulator for the Ising model on arbitrary graph topologies. The project targets publication in Physical Review E. Figures are grouped by topic, with plain-English captions describing what each one shows.

---

## Paper Figures: Finite-Size Scaling (CPU Wolff Algorithm)

These are the main publication-quality figures, generated using the Wolff cluster algorithm on 3D simple-cubic lattices ranging from L=8 to L=48.

![Energy, magnetization, heat capacity, and susceptibility vs temperature for several lattice sizes](../paper/figures/fss_observables.png)
**Thermodynamic observables.** Energy, magnetization, heat capacity, and susceptibility plotted against temperature for lattice sizes L=8 through L=48. The phase transition shows up as sharp features that grow with system size.

![Binder cumulant crossings pinning down the critical temperature](../paper/figures/fss_binder.png)
**Binder cumulant crossings.** The Binder cumulant is a dimensionless ratio that crosses at the same point for all lattice sizes — right at the critical temperature. This crossing pins down Tc = 4.512(4) J/kB.

![Histogram-reweighted observables near the critical point](../paper/figures/fss_reweighted.png)
**Histogram reweighting near Tc.** By reweighting energy histograms, we get much finer temperature resolution near the critical point than raw simulation data alone would provide.

![Log-log peak scaling for extracting critical exponents](../paper/figures/fss_peak_scaling.png)
**Peak scaling.** The heights and positions of thermodynamic peaks follow power laws in system size. Log-log fits extract the critical exponent ratios gamma/nu and beta/nu.

![Finite-size scaling collapse onto a universal curve](../paper/figures/fss_collapse.png)
**Scaling collapse.** When temperature and observable axes are rescaled using the critical exponents, data from all lattice sizes collapse onto a single universal curve. This is the hallmark of a continuous phase transition.

![Global scaling collapse of susceptibility and Binder cumulant](../paper/figures/fss_collapse_global.png)
**Global scaling collapse.** Susceptibility and Binder cumulant data from all sizes, collapsed simultaneously. A clean collapse confirms that the extracted exponents are self-consistent.

![Peak scaling from histogram-reweighted data](../paper/figures/fss_reweighted_scaling.png)
**Reweighted peak scaling.** Same peak-scaling analysis as above, but using the finer-grained histogram-reweighted data for improved precision on the critical exponents.

---

## GPU FSS Results: Three Universality Classes

These figures come from GPU-accelerated runs on an RTX 2060, comparing three spin models — Ising, Heisenberg, and XY — each belonging to a different universality class.

### Ising Model

![Ising model thermodynamic observables vs temperature](../analysis/figures/gpu_fss/ising_observables.png)
**Ising observables.** Temperature dependence of energy, magnetization, heat capacity, and susceptibility for the discrete-spin Ising model on the GPU.

![Ising model Binder cumulant crossing](../analysis/figures/gpu_fss/ising_binder.png)
**Ising Binder cumulant.** Crossing analysis to locate the Ising critical temperature from GPU data.

![Ising model histogram-reweighted curves near Tc](../analysis/figures/gpu_fss/ising_reweighted.png)
**Ising reweighted curves.** Histogram-reweighted observables near the Ising critical point, generated from GPU run data.

![Ising model scaling collapse](../analysis/figures/gpu_fss/ising_collapse.png)
**Ising scaling collapse.** All lattice sizes fall onto one curve when rescaled with Ising critical exponents.

![Ising model peak scaling for exponent extraction](../analysis/figures/gpu_fss/ising_peak_scaling.png)
**Ising peak scaling.** Log-log fits of peak heights and positions to extract Ising universality class exponents.

### Heisenberg Model

![Heisenberg model thermodynamic observables vs temperature](../analysis/figures/gpu_fss/heisenberg_observables.png)
**Heisenberg observables.** Temperature dependence of thermodynamic quantities for the three-component continuous-spin Heisenberg model.

![Heisenberg model Binder cumulant crossing](../analysis/figures/gpu_fss/heisenberg_binder.png)
**Heisenberg Binder cumulant.** Crossing analysis for the Heisenberg critical temperature, which differs from the Ising value due to the different spin symmetry.

![Heisenberg model histogram-reweighted curves near Tc](../analysis/figures/gpu_fss/heisenberg_reweighted.png)
**Heisenberg reweighted curves.** Fine-grained view of Heisenberg observables near the critical point.

![Heisenberg model scaling collapse](../analysis/figures/gpu_fss/heisenberg_collapse.png)
**Heisenberg scaling collapse.** Data collapse using Heisenberg universality class exponents.

![Heisenberg model peak scaling for exponent extraction](../analysis/figures/gpu_fss/heisenberg_peak_scaling.png)
**Heisenberg peak scaling.** Exponent extraction from peak heights and positions for the Heisenberg model.

### XY Model

![XY model thermodynamic observables vs temperature](../analysis/figures/gpu_fss/xy_observables.png)
**XY observables.** Temperature dependence for the two-component planar-spin XY model.

![XY model Binder cumulant crossing](../analysis/figures/gpu_fss/xy_binder.png)
**XY Binder cumulant.** Crossing analysis to locate the XY critical temperature.

![XY model histogram-reweighted curves near Tc](../analysis/figures/gpu_fss/xy_reweighted.png)
**XY reweighted curves.** Histogram-reweighted observables near the XY critical point.

![XY model scaling collapse](../analysis/figures/gpu_fss/xy_collapse.png)
**XY scaling collapse.** Data collapse confirming XY universality class exponents.

![XY model peak scaling for exponent extraction](../analysis/figures/gpu_fss/xy_peak_scaling.png)
**XY peak scaling.** Log-log peak scaling for the XY model's critical exponents.

---

## Validation Tests

These figures demonstrate that the simulator produces correct results by comparing against known analytical solutions and consistency checks.

![Monte Carlo results vs Onsager exact 2D solution](../analysis/figures/generated/validation/onsager_comparison.png)
**Onsager comparison.** The gold standard validation: our Monte Carlo results for the 2D Ising model match Onsager's exact analytical solution.

![Comparison with exact 4x4 lattice enumeration](../analysis/figures/generated/validation/exact_enumeration.png)
**Exact enumeration.** For a small 4x4 lattice, every possible spin configuration can be enumerated exactly. Our Monte Carlo averages agree with the exact answers.

![Heat capacity from fluctuations matches direct derivative](../analysis/figures/generated/validation/fluctuation_dissipation.png)
**Fluctuation-dissipation check.** The heat capacity computed from energy fluctuations matches the one computed from a direct numerical derivative — confirming the fluctuation-dissipation theorem holds in the simulation.

![Autocorrelation time scaling with system size](../analysis/figures/generated/validation/autocorrelation_scaling.png)
**Autocorrelation scaling.** Autocorrelation times grow with system size as expected. This validates that our statistical error estimates account for correlated samples.

![Ground state and high-temperature limits](../analysis/figures/generated/validation/known_limits.png)
**Known limits.** At zero temperature the system is fully ordered; at infinite temperature it is fully disordered. The simulator reproduces both limits correctly.

---

## Coarsening Dynamics

After a sudden quench from high temperature into the ordered phase, domains of aligned spins grow over time.

![Raw domain growth snapshots after quench](../analysis/coarsening_raw.png)
**Raw coarsening data.** Snapshots of domain growth at successive times after a temperature quench. Domains start small and gradually merge.

![Domain size grows as t^(1/2) confirming Allen-Cahn coarsening](../analysis/coarsening_fit.png)
**Coarsening fit.** Domain size grows as the square root of time, consistent with Allen-Cahn theory for non-conserved order parameter dynamics.

---

## Kibble-Zurek Mechanism

When the system is cooled through the phase transition at a finite rate, it cannot keep up with the changing equilibrium and defects freeze in. Slower cooling means fewer defects.

![Defect density vs quench time for different lattice sizes](../analysis/figures/generated/kz/kz_raw.png)
**Kibble-Zurek raw data.** Defect density measured after cooling through Tc at various rates, for several lattice sizes.

![Power-law fit giving kappa approximately 0.25](../analysis/figures/generated/kz/kz_fit.png)
**Kibble-Zurek fit.** A power-law fit to defect density vs quench time gives an exponent kappa of approximately 0.25, close to the theoretical prediction of 0.279 for the 3D Ising universality class with Metropolis dynamics.

---

## Bond Dilution

Randomly removing bonds from the lattice weakens the magnetic coupling and shifts the phase transition.

![Effect of random bond removal on the phase transition](../analysis/figures/generated/dilution/dilution_observables.png)
**Dilution observables.** Thermodynamic observables for lattices with increasing fractions of randomly removed bonds. The transition broadens and shifts to lower temperatures.

![Critical temperature drops linearly with dilution](../analysis/figures/generated/dilution/dilution_tc_summary.png)
**Dilution Tc summary.** The critical temperature decreases approximately linearly with the concentration of removed bonds, consistent with mean-field expectations for weak dilution.

---

## Machine Learning Phase Detection

A neural network trained on raw spin configurations can learn to identify the phase transition without being told any physics.

![PCA of spin configurations separates phases](../analysis/data/ml_pca.png)
**PCA projection.** Principal component analysis of spin configurations. Ordered (low-temperature) and disordered (high-temperature) configurations separate cleanly in the first two principal components.

![Neural network identifies the phase boundary](../analysis/data/ml_phase_boundary.png)
**ML phase boundary.** A simple neural network classifier identifies the phase boundary by finding the temperature where its classification confidence drops to 50%.

![Training dynamics of the phase classifier](../analysis/data/ml_training_curve.png)
**Training curve.** Loss and accuracy during training of the phase classifier. The network converges quickly, indicating that the ordered-vs-disordered distinction is an easy classification problem.
