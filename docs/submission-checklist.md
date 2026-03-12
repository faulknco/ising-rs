# Submission Checklist — ising-rs Paper

**Target:** Physical Review E (primary) / Computer Physics Communications (backup)
**Current status:** Phase 1 data complete for all three universality classes. Ising and XY at publication quality. Heisenberg usable (N=192 re-run with narrower T window backburnered). Phase 2 anisotropy crossover work starting.

---

## Phase 1: Data & Figures

### GPU FSS Pipeline (completed 2026-03-09)
- [x] GPU parallel tempering pipeline running on RTX 2060
- [x] Three models: Ising (Z2), Heisenberg (O(3)), XY (O(2))
- [x] Sizes N=8,16,32,64,128 with 100k samples, 32 replicas (Ising) / 20 replicas (Heis/XY)
- [x] Publication data: `analysis/data/gpu_windows_pipeline/publication/` (30 files, 128 MB)
- [x] Analysis script with single-histogram reweighting: `analysis/scripts/analyze_gpu_fss.py`
- [x] 15 publication figures in `analysis/figures/gpu_fss/`
- [x] Reproducibility guide: `analysis/REPRODUCIBILITY.md`
- [x] Summary CSVs + figures committed to `gpu-windows-pipeline` branch

### Results (updated 2026-03-12)
| Model | Tc error | gamma/nu | nu | beta/nu | Sizes | Status |
|-------|----------|----------|-----|---------|-------|--------|
| Ising | **0.015%** | 2.6% | **0.1%** | **0.3%** | 8-192 | **Publication quality** |
| XY | **0.001%** | **0.8%** | **2.3%** | **1.0%** | 8-128 | **Publication quality** |
| Heisenberg | **0.3%** | **1.1%** | 11.6% | **2.0%**† | 16-128 | Usable (ν weak) |

† Heisenberg beta/nu via hyperscaling (d=3). Direct fit fails due to coarse T grid.

### Heisenberg N=192 issue
- [x] Run completed (50k samples, 16 replicas, ~24h with Wolff embedding)
- [ ] **Backburnered:** T grid too coarse (dT=0.013, 16 replicas over [1.4,1.6]). No replica in critical fluctuation zone. Fix: re-run with `--tmin 1.42 --tmax 1.48` (~24h).
- Best fit sizes: `--fit-sizes 16,32,64,128` (exclude N=8 finite-size corrections, N=192 coarse grid)

### Completed data improvements
- [x] Ising N=192 with MSC batched kernel
- [x] Heisenberg N=128 with Wolff embedding (50k samples)
- [x] Heisenberg N=192 with Wolff embedding (50k samples, 16 replicas)
- [x] XY all sizes (N=8-128, 200k samples each)
- [x] All analysis figures updated in `analysis/figures/gpu_fss/`

### Remaining figure tasks
- [ ] Re-run `analysis/kz.ipynb` if KZ data changed → regenerate `kz_fit.png`
- [ ] Confirm all figures in `paper/figures/` are up to date with GPU data
- [ ] Re-run `paper/build.sh` and verify clean compile

---

## Phase 2: Paper Content

- [ ] **Fix RevTeX class:** change `pra` → `pre` in `\documentclass` for Physical Review E
- [ ] **ML phase detection:** decide whether to include results in paper or drop from claims
  - Currently: `ml_phase.ipynb` exists but results not in `draft.tex`
  - Option A: add a short ML results subsection
  - Option B: move ML to future work in conclusion (currently implied)
- [ ] **Coarsening section:** not in paper — decide include or future work
- [ ] **GitHub URL:** verify `https://github.com/faulknco/ising-rs` is correct username
- [ ] **Tc(Ni) typo:** paper says 631 K, literature notes say 627 K — check and fix
- [ ] **Data availability statement:** add before acknowledgments
- [ ] **Code availability statement:** point to GitHub repo

---

## Phase 3: Pre-submission Technical

- [ ] Make GitHub repo **public** (paper cites it — must be public before submission)
- [ ] Add `LICENSE` file to repo if not present
- [ ] ORCID: obtain and add to author affiliation in `draft.tex`
- [ ] Final full LaTeX build: `pdflatex → bibtex → pdflatex → pdflatex`
- [ ] Check no `??` citations in final PDF
- [ ] Check no missing figure warnings
- [ ] Page count acceptable (currently 11 pages — PRE has no strict limit)

---

## Phase 4: PRE Submission

- [ ] Create account at journals.aps.org
- [ ] Prepare cover letter (1 page: novelty, target section, suggested reviewers)
- [ ] Upload: `draft.tex`, `references.bib`, all figure PNGs
- [ ] Suggested reviewers: experts in GPU MC, FSS, Kibble-Zurek
- [ ] Suggested section: **Computational Physics** or **Statistical Physics**

---

## Known Issues (minor, not blockers)

- Two RevTeX "float stuck" warnings during `\clearpage` — benign, all figures place correctly
- `paper/fss_collapse.png` duplicate (gitignored, not a problem)
- `build.sh` only runs `pdflatex` twice, not the full `bibtex` sequence — fix or document

---

## Future Work (Paper 2 / extensions — do not block submission)

- **Phase 2 (active):** Anisotropy-driven crossover in 3D Heisenberg (branch: `feature/gpu-anisotropy-port`)
  - CPU implementation complete (Phases A-B), GPU porting needed (Phase C)
  - Campaign scripts ready: 7 D values × 5 sizes × 49 temps
- Multi-GPU support for N≥256 lattices
- GPU Wolff cluster algorithm (currently CPU Wolff via embedding)
- GPU-side parallel tempering exchange (currently CPU-mediated)
- Kibble-Zurek quench on GPU for larger N
- Domain coarsening dynamics on GPU
- BCC/FCC lattice GPU kernels for J-fitting
- Spin glass on network topologies
