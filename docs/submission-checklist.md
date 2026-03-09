# Submission Checklist — ising-rs Paper

**Target:** Physical Review E (primary) / Computer Physics Communications (backup)
**Current status:** Draft complete, compiles cleanly, awaiting cleaner GPU data from Windows machine.

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

### Results (current)
| Model | Tc error | gamma/nu | nu | beta/nu |
|-------|----------|----------|-----|---------|
| Ising | 0.02% | 2.5% | 1.2% | 12.6% |
| Heisenberg | 0.04% | 1.6% | 18% | 1.9% |
| XY | 0.001% | 0.5% | 3.7% | 2.0% |

### Improving weak exponents (in progress)
- [ ] Run Ising N=192 (100k samples) — running now
- [ ] Rerun Ising N=128 with 500k samples for better statistics
- [ ] Rerun Heisenberg N=128 with 500k samples
- [ ] Re-run analysis with additional sizes → update figures and tables
- [ ] Target: Ising beta/nu < 5%, Heisenberg nu < 5%

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

- Multi-GPU support for N≥256 lattices
- Wolff cluster algorithm on GPU (currently CPU Wolff, GPU checkerboard Metropolis)
- GPU-side parallel tempering exchange (currently CPU-mediated)
- Kibble-Zurek quench on GPU for larger N
- Domain coarsening dynamics on GPU
- BCC/FCC lattice GPU kernels for J-fitting
- Heisenberg model on same BCC/FCC graphs → better J-fitting for Fe/Ni
- Spin glass on network topologies → cybersecurity bridge paper
