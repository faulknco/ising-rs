# Submission Checklist — ising-rs Paper

**Target:** Physical Review E (primary) / Computer Physics Communications (backup)
**Current status:** Draft complete, compiles cleanly, awaiting cleaner GPU data from Windows machine.

---

## Phase 1: Data & Figures (blocked on Windows run)

- [ ] `git pull` new data from Windows machine
- [ ] Check if new CSVs differ from current (larger N, more samples, better statistics)
- [ ] Re-run `analysis/fss.ipynb` if FSS data changed → regenerate FSS figures
- [ ] Re-run `analysis/kz.ipynb` if KZ data changed → regenerate `kz_fit.png`
- [ ] Verify KZM exponent κ improves toward 0.279 with cleaner data
- [ ] Confirm all figures in `paper/figures/` are up to date
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

- Heisenberg model on same BCC/FCC graphs → better J-fitting for Fe/Ni
- KZM on diluted/BCC/FCC graphs → disorder modification of κ
- Spin glass on network topologies → cybersecurity bridge paper
- Quantum transverse-field Ising → quantum KZM on arbitrary graphs
