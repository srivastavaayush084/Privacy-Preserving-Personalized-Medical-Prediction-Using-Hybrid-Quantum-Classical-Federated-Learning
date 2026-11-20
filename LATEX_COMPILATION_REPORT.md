# Phase 14 — LaTeX Compilation and Typesetting Certification Report

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Manuscript Source:** `IEEE_Research_Paper.tex`  
**Output Document:** `IEEE_Research_Paper.pdf`  
**Compiler:** Tectonic Modern TeX Engine (v0.15.0, x86_64-pc-windows-msvc)  
**Document Class:** `IEEEtran` (Conference Format, Two-Column)  
**Date:** November 20, 2026  
**Auditor:** LaTeX Specialist & IEEE Publications Technical Reviewer  
**Compilation Status:** **CLEAN BUILD (Exit Code 0) — CERTIFIED**  

---

## 1. Toolchain and Build Environment

| Component | Specification / Version | Status |
| :--- | :--- | :---: |
| **TeX Engine** | Tectonic v0.15.0 (Automated multi-pass XeTeX wrapper) | **PASS** |
| **Document Class** | `IEEEtran.cls` (IEEE Conference proceedings standard) | **PASS** |
| **BibTeX Engine** | Tectonic integrated bibtex parser | **PASS** |
| **Graphics Driver** | `xdvipdfmx` (Direct vector PDF synthesis) | **PASS** |
| **Operating System** | Microsoft Windows 11 Enterprise (x86_64) | **PASS** |
| **Source File** | `IEEE_Research_Paper.tex` (437 lines, UTF-8) | **PASS** |
| **Bibliography** | `references.bib` (265 lines, 26 authentic entries) | **PASS** |
| **Output File** | `IEEE_Research_Paper.pdf` (1.97 MiB) | **PASS** |

---

## 2. Page Budget and Layout Verification

- **IEEE Page Constraint:** Strictly 6 to 8 pages for conference submissions.
- **Compiled Document Length:** **EXACTLY 8 PAGES** ($100\%$ compliant).
- **Overfull `\hbox` Errors:** **ZERO.** All equations, tables, and paragraphs fit within column boundaries.
- **Underfull Warnings:** Minimal typographic warnings consistent with strict IEEE two-column formatting; zero visual artifacts.

### Page-by-Page Content and Geometry Distribution:

| Page | Character Count | Graphical Assets | Tabular Assets | Content Description |
| :---: | :---: | :---: | :---: | :--- |
| **Page 1** | 5,299 | 0 | 0 | Title, author metadata, abstract, index terms, Section I (Introduction), Section II (Related Work). |
| **Page 2** | 5,256 | 0 | 0 | Section II (Related Work cont.), Section III (Research Gap & Contributions), Section IV (Proposed Hybrid Architecture). |
| **Page 3** | 3,326 | 2 (Fig. 1, Fig. 2) | 0 | **Fig. 1** (Double-column framework architecture), **Fig. 2** (4-qubit quantum variational circuit), Section IV.B (Strongly entangling layers & observables). |
| **Page 4** | 4,619 | 0 | 1 (Tab. I) | Section V (Dataset & Cohort), **Table I** (Canonical cohort statistics), Section VI (Privacy & Security Framework: DP-SGD & 4-round SecAgg). |
| **Page 5** | 4,270 | 2 (Fig. 3, Fig. 4) | 1 (Tab. II) | **Table II** (Double-column comprehensive metrics across $\varepsilon$), **Fig. 3** (ROC/PR curves), **Fig. 4** (Privacy-utility Pareto frontier), Section VII (Experimental Methodology). |
| **Page 6** | 4,635 | 2 (Fig. 5, Fig. 7) | 1 (Tab. III) | Section VIII (Results & Analysis), **Table III** (Bootstrap paired differences), **Fig. 7** (Bootstrap distributions), **Fig. 5** (Reliability curves), Section VIII.D (Parameter footprint). |
| **Page 7** | 4,981 | 1 (Fig. 6) | 1 (Tab. IV) | Section IX (Calibration & DCA), **Table IV** (Decision Curve Analysis net benefits), **Fig. 6** (DCA net benefit curves), Section X (Discussion), Section XI (Limitations & Threats). |
| **Page 8** | 7,006 | 0 | 0 | Section XII (Conclusion and Future Work), Acknowledgment, **References [1] through [26]** (Two balanced columns). |

---

## 3. Image and Graphical Assets Verification

All figure assets were rendered at high resolution ($300+$ DPI) and staged in `figures/`:

| Figure Label | File Path | Resolution / Dimensions | Color Space | Content Verified |
| :---: | :--- | :---: | :---: | :--- |
| **Fig. 1** | `figures/fig1_overall_framework_architecture.png` | $3300 \times 1350$ px | RGB (Lossless PNG) | End-to-end federated quantum-classical pipeline with DP and SecAgg. |
| **Fig. 2** | `figures/fig2_hybrid_quantum_circuit_architecture.png` | $2700 \times 1350$ px | RGB (Lossless PNG) | 4-qubit register, $R_Y$ angle embedding, depth-2 Strongly Entangling, Pauli expectation readouts. |
| **Fig. 3** | `figures/fig3_roc_prc_comparison.png` | $2400 \times 1200$ px | RGB (Lossless PNG) | ROC and PR curves on test cohort comparing QNN, MLP, and baselines. |
| **Fig. 4** | `figures/fig4_privacy_utility_tradeoff.png` | $2400 \times 1200$ px | RGB (Lossless PNG) | AUROC Pareto frontier across $\varepsilon \in \{1.0, 2.0, 4.0, 8.0, \infty\}$. |
| **Fig. 5** | `figures/fig5_reliability_curves.png` | $2400 \times 1200$ px | RGB (Lossless PNG) | Test reliability curves: uncalibrated, Platt scaled, and prevalence baselines. |
| **Fig. 6** | `figures/fig6_decision_curve_analysis.png` | $2400 \times 1200$ px | RGB (Lossless PNG) | Net Benefit curves across $p_t \in [0.01, 0.35]$ with Treat-All and Treat-None baselines. |
| **Fig. 7** | `figures/fig7_bootstrap_paired_differences.png` | $2400 \times 1200$ px | RGB (Lossless PNG) | Empirical paired bootstrap difference histograms ($B=2,000$) with $95\%$ CIs spanning zero. |

---

## 4. Cross-Reference and Citation Integrity Check

1. **Undefined References (`LaTeX Warning: Reference '...' on page ... undefined`):**
   - **0 warnings.** All `\ref{tab:cohort}`, `\ref{tab:main_results}`, `\ref{tab:bootstrap}`, `\ref{tab:dca}`, `\ref{fig:framework}`, `\ref{fig:quantum_circuit}`, `\ref{fig:roc_prc}`, `\ref{fig:privacy_tradeoff}`, `\ref{fig:calibration}`, `\ref{fig:dca}`, and `\ref{fig:bootstrap_distributions}` resolve accurately.
2. **Undefined Citations (`LaTeX Warning: Citation '...' on page ... undefined`):**
   - **0 warnings.** All 26 references resolve cleanly to `[1]` through `[26]`.
3. **Hyperlink Functionality:**
   - Active hyperref package configured with discrete color scheme (`linkcolor=black`, `citecolor=black`, `urlcolor=blue`), ensuring clean black-and-white print output while providing clickable DOIs and URLs for digital readers.

---

## 5. IEEE Xplore Compliance Checklist

- [x] Two-column conference format using standard `IEEEtran.cls`.
- [x] Title in 24 pt Times/Roman font, author block in standard IEEE conference geometry.
- [x] Page count is exactly 8 pages (within the 6–8 page maximum conference requirement).
- [x] No page numbers or headers in final submission mode (managed by `\pagestyle{empty}` per IEEE instructions).
- [x] All fonts embedded via `xdvipdfmx` PDF synthesis.
- [x] Math formatted in standard Computer Modern Roman / AMS math fonts.
- [x] Standard booktabs line weights on all tabular data.
- [x] Clean exit code 0 on build.

---

## 6. Certification

The PDF output `IEEE_Research_Paper.pdf` meets all IEEE conference submission requirements for formatting, visual asset clarity, mathematical typography, and bibliography compliance.
