# IEEE Research Manuscript & Publication Package

## Project Title
**Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning**

---

### 1. Overview and Package Contents

This directory contains the complete, technically rigorous IEEE conference manuscript package prepared in **Phase 14**, integrating the verified and audited empirical findings from Phases 1–13.

```
paper/
├── main.tex                            # Complete IEEE-style LaTeX manuscript (26 sections)
├── references.bib                      # 26 verified IEEE-format bibliographic citations
├── figures/                            # 10 publication-quality high-res figures (300 DPI)
│   ├── fig1_overall_framework_architecture.png
│   ├── fig2_hybrid_quantum_circuit_architecture.png
│   ├── fig3_federated_learning_workflow.png
│   ├── fig4_differential_privacy_workflow.png
│   ├── fig5_secure_aggregation_workflow.png
│   ├── fig6_roc_curves_comparison.png
│   ├── fig7_precision_recall_curves.png
│   ├── fig8_privacy_utility_tradeoff.png
│   ├── fig9_communication_overhead_comparison.png
│   └── fig10_quantum_noise_sensitivity.png
├── tables/                             # 10 publication-quality LaTeX table source files
│   ├── tableA_cohort_statistics.tex
│   ├── tableB_centralized_models.tex
│   ├── tableC_federated_models.tex
│   ├── tableD_dual_metric_rounds.tex
│   ├── tableE_differential_privacy.tex
│   ├── tableF_personalization.tex
│   ├── tableG_model_complexity.tex
│   ├── tableH_communication_overhead.tex
│   ├── tableI_secure_aggregation.tex
│   └── tableJ_statistical_confidence_intervals.tex
├── EVIDENCE_TRACEABILITY_MATRIX.csv    # Master register mapping every number to source files
├── CLAIM_TO_EVIDENCE_MATRIX.csv        # Matrix linking each scientific claim to audited proof
├── FINAL_MANUSCRIPT_AUDIT.md           # Pre-submission editorial & scientific checklist
└── IEEE_MANUSCRIPT_README.md           # This compilation and submission guide
```

---

### 2. Step-by-Step LaTeX Compilation Instructions

The manuscript `main.tex` is authored using standard `IEEEtran.cls` conference formatting. It can be compiled using any modern TeX distribution or cloud compiler:

#### Option A: Local TeX Live / MiKTeX
Run the following sequence in the `paper/` directory:
```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

#### Option B: Latexmk (One-Step Build)
```bash
cd paper
latexmk -pdf main.tex
```

#### Option C: Overleaf Cloud Platform
1. Create a new blank project on [Overleaf](https://www.overleaf.com/).
2. Upload the entire contents of the `paper/` folder (including `figures/`, `tables/`, and `references.bib`).
3. Set the compiler to **pdfLaTeX** and main document to **`main.tex`**.
4. Click **Recompile**.

#### Option D: Docker Container
```bash
docker run --rm -v "${PWD}/paper:/work" -w /work texlive/texlive:latest sh -c "pdflatex main && bibtex main && pdflatex main && pdflatex main"
```

---

### 3. Traceability to Evidence Register

Every numerical value reported in the manuscript originates directly from frozen, reproducible CSV files in the workspace:

- **Cohort Counts (2,492 Stays, 212 Deaths):** Sourced from `results/phase13/cohort_flow.csv` and `results/phase13/split_integrity.csv`.
- **Centralized Model Baselines (XGBoost 0.8445, QNN 0.8346):** Sourced from `results/metrics/classical_baseline_metrics.csv` and `results/metrics/quantum_experiments_metrics.csv`.
- **Model Parameters (957 Total vs 5,537 MLP):** Sourced from `results/phase13/model_complexity_comparison.csv`.
- **Communication Bandwidth (0.80 MB vs 2.88 MB, 72.4% Reduction):** Sourced from `results/phase13/communication_recalculation.csv`.
- **Formal Differential Privacy ($\epsilon \in \{1, 2, 4, 8\}, \delta=10^{-5}$):** Sourced from `results/phase12/metrics/formal_dp_metrics.csv` and `results/phase13/privacy_accounting_audit.csv`.
- **Secure Aggregation ($\text{RMSE} \le 4.36 \times 10^{-7}$):** Sourced from `results/phase13/secure_aggregation_audit.csv`.

---

### 4. Mandatory Ethical, Editorial & Submission Disclosures

1. **Exploratory Framing:** The paper presents an exploratory hybrid quantum-classical federated learning architecture for communication-efficient ICU prognostication.
2. **No Claim of Predictive Quantum Advantage:** Gradient-boosted decision trees (XGBoost AUROC 0.8445, Random Forest 0.8411) outperform the hybrid quantum model (0.8346). The true quantum contribution is compact parameterization (957 parameters vs 5,537) saving 72.4% network payload.
3. **No Claim of Rural Hospital Rescue:** The claim of rescuing rural hospitals is formally withdrawn due to small positive death counts ($N_{\text{pos}} = 6$) in the test partition.
4. **Record-Level Privacy Scope:** Privacy guarantees apply per ICU unit stay record, not per patient lifetime.
5. **Simulation Scope:** Secure aggregation is evaluated as an algorithmic cryptographic simulation in shared memory, not a live hospital WAN network deployment.
6. **Single-Cohort Study:** Evaluated solely on eICU Demo v2.0.1; external validation on MIMIC-IV or the full eICU database remains future work under PhysioNet DUA.
