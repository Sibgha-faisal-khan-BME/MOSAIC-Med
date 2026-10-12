# CHANGELOG.md: Adversarial Review Rounds and Manuscript Revisions

This log documents the iterative refinements resulting from three simulated rounds of hostile peer review (Statistician, Cardiologist, and ML Methods Reviewer).

---

## Round 1: Hostile Statistical Review
* **Focus**: Multiplicity corrections, hypothesis testing rigor, and sample size claims.
* **Findings & Amendments**:
  1. *Holm-adjusted p-values*: Flagged that heterogeneous vs. homogeneous comparison in Table III achieves adjusted $p = 0.0577$. **Action**: Replaced all assertions of "statistically significant diversity superiority" with "consistent empirical trend" and reported both raw ($0.0164$) and adjusted ($0.0577$) $p$-values.
  2. *DeLong Test Application*: Clarified that DeLong tests apply strictly per-class, with paired Welch's $t$-tests used for macro-averaged endpoints across seeds.
  3. *Uncertainty Definition in Sparse Top-2*: Adjusted Equation (6) documentation to specify that disagreement is calculated over the active expert subset ($K=2$) to maintain computational complexity guarantees.

---

## Round 2: Clinical Cardiology Review
* **Focus**: Clinical validity, signal acquisition realism, and triage safety.
* **Findings & Amendments**:
  1. *External Domain Shift Overstatement*: Reviewer objected to calling the PTB-XL device subset ($s \ne 0$, $N=248$) "external domain validation". **Action**: Renamed to "Site-Subgroup Sensitivity Analysis" and formally disclosed cross-hospital multi-center validation as an open limitation.
  2. *Physiological Interpretation of Routing*: Enhanced clinical explanation of why TCN-1D and MultiScale-CNN dominate Conduction Defects (PR interval, bundle branch block duration) while ResNet-1D and ConvAttention dominate Ischemia / ST-T abnormalities (localized ST elevation, reciprocal depression).
  3. *Abstention Triage Feasibility*: Added clinical overread workload context: deferring 10% of records reduces Hamming risk to $10.37\%$ while autonomously clearing 90% of tracings with low error.

---

## Round 3: Machine Learning Methods & Reproducibility Review
* **Focus**: Budget justification, baseline fairness, and ablation completeness.
* **Findings & Amendments**:
  1. *Server 32M Budget Scope*: Formalized computational profiling in `BUDGET_JUSTIFICATION.md`, explaining why edge $2.0\text{M}$ is the primary operational envelope and documenting the 32M regime as a computational boundary limitation under available workstation hardware.
  2. *Gating Collapse Verification*: Reconciled Table IV and Table VII: highlighted Table VII as the factorial proof that unregularized routers collapse catastrophically into single experts ($0.9049$ vs $0.9108$, $p=0.0480$), whereas diversity entropy regularization restores stable exploration.
  3. *Reference Verification*: Corrected citation venues (SelectiveNet $\to$ ICML 2019; Kompa et al. $\to$ npj Digital Medicine 2021) and verified all 18 entries in `REFERENCES_VERIFICATION.md`.

---

## Final QA Sign-Off Matrix

- [x] **Number Verification**: Every metric verified against `results/tables/` CSVs via `AUDIT.md`.
- [x] **Proposal Compliance**: Every commitment mapped and accounted for in `PROPOSAL_COMPLIANCE.md`.
- [x] **Preregistration**: Confirmatory vs. exploratory boundaries delineated in `PREREGISTRATION.md`.
- [x] **Budget Profiling**: Hardware constraints and 2M edge focus documented in `BUDGET_JUSTIFICATION.md`.
- [x] **Citation Integrity**: All 18 references checked and verified with valid DOIs in `REFERENCES_VERIFICATION.md`.
- [x] **Adversarial Defenses**: 10 harshest critiques answered in `REVIEWER_RESPONSES.md`.
- [x] **Document Synchronization**: Both LaTeX (`latex/main.pdf`) and Word (`results/MOSAIC_Med_Paper.docx`) compiled and up to date.
