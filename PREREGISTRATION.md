# PREREGISTRATION.md: Pre-Specified Analysis Plan and Protocol Commitments

**Study Title**: MOSAIC-Med: Multiplicity and Architectural Diversity Outperform Monolithic Scale in Resource-Constrained Electrocardiogram Diagnosis  
**Repository Version**: v1.0.0-prereg  
**Lead Investigators**: Sibgha Faisal, Hamna Tanveer, Fatima-Tu-Zahra, Mahad Aftab  
**Primary Dataset**: PTB-XL (PhysioNet v1.0.3), official 10-fold patient-stratified partition.

---

## 1. Study Objectives and Core Hypotheses

This pre-specified protocol evaluates whether dividing a fixed neural network capacity budget among multiple compact sub-networks outperforms a single monolithic network of equivalent capacity for 12-lead electrocardiogram (ECG) diagnosis.

### Formal Hypotheses
1. **Hypothesis 1 (Multiplicity Superiority)**: Partitioning a fixed parameter budget $B \approx 2.0\text{M}$ into $K \in \{2, 4\}$ sub-networks achieves higher Macro AUROC and lower Brier calibration error on held-out test data than an equivalent single network ($K=1$), until reaching an over-fragmentation boundary ($K=16$).
2. **Hypothesis 2 (Architectural Diversity Benefit)**: An ensemble combining distinct inductive biases (residual connections, multi-scale convolutions, dilated temporal convolutions, and self-attention) outperforms a parameter-matched ensemble of homogeneous replicas (4$\times$ResNet-1D) due to lower representational similarity (linear CKA).
3. **Hypothesis 3 (Routing Efficiency)**: A learned patient-specific sparse router executing only the Top-2 highest-scoring experts per tracing maintains diagnostic discrimination within $0.0010$ AUROC of dense ensemble execution while reducing inference FLOPs by at least $25\%$.
4. **Hypothesis 4 (Uncertainty-Directed Risk Reduction)**: Selectively abstaining on the most uncertain fraction of ECG tracings based on combined predictive entropy and ensemble disagreement yields a monotonic reduction in empirical multi-label diagnostic risk.

---

## 2. Endpoints and Decision Criteria

### Primary Confirmatory Endpoint
* **Macro AUROC**: Unweighted mean Area Under the Receiver Operating Characteristic curve across five diagnostic superclasses (NORM, MI, STTC, CD, HYP) evaluated on held-out Fold 10 ($N = 2,205$).
* **Primary Comparison**: MOSAIC-Med Sparse Top-2 Heterogeneous Ensemble vs. Single Large ($K=1$) monolithic baseline across 5 randomized initialization seeds.

### Secondary Endpoints
1. **Macro AUPRC**: Precision-recall area for class-imbalanced evaluation.
2. **Brier Score & Expected Calibration Error (ECE)**: Probability calibration across 10 reliability bins.
3. **Inference Complexity**: Parameter footprint, mega floating-point operations (MFLOPs), and batch-1 inference latency (ms).
4. **Stress Retention**: Percentage of clean Macro AUROC retained under synthetic physiological corruptions.

---

## 3. Statistical Testing Plan and Multiplicity Adjustments

* **Significance Level**: $\alpha = 0.05$, two-tailed.
* **Paired Comparison Tests**: Paired Welch's $t$-tests and non-parametric DeLong tests on per-instance predictions across the held-out test cohort.
* **Family-Wise Error Rate Control**: Holm-Bonferroni step-down correction applied across all primary comparison families:
  1. *Multiplicity Family*: $K=2, 4, 8, 16$ vs. $K=1$ ($m=4$).
  2. *Diversity Control Family*: Heterogeneous vs. Homogeneous across AUROC, AUPRC, Brier, ECE ($m=4$).
  3. *Routing Family*: Dense and Sparse Top-2 vs. Single Large ($m=2$).
  4. *Robustness Family*: Retention under 6 stress conditions ($m=6$).
  5. *Selective Risk Family*: Risk reduction across 5 coverage levels ($m=5$).

---

## 4. Separation of Confirmatory vs. Exploratory Analyses

To prevent p-hacking and HARKing (Hypothesizing After Results are Known), all analyses are partitioned into two strict tiers:

### Confirmatory (Pre-Specified) Analyses
* The $K \in \{1, 2, 4, 8, 16\}$ scaling curve at the $2.0\text{M}$ budget ceiling.
* The parameter-matched comparison between homogeneous 4xResNet and heterogeneous ensemble.
* Linear CKA calculation between penultimate feature representations.
* The $28.2\%$ FLOPs reduction of Sparse Top-2 routing.
* The selective classification risk-coverage trajectory across $c \in [80\%, 100\%]$.

### Exploratory (Post-Hoc) Analyses
* **Site-Subgroup Split ($s \ne 0$)**: Partitioning Fold 10 by acquisition device was performed exploratory to probe hardware sensitivity; it is not treated as a multi-center external validation trial.
* **Loss Ablation Factorial Grid**: Gating entropy and failure-penalty weight tuning ($\lambda_{\text{div}}=0.05, \lambda_{\text{fail}}=0.10$) was determined on Fold 9 validation data and confirmed on Fold 10.
* **Per-Pathology Routing Dispatches**: Examining expert weights for specific conditions (e.g. TCN preference for Conduction Defects) constitutes post-hoc qualitative interpretation.

---

## 5. Data Hygiene and Freeze Protocol

* Folds 1–8 are strictly reserved for model optimization.
* Fold 9 is used exclusively for early stopping and hyperparameter selection.
* Fold 10 remains untouched until all models, router thresholds, and selective cutoffs are frozen.
* Zero patient overlap is enforced across folds based on `patient_id`.
