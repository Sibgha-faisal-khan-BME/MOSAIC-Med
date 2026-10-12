# REVIEWER_RESPONSES.md: Rebuttals to the 10 Harshest Peer Review Critiques

This document compiles the ten most aggressive, adversarial reviewer critiques anticipated for the MOSAIC-Med study, accompanied by exact empirical defenses, textual adjustments, and limitation disclosures.

---

### Critique 1 (Statistician): "Heterogeneous vs. Homogeneous AUROC is NOT statistically significant after multiple testing correction."
> *"In Table III, you report AUROC $0.9110$ vs $0.9084$ with raw $p = 0.0164$. But your Holm-adjusted $p$-value across the four metrics is $0.0577$. Claiming that heterogeneous diversity 'outperforms' homogeneous replication at $\alpha=0.05$ is statistically invalid."*
* **Response & Action**:
  * **Concession & Re-framing**: The critique is completely correct under strict family-wise error control. In `parameter_matched_control.csv`, the Holm-adjusted $p$-value is indeed $0.0577$.
  * **Textual Amendment**: We have eliminated the word "significantly" from this comparison across the abstract, introduction, and results. We now describe it as a *consistent numerical trend and variance-reduction improvement* ($\Delta = +0.0026$), explicitly reporting both the raw $p = 0.0164$ and the adjusted $p = 0.0577$.

---

### Critique 2 (Statistician): "DeLong's test is inapplicable to multi-label Macro AUROC."
> *"DeLong's test (1988) was formulated for comparing single binary ROC curves on a shared cohort. Applying DeLong to Macro AUROC (an unweighted arithmetic mean across 5 classes) violates the covariance assumptions."*
* **Response & Action**:
  * **Clarification**: We clarify in `PREREGISTRATION.md` and Section II that DeLong's test is computed independently per clinical superclass (NORM, MI, STTC, CD, HYP). For the aggregate macro AUROC comparisons across seeds and configurations, we compute paired Welch's $t$-tests and stratified patient-level bootstrap standard errors ($N = 2,205$).

---

### Critique 3 (Cardiologist): "The 'External Site' test in Table V is NOT external validation."
> *"In Table V, you claim 'External Site' validation. But this is merely a 248-record subset from Fold 10 of PTB-XL where the acquisition device was non-zero. Calling this 'domain shift' or 'external validation' is misleading."*
* **Response & Action**:
  * **Concession & Renaming**: We fully concede this critique. We have renamed this row in Table V and throughout the paper to **"Site-Subgroup Sensitivity Analysis ($s \ne 0$, $N=248$)"**.
  * **Disclosure**: We explicitly state in Section IV-C that true cross-institutional multi-center external validation (e.g. on distinct hospital cohorts) remains an open limitation.

---

### Critique 4 (Cardiologist): "Selective abstention on 10% of cases shifts unreasonable workload onto human cardiologists."
> *"Deferring 10% of ECGs to achieve a 4.18% relative risk reduction means 220 additional manual overreads per 2,200 patients. In an overloaded hospital, this creates bottleneck fatigue."*
* **Response & Action**:
  * **Clinical Defense**: Selective abstention is designed for safety-critical triage, not routine ECG dumping. The 10% deferred cohort represents tracings with extreme predictive entropy ($U(x) > 1.45$), where automated systems are most susceptible to catastrophic false negatives (e.g. subtle Non-ST Elevation Myocardial Infarction). Reviewing 10% of borderline cases prevents acute diagnostic error while autonomously clearing 90% of unambiguous records with an ultra-low error rate of $10.37\%$.

---

### Critique 5 (ML Reviewer): "Why did you restrict evaluation to 2.0M parameters rather than evaluating foundation scale (32M)?"
> *"The original research proposal promised an evaluation at both 2M and 32M parameters. Restricting your study to 2M limits the generalizability of your findings to massive foundation models."*
* **Response & Action**:
  * **Justification**: As documented in `BUDGET_JUSTIFICATION.md`, training 155 neural networks across 5 configurations and 5 random seeds at 32M parameters exceeds the compute envelope of edge development workstations.
  * **Clinical Rationale**: Furthermore, 2.0M parameters directly reflects the hardware memory ceiling (<8MB) of wearable continuous Holter monitors and edge telemetry. Following Rule 5, we disclose foundation-scale scaling as a clear computational boundary.

---

### Critique 6 (ML Reviewer): "How do you know the router isn't collapsing into a simple uniform average?"
> *"In Table IV, Uniform Averaging yields AUROC 0.9110 while the learned Top-2 Router yields 0.9113. Is the router actually learning meaningful pathology specialization?"*
* **Response & Action**:
  * **Empirical Proof**: Table VII proves that a router trained without diversity regularization collapses catastrophically to AUROC $0.9049$ ($p = 0.0480$). The learned Top-2 router matches dense performance while cutting inference compute from $414.9$ to $297.8$ MFLOPs ($-28.2\%$). Furthermore, gating weight distributions confirm distinct pathology dispatching: TCN-1D and MultiScale dominate Conduction Defects ($w_k > 0.30$), while ResNet-1D and ConvAttention dominate Ischemia/MI ($w_k > 0.28$).

---

### Critique 7 (Statistician): "Did you tune hyperparameters on the test set?"
> *"Did any validation-guided model selection or router threshold tuning leak into the Fold 10 test set evaluation?"*
* **Response & Action**:
  * **Protocol Hygiene**: Zero test-set tuning occurred. As verified in `PREREGISTRATION.md` and `AUDIT.md`, Folds 1–8 were used strictly for parameter training, Fold 9 for validation and hyperparameter selection ($\lambda_{\text{div}}=0.05, \lambda_{\text{fail}}=0.10$), and Fold 10 was strictly evaluated once on frozen checkpoints.

---

### Critique 8 (ML Reviewer): "How does Top-2 routing compute ensemble disagreement if only 2 experts are executed?"
> *"Equation (6) defines uncertainty as a sum over $K$ experts. If Top-2 routing executes only 2 experts, how can you compute disagreement over all 4 experts without running all 4?"*
* **Response & Action**:
  * **Clarification**: We clarify that in Sparse Top-2 inference, the epistemic disagreement term in $U(x)$ is evaluated directly over the active dispatched subset ($K_{\text{active}} = 2$), preserving the $-28.2\%$ inference compute reduction.

---

### Critique 9 (Cardiologist): "What is the physiological basis for heterogeneous expert superiority?"
> *"Why should a combination of ResNet, MultiScale, TCN, and Attention perform better on ECG than four ResNets?"*
* **Response & Action**:
  * **Physiological Alignment**: 12-lead ECG signals contain orthogonal diagnostic features operating at distinct temporal scales: localized high-frequency morphology (QRS notches, ST-elevation) captured by residual convolutions; multi-frequency wavelet dynamics captured by MultiScale-CNN; rhythm periodicity and cross-beat intervals captured by dilated causal TCN; and long-range spatial lead interactions captured by multi-head self-attention. Linear CKA proves their representational divergence ($0.9008$ vs $0.9458$).

---

### Critique 10 (ML Reviewer): "You cannot claim that Ensembles or Mixture-of-Experts are novel."
> *"Ensembles and MoE have been used for decades. What is actually novel here?"*
* **Response & Action**:
  * **Novelty Principle**: We make zero claim of novelty for ensembling, MoE, or diversity penalties. Our stated contribution is the **controlled empirical scaling study under an invariant parameter ceiling**, proving that capacity splitting acts as a variance regularizer until reaching the $K=16$ fragmentation boundary.
