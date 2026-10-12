"""
Script to generate the 8 verified research Jupyter notebooks for MOSAIC-Med using nbformat.
Incorporates all methodological, statistical, and computational reconciliations.
"""

import os
import nbformat as nbf

def create_cell(cell_type, content):
    if cell_type == "markdown":
        return nbf.v4.new_markdown_cell(content)
    elif cell_type == "code":
        return nbf.v4.new_code_cell(content)

# -----------------------------------------------------------------------------
# 01_data_preprocessing.ipynb
# -----------------------------------------------------------------------------
def make_nb_01():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        create_cell("markdown", """# Notebook 01: PTB-XL Dataset Loading, Deduplication & Patient-Level Splitting
### MOSAIC-Med Research Study: Investigating Small-Model Multiplicity vs Scale in Healthcare AI

---

### Overview & Objectives
This notebook establishes the clinical testbed for the MOSAIC-Med study using the PTB-XL 12-lead electrocardiography benchmark (Wagner et al., *Scientific Data* 2020; Strodthoff et al., *IEEE JBHI* 2021).
Specifically, we:
1. **Document Dataset Version & Deduplication**:
   - Raw PTB-XL v1.0.1 contained 21,837 records across 18,885 patients.
   - Official PTB-XL v1.0.2 / v1.0.3 changelogs identified and removed **38 duplicate recordings across 16 patients** (`ptbxl/ptbxl_v103_changelog.txt`), yielding exactly **21,799 unique recordings across 18,869 distinct patients**.
2. **Implement Patient-Stratified Splits**:
   - **Training Set (Folds 1–8)**: 17,418 records (15,023 patients)
   - **Validation Set (Fold 9)**: 2,183 records (1,942 patients)
   - **Test Set (Fold 10)**: 2,198 records (1,904 patients)
   - Programmatically verify strict patient separation (zero patient identity leakage).
3. **Verify Signal Preprocessing & Normalization**:
   - Zero-mean unit-variance standardization evaluated per 10-second recording instance across time ($L=1000$) independently per lead:
     $$x'_c(t) = \\frac{x_c(t) - \\mu_c}{\\max(\\sigma_c, 10^{-6})}, \\quad \\text{clipped to } [-10, 10]$$
   - No dataset-level training statistics are used during inference, ensuring total independence.
4. **Clarify Acquisition Site Shift**:
   - Non-primary acquisition sites (`site != 0.0`) within held-out Test Fold 10 serve as an internal domain shift stress test.
"""),
        create_cell("code", """import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scratch.py.data import (
    load_ptbxl_metadata, cache_ptbxl_dataset,
    CLASSES, CLASS_NAMES
)
from scratch.py.plotting import set_ieee_style, COLORS

set_ieee_style()
print("Libraries and MOSAIC-Med data modules loaded successfully.")
"""),
        create_cell("markdown", """## 1. Metadata Inspection & Deduplication Verification
Let's inspect the database metadata and verify the 21,799 records and 18,869 unique patients.
"""),
        create_cell("code", """# Load PTB-XL database metadata
df_meta = load_ptbxl_metadata('ptbxl')
print(f"Total Unique Records: {len(df_meta):,}")
print(f"Distinct Patients:    {df_meta['patient_id'].nunique():,}")
print(f"Acquisition Sites:    {df_meta['site'].nunique()} distinct sites")

# Inspect superclass prevalence summary
summary_table = pd.read_csv('results/tables/data_summary.csv')
summary_table
"""),
        create_cell("markdown", """## 2. Patient-Level Stratification Verification (Zero Leakage Check)
To guarantee clinical validity and eliminate optimistic bias, records from the same patient must never cross train, validation, or test partitions.
Let's verify mathematically that there is zero patient overlap.
"""),
        create_cell("code", """train_patients = set(df_meta[df_meta['strat_fold'] <= 8]['patient_id'])
val_patients = set(df_meta[df_meta['strat_fold'] == 9]['patient_id'])
test_patients = set(df_meta[df_meta['strat_fold'] == 10]['patient_id'])

overlap_train_val = len(train_patients.intersection(val_patients))
overlap_train_test = len(train_patients.intersection(test_patients))
overlap_val_test = len(val_patients.intersection(test_patients))

print(f"Train Records: {len(df_meta[df_meta['strat_fold'] <= 8]):,} | Distinct Patients: {len(train_patients):,}")
print(f"Val Records:   {len(df_meta[df_meta['strat_fold'] == 9]):,}  | Distinct Patients: {len(val_patients):,}")
print(f"Test Records:  {len(df_meta[df_meta['strat_fold'] == 10]):,}  | Distinct Patients: {len(test_patients):,}")
print("---")
print(f"Train/Val Patient Overlap:  {overlap_train_val} (Required: 0)")
print(f"Train/Test Patient Overlap: {overlap_train_test} (Required: 0)")
print(f"Val/Test Patient Overlap:   {overlap_val_test} (Required: 0)")

assert overlap_train_val == 0 and overlap_train_test == 0 and overlap_val_test == 0, "Patient leakage detected!"
print(">> Patient-level stratification verified with zero patient overlap across splits.")
"""),
        create_cell("markdown", """## 3. Signal Standardization & Cached Arrays
Raw ECG waveforms are standardized instance-by-instance per lead across the temporal axis ($L=1000$).
Let's inspect preprocessed arrays from disk cache.
"""),
        create_cell("code", """# Load preprocessed arrays from cache
data = cache_ptbxl_dataset('ptbxl', 'scratch/cache')
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

print(f"X_train shape: {X_train.shape} (float32)")
print(f"X_val shape:   {X_val.shape} (float32)")
print(f"X_test shape:  {X_test.shape} (float32)")

# Plot 12 leads of a representative sample
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
fig, axes = plt.subplots(6, 2, figsize=(7.16, 5.0), sharex=True)
sample_idx = 0
time_axis = np.linspace(0, 10, 1000)

for idx, lead in enumerate(lead_names):
    r, c = divmod(idx, 2)
    ax = axes[r, c]
    ax.plot(time_axis, X_train[sample_idx, idx, :], color=COLORS['primary'], linewidth=0.9)
    ax.set_ylabel(lead, fontsize=7.5)
    ax.grid(True, alpha=0.2)
axes[-1, 0].set_xlabel("Time (seconds)")
axes[-1, 1].set_xlabel("Time (seconds)")
fig.suptitle("Preprocessed 12-Lead ECG Sample (100 Hz, 1000 Samples)", fontsize=9.5)
plt.tight_layout()
plt.show()
"""),
        create_cell("markdown", """## 4. Internal Acquisition Site Domain Shift Documentation
> [!NOTE]
> In held-out Test Fold 10 ($N=2,198$), records acquired at non-primary recording sites (`site != 0.0`, $N = 1,711$) exhibit hardware and sensor configuration variations.
> This subset is used as an **internal acquisition site transferability stress test**, but does NOT constitute an independent external hospital cohort.
""")
    ]
    with open("01_data_preprocessing.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created 01_data_preprocessing.ipynb")

# -------------------------------------------------------------
# 02_fixed_budget_scaling.ipynb
# -------------------------------------------------------------
def make_nb_02():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        create_cell("markdown", """# Notebook 02: Fixed-Budget Model Scaling (1 vs 2 vs 4 vs 8 vs 16 Experts)
### MOSAIC-Med Research Study: Capacity Budget Allocation under 2.0M Total Parameters

---

### Core Scientific Question & Invariance Envelope
Under a fixed total parameter budget of **~2.0 Million parameters**, how should capacity be distributed?
- **Parameter Invariance Envelope**: $\\mathcal{B}_{\\text{target}} = 2.00 \\times 10^6 \\pm 0.06 \\times 10^6$ parameters (tolerance $\\pm 3.0\\%$).
- Channel widths $W$ are discrete integers ($W \\in \\{16, 23, 32, 46, 65\\}$), preventing exact integer convergence to $2,000,000$.
- Configurations:
  - **$K = 1$** ($W=65$): $2,019,945$ params ($+0.997\\%$) | 123.9 MFLOPs
  - **$K = 2$** ($W=46$): $2 \\times 1,015,225 = 2,030,450$ params ($+1.523\\%$) | 126.9 MFLOPs
  - **$K = 4$** ($W=32$): $4 \\times 493,893 = 1,975,572$ params ($-1.221\\%$) | 126.8 MFLOPs
  - **$K = 8$** ($W=23$): $8 \\times 256,869 = 2,054,952$ params ($+2.748\\%$) | 136.2 MFLOPs
  - **$K = 16$** ($W=16$): $16 \\times 125,605 = 2,009,680$ params ($+0.484\\%$) | 139.8 MFLOPs

All configurations are trained across **5 independent random seeds (42–46)** to isolate structural capacity allocation from initialization stochasticity.
"""),
        create_cell("code", """import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display

print("Modules loaded.")
"""),
        create_cell("markdown", """## 1. Scaling Results Table (Seed-Level and Aggregated)
Below are the individual seed results and the aggregated summary statistics (mean ± SD) for all 5 configurations.
"""),
        create_cell("code", """scaling_df = pd.read_csv('results/tables/scaling_results.csv')
scaling_summary = pd.read_csv('results/tables/scaling_summary.csv')

# Add explicit percentage deviation column from 2.0M target
target_budget = 2_000_000
scaling_summary['Budget Deviation (%)'] = ((scaling_summary['Total Parameters'] - target_budget) / target_budget * 100).round(3)

print("Fixed-Budget Scaling Performance Across 5 Seeds:")
scaling_summary[['Configuration', 'Total Parameters', 'Budget Deviation (%)', 'MFLOPs', 'Macro AUROC', 'Macro AUPRC', 'Brier Score', 'Latency (ms)']]
"""),
        create_cell("markdown", """## 2. Discrimination and Calibration Scaling Curves
Let's inspect the scaling curves for AUROC discrimination and Brier score calibration under the fixed 2.0M parameter budget.
"""),
        create_cell("code", """display(Image('results/images/scaling_tradeoff.png'))
"""),
        create_cell("markdown", """## 3. Analysis & Key Insights
1. **Multiplicity Superiority (H1 Supported)**: Partitioning the 2.0M parameter budget into $K=2$ ($0.9085 \\pm 0.0007$) and $K=4$ ($0.9084 \\pm 0.0009$) compact experts statistically outperforms the monolithic $K=1$ architecture ($0.9059 \\pm 0.0010$, DeLong $p = 3.63 \\times 10^{-4}$).
2. **Probability Calibration**: Brier scores improve from $0.0900$ ($K=1$) to $0.0881$ ($K=4$), confirming that model splitting acts as an effective variance regularizer.
3. **Over-Fragmentation Boundary ($K=16$)**: At $K=16$ (125K parameters per expert), individual network capacity is insufficient to resolve multi-lead morphological patterns, degrading AUROC to $0.9048 \\pm 0.0005$ and tripling latency to $6.78$ ms.
4. **Capacity vs Compute**: Parameter counts are matched within $\\pm 3.0\\%$, but sequential inference FLOPs scale slightly from 123.9 to 139.8 MFLOPs due to per-network stem overheads.
""")
    ]
    with open("02_fixed_budget_scaling.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created 02_fixed_budget_scaling.ipynb")

# -------------------------------------------------------------
# 03_architectural_diversity.ipynb
# -------------------------------------------------------------
def make_nb_03():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        create_cell("markdown", """# Notebook 03: Architectural Diversity & Centered Kernel Alignment (CKA)
### MOSAIC-Med Research Study: Homogeneous vs Heterogeneous Expert Populations

---

### Hypothesis 2 & Parameter-Matched Control
Does combining models with **distinct inductive biases** (heterogeneous architectures) produce complementary feature representations that surpass homogeneous ensembles of identical copies?

We compare two 4-expert populations under an invariant ~2.0M parameter budget (+0.92% matching margin):
1. **Homogeneous Population (4$\\times$ResNet)**: $1,975,572$ parameters (126.8 MFLOPs)
2. **Heterogeneous Population**: $1,993,688$ parameters (413.7 MFLOPs)
   - **ResNet-1D** (493,893 params, 31.69 MFLOPs) — Deep residual temporal convolutions
   - **MultiScale-CNN** (508,909 params, 50.72 MFLOPs) — Parallel multi-resolution temporal kernels (5, 11, 23 samples)
   - **TCN-1D** (490,661 params, 245.86 MFLOPs) — Exponentially dilated causal convolutions ($d \\in \\{1, 2, 4, 8\\}$)
   - **ConvAttention-1D** (500,225 params, 85.40 MFLOPs) — Convolutional stem + 2-layer self-attention Transformer

Representation diversity is quantified using Linear **Centered Kernel Alignment (CKA)** on penultimate activations over the held-out test cohort.
"""),
        create_cell("code", """import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from IPython.display import Image, display

print("Modules loaded.")
"""),
        create_cell("markdown", """## 1. Parameter-Matched Statistical Comparison (Seeds 42–46)
"""),
        create_cell("code", """matched_df = pd.read_csv('results/tables/parameter_matched_control.csv')
print("Parameter-Matched Control Results (5 Seeds, df=4):")
matched_df[['Metric', 'Homogeneous 4xResNet (Mean ± SD)', 'Heterogeneous Matched (Mean ± SD)', 'Delta (Hetero - Homo)', 'Raw p-value', 'Holm-adjusted p-value', 'Statistically Significant (α=0.05)']]
"""),
        create_cell("markdown", """## 2. Centered Kernel Alignment (CKA) Representation Analysis
A lower CKA score between two models indicates higher feature orthogonality.
"""),
        create_cell("code", """cka_df = pd.read_csv('results/tables/cka_comparison.csv')
print("Mean Pairwise Linear CKA Scores:")
print(f"Homogeneous 4xResNet Mean CKA:  {cka_df[cka_df['population'] == 'Homogeneous']['mean_cka'].values[0]:.4f}")
print(f"Heterogeneous Matched Mean CKA: {cka_df[cka_df['population'] == 'Heterogeneous']['mean_cka'].values[0]:.4f}")

display(Image('results/images/cka_representation_diversity.png'))
"""),
        create_cell("markdown", """## 3. Scientific Findings & Statistical Rigor
- **Observed Trend (H2)**: The heterogeneous ensemble achieves higher discrimination (AUROC $0.9110 \\pm 0.0006$ vs $0.9084 \\pm 0.0008$) and calibration (Brier $0.0867 \\pm 0.0005$ vs $0.0881 \\pm 0.0002$).
- **Statistical Significance Status**: Across 5 seeds, paired $t$-test yields raw $p = 0.0164$, which adjusts to **Holm $p = 0.0577$** across the $m=4$ metric family (AUROC, AUPRC, Brier, ECE). Because $p > 0.05$, this result represents a **consistent descriptive improvement trend**, but does not achieve strict significance at the conventional $\\alpha = 0.05$ threshold.
- **CKA Diversity**: Heterogeneous models achieve lower representational similarity ($0.9008$ vs $0.9458$), indicating orthogonal feature spaces. However, CKA characterizes representation structure and should not be treated as a causal proof of performance.
""")
    ]
    with open("03_architectural_diversity.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created 03_architectural_diversity.ipynb")

# -------------------------------------------------------------
# 04_adaptive_routing_and_diversity.ipynb
# -------------------------------------------------------------
def make_nb_04():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        create_cell("markdown", """# Notebook 04: MOSAIC-Med Adaptive Routing & Computational Reconciliation
### Dynamic Patient-Specific Expert Gating, Loss Regularization, and FLOP Accounting

---

### Computational Accounting Reconciliation
A critical distinction must be made between computational savings **relative to dense 4-expert ensembling** versus **relative to the monolithic baseline**:
- **Single Large Model (1$\\times$2M)**: 123.9 MFLOPs
- **Heterogeneous Uniform Average**: $31.69 + 50.72 + 245.86 + 85.40 = \\mathbf{413.7\\text{ MFLOPs}}$
- **Dense Learned Router**: $413.7 + 1.19\\text{ (router)} = \\mathbf{414.9\\text{ MFLOPs}}$
- **Sparse Top-2 Router**: $1.19\\text{ (router)} + 296.6\\text{ (mean active Top-2 pair)} = \\mathbf{297.8\\text{ MFLOPs}}$

**Key Insight**: Sparse Top-2 routing cuts inference computation by **28.2% relative to dense heterogeneous ensembling** (297.8 vs 414.9 MFLOPs). However, it requires **$2.40\\times$ more computation than the single monolithic model** (123.9 MFLOPs).

The router objective is regularized by:
$$\\mathcal{L}_{\\text{total}} = \\mathcal{L}_{\\text{diag}} + \\lambda_{\\text{div}} \\mathcal{L}_{\\text{div}} + \\lambda_{\\text{fail}} \\mathcal{L}_{\\text{fail}}$$
where $\\mathcal{L}_{\\text{div}} = -\\mathcal{H}(\\bar{w})$ prevents expert collapse by maximizing mean gating entropy.
"""),
        create_cell("code", """import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
from IPython.display import Image, display

print("Modules loaded.")
"""),
        create_cell("markdown", """## 1. Routing Strategies Comparison Table Across 5 Seeds
"""),
        create_cell("code", """routing_summary = pd.read_csv('results/tables/routing_multiseed_summary.csv')
print("Adaptive Routing Comparison Across 5 Seeds (Mean ± SD):")
routing_summary[['Routing Strategy', 'Active Experts', 'MFLOPs', 'Macro AUROC', 'Macro AUPRC', 'Holm p-value', 'Significant (α=0.05)']]
"""),
        create_cell("markdown", """## 2. Learned Expert Specialization Across Clinical Conditions
Let's inspect how the router distributes weights across the 5 diagnostic classes:
"""),
        create_cell("code", """display(Image('results/images/adaptive_routing_analysis.png'))
"""),
        create_cell("markdown", """## 3. Findings & Rigorous Claim Calibration
- **Descriptive Equivalence (H3)**: The Sparse Top-2 Router ($0.9113 \\pm 0.0008$) achieves diagnostic parity with dense averaging ($0.9110 \\pm 0.0007$) while dynamically reducing FLOPs by 28.2% relative to dense 4-expert inference.
- **Compute Trade-off**: The router enables computational pruning for multi-expert ensembles, but does NOT reduce compute relative to the single monolithic model (297.8 vs 123.9 MFLOPs).
- **Specialization**: The router routes Conduction Disturbances (CD) and Hypertrophy (HYP) predominantly to TCN-1D and MultiScale-CNN ($w_k > 0.30$), while ResNet-1D and ConvAttention-1D receive higher weight for MI and STTC.
""")
    ]
    with open("04_adaptive_routing_and_diversity.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created 04_adaptive_routing_and_diversity.ipynb")

# -------------------------------------------------------------
# 05_uncertainty_escalation_abstention.ipynb
# -------------------------------------------------------------
def make_nb_05():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        create_cell("markdown", """# Notebook 05: Predictive Uncertainty Estimation & Selective Abstention
### Multi-Label Predictive Entropy, Ensemble Variance, and Risk-Coverage Optimization

---

### Uncertainty Formulation & Objective
Predictive uncertainty $U(x)$ combines aleatoric predictive entropy and epistemic ensemble disagreement:
$$U(x) = \\sum_{m=1}^M \\mathcal{H}(\\hat{y}_m(x)) + \\alpha \\cdot \\sum_{m=1}^M \\frac{1}{K}\\sum_{k=1}^K (\\sigma(z_{k, m}(x)) - \\hat{y}_m(x))^2$$

- **First Term (Aleatoric)**: Multi-label binary predictive entropy $\\mathcal{H}(p) = -p \\log_2 p - (1-p) \\log_2 (1-p)$.
- **Second Term (Epistemic)**: Variance across expert predictions relative to ensemble mean $\\hat{y}_m$.
- **Validation Tuning**: The scaling coefficient $\\alpha = 2.0$ was tuned strictly on **Validation Fold 9** to **minimize the Area Under the Risk-Coverage Curve (AURC)**, since risk corresponds to Hamming classification error rate.
- **Threshold Selection**: Deferral thresholds corresponding to coverage levels $c \\in \\{0.80, 0.85, 0.90, 0.95\\}$ were fitted on Validation Fold 9 quantiles and subsequently applied to held-out Test Fold 10.
"""),
        create_cell("code", """import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
from IPython.display import Image, display

print("Modules loaded.")
"""),
        create_cell("markdown", """## 1. Selective Classification Risk Summary Across 5 Seeds
"""),
        create_cell("code", """selective_summary = pd.read_csv('results/tables/selective_risk_multiseed_summary.csv')
print("Multi-Label Hamming Risk (%) and Relative Risk Reduction Across 5 Seeds:")
selective_summary[['Coverage (c)', 'Single Large Risk (%)', 'MOSAIC-Med Risk (%)', 'Relative Risk Reduction (%)', 'Holm p-value', 'Significant (α=0.05)']]
"""),
        create_cell("markdown", """## 2. Risk vs Coverage Curve
"""),
        create_cell("code", """display(Image('results/images/risk_coverage.png'))
"""),
        create_cell("markdown", """## 3. Clinical Abstention Insights (H4 Supported)
- **Empirical Risk Reduction**: Deferring the $10\\%$ most ambiguous records ($c=90\\%$) yields an empirical **$4.18\\% \\pm 2.02\\%$ relative risk reduction** relative to the single large model (Hamming risk drops from $10.83\\%$ to $10.37\\%$, Holm $p = 9.34 \\times 10^{-3}$).
- **Monotonic Ordering**: Diagnostic risk decreases monotonically as coverage is reduced, confirming that $U(x)$ effectively isolates ambiguous tracings.
- **Scope Note**: Selective deferral is demonstrated as an algorithmic evaluation on retrospective data, not a clinically deployed or validated bedside triage system.
""")
    ]
    with open("05_uncertainty_escalation_abstention.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created 05_uncertainty_escalation_abstention.ipynb")

# -------------------------------------------------------------
# 06_robustness_and_external_validation.ipynb
# -------------------------------------------------------------
def make_nb_06():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        create_cell("markdown", """# Notebook 06: Clinical Robustness Stress Testing & Domain Shift
### Evaluating High-Frequency Noise, Baseline Wander, Missing Leads, and Held-Out Acquisition Sites

---

### Mathematical Formulations of Stress Corruptions
All perturbation parameters were pre-specified based on clinical ECG artifact literature (Strodthoff et al., 2021) prior to test cohort evaluation:
1. **High-Frequency EMG / Tremor Noise**:
   $$X_{\\text{EMG}} = X + \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, \\sigma^2), \\quad \\sigma = 0.15$$
2. **Low-Frequency Baseline Wander**:
   $$X_{\\text{drift}}[c, t] = X[c, t] + A \\sin\\left(2\\pi f \\frac{t}{f_s} + \\phi_c\\right)$$
   where $A = 0.50$, $f = 0.20$ Hz, $f_s = 100$ Hz, and $\\phi_c \\sim \\mathcal{U}(0, 2\\pi)$ per lead.
3. **Missing Precordial Leads (V1–V3)**:
   $$X_{\\text{precordial}}[c, t] = 0 \\quad \\forall c \\in \\{6, 7, 8\\}, \\quad t \\in [0, 1000)$$
4. **Missing Limb Leads (I–III)**:
   $$X_{\\text{limb}}[c, t] = 0 \\quad \\forall c \\in \\{0, 1, 2\\}, \\quad t \\in [0, 1000)$$
5. **Acquisition Site Domain Shift**:
   Held-out test records acquired from non-primary clinical recording sites (`site != 0.0`) within PTB-XL.
"""),
        create_cell("code", """import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
from IPython.display import Image, display

print("Modules loaded.")
"""),
        create_cell("markdown", """## 1. Robustness Benchmark Table Across 5 Seeds
"""),
        create_cell("code", """robustness_summary = pd.read_csv('results/tables/robustness_multiseed_summary.csv')
print("Diagnostic Retention Under Clinical Degradations Across 5 Seeds (Mean ± SD):")
robustness_summary[['Condition', 'Single Large AUROC', 'MOSAIC-Med AUROC', 'Delta AUROC (M - S)', 'Holm p-value', 'Significant (α=0.05)']]
"""),
        create_cell("markdown", """## 2. Retention Comparison Under Stress Corruptions
"""),
        create_cell("code", """display(Image('results/images/robustness_stress.png'))
"""),
        create_cell("markdown", """## 3. Robustness Findings & Scope Calibration
- **Resilience to Sensor Loss**: Under precordial lead loss (V1–V3), Single Large drops to $0.8694 \\pm 0.0021$, whereas MOSAIC-Med retains $0.8801 \\pm 0.0030$ ($\\Delta = +0.0106$, Holm $p = 8.82 \\times 10^{-4}$). Ensembling orthogonal inductive biases provides compensatory feature extraction across unaffected leads.
- **Acquisition Site Shift**: Under held-out non-primary site records (`site != 0`), MOSAIC-Med achieves AUROC $0.9033 \\pm 0.0009$ vs $0.8978 \\pm 0.0008$ ($p = 3.72 \\times 10^{-5}$).
- **Validation Scope**: This held-out site evaluation tests internal acquisition transferability within PTB-XL; prospective validation on independent hospital cohorts remains required before clinical claims can be made.
""")
    ]
    with open("06_robustness_and_external_validation.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created 06_robustness_and_external_validation.ipynb")

# -------------------------------------------------------------
# 07_ablation_studies.ipynb
# -------------------------------------------------------------
def make_nb_07():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        create_cell("markdown", """# Notebook 07: Factorial Regularization Ablation Studies
### Step-by-Step Analysis of Diagnostic (L_diag), Diversity (L_div), and Failure (L_fail) Losses

---

### Objective & Experimental Design
To evaluate the contribution of each regularization loss term on frozen heterogeneous experts across 3 random seeds:
1. **Variant 1: Uniform Averaging (Baseline)** — Equal 0.25 weighting across all 4 experts.
2. **Variant 2: Router ($\\mathcal{L}_{\\text{diag}}$ only)** — Router trained with standard BCE loss.
3. **Variant 3: Router ($\\mathcal{L}_{\\text{diag}} + \\mathcal{L}_{\\text{div}}$)** — Adds mean gating entropy maximization to prevent expert collapse.
4. **Variant 4: Router ($\\mathcal{L}_{\\text{diag}} + \\mathcal{L}_{\\text{fail}}$)** — Adds expert failure penalty.
5. **Variant 5: All Terms ($\\mathcal{L}_{\\text{diag}} + \\mathcal{L}_{\\text{div}} + \\mathcal{L}_{\\text{fail}}$)** — Complete composite objective.
"""),
        create_cell("code", """import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
from IPython.display import Image, display

print("Modules loaded.")
"""),
        create_cell("markdown", """## 1. Loss Ablation Results Table Across 3 Seeds
"""),
        create_cell("code", """loss_ablation = pd.read_csv('results/tables/loss_ablation_summary.csv')
print("Loss-Term Regularization Ablation Across 3 Seeds (Mean ± SD):")
loss_ablation[['Ablation Variant', 'Macro AUROC', 'Macro AUPRC', 'Brier Score', 'Holm-adjusted p-value', 'Significant (α=0.05)']]
"""),
        create_cell("markdown", """## 2. Factorial Ablation Waterfall Diagram
"""),
        create_cell("code", """display(Image('results/images/ablation_waterfall.png'))
"""),
        create_cell("markdown", """## 3. Conclusions & Critical Nuance
- **Mitigating Router Collapse**: Training the router with diagnostic loss alone drops AUROC to $0.9049 \\pm 0.0017$ due to gating collapse into a single dominant expert. Adding $\\mathcal{L}_{\\text{div}}$ recovers performance to $0.9088 \\pm 0.0015$.
- **Router vs Uniform Averaging**: When all 4 experts are executed, dense Uniform Averaging ($0.9108 \\pm 0.0009$) achieves slightly higher AUROC than the full router ($0.9090 \\pm 0.0013$). The primary role of learned routing is NOT to outperform dense averaging when compute is unconstrained, but to enable **sparse dynamic Top-2 execution** (297.8 MFLOPs), pruning compute by 28.2%.
""")
    ]
    with open("07_ablation_studies.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created 07_ablation_studies.ipynb")

# -------------------------------------------------------------
# 08_statistical_analysis_and_synthesis.ipynb
# -------------------------------------------------------------
def make_nb_08():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        create_cell("markdown", """# Notebook 08: Formal Statistical Testing & Executive Synthesis
### DeLong Tests, McNemar Disagreements, Stratified Bootstrap CIs, and Hypothesis Alignment

---

### Hierarchical Statistical Testing Protocol
To address peer review rigor:
1. **Within-Run Cohort Discrimination**: DeLong's test on paired test-set predictions ($N = 2,198$) comparing correlated ROC curves.
2. **Sample-Level Decision Agreement**: Continuity-corrected McNemar's $\\chi^2$ test on binary decisions at threshold 0.5.
3. **Patient-Level Bootstrap**: 2,000 resamples computing empirical 95% percentile confidence intervals for AUROC, AUPRC, F1, Brier score, and paired differences.
4. **Across-Seed Variance**: Paired Student's $t$-test across the 5 independent seeds ($df = 4$) with Holm-Bonferroni correction over the $m=4$ metric family.
"""),
        create_cell("code", """import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
from IPython.display import Image, display

print("Modules loaded.")
"""),
        create_cell("markdown", """## 1. Primary Statistical Comparison Table
"""),
        create_cell("code", """stat_df = pd.read_csv('results/tables/main_statistical_comparison.csv')
print("Primary Statistical Comparison (MOSAIC-Med vs Single Large):")
stat_df[['Comparison', 'Metric', 'MOSAIC-Med (95% CI)', 'Single Large (95% CI)', 'Difference (95% CI)', 'Test Method', 'p-value', 'Significant (α=0.05)']]
"""),
        create_cell("markdown", """## 2. Reliability & Calibration Analysis
Reliability diagram and empirical confidence histogram comparing Single Large (2M) vs MOSAIC-Med:
"""),
        create_cell("code", """display(Image('results/images/reliability_calibration.png'))
"""),
        create_cell("markdown", """## 3. Executive Top-Line Figure (IEEE Double-Column)
"""),
        create_cell("code", """display(Image('results/images/topline_mosaic_vs_single.png'))
"""),
        create_cell("markdown", """## 4. Final Hypothesis Alignment Matrix

| Hypothesis | Theoretical Focus | Empirical Status | Formal Stance |
|---|---|---|---|
| **H1: Multiplicity Superiority** | $K=2, 4$ vs Monolithic $K=1$ | **Statistically Validated** | Supported ($p = 3.63 \\times 10^{-4}$ DeLong, $p = 1.09 \\times 10^{-3}$ Holm). Intermediate capacity splitting acts as a variance regularizer. |
| **H2: Diversity Benefit** | Heterogeneous vs Homogeneous | **Descriptive Trend** | Consistent improvement trend ($p = 0.0577$ Holm), non-significant at $\\alpha=0.05$. Lower linear CKA ($0.9008$ vs $0.9458$) characterizes feature orthogonality. |
| **H3: Sparse Routing Efficiency** | Top-2 Router vs Dense Hetero | **Supported Trade-off** | Dynamic Top-2 gating achieves descriptive parity with dense ensembling ($0.9113$ vs $0.9110$) while reducing inference FLOPs by $28.2\\%$. Requires $2.40\\times$ more FLOPs than monolithic baseline. |
| **H4: Selective Risk Reduction** | Uncertainty Abstention | **Supported in Retrospective Cohort** | Deferring $10\\%$ most ambiguous tracings achieves $4.18\\%$ relative risk reduction relative to monolithic baseline at $90\\%$ coverage ($p = 0.0093$). |
""")
    ]
    with open("08_statistical_analysis_and_synthesis.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created 08_statistical_analysis_and_synthesis.ipynb")

if __name__ == '__main__':
    make_nb_01()
    make_nb_02()
    make_nb_03()
    make_nb_04()
    make_nb_05()
    make_nb_06()
    make_nb_07()
    make_nb_08()
    print("All 8 research notebooks created and synchronized successfully.")
