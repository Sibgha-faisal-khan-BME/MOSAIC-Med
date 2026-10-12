# MOSAIC-Med: Multiplicity and Architectural Diversity Outperform Monolithic Scale in Resource-Constrained Electrocardiogram Diagnosis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PTB-XL](https://img.shields.io/badge/Dataset-PTB--XL_v1.0.3-green.svg)](https://physionet.org/content/ptb-xl/1.0.3/)
[![Release](https://img.shields.io/badge/release-v1.0.0-purple.svg)](https://github.com/Sibgha-faisal-khan-BME/MOSAIC-Med/releases/tag/v1.0.0)

Official code, configurations, data processing pipelines, and reproduction scripts for the paper:
> **MOSAIC-Med: Multiplicity and Architectural Diversity Outperform Monolithic Scale in Resource-Constrained Electrocardiogram Diagnosis**  
> *Sibgha Faisal, Hamna Tanveer, Fatima-Tu-Zahra†, Mahad Aftab†* († Supervisors)  
> *International Conference on Intelligent Computing (ICIC 2026)*

---

## 1. Abstract & Key Findings

Deep learning models for medical diagnostics predominantly emphasize monolithic parameter scaling. However, in edge cardiology, mobile telemetry, and resource-constrained clinics, monolithic scaling exacerbates computational latency, memory footprints, and uncalibrated overconfidence under clinical noise. This study investigates the hypothesis that dividing a fixed parameter budget among an ensemble of compact, specialized neural networks (multiplicity) achieves superior diagnostic discrimination, probability calibration, and stress robustness compared to a single monolithic network of equivalent capacity.

Using the standardized **PTB-XL 12-lead electrocardiography (ECG)** benchmark across 21,799 clinical records and five diagnostic superclasses, a fixed total capacity budget of **~2.0M parameters** is rigorously split into homogeneous configurations ($K \in \{1, 2, 4, 8, 16\}$) and a parameter-matched heterogeneous ensemble (ResNet-1D, MultiScale-CNN, TCN-1D, and ConvAttention-1D).

### Key Empirical Findings:
1. **Multiplicity Hypothesis Confirmed**: Intermediate model splitting ($K=2$: AUROC $0.9085 \pm 0.0007$; $K=4$: $0.9084 \pm 0.0009$) statistically outperforms the monolithic baseline ($K=1$: $0.9059 \pm 0.0010$, DeLong $p = 3.63 \times 10^{-4}$) under an invariant ~2.0M parameter budget.
2. **Diversity Benefit Trend**: Combining four distinct inductive biases shows a consistent improvement trend (adjusted $p = 0.058$) over homogeneous replication (AUROC $0.9110$ vs $0.9084$; Brier score $0.0867$ vs $0.0881$), driven by reduced representational similarity (linear CKA $0.9008$ vs $0.9458$).
3. **Over-Fragmentation Boundary**: Dividing the budget into $K=16$ (125k parameters/expert) erodes individual feature quality, dropping AUROC to $0.9048$ and tripling latency to $6.78$ ms.
4. **Adaptive Sparse Routing**: A learned Top-2 router matches dense ensemble accuracy ($0.9113$ vs $0.9110$) while reducing inference computation by $28.2\%$ (297.8 MFLOPs vs 414.9 MFLOPs).
5. **Selective Abstention Safety**: Deferring the $10\%$ most ambiguous records based on predictive uncertainty achieves a **4.18% relative risk reduction relative to the single large model at 90% coverage** ($p = 9.34 \times 10^{-3}$).
6. **Clinical Stress Robustness**: MOSAIC-Med retains superior discrimination across 5 clinical stress tests (EMG noise, baseline wander, precordial lead loss, limb lead loss, and acquisition site shift).

---

## 2. Repository Structure

```
MOSAIC-Med/
├── 01_data_preprocessing.ipynb                 # Step 1: PTB-XL loading, folds & preprocessing
├── 02_fixed_budget_scaling.ipynb               # Step 2: 1 vs 2 vs 4 vs 8 vs 16 scaling (5 seeds)
├── 03_architectural_diversity.ipynb            # Step 3: Homogeneous vs Heterogeneous & CKA
├── 04_adaptive_routing_and_diversity.ipynb     # Step 4: Learned router, L_div, L_fail & Top-2
├── 05_uncertainty_escalation_abstention.ipynb  # Step 5: Risk-coverage curves & clinical deferral
├── 06_robustness_and_external_validation.ipynb # Step 6: Noise, baseline drift & external sites
├── 07_ablation_studies.ipynb                   # Step 7: Factorial component ablations
├── 08_statistical_analysis_and_synthesis.ipynb # Step 8: DeLong, McNemar, Bootstrap CIs & figures
├── CITATION.cff                                # Formal Citation File Format metadata
├── DISCREPANCIES.md                            # Comprehensive numerical audit log
├── LICENSE                                     # MIT License
├── README.md                                   # Project documentation and reproduction guide
├── requirements.txt                            # Python dependencies
├── latex/                                      # Complete LaTeX manuscript source and class
│   ├── ICIC2026.cls                            # Official ICIC 2026 class file
│   ├── main.tex                                # Paper source file
│   ├── references.bib                          # Verified literature references
│   ├── figures/                                # Figures 1-4 embedded in paper
│   └── main.pdf                                # Compiled PDF manuscript (6 pages)
├── results/
│   ├── images/                                 # 300+ DPI high-resolution figures (PNG & PDF)
│   ├── tables/                                 # Exact numeric result tables (CSV)
│   └── MOSAIC_Med_Paper.docx                   # Word document manuscript version
├── scripts/
│   └── download_ptbxl.py                       # Automated PTB-XL dataset downloader
└── scratch/py/                                 # Modular reproduction and model code
    ├── data.py                                 # Signal pre-processing and clinical artifacts
    ├── models.py                               # Architectures (ResNet, MultiScale, TCN, Attention)
    ├── training.py                             # AMP training routines
    ├── metrics.py                              # AUROC, AUPRC, Brier, ECE, Hamming risk
    ├── stats.py                                # DeLong test, McNemar test, Bootstrap CIs
    ├── generate_hierarchical_prisma.py         # Fig 1 generator
    ├── generate_standalone_figures.py          # Figs 2, 3, 4 generator
    └── generate_word_doc.py                    # Word document generator
```

---

## 3. Installation & Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.11 and 3.12)
- NVIDIA GPU with CUDA support recommended (CPU execution supported)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Sibgha-faisal-khan-BME/MOSAIC-Med.git
cd MOSAIC-Med
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Dataset Download (PTB-XL v1.0.3)

The PTB-XL dataset contains 21,799 12-lead ECG records from 18,869 patients, digitized at 100 Hz and 500 Hz:

### Automated Download:
```bash
python scripts/download_ptbxl.py
```
This script will automatically retrieve PTB-XL v1.0.3 using `wfdb` or fallback to direct PhysioNet zip archive download and extract into `ptbxl/`.

### Manual Download:
Alternatively, download directly from PhysioNet:
1. Visit: https://physionet.org/content/ptb-xl/1.0.3/
2. Download the database archive.
3. Extract into the `ptbxl/` root folder so that `ptbxl/ptbxl_database.csv` and `ptbxl/records100/` are present.

---

## 5. How to Reproduce Each Table and Figure

All results in the paper can be inspected and reproduced either via the interactive Jupyter notebooks or via standalone Python scripts.

### Reproducing Tables:

| Table | Content | Source CSV File | Reproduction Command / Notebook |
|---|---|---|---|
| **Table I** | Cohort Demographics & Diagnostic Prevalence | `results/tables/data_summary.csv` | `01_data_preprocessing.ipynb` |
| **Table II** | Fixed-Budget Multiplicity Performance ($K=1..16$) | `results/tables/scaling_summary.csv` | `02_fixed_budget_scaling.ipynb` |
| **Table III** | Parameter-Matched Control (Homo vs Hetero) | `results/tables/parameter_matched_control.csv` | `python scratch/py/run_parameter_matched_control.py` or `03_architectural_diversity.ipynb` |
| **Table IV** | Adaptive Routing Strategy Comparison | `results/tables/routing_multiseed_summary.csv` | `04_adaptive_routing_and_diversity.ipynb` |
| **Table V** | Clinical Robustness Stress Testing Across 5 Seeds | `results/tables/robustness_multiseed_summary.csv` | `06_robustness_and_external_validation.ipynb` |
| **Table VI** | Selective Classification Multi-Label Risk | `results/tables/selective_risk_multiseed_summary.csv` | `05_uncertainty_escalation_abstention.ipynb` |
| **Table VII** | Loss-Term Regularization Ablation | `results/tables/loss_ablation_summary.csv` | `python scratch/py/run_loss_term_ablation.py` or `07_ablation_studies.ipynb` |

### Reproducing Figures:

| Figure | Content | Output Path | Reproduction Script |
|---|---|---|---|
| **Figure 1** | Fifteen-Step Methodological Flowchart & Real Waveforms | `latex/figures/methodology_flowchart.png` | `python scratch/py/generate_hierarchical_prisma.py` |
| **Figure 2** | Multiplicity Scaling Trade-off ($K=1..16$) | `latex/figures/scaling_tradeoff.png` | `python scratch/py/generate_standalone_figures.py` |
| **Figure 3** | Diagnostic Retention Under Clinical Stress Tests | `latex/figures/robustness_stress.png` | `python scratch/py/generate_standalone_figures.py` |
| **Figure 4** | Selective Classification Risk vs Coverage Curve | `latex/figures/risk_coverage.png` | `python scratch/py/generate_standalone_figures.py` |

To generate Figures 2, 3, and 4 in one run:
```bash
python scratch/py/generate_standalone_figures.py
```

---

## 6. Compiling the LaTeX Manuscript

The paper is typeset using the custom `ICIC2026.cls` class file for the International Conference on Intelligent Computing (ICIC 2026).

To compile the paper using tectonic or standard pdfLaTeX:
```bash
# Using Tectonic (standalone):
scratch/bin/tectonic.exe -o latex/ latex/main.tex

# Using standard pdfLaTeX / latexmk:
cd latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
The resulting compiled PDF is located at `latex/main.pdf` (strictly 6 pages, formatted in IEEE conference layout).

---

## 7. Data and Code Availability

- **PTB-XL Dataset**: Publicly available on PhysioNet at [https://physionet.org/content/ptb-xl/1.0.3/](https://physionet.org/content/ptb-xl/1.0.3/) under the Open Data Commons Attribution License v1.0.
- **Code & Reproducibility**: All models, training scripts, routing algorithms, evaluation notebooks, and figure generation routines are open-source under the MIT License in this repository.

---

## 8. Citation

If you use MOSAIC-Med in your research, please cite our paper:

```bibtex
@inproceedings{faisal2026mosaic,
  title={MOSAIC-Med: Multiplicity and Architectural Diversity Outperform Monolithic Scale in Resource-Constrained Electrocardiogram Diagnosis},
  author={Faisal, Sibgha and Tanveer, Hamna and Fatima-Tu-Zahra and Aftab, Mahad},
  booktitle={International Conference on Intelligent Computing (ICIC)},
  year={2026}
}
```

---

## 9. License

This repository is licensed under the [MIT License](LICENSE).
