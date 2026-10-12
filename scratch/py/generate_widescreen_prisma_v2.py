import subprocess
import pymupdf

prisma_tex = r"""\documentclass[tikz,border=10pt]{standalone}
\usepackage{times}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc}

\begin{document}
\begin{tikzpicture}[
    font=\rmfamily,
    >=Stealth,
    header/.style={rectangle, fill=blue!14, draw=blue!85!black, line width=1.1pt, rounded corners=3pt, text width=7.0cm, minimum height=0.7cm, align=center, font=\bfseries\rmfamily\small},
    headerMid/.style={rectangle, fill=teal!14, draw=teal!85!black, line width=1.1pt, rounded corners=3pt, text width=8.6cm, minimum height=0.7cm, align=center, font=\bfseries\rmfamily\small},
    headerRight/.style={rectangle, fill=purple!14, draw=purple!85!black, line width=1.1pt, rounded corners=3pt, text width=8.6cm, minimum height=0.7cm, align=center, font=\bfseries\rmfamily\small},
    boxLeft/.style={rectangle, fill=blue!3, draw=blue!75!black, line width=1.0pt, rounded corners=4pt, text width=7.0cm, align=left, inner sep=6pt, font=\rmfamily\scriptsize},
    boxA/.style={rectangle, fill=teal!5, draw=teal!85!black, line width=1.0pt, rounded corners=4pt, text width=8.6cm, align=left, inner sep=6pt, font=\rmfamily\scriptsize},
    boxB/.style={rectangle, fill=cyan!5, draw=cyan!85!black, line width=1.0pt, rounded corners=4pt, text width=8.6cm, align=left, inner sep=6pt, font=\rmfamily\scriptsize},
    boxRouter/.style={rectangle, fill=orange!6, draw=orange!85!black, line width=1.0pt, rounded corners=4pt, text width=8.6cm, align=left, inner sep=6pt, font=\rmfamily\scriptsize},
    boxEval/.style={rectangle, fill=purple!4, draw=purple!85!black, line width=1.0pt, rounded corners=4pt, text width=8.6cm, align=left, inner sep=6pt, font=\rmfamily\scriptsize},
    arrow/.style={->, line width=1.4pt, draw=gray!75!black},
    mainflow/.style={->, line width=1.8pt, draw=blue!85!black}
]

% COLUMN 1: DATA PIPELINE (X = 0)
\node[header] (h1) at (0, 0) {PHASES 1 \& 2: Cohort Selection \& Partitioning};

\node[boxLeft, below=0.3cm of h1] (b1) {
    \textbf{\normalsize Phase 1: PTB-XL Clinical ECG Repository}\\
    $\bullet$ $N = 21,799$ recordings from $18,869$ patients\\
    $\bullet$ Standard 12-lead ECG, 10-s, 100\,Hz ($L=1000$)\\
    $\bullet$ 5 Superclasses: NORM (9,514), MI (5,469), STTC (5,235),\\
    \quad CD (4,907), HYP (2,649) $\to$ Target $y \in \{0, 1\}^5$
};

\node[boxLeft, below=0.35cm of b1] (b2) {
    \textbf{\normalsize Phase 2: Patient-Stratified 10-Fold Splitting}\\
    $\bullet$ Strict zero patient identity leakage across partitions\\
    $\bullet$ Per-lead $z$-score normalization: $x_c' = (x_c - \mu_c) / \sigma_c$\\
    $\bullet$ \textbf{Train (Folds 1--8):} $n = 17,411$ (15,082 patients)\\
    $\bullet$ \textbf{Validation (Fold 9):} $n = 2,183$ (1,894 patients)\\
    $\bullet$ \textbf{Held-Out Test (Fold 10):} $n = 2,205$ (1,893 patients)
};
\draw[arrow] (b1) -- (b2);

% COLUMN 2: FIXED-BUDGET ARCHITECTURES (X = 8.6)
\node[headerMid] (h2) at (8.6, 0) {PHASE 3: Fixed-Budget Architectural Allocation};

\node[boxA, below=0.3cm of h2] (b3a) {
    \textbf{\normalsize Arm A: Homogeneous Capacity Scaling ($B \approx 2.0\text{M}$)}\\
    $\bullet$ Divided equally into $K$ identical ResNet-1D sub-networks ($2.0\text{M}/K$)\\
    $\bullet$ $K=1$ ($W_1=65$, 2.02M) $\bullet$ $K=2$ ($W_2=46$, 2.03M)\\
    $\bullet$ $K=4$ ($W_4=32$, 1.98M) $\bullet$ $K=8$ ($W_8=23$, 2.05M)\\
    $\bullet$ $K=16$ ($W_{16}=16$, 2.01M) $\to$ \textit{High CKA redundancy ($0.9458$)}
};

\node[boxB, below=0.35cm of b3a] (b3b) {
    \textbf{\normalsize Arm B: Parameter-Matched Heterogeneous Arm (1.99M)}\\
    $\bullet$ 4 Diverse inductive biases matched to homogeneous $K=4$ ($+0.92\%$):\\
    \quad $\star$ \textbf{ResNet-1D} (494k): 3-stage residual blocks, skip connections\\
    \quad $\star$ \textbf{MultiScale-CNN} (509k): Parallel receptive fields (kernels 3, 7, 15, 31)\\
    \quad $\star$ \textbf{TCN-1D} (491k): Dilated causal convolutions ($d \in \{1, 2, 4, 8\}$)\\
    \quad $\star$ \textbf{ConvAttention-1D} (500k): Downsampling stem + 2-layer Transformer\\
    $\bullet$ \textit{Orthogonal feature representations: Mean Linear CKA = 0.9008}
};

% Connecting Column 1 to Column 2
\draw[mainflow] (b2.east) -- ++(0.4, 0) |- (b3a.west);
\draw[mainflow] (b2.east) -- ++(0.4, 0) |- (b3b.west);

% COLUMN 3: ROUTING & EVALUATION (X = 18.0)
\node[headerRight] (h3) at (18.0, 0) {PHASES 4 \& 5: Adaptive Gating, Stress \& Deferral};

\node[boxRouter, below=0.3cm of h3] (b4) {
    \textbf{\normalsize Phase 4: Dynamic Gating \& Sparse Inference}\\
    $\bullet$ Patient-Specific Router $g_\phi(x)$ ($\sim 16\text{K}$ params, Conv1D + MLP)\\
    $\bullet$ Loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diag}} + 0.05\,\mathcal{L}_{\text{div}} + 0.10\,\mathcal{L}_{\text{fail}}$ (prevents collapse)\\
    $\bullet$ \textbf{Dense Learned:} 414.9 MFLOPs (AUROC $0.9118 \pm 0.0004$)\\
    $\bullet$ \textbf{Sparse Top-2 (MOSAIC-Med):} \textbf{297.8 MFLOPs} ($-28.2\%$ FLOPs)
};

\node[boxEval, below=0.35cm of b4] (b5) {
    \textbf{\normalsize Phase 5: Clinical Stress Testing \& Selective Deferral}\\
    $\bullet$ \textbf{6-Stress Benchmark:} Clean baseline (0.9113), EMG noise, wander,\\
    \quad precordial loss (V1--V3), limb loss (I--III), external site shift ($s \ne 0$)\\
    $\bullet$ \textbf{Selective Deferral:} Uncertainty $U(x) = \mathcal{H}(\hat{y}) + 2.0\,\text{Var}_k(\hat{y}_k)$\\
    \quad Deferring $10\%$ most uncertain ($c=90\%$): risk drops $11.95\% \to 10.37\%$\\
    \quad Empirical relative risk reduction: \textbf{4.18\%} ($p = 0.0093$, 5 seeds)
};
\draw[arrow] (b4) -- (b5);

% Connecting Column 2 to Column 3
\draw[mainflow] (b3a.east) -- ++(0.4, 0) |- ($(b4.west)+(0, 0.4)$);
\draw[mainflow] (b3b.east) -- ++(0.4, 0) |- ($(b4.west)+(0, -0.4)$);

\end{tikzpicture}
\end{document}
"""

with open("latex/standalone_prisma.tex", "w", encoding="utf-8") as f:
    f.write(prisma_tex)

res = subprocess.run(["scratch/bin/tectonic.exe", "latex/standalone_prisma.tex"], capture_output=True, text=True)
print("Tectonic returncode:", res.returncode)
if res.returncode == 0:
    doc = pymupdf.open("latex/standalone_prisma.pdf")
    page = doc[0]
    print("Rect:", page.rect, "w/h ratio:", page.rect.width / page.rect.height)
    pix = page.get_pixmap(dpi=300)
    pix.save("latex/figures/methodology_flowchart.png")
    pix.save("results/images/methodology_flowchart.png")
    print(f"Rendered PNG: {pix.width} x {pix.height}")
else:
    print("Error:", res.stderr)
