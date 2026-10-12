import subprocess
import pymupdf

tikz_code = r"""\documentclass[tikz,border=10pt]{standalone}
\usepackage{times}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc}

\begin{document}
\begin{tikzpicture}[
    font=\rmfamily,
    >=Stealth,
    phasehead/.style={rectangle, fill=blue!15, draw=blue!80!black, line width=1.1pt, rounded corners=3pt, text width=4.8cm, minimum height=0.75cm, align=center, font=\bfseries\rmfamily\small},
    phasebox/.style={rectangle, fill=blue!2, draw=blue!70!black, line width=1.0pt, rounded corners=4pt, text width=4.8cm, align=left, inner sep=6pt, font=\rmfamily\scriptsize},
    branchboxA/.style={rectangle, fill=teal!5, draw=teal!80!black, line width=1.0pt, rounded corners=4pt, text width=4.8cm, align=left, inner sep=6pt, font=\rmfamily\scriptsize},
    branchboxB/.style={rectangle, fill=cyan!5, draw=cyan!80!black, line width=1.0pt, rounded corners=4pt, text width=4.8cm, align=left, inner sep=6pt, font=\rmfamily\scriptsize},
    sidebox/.style={rectangle, fill=orange!6, draw=orange!80!black, line width=1.0pt, rounded corners=4pt, text width=4.8cm, align=left, inner sep=6pt, font=\rmfamily\scriptsize},
    evalbox/.style={rectangle, fill=purple!4, draw=purple!80!black, line width=1.0pt, rounded corners=4pt, text width=4.8cm, align=left, inner sep=6pt, font=\rmfamily\scriptsize},
    arrow/.style={->, line width=1.2pt, draw=gray!80!black},
    mainarrow/.style={->, line width=1.5pt, draw=blue!80!black}
]

% Five Horizontal Columns (Phases 1 to 5)
% Column Centers: X = 0, 5.5, 11.0, 16.5, 22.0 (Total width ~ 27 cm)

% --- COLUMN 1: PHASE 1 ---
\node[phasehead] (h1) at (0, 0) {PHASE 1: Cohort Selection};
\node[phasebox, below=0.3cm of h1] (b1) {
    \textbf{PTB-XL Benchmark Database}\\
    $\bullet$ $N = 21,799$ clinical 12-lead ECGs\\
    $\bullet$ 18,869 patients, 10-s duration\\
    $\bullet$ Sampling: 100\,Hz ($L=1000$ points)\\[4pt]
    \textbf{5 Diagnostic Superclasses:}\\
    $\bullet$ NORM: 9,514 (43.7\%)\\
    $\bullet$ MI: 5,469 (25.1\%)\\
    $\bullet$ STTC: 5,235 (24.3\%)\\
    $\bullet$ CD: 4,907 (22.5\%)\\
    $\bullet$ HYP: 2,649 (12.2\%)\\[3pt]
    \textit{Multi-label target: $y \in \{0, 1\}^5$}
};

% --- COLUMN 2: PHASE 2 ---
\node[phasehead] (h2) at (5.5, 0) {PHASE 2: Stratified Partitioning};
\node[phasebox, below=0.3cm of h2] (b2) {
    \textbf{Patient-Stratified 10-Fold Split}\\
    $\bullet$ Strict zero patient overlap\\
    $\bullet$ Per-lead $z$-score normalization:\\[2pt]
    \hspace*{0.5cm}$x'_{c} = (x_c - \mu_c) / \sigma_c$\\[4pt]
    \textbf{Cohort Partitions:}\\
    $\bullet$ \textbf{Train (Folds 1--8):}\\
    \quad 17,411 ECGs (15,082 patients)\\
    $\bullet$ \textbf{Validation (Fold 9):}\\
    \quad 2,183 ECGs (1,894 patients)\\
    $\bullet$ \textbf{Held-Out Test (Fold 10):}\\
    \quad 2,205 ECGs (1,893 patients)\\[3pt]
    \textit{Standardized evaluation protocol}
};

% --- COLUMN 3: PHASE 3 ---
\node[phasehead] (h3) at (11.0, 0) {PHASE 3: Fixed-Budget Scaling};
\node[branchboxA, below=0.3cm of h3] (b3a) {
    \textbf{Arm A: Homogeneous ResNet-1D}\\
    $\bullet$ Invariant $B \approx 2.0\text{M}$ parameter budget\\
    $\bullet$ $K \in \{1, 2, 4, 8, 16\}$ sub-networks\\
    $\bullet$ Equal capacity split: $2.0\text{M} / K$\\
    $\bullet$ Mean Linear CKA $= 0.9458$\\[2pt]
    \textit{High representational redundancy}
};
\node[branchboxB, below=0.25cm of b3a] (b3b) {
    \textbf{Arm B: Heterogeneous Arm (1.99M)}\\
    $\bullet$ \textbf{ResNet-1D} (494k): 3-stage residual\\
    $\bullet$ \textbf{MultiScale} (509k): Kernels 3, 7, 15, 31\\
    $\bullet$ \textbf{TCN-1D} (491k): Dilations $1, 2, 4, 8$\\
    $\bullet$ \textbf{ConvAttention} (500k): Self-attention\\[2pt]
    \textit{Orthogonal representations (CKA 0.9008)}
};

% --- COLUMN 4: PHASE 4 ---
\node[phasehead] (h4) at (16.5, 0) {PHASE 4: Adaptive Gating};
\node[sidebox, below=0.3cm of h4] (b4) {
    \textbf{Patient-Specific Router $g_\phi(x)$}\\
    $\bullet$ Conv1D + MaxPool + 2-layer MLP\\
    $\bullet$ Parameters: $\sim 16\text{K}$ ($<0.8\%$ of total)\\
    $\bullet$ Gating weights: $w(x) \in \Delta^3$\\[3pt]
    \textbf{Composite Loss Function:}\\
    $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diag}} + 0.05\,\mathcal{L}_{\text{div}} + 0.10\,\mathcal{L}_{\text{fail}}$\\[4pt]
    \textbf{Inference Regimes:}\\
    $\bullet$ \textbf{Dense Learned:} 414.9 MFLOPs\\
    $\bullet$ \textbf{Sparse Top-2 (MOSAIC-Med):}\\
    \quad \textbf{297.8 MFLOPs} ($-28.2\%$ compute)\\
    \quad Full AUROC $0.9113$ retained
};

% --- COLUMN 5: PHASE 5 ---
\node[phasehead] (h5) at (22.0, 0) {PHASE 5: Clinical Stress \& Deferral};
\node[evalbox, below=0.3cm of h5] (b5) {
    \textbf{6 Clinical Stress Benchmarks:}\\
    $\bullet$ Clean Test Baseline (0.9113)\\
    $\bullet$ EMG Noise ($\sigma=0.15$) $\bullet$ Wander ($0.5\,$Hz)\\
    $\bullet$ Precordials V1--V3 missing\\
    $\bullet$ Limb leads I--III missing\\
    $\bullet$ External Site Shift ($s \ne 0$)\\[3pt]
    \textbf{Selective Clinical Deferral:}\\
    $\bullet$ $U(x) = \sum \mathcal{H}(\hat{y}_m) + 2.0\,\text{Var}_k(p_k)$\\
    $\bullet$ Coverage $c \in [80\%, 100\%]$\\
    $\bullet$ Risk drops $11.95\% \to 10.37\%$ at $c=90\%$\\
    $\bullet$ Relative risk reduction: \textbf{4.18\%}\\[3pt]
    \textit{Paired DeLong \& Holm-Welch tests}
};

% Connecting Arrows Between Headers
\draw[mainarrow] (h1) -- (h2);
\draw[mainarrow] (h2) -- (h3);
\draw[mainarrow] (h3) -- (h4);
\draw[mainarrow] (h4) -- (h5);

% Inter-box Connecting Arrows
\draw[arrow] (b1.east) -- (b2.west);
\draw[arrow] (b2.east) -- ++(0.25,0) |- (b3a.west);
\draw[arrow] (b2.east) -- ++(0.25,0) |- (b3b.west);
\draw[arrow] (b3a.east) -| ++(0.25,-0.6) -| ($(b4.west)+(0,0.8)$);
\draw[arrow] (b3b.east) -| ++(0.25,0.6) -| ($(b4.west)+(0,-0.8)$);
\draw[arrow] (b4.east) -- (b5.west);

\end{tikzpicture}
\end{document}
"""

with open("scratch/py/test_prisma_widescreen.tex", "w", encoding="utf-8") as f:
    f.write(tikz_code)

res = subprocess.run(["scratch/bin/tectonic.exe", "scratch/py/test_prisma_widescreen.tex"], capture_output=True, text=True)
print("Tectonic returncode:", res.returncode)
if res.returncode == 0:
    doc = pymupdf.open("scratch/py/test_prisma_widescreen.pdf")
    page = doc[0]
    print("Rect:", page.rect, "w/h ratio:", page.rect.width / page.rect.height)
    pix = page.get_pixmap(dpi=300)
    pix.save("scratch/test_prisma_widescreen.png")
    print("Rendered PNG:", pix.width, "x", pix.height)
else:
    print("STDERR:", res.stderr)
