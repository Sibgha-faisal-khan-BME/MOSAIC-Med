import subprocess
import pymupdf

prisma_tex = r"""\documentclass[tikz,border=12pt]{standalone}
\usepackage{times}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc}

\begin{document}
\begin{tikzpicture}[
    font=\rmfamily,
    >=Stealth,
    phasehead/.style={rectangle, fill=blue!12, draw=blue!80!black, line width=1.1pt, rounded corners=3pt, text width=2.6cm, minimum height=1.1cm, align=center, font=\bfseries\rmfamily\footnotesize},
    box/.style={rectangle, fill=blue!2, draw=blue!75!black, line width=1.0pt, rounded corners=4pt, text width=5.8cm, align=left, inner sep=6pt, font=\rmfamily\scriptsize},
    branchA/.style={rectangle, fill=teal!5, draw=teal!85!black, line width=1.0pt, rounded corners=4pt, text width=5.6cm, align=left, inner sep=6pt, font=\rmfamily\scriptsize},
    branchB/.style={rectangle, fill=cyan!5, draw=cyan!85!black, line width=1.0pt, rounded corners=4pt, text width=5.6cm, align=left, inner sep=6pt, font=\rmfamily\scriptsize},
    sidebox/.style={rectangle, fill=orange!6, draw=orange!85!black, line width=1.0pt, rounded corners=4pt, text width=4.8cm, align=left, inner sep=6pt, font=\rmfamily\scriptsize},
    evalA/.style={rectangle, fill=purple!4, draw=purple!85!black, line width=1.0pt, rounded corners=4pt, text width=5.6cm, align=left, inner sep=6pt, font=\rmfamily\scriptsize},
    evalB/.style={rectangle, fill=violet!4, draw=violet!85!black, line width=1.0pt, rounded corners=4pt, text width=5.6cm, align=left, inner sep=6pt, font=\rmfamily\scriptsize},
    arrow/.style={->, line width=1.3pt, draw=gray!80!black},
    mainarrow/.style={->, line width=1.6pt, draw=blue!85!black},
    dasharrow/.style={->, dashed, line width=1.1pt, draw=orange!85!black}
]

% Phase labels column at X = -7.2
\node[phasehead] (p1) at (-7.2, 0) {PHASE 1\\Identification};
\node[phasehead] (p2) at (-7.2, -2.5) {PHASE 2\\Screening \& Split};
\node[phasehead] (p3) at (-7.2, -5.6) {PHASE 3\\Fixed-Budget\\Allocation};
\node[phasehead] (p4) at (-7.2, -9.0) {PHASE 4\\Adaptive Gating\\\& Routing};
\node[phasehead] (p5) at (-7.2, -12.3) {PHASE 5\\Clinical Stress\\\& Deferral};

% Connect Phase Labels vertically
\draw[mainarrow] (p1) -- (p2);
\draw[mainarrow] (p2) -- (p3);
\draw[mainarrow] (p3) -- (p4);
\draw[mainarrow] (p4) -- (p5);

% ROW 1: IDENTIFICATION
\node[box] (b1) at (-0.5, 0) {
    \textbf{\normalsize PTB-XL Benchmark Repository}\\
    $\bullet$ $N = 21,799$ clinical recordings ($18,869$ patients)\\
    $\bullet$ Standard 12-lead ECG, 10-s duration, 100\,Hz\\
    $\bullet$ Multi-label ground truth: $y \in \{0, 1\}^5$
};
\node[sidebox] (s1) at (5.8, 0) {
    \textbf{Cohort Diagnostic Prevalence:}\\
    $\bullet$ NORM: 9,514 (43.7\%) $\bullet$ MI: 5,469 (25.1\%)\\
    $\bullet$ STTC: 5,235 (24.3\%) $\bullet$ CD: 4,907 (22.5\%)\\
    $\bullet$ HYP: 2,649 (12.2\%)
};
\draw[dasharrow] (b1) -- (s1);

% ROW 2: PARTITIONING
\node[box] (b2) at (-0.5, -2.5) {
    \textbf{\normalsize Patient-Stratified 10-Fold Split}\\
    $\bullet$ Zero patient overlap across splits\\
    $\bullet$ Lead $z$-score normalization: $x_c' = (x_c - \mu_c) / \sigma_c$\\
    $\bullet$ Standardized PTB-XL evaluation protocol
};
\node[sidebox] (s2) at (5.8, -2.5) {
    \textbf{Cohort Partition Sizes:}\\
    $\bullet$ \textbf{Train (Folds 1--8):} $n = 17,411$ (15,082 pts)\\
    $\bullet$ \textbf{Val (Fold 9):} $n = 2,183$ (1,894 pts)\\
    $\bullet$ \textbf{Test (Fold 10):} $n = 2,205$ (1,893 pts)
};
\draw[arrow] (b1) -- (b2);
\draw[dasharrow] (b2) -- (s2);

% ROW 3: ALLOCATION
\node[branchA] (b3a) at (-3.4, -5.6) {
    \textbf{\normalsize Arm A: Homogeneous Scaling}\\
    \textit{Fixed 2.0M budget divided equally ($2.0\text{M}/K$)}\\
    $\bullet$ $K=1$: 1$\times$2.0M ($W_1=65$, 2.02M params)\\
    $\bullet$ $K=2$: 2$\times$1.0M ($W_2=46$, 2.03M params)\\
    $\bullet$ $K=4$: 4$\times$500K ($W_4=32$, 1.98M params)\\
    $\bullet$ $K=8$: 8$\times$250K ($W_8=23$, 2.05M params)\\
    $\bullet$ $K=16$: 16$\times$125K ($W_{16}=16$, 2.01M params)\\
    \textit{High redundancy: Mean CKA = 0.9458}
};
\node[branchB] (b3b) at (3.0, -5.6) {
    \textbf{\normalsize Arm B: Heterogeneous Arm (1.99M)}\\
    \textit{4 Parameter-matched inductive biases ($\sim 500\text{K}$)}\\
    $\bullet$ \textbf{ResNet-1D} (494k): 3-stage residual stages\\
    $\bullet$ \textbf{MultiScale-CNN} (509k): Parallel kernels 3, 7, 15, 31\\
    $\bullet$ \textbf{TCN-1D} (491k): Dilations $d \in \{1, 2, 4, 8\}$\\
    $\bullet$ \textbf{ConvAttention-1D} (500k): 2-layer Transformer\\
    \textit{Orthogonal diversity: Mean CKA = 0.9008}
};
\draw[arrow] (b2.south) -- ++(0,-0.3) -| (b3a.north);
\draw[arrow] (b2.south) -- ++(0,-0.3) -| (b3b.north);

% ROW 4: ROUTING
\node[box] (b4) at (-0.5, -9.0) {
    \textbf{\normalsize Adaptive Router $g_\phi(x)$ ($\sim 16\text{K}$ parameters)}\\
    $\bullet$ Gating function: $w(x) = \text{Softmax}(W_2 \cdot \text{ReLU}(W_1 \cdot \text{Pool}(\text{Conv1D}(x))))$\\
    $\bullet$ Loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diag}} + 0.05\,\mathcal{L}_{\text{div}} + 0.10\,\mathcal{L}_{\text{fail}}$ (prevents collapse)
};
\node[sidebox] (s4) at (5.8, -9.0) {
    \textbf{Execution Regimes:}\\
    $\bullet$ \textbf{Dense Learned:} 414.9 MFLOPs\\
    $\bullet$ \textbf{Sparse Top-2 (MOSAIC-Med):}\\
    \quad \textbf{297.8 MFLOPs} ($-28.2\%$ compute)\\
    \quad AUROC $0.9113 \pm 0.0008$ retained
};
\draw[arrow] (b3a.south) -- ++(0,-0.3) -| ($(b4.north)+(-1.8,0)$);
\draw[arrow] (b3b.south) -- ++(0,-0.3) -| ($(b4.north)+(1.8,0)$);
\draw[dasharrow] (b4) -- (s4);

% ROW 5: STRESS & DEFERRAL
\node[evalA] (b5a) at (-3.4, -12.3) {
    \textbf{\normalsize 6 Clinical Stress Benchmarks}\\
    $\bullet$ Clean Baseline: AUROC 0.9113\\
    $\bullet$ EMG Noise ($\sigma=0.15$) $\bullet$ Baseline Wander (0.5\,Hz)\\
    $\bullet$ Precordial Loss (V1--V3) $\bullet$ Limb Loss (I--III)\\
    $\bullet$ External Site Shift ($s \ne 0$, $N=248$)\\
    \textit{Significantly superior retention vs Single Large}
};
\node[evalB] (b5b) at (3.0, -12.3) {
    \textbf{\normalsize Selective Clinical Deferral}\\
    $\bullet$ Uncertainty: $U(x) = \mathcal{H}(\hat{y}) + 2\,\text{Var}_k(\hat{y}_k)$\\
    $\bullet$ Selective coverage thresholds $c \in [80\%, 100\%]$\\
    $\bullet$ Deferring $10\%$ most uncertain ($c=90\%$):\\
    \quad Hamming risk drops $11.95\% \to 10.37\%$\\
    \quad Relative risk reduction: \textbf{4.18\%} ($p=0.0093$)
};
\draw[arrow] (b4.south) -- ++(0,-0.3) -| (b5a.north);
\draw[arrow] (b4.south) -- ++(0,-0.3) -| (b5b.north);

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
