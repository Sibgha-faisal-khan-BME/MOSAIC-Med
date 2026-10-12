"""
Publication-quality IEEE visualization styling and figure generators.
Sized for IEEE format:
- Single column: ~3.5 in
- Double column: ~7.16 in
300+ DPI, refined color palettes, polished styling, PDF and PNG exports.
"""

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image, ImageFilter
from typing import Optional, List, Dict, Tuple

# -------------------------------------------------------------
# Matplotlib Aesthetic Configuration for IEEE Publications
# -------------------------------------------------------------
def set_ieee_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 8.5,
        'axes.labelsize': 9.5,
        'axes.titlesize': 10.5,
        'xtick.labelsize': 8.0,
        'ytick.labelsize': 8.0,
        'legend.fontsize': 8.0,
        'figure.titlesize': 11.5,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'lines.linewidth': 1.6,
        'lines.markersize': 5.0
    })

# Curated Palette
COLORS = {
    'primary': '#1E40AF',       # Deep Blue
    'secondary': '#0D9488',     # Teal / Emerald
    'accent': '#EA580C',        # Vibrant Amber/Orange
    'purple': '#7C3AED',        # Violet
    'rose': '#E11D48',          # Rose Red
    'dark': '#0F172A',          # Slate 900
    'gray': '#64748B',          # Slate 500
    'light': '#F8FAFC',         # Slate 50
    'gold': '#D97706'           # Amber Gold
}

PALETTE = ['#1E40AF', '#0D9488', '#EA580C', '#7C3AED', '#E11D48', '#0284C7', '#16A34A', '#D97706']

def save_ieee_fig(fig: plt.Figure, base_filename: str, output_dir: str = 'results/images'):
    """Saves figure in both high-res PNG (300 dpi) and vector PDF format."""
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {png_path} and {pdf_path}")

# -------------------------------------------------------------
# Figure 1: Scaling Trade-off (1 vs 2 vs 4 vs 8 vs 16 Experts)
# -------------------------------------------------------------
def plot_scaling_tradeoff(
    num_experts: List[int],
    auroc_means: List[float],
    auroc_stds: List[float],
    auprc_means: List[float],
    auprc_stds: List[float],
    f1_means: List[float],
    f1_stds: List[float],
    brier_means: List[float],
    brier_stds: List[float],
    save_name: str = 'scaling_tradeoff'
):
    set_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))
    
    x = np.arange(len(num_experts))
    
    # Left: Discrimination (AUROC & AUPRC)
    ax1 = axes[0]
    ax1.plot(x, auroc_means, marker='o', color=COLORS['primary'], label='Macro AUROC', linewidth=2.0)
    ax1.fill_between(x, np.array(auroc_means) - np.array(auroc_stds), np.array(auroc_means) + np.array(auroc_stds),
                     color=COLORS['primary'], alpha=0.15)
    
    ax1.plot(x, auprc_means, marker='s', color=COLORS['secondary'], label='Macro AUPRC', linewidth=2.0)
    ax1.fill_between(x, np.array(auprc_means) - np.array(auprc_stds), np.array(auprc_means) + np.array(auprc_stds),
                     color=COLORS['secondary'], alpha=0.15)
    
    ax1.plot(x, f1_means, marker='^', color=COLORS['accent'], label='Macro F1', linewidth=1.8, linestyle='--')
    ax1.fill_between(x, np.array(f1_means) - np.array(f1_stds), np.array(f1_means) + np.array(f1_stds),
                     color=COLORS['accent'], alpha=0.12)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"K={k}\n({2000//k}k)" for k in num_experts])
    ax1.set_xlabel("Number of Experts (Budget Split at ~2M Total)")
    ax1.set_ylabel("Diagnostic Performance Metric")
    ax1.set_title("A. Discrimination Scaling Under Fixed 2M Budget")
    ax1.legend(loc='lower left', frameon=True, facecolor='#FFFFFF', framealpha=0.9)
    
    # Right: Calibration (Brier Score - lower is better)
    ax2 = axes[1]
    ax2.plot(x, brier_means, marker='D', color=COLORS['purple'], label='Brier Score (Lower = Better)', linewidth=2.0)
    ax2.fill_between(x, np.array(brier_means) - np.array(brier_stds), np.array(brier_means) + np.array(brier_stds),
                     color=COLORS['purple'], alpha=0.15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"K={k}\n({2000//k}k)" for k in num_experts])
    ax2.set_xlabel("Number of Experts (Budget Split at ~2M Total)")
    ax2.set_ylabel("Brier Score")
    ax2.set_title("B. Probability Calibration vs Expert Multiplicity")
    ax2.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', framealpha=0.9)
    
    plt.tight_layout()
    save_ieee_fig(fig, save_name)

# -------------------------------------------------------------
# Figure 2: CKA Representation Diversity Matrix
# -------------------------------------------------------------
def plot_cka_heatmap(
    cka_matrix_homo: np.ndarray,
    cka_matrix_hetero: np.ndarray,
    labels: List[str],
    save_name: str = 'cka_representation_diversity'
):
    set_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2))
    
    im1 = axes[0].imshow(cka_matrix_homo, cmap='Blues', vmin=0.4, vmax=1.0)
    axes[0].set_title("A. Homogeneous Experts (4x ResNet)")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=35, ha='right')
    axes[0].set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cka_matrix_homo[i, j]
            axes[0].text(j, i, f"{val:.2f}", ha="center", va="center",
                         color="white" if val > 0.75 else "black", fontsize=8.0, fontweight='bold')
            
    im2 = axes[1].imshow(cka_matrix_hetero, cmap='viridis', vmin=0.4, vmax=1.0)
    axes[1].set_title("B. Heterogeneous Experts (ResNet, MS, TCN, Att)")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=35, ha='right')
    axes[1].set_yticklabels([])
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cka_matrix_hetero[i, j]
            axes[1].text(j, i, f"{val:.2f}", ha="center", va="center",
                         color="white" if val > 0.75 else "black", fontsize=8.0, fontweight='bold')
            
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.20, 0.02, 0.65])
    fig.colorbar(im2, cax=cbar_ax, label='Centered Kernel Alignment (CKA)')
    
    save_ieee_fig(fig, save_name)

# -------------------------------------------------------------
# Figure 3: Selective Risk vs Coverage and Abstention
# -------------------------------------------------------------
def plot_risk_coverage(
    coverages: np.ndarray,
    risks_single: np.ndarray,
    risks_mosaic: np.ndarray,
    save_name: str = 'risk_coverage_abstention'
):
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    
    ax.plot(coverages * 100, risks_single * 100, label='Single Large (2M)', color=COLORS['gray'], linestyle='--', linewidth=1.8)
    ax.plot(coverages * 100, risks_mosaic * 100, label='MOSAIC-Med (Adaptive)', color=COLORS['secondary'], linewidth=2.2)
    
    ax.set_xlabel('Clinical Coverage (%)')
    ax.set_ylabel('Selective Diagnostic Risk (Hamming Loss %)')
    ax.set_title('Selective Classification & Abstention')
    ax.legend(loc='upper left', frameon=True, facecolor='#FFFFFF')
    ax.set_xlim(10, 100)
    
    plt.tight_layout()
    save_ieee_fig(fig, save_name)

# -------------------------------------------------------------
# Figure 4: Robustness Stress Testing
# -------------------------------------------------------------
def plot_robustness_comparison(
    perturbations: List[str],
    single_retention: List[float],
    mosaic_retention: List[float],
    save_name: str = 'robustness_stress_test'
):
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    
    x = np.arange(len(perturbations))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, [v * 100 for v in single_retention], width, label='Single Large Baseline (2M)',
                    color='#94A3B8', edgecolor='#475569', linewidth=0.8)
    rects2 = ax.bar(x + width/2, [v * 100 for v in mosaic_retention], width, label='MOSAIC-Med (Heterogeneous Adaptive)',
                    color=COLORS['secondary'], edgecolor='#0F766E', linewidth=0.8)
    
    ax.set_ylabel('AUROC Retention Relative to Clean (%)')
    ax.set_title('Robustness Stress Test Across Real-World ECG Corruptions & External Domain')
    ax.set_xticks(x)
    ax.set_xticklabels(perturbations, rotation=20, ha='right')
    ax.legend(loc='lower left', frameon=True, facecolor='#FFFFFF')
    ax.set_ylim(50, 105)
    
    for r in rects1:
        h = r.get_height()
        ax.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2),
                    textcoords="offset points", ha='center', va='bottom', fontsize=7.0)
    for r in rects2:
        h = r.get_height()
        ax.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2),
                    textcoords="offset points", ha='center', va='bottom', fontsize=7.0, fontweight='bold')
        
    plt.tight_layout()
    save_ieee_fig(fig, save_name)

# -------------------------------------------------------------
# Figure 5: Reliability & Calibration Curve
# -------------------------------------------------------------
def plot_calibration_curves(
    rel_single: Dict[str, np.ndarray],
    rel_mosaic: Dict[str, np.ndarray],
    save_name: str = 'reliability_calibration'
):
    set_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), gridspec_kw={'width_ratios': [1.2, 1.0]})
    
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    ax1.plot(rel_single['bin_confs'], rel_single['bin_accs'], marker='s', color=COLORS['gray'], label='Single Large (2M)')
    ax1.plot(rel_mosaic['bin_confs'], rel_mosaic['bin_accs'], marker='o', color=COLORS['primary'], linewidth=2.0, label='MOSAIC-Med')
    ax1.set_xlabel('Predicted Probability (Confidence)')
    ax1.set_ylabel('Observed Empirical Frequency')
    ax1.set_title('A. Reliability Diagram')
    ax1.legend(loc='upper left', frameon=True)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    ax2 = axes[1]
    width = 0.04
    ax2.bar(rel_mosaic['bin_centers'] - width/2, rel_single['counts'], width=width, color='#CBD5E1', label='Single Model')
    ax2.bar(rel_mosaic['bin_centers'] + width/2, rel_mosaic['counts'], width=width, color='#60A5FA', label='MOSAIC-Med')
    ax2.set_xlabel('Predicted Confidence Bin')
    ax2.set_ylabel('Sample Count')
    ax2.set_title('B. Confidence Histogram')
    ax2.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    save_ieee_fig(fig, save_name)

# -------------------------------------------------------------
# Figure 6: Top-Line Summary Figure (Novelty Highlight)
# -------------------------------------------------------------
def plot_topline_mosaic_comparison(
    models: List[str],
    auroc: List[float],
    auprc: List[float],
    brier: List[float],
    flops: List[float],
    save_name: str = 'topline_mosaic_vs_single'
):
    set_ieee_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8))
    
    x = np.arange(len(models))
    palette = [COLORS['gray'], '#38BDF8', '#818CF8', COLORS['secondary']]
    
    # AUROC
    axes[0].bar(x, auroc, color=palette, edgecolor='#1E293B', linewidth=0.8)
    axes[0].set_title('A. Macro AUROC (Discrimination)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=25, ha='right', fontsize=7.5)
    axes[0].set_ylim(min(auroc) - 0.03, max(auroc) + 0.02)
    for i, v in enumerate(auroc):
        axes[0].text(i, v + 0.002, f"{v:.3f}", ha='center', fontsize=7.5, fontweight='bold')
        
    # Calibration (Brier)
    axes[1].bar(x, brier, color=palette, edgecolor='#1E293B', linewidth=0.8)
    axes[1].set_title('B. Brier Score (Calibration: ↓)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=25, ha='right', fontsize=7.5)
    axes[1].set_ylim(0, max(brier) * 1.25)
    for i, v in enumerate(brier):
        axes[1].text(i, v + 0.003, f"{v:.3f}", ha='center', fontsize=7.5, fontweight='bold')
        
    # Efficiency (FLOPs)
    axes[2].bar(x, flops, color=palette, edgecolor='#1E293B', linewidth=0.8)
    axes[2].set_title('C. Effective MFLOPs / Sample (↓)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=25, ha='right', fontsize=7.5)
    axes[2].set_ylim(0, max(flops) * 1.25)
    for i, v in enumerate(flops):
        axes[2].text(i, v + 2.0, f"{v:.1f}M", ha='center', fontsize=7.5, fontweight='bold')
        
    plt.tight_layout()
    save_ieee_fig(fig, save_name)
