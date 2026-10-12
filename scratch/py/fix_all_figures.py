import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("latex/figures", exist_ok=True)
os.makedirs("results/images", exist_ok=True)

# Aesthetic parameters - Serif / Times New Roman for IEEE publication
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'font.size': 9.0,
    'axes.labelsize': 9.5,
    'axes.titlesize': 10.0,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.8,
    'lines.markersize': 5.5
})

COLORS = {
    'primary': '#1E40AF',
    'secondary': '#0D9488',
    'accent': '#EA580C',
    'purple': '#7C3AED',
    'gray': '#64748B',
    'slate': '#334155'
}

def save_fig_both(fig, name):
    for d in ['results/images', 'latex/figures']:
        fig.savefig(os.path.join(d, f"{name}.png"), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(d, f"{name}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {name}.png and .pdf")

# =========================================================================
# 1. SCALING TRADEOFF
# =========================================================================
df_scale = pd.read_csv("results/tables/scaling_results.csv")
scale_summary = df_scale.groupby('K').agg({
    'auroc': ['mean', 'std'],
    'auprc': ['mean', 'std'],
    'f1': ['mean', 'std'],
    'brier': ['mean', 'std']
}).reset_index()

K_vals = scale_summary['K'].values
x = np.arange(len(K_vals))
auroc_m = scale_summary[('auroc', 'mean')].values
auroc_s = scale_summary[('auroc', 'std')].values
auprc_m = scale_summary[('auprc', 'mean')].values
auprc_s = scale_summary[('auprc', 'std')].values
f1_m = scale_summary[('f1', 'mean')].values
f1_s = scale_summary[('f1', 'std')].values
brier_m = scale_summary[('brier', 'mean')].values
brier_s = scale_summary[('brier', 'std')].values

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7), gridspec_kw={'wspace': 0.30})

# Panel A
axes[0].plot(x, auroc_m, marker='o', color=COLORS['primary'], label='Macro AUROC', lw=1.8)
axes[0].fill_between(x, auroc_m - auroc_s, auroc_m + auroc_s, color=COLORS['primary'], alpha=0.15)
axes[0].plot(x, auprc_m, marker='s', color=COLORS['secondary'], label='Macro AUPRC', lw=1.8)
axes[0].fill_between(x, auprc_m - auprc_s, auprc_m + auprc_s, color=COLORS['secondary'], alpha=0.15)
axes[0].plot(x, f1_m, marker='^', color=COLORS['accent'], label='Macro F1', lw=1.6, ls='--')
axes[0].fill_between(x, f1_m - f1_s, f1_m + f1_s, color=COLORS['accent'], alpha=0.12)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"K={k}\n({2000//k}k)" for k in K_vals])
axes[0].set_xlabel("Number of Experts (Split at ~2M Budget)")
axes[0].set_ylabel("Diagnostic Score")
axes[0].set_title("(a) Discrimination Scaling", pad=8, fontweight='bold')
axes[0].legend(loc='lower left', frameon=True, facecolor='#FFFFFF', framealpha=0.9)

# Panel B
axes[1].plot(x, brier_m, marker='D', color=COLORS['purple'], label='Brier Score (Lower=Better)', lw=1.8)
axes[1].fill_between(x, brier_m - brier_s, brier_m + brier_s, color=COLORS['purple'], alpha=0.15)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"K={k}\n({2000//k}k)" for k in K_vals])
axes[1].set_xlabel("Number of Experts (Split at ~2M Budget)")
axes[1].set_ylabel("Brier Calibration Loss")
axes[1].set_title("(b) Probability Calibration", pad=8, fontweight='bold')
axes[1].legend(loc='upper right', frameon=True, facecolor='#FFFFFF', framealpha=0.9)

save_fig_both(fig, "scaling_tradeoff")

# =========================================================================
# 2. CKA REPRESENTATION DIVERSITY
# =========================================================================
df_cka = pd.read_csv("results/tables/cka_comparison.csv")
homo_labels = ['ResNet #1', 'ResNet #2', 'ResNet #3', 'ResNet #4']
hetero_labels = ['ResNet-1D', 'MultiScale', 'TCN', 'ConvAttn']

cka_homo = np.eye(4)
cka_hetero = np.eye(4)
idx = 0
for i in range(4):
    for j in range(i+1, 4):
        val_h = df_cka.loc[idx, 'Homogeneous CKA']
        val_e = df_cka.loc[idx, 'Heterogeneous CKA']
        cka_homo[i, j] = cka_homo[j, i] = val_h
        cka_hetero[i, j] = cka_hetero[j, i] = val_e
        idx += 1

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9), gridspec_kw={'wspace': 0.38})

im1 = axes[0].imshow(cka_homo, cmap='Blues', vmin=0.4, vmax=1.0)
axes[0].set_title("(a) Homogeneous (4x ResNet)", pad=8, fontweight='bold')
axes[0].set_xticks(range(4))
axes[0].set_yticks(range(4))
axes[0].set_xticklabels(homo_labels, rotation=30, ha='right')
axes[0].set_yticklabels(homo_labels)
for i in range(4):
    for j in range(4):
        v = cka_homo[i, j]
        axes[0].text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white" if v > 0.75 else "black", fontsize=8.0, fontweight='bold')

im2 = axes[1].imshow(cka_hetero, cmap='viridis', vmin=0.4, vmax=1.0)
axes[1].set_title("(b) Heterogeneous Inductive Biases", pad=8, fontweight='bold')
axes[1].set_xticks(range(4))
axes[1].set_yticks(range(4))
axes[1].set_xticklabels(hetero_labels, rotation=30, ha='right')
axes[1].set_yticklabels(hetero_labels)
for i in range(4):
    for j in range(4):
        v = cka_hetero[i, j]
        axes[1].text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white" if v > 0.75 else "black", fontsize=8.0, fontweight='bold')

fig.subplots_adjust(right=0.87)
cbar_ax = fig.add_axes([0.89, 0.20, 0.02, 0.65])
fig.colorbar(im2, cax=cbar_ax, label='Linear CKA Similarity')

save_fig_both(fig, "cka_representation_diversity")

# =========================================================================
# 3. ROBUSTNESS STRESS TEST (NON-OVERLAPPING ANNOTATIONS)
# =========================================================================
df_rob = pd.read_csv("results/tables/robustness_multiseed_summary.csv")
cond_labels = [
    'Clean\nBaseline',
    'EMG Noise\n(σ=0.15)',
    'Baseline\nWander',
    'Missing V1-V3\n(Precordial)',
    'Missing I-III\n(Limb)',
    'External Site\n(Site != 0)'
]
single_ret = [float(x.split('%')[0].strip()) for x in df_rob['Single Retention (%)']]
mosaic_ret = [float(x.split('%')[0].strip()) for x in df_rob['MOSAIC Retention (%)']]

fig, ax = plt.subplots(figsize=(7.16, 2.9))
x = np.arange(len(cond_labels))
w = 0.35

rects1 = ax.bar(x - w/2, single_ret, w, label='Single Large (2M)',
                color='#94A3B8', edgecolor='#475569', lw=0.8)
rects2 = ax.bar(x + w/2, mosaic_ret, w, label='MOSAIC-Med (Heterogeneous Top-2)',
                color=COLORS['secondary'], edgecolor='#0F766E', lw=0.8)

ax.set_ylabel('AUROC Retention Relative to Clean (%)')
ax.set_title('Robustness Under Real-World ECG Corruptions and Acquisition Site Shift', pad=8, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cond_labels, ha='center', fontsize=8.0)
ax.legend(loc='lower left', frameon=True, facecolor='#FFFFFF', framealpha=0.9)
ax.set_ylim(85, 107)

for i, r in enumerate(rects1):
    h = r.get_height()
    ax.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2),
                textcoords="offset points", ha='center', va='bottom', fontsize=7.2, color='#334155')

for i, r in enumerate(rects2):
    h = r.get_height()
    offset_y = 6 if abs(mosaic_ret[i] - single_ret[i]) < 1.0 else 2
    ax.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, offset_y),
                textcoords="offset points", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#0F766E')

save_fig_both(fig, "robustness_stress_test")

# =========================================================================
# 4. SELECTIVE RISK VS COVERAGE
# =========================================================================
df_sel = pd.read_csv("results/tables/selective_risk_per_seed.csv")
sel_summary = df_sel.groupby('coverage').agg({
    'single_risk_pct': ['mean', 'std'],
    'mosaic_risk_pct': ['mean', 'std']
}).reset_index()

cov_pct = sel_summary['coverage'].values * 100
s_mean = sel_summary[('single_risk_pct', 'mean')].values
s_std = sel_summary[('single_risk_pct', 'std')].values
m_mean = sel_summary[('mosaic_risk_pct', 'mean')].values
m_std = sel_summary[('mosaic_risk_pct', 'std')].values

fig, ax = plt.subplots(figsize=(3.6, 2.8))
ax.plot(cov_pct, s_mean, marker='o', label='Single Large (2M)', color=COLORS['gray'], ls='--', lw=1.8)
ax.fill_between(cov_pct, s_mean - s_std, s_mean + s_std, color=COLORS['gray'], alpha=0.15)
ax.plot(cov_pct, m_mean, marker='s', label='MOSAIC-Med (Sparse Top-2)', color=COLORS['secondary'], lw=2.0)
ax.fill_between(cov_pct, m_mean - m_std, m_mean + m_std, color=COLORS['secondary'], alpha=0.15)

ax.set_xlabel('Cohort Coverage Retention (%)')
ax.set_ylabel('Multi-Label Hamming Risk (%)')
ax.set_title('Selective Risk vs Coverage', pad=8, fontweight='bold')
ax.legend(loc='upper left', frameon=True, facecolor='#FFFFFF', framealpha=0.9)
ax.set_xlim(78, 102)

save_fig_both(fig, "risk_coverage_abstention")

# =========================================================================
# 5. ABLATION WATERFALL
# =========================================================================
df_ab = pd.read_csv("results/tables/ablation_summary.csv")
stages = [
    'Single Large\n(2M)',
    'Homo\n(4x500K)',
    'Hetero\n(4 Biases)',
    'Dense\nRouter',
    'Sparse\nTop-2',
    'Deferral\n(90%)'
]
auroc_vals = [float(v) for v in df_ab['AUROC'].values]

fig, ax = plt.subplots(figsize=(7.16, 2.8))
colors_bar = ['#94A3B8', '#38BDF8', '#0284C7', '#0D9488', '#059669', '#10B981']
x = np.arange(len(stages))
bars = ax.bar(x, auroc_vals, color=colors_bar, edgecolor='#1E293B', lw=0.8, width=0.55)

ax.set_xticks(x)
ax.set_xticklabels(stages, ha='center', fontsize=8.0)
ax.set_ylabel('Macro AUROC')
ax.set_title('Empirical Step-by-Step System Progression Waterfall', pad=8, fontweight='bold')
ax.set_ylim(0.895, 0.925)

for b in bars:
    h = b.get_height()
    ax.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=8.0, fontweight='bold')

save_fig_both(fig, "ablation_waterfall")

# =========================================================================
# 6. LOSS ABLATION TRADEOFF
# =========================================================================
df_loss = pd.read_csv("results/tables/loss_ablation_per_seed.csv")
variants = [
    'Uniform\nAverage',
    'Router\n(L_diag only)',
    'Router\n(+L_div)',
    'Router\n(+L_fail)',
    'All Terms\n(MOSAIC-Med)'
]

loss_summary = df_loss.groupby('variant').agg({
    'auroc': ['mean', 'std'],
    'brier': ['mean', 'std']
}).reindex([
    'Uniform Average (Baseline)',
    'Router with L_diag only (BCE)',
    'Router with L_diag + L_div (Diversity)',
    'Router with L_diag + L_fail (Failure-Aware)',
    'All Terms (L_diag + L_div + L_fail)'
]).reset_index()

auroc_loss_m = loss_summary[('auroc', 'mean')].values
auroc_loss_s = loss_summary[('auroc', 'std')].values
brier_loss_m = loss_summary[('brier', 'mean')].values
brier_loss_s = loss_summary[('brier', 'std')].values

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7), gridspec_kw={'wspace': 0.32})
x = np.arange(len(variants))

# Panel A
axes[0].bar(x, auroc_loss_m, yerr=auroc_loss_s, capsize=3, color='#0284C7', edgecolor='#0369A1', lw=0.8, width=0.55)
axes[0].set_xticks(x)
axes[0].set_xticklabels(variants, fontsize=7.5)
axes[0].set_ylabel('Macro AUROC (Higher = Better)')
axes[0].set_title('(a) Discrimination vs Regularization', pad=8, fontweight='bold')
axes[0].set_ylim(0.900, 0.916)
for i, v in enumerate(auroc_loss_m):
    axes[0].text(i, v + 0.0015, f"{v:.4f}", ha='center', fontsize=7.2, fontweight='bold')

# Panel B
axes[1].bar(x, brier_loss_m, yerr=brier_loss_s, capsize=3, color='#8B5CF6', edgecolor='#6D28D9', lw=0.8, width=0.55)
axes[1].set_xticks(x)
axes[1].set_xticklabels(variants, fontsize=7.5)
axes[1].set_ylabel('Brier Score (Lower = Better)')
axes[1].set_title('(b) Probability Calibration Loss', pad=8, fontweight='bold')
axes[1].set_ylim(0.084, 0.094)
for i, v in enumerate(brier_loss_m):
    axes[1].text(i, v + 0.0010, f"{v:.4f}", ha='center', fontsize=7.2, fontweight='bold')

save_fig_both(fig, "loss_ablation_tradeoff")

print("SUCCESS: All figures regenerated cleanly with Times New Roman / Serif styling!")
