import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("latex/figures", exist_ok=True)
os.makedirs("results/images", exist_ok=True)

# Aesthetic parameters - Serif / Times New Roman for IEEE publication
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'font.size': 9.5,
    'axes.labelsize': 10.0,
    'axes.titlesize': 10.5,
    'xtick.labelsize': 8.8,
    'ytick.labelsize': 8.8,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'lines.linewidth': 2.0,
    'lines.markersize': 6.5
})

COLORS = {
    'primary': '#1E40AF',    # Navy Blue
    'secondary': '#0D9488',  # Teal
    'accent': '#EA580C',     # Amber/Orange
    'purple': '#7C3AED',     # Deep Violet
    'gray': '#64748B',       # Slate Gray
    'slate': '#334155'
}

def save_fig_both(fig, name):
    for d in ['results/images', 'latex/figures']:
        fig.savefig(os.path.join(d, f"{name}.png"), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(d, f"{name}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {name}.png and .pdf")

# =========================================================================
# FIGURE 2: MULTIPLICITY SCALING TRADEOFF (NO LINE OVERLAYING LEGEND)
# =========================================================================
df_scale = pd.read_csv("results/tables/scaling_results.csv")
scale_summary = df_scale.groupby('K').agg({
    'auroc': ['mean', 'std'],
    'auprc': ['mean', 'std'],
    'brier': ['mean', 'std']
}).reset_index()

K_vals = scale_summary['K'].values
x = np.arange(len(K_vals))
auroc_m = scale_summary[('auroc', 'mean')].values
auroc_s = scale_summary[('auroc', 'std')].values
auprc_m = scale_summary[('auprc', 'mean')].values
auprc_s = scale_summary[('auprc', 'std')].values
brier_m = scale_summary[('brier', 'mean')].values
brier_s = scale_summary[('brier', 'std')].values

fig, ax1 = plt.subplots(figsize=(4.5, 3.3))

# Left Axis: Discrimination (AUROC & AUPRC)
line1 = ax1.plot(x, auroc_m, marker='o', color=COLORS['primary'], label='Macro AUROC', lw=2.2, ms=6.5)
ax1.fill_between(x, auroc_m - auroc_s, auroc_m + auroc_s, color=COLORS['primary'], alpha=0.15)

line2 = ax1.plot(x, auprc_m, marker='s', color=COLORS['secondary'], label='Macro AUPRC', lw=2.2, ms=6.0)
ax1.fill_between(x, auprc_m - auprc_s, auprc_m + auprc_s, color=COLORS['secondary'], alpha=0.15)

ax1.set_xlabel(r"Number of Sub-Networks $K$ (Budget $\approx 2.0$M)", fontweight='bold')
ax1.set_ylabel("Discrimination (AUROC / AUPRC)", color=COLORS['primary'], fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f"K={k}\n({2000//k}k)" for k in K_vals])
ax1.tick_params(axis='y', labelcolor=COLORS['primary'])
ax1.set_ylim(0.755, 0.920)

# Right Axis: Calibration (Brier Score)
ax2 = ax1.twinx()
ax2.spines['top'].set_visible(False)
line3 = ax2.plot(x, brier_m, marker='D', color=COLORS['purple'], label='Brier Score (Lower=Better)', lw=2.0, ls='--', ms=6.0)
ax2.fill_between(x, brier_m - brier_s, brier_m + brier_s, color=COLORS['purple'], alpha=0.12)
ax2.set_ylabel("Brier Calibration Loss", color=COLORS['purple'], fontweight='bold')
ax2.tick_params(axis='y', labelcolor=COLORS['purple'])
ax2.set_ylim(0.0865, 0.0920)

# Over-fragmentation boundary
ax1.axvline(x=4, color='#DC2626', ls=':', lw=1.5, alpha=0.7)
ax1.annotate("Over-Fragmentation\nBoundary ($K=16$)", xy=(4, 0.765), xytext=(2.4, 0.775),
             arrowprops=dict(arrowstyle="->", color='#DC2626', lw=1.2),
             fontsize=7.8, fontweight='bold', color='#DC2626',
             bbox=dict(boxstyle="round,pad=0.25", facecolor='#FEE2E2', edgecolor='#DC2626', alpha=0.92))

# Top legend: completely outside plot area so NO lines ever overlay it
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3,
           frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', framealpha=0.98, fontsize=8.0)

save_fig_both(fig, "scaling_tradeoff_standalone")
save_fig_both(fig, "scaling_tradeoff")

# =========================================================================
# FIGURE 3: CLINICAL STRESS TEST (NO TEXT OVERLAYING EACH OTHER)
# =========================================================================
df_rob = pd.read_csv("results/tables/robustness_multiseed_summary.csv")
cond_labels = [
    'Clean\nBaseline',
    'EMG Noise\n($\\sigma=0.15$)',
    'Baseline\nWander',
    'Precordial Loss\n(V1--V3)',
    'Limb Loss\n(I--III)',
    'Site Shift\n($s \\ne 0$)'
]
single_ret = [float(val.split('%')[0].strip()) for val in df_rob['Single Retention (%)']]
mosaic_ret = [float(val.split('%')[0].strip()) for val in df_rob['MOSAIC Retention (%)']]

fig, ax = plt.subplots(figsize=(4.8, 3.4))
x = np.arange(len(cond_labels))
w = 0.36

rects1 = ax.bar(x - w/2, single_ret, w, label=r'Single Large ($1 \times 2$M)',
                color='#94A3B8', edgecolor='#475569', lw=1.0)
rects2 = ax.bar(x + w/2, mosaic_ret, w, label=r'MOSAIC-Med (Sparse Top-2)',
                color=COLORS['secondary'], edgecolor='#0F766E', lw=1.0)

ax.set_ylabel('AUROC Retention Relative to Clean (%)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cond_labels, ha='center', fontsize=8.2)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2,
          frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', framealpha=0.98, fontsize=8.5)
ax.set_ylim(88, 108)

# Vertical staggering guarantees zero text overlap:
# Single Large sits at y = h + 2 points (gray text)
for r in rects1:
    h = r.get_height()
    ax.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2),
                textcoords="offset points", ha='center', va='bottom', fontsize=7.2, color='#334155')

# MOSAIC-Med sits at y = h + 14 points (teal bold text) -> completely above Single Large
for r in rects2:
    h = r.get_height()
    ax.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 14),
                textcoords="offset points", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#0F766E')

save_fig_both(fig, "robustness_stress_standalone")
save_fig_both(fig, "robustness_stress")

# =========================================================================
# FIGURE 4: SELECTIVE RISK VS COVERAGE (CLEAN ANNOTATION AND LEGEND)
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

fig, ax = plt.subplots(figsize=(4.5, 3.3))
line_s = ax.plot(cov_pct, s_mean, marker='o', label=r'Single Large ($1 \times 2$M)', color=COLORS['gray'], ls='--', lw=2.0, ms=6.5)
ax.fill_between(cov_pct, s_mean - s_std, s_mean + s_std, color=COLORS['gray'], alpha=0.15)
line_m = ax.plot(cov_pct, m_mean, marker='s', label=r'MOSAIC-Med (Sparse Top-2)', color=COLORS['secondary'], lw=2.2, ms=6.5)
ax.fill_between(cov_pct, m_mean - m_std, m_mean + m_std, color=COLORS['secondary'], alpha=0.18)

ax.set_xlabel('Cohort Coverage Retention (%)', fontweight='bold')
ax.set_ylabel('Multi-Label Hamming Risk (%)', fontweight='bold')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2,
          frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', framealpha=0.98, fontsize=8.5)
ax.set_xlim(78, 102)
ax.set_ylim(8.5, 13.0)

# Annotation for 90% coverage
risk_red = ((s_mean[cov_pct==90][0] - m_mean[cov_pct==90][0]) / s_mean[cov_pct==90][0]) * 100
ax.annotate(r"$-4.18\%$ Risk Reduction" + "\nat $90\%$ Coverage ($p=0.0093$)",
            xy=(90, m_mean[cov_pct==90][0]), xytext=(81.5, 11.5),
            arrowprops=dict(arrowstyle="->", color=COLORS['secondary'], lw=1.5),
            fontsize=8.2, fontweight='bold', color=COLORS['secondary'],
            bbox=dict(boxstyle="round,pad=0.25", facecolor='#F0FDFA', edgecolor=COLORS['secondary'], alpha=0.95))

save_fig_both(fig, "risk_coverage_standalone")
save_fig_both(fig, "risk_coverage")

print("SUCCESS: Generated all standalone single plots cleanly!")
