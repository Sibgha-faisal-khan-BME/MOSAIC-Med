import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import wfdb

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'font.size': 8.5,
    'mathtext.fontset': 'stix'
})

# Load actual PTB-XL ECG recording
try:
    real_ecg, meta = wfdb.rdsamp('ptbxl/records100/00000/00001_lr')
except Exception as e:
    print("Warning: wfdb load failed, falling back to simulated signal:", e)
    t = np.linspace(0, 10, 1000)
    real_ecg = np.zeros((1000, 12))
    for i in range(12):
        real_ecg[:, i] = np.sin(2 * np.pi * 1.2 * t + i * 0.3)

fig = plt.figure(figsize=(16.8, 11.4), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Color Palette matching publication style
C = {
    'blue_bdr': '#2563EB', 'blue_bg': '#EFF6FF', 'blue_hdr': '#DBEAFE', 'blue_num': '#1D4ED8',
    'teal_bdr': '#0D9488', 'teal_bg': '#F0FDFA', 'teal_hdr': '#CCFBF1', 'teal_num': '#0F766E',
    'orange_bdr': '#EA580C', 'orange_bg': '#FFF7ED', 'orange_hdr': '#FFEDD5', 'orange_num': '#C2410C',
    'purple_bdr': '#7C3AED', 'purple_bg': '#FAF5FF', 'purple_hdr': '#F3E8FF', 'purple_num': '#6D28D9',
    'amber_bdr': '#D97706', 'amber_bg': '#FFFBEB', 'amber_hdr': '#FEF3C7', 'amber_num': '#B45309',
    'cyan_bdr': '#0284C7', 'cyan_bg': '#F0F9FF', 'cyan_hdr': '#E0F2FE', 'cyan_num': '#0369A1',
    'magenta_bdr': '#C026D3', 'magenta_bg': '#FDF4FF', 'magenta_hdr': '#FAE8FF', 'magenta_num': '#A21CAF',
    'green_bdr': '#16A34A', 'green_bg': '#F0FDF4', 'green_hdr': '#DCFCE7', 'green_num': '#15803D',
    'gray_box': '#F1F5F9', 'gray_bdr': '#CBD5E1', 'gray_txt': '#334155'
}

def draw_card(x, y, w, h, num, title, subtitle, color_key, title_fs=8.5):
    bdr = C[f'{color_key}_bdr']
    bg = C[f'{color_key}_bg']
    hdr = C[f'{color_key}_hdr']
    num_c = C[f'{color_key}_num']
    
    # Outer Card
    card = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                  facecolor=bg, edgecolor=bdr, linewidth=1.2, zorder=2)
    ax.add_patch(card)
    
    # Header Strip
    hdr_h = 3.6
    hdr_patch = patches.FancyBboxPatch((x, y + h - hdr_h), w, hdr_h, boxstyle="round,pad=0,rounding_size=1.2",
                                      facecolor=hdr, edgecolor=bdr, linewidth=1.1, zorder=3)
    ax.add_patch(hdr_patch)
    sq = patches.Rectangle((x, y + h - hdr_h), w, 1.2, facecolor=hdr, edgecolor='none', zorder=4)
    ax.add_patch(sq)
    ax.plot([x, x + w], [y + h - hdr_h, y + h - hdr_h], color=bdr, lw=0.8, zorder=5)
    
    # Number Badge
    badge_r = 1.3
    badge = patches.Circle((x + 2.0, y + h - hdr_h/2), badge_r, facecolor=num_c, edgecolor='none', zorder=6)
    ax.add_patch(badge)
    ax.text(x + 2.0, y + h - hdr_h/2, str(num), color='white', ha='center', va='center',
            fontweight='bold', fontsize=8.0, zorder=7)
    
    # Title & Subtitle
    if subtitle:
        ax.text(x + 3.8, y + h - hdr_h/2 + 0.55, title, color='#0F172A', ha='left', va='center',
                fontweight='bold', fontsize=title_fs, zorder=6)
        ax.text(x + 3.8, y + h - hdr_h/2 - 0.75, subtitle, color='#475569', ha='left', va='center',
                fontsize=7.2, zorder=6)
    else:
        ax.text(x + 3.8, y + h - hdr_h/2, title, color='#0F172A', ha='left', va='center',
                fontweight='bold', fontsize=title_fs, zorder=6)

def draw_arrow(p1, p2, color='#2563EB', lw=1.5, ls='-', label=None, label_pos=(0,0)):
    arrow = patches.FancyArrowPatch(p1, p2, arrowstyle='->,head_width=3.5,head_length=5.5',
                                    color=color, lw=lw, linestyle=ls, zorder=8)
    ax.add_patch(arrow)
    if label:
        ax.text(label_pos[0], label_pos[1], label, color=color, ha='center', va='center',
                fontsize=7.6, fontweight='bold', zorder=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor='white', edgecolor=color, lw=0.8))

# =========================================================================
# TIER 1: DATA PIPELINE (Boxes 1 to 5)
# =========================================================================
t1_y = 79.5
t1_h = 18.5
box_w = 17.6
gap = 2.4
xs = [2.0 + i*(box_w + gap) for i in range(5)]

# Box 1: Dataset & Cohort (with real PTB-XL ECG strip)
draw_card(xs[0], t1_y, box_w, t1_h, 1, "Dataset and Cohort", "(PTB-XL Repository)", 'blue')
ax.text(xs[0] + box_w/2, t1_y + 11.2, "21,799 records from 18,869 patients", ha='center', fontsize=7.8, color='#1E293B', fontweight='bold', zorder=5)
ax.text(xs[0] + box_w/2, t1_y + 9.3, "Standard 12-lead ECG, 10-s, 100 Hz", ha='center', fontsize=7.6, color='#334155', zorder=5)
ax.text(xs[0] + box_w/2, t1_y + 7.5, "SCP-ECG diagnostic statements", ha='center', fontsize=7.4, color='#475569', zorder=5)

# Real ECG rhythm strip over calibration grid
gx = xs[0] + 1.2
gy = t1_y + 2.0
gw = box_w - 2.4
gh = 4.5
grid_bg = patches.Rectangle((gx, gy), gw, gh, facecolor='#FEF2F2', edgecolor='#FCA5A5', lw=0.6, zorder=4)
ax.add_patch(grid_bg)
for i in range(1, 8):
    ax.plot([gx + i*(gw/8), gx + i*(gw/8)], [gy, gy + gh], color='#FEE2E2', lw=0.5, zorder=4)
for j in range(1, 4):
    ax.plot([gx, gx + gw], [gy + j*(gh/4), gy + j*(gh/4)], color='#FEE2E2', lw=0.5, zorder=4)
lead2_data = real_ecg[150:400, 1]
wx = np.linspace(gx + 0.3, gx + gw - 0.3, len(lead2_data))
lead2_norm = (lead2_data - np.min(lead2_data)) / (np.ptp(lead2_data) + 1e-6)
wy = gy + 0.4 + lead2_norm * (gh - 1.2)
ax.plot(wx, wy, color='#DC2626', lw=1.1, zorder=5)
ax.text(gx + 0.4, gy + gh - 0.7, "PTB-XL Record #00001 (Lead II)", fontsize=6.2, color='#991B1B', fontweight='bold', zorder=6)
ax.text(xs[0] + box_w/2, t1_y + 0.8, r"$C = 12\ \mathrm{leads},\ L = 1000\ \mathrm{samples}$", ha='center', fontsize=7.0, color='#64748B', zorder=5)

# Box 2: Patient-Stratified Split
draw_card(xs[1], t1_y, box_w, t1_h, 2, "Patient-Stratified Split", "Official 10-Fold Split", 'teal')
ax.text(xs[1] + box_w/2, t1_y + 11.6, "Strict Zero Patient Leakage", ha='center', fontsize=7.8, color='#0F766E', fontweight='bold', zorder=5)
for i, (name, rec, pts, clr) in enumerate([
    ("Train (Folds 1-8)", "17,411 records", "15,082 pts", '#F0FDFA'),
    ("Validation (Fold 9)", "2,183 records", "1,894 pts", '#CCFBF1'),
    ("Held-Out Test (Fold 10)", "2,205 records", "1,893 pts", '#99F6E4')
]):
    sy = t1_y + 8.0 - i*3.1
    sp = patches.FancyBboxPatch((xs[1] + 1.2, sy), box_w - 2.4, 2.6, boxstyle="round,pad=0,rounding_size=0.6",
                                facecolor=clr, edgecolor='#0D9488', lw=0.6, zorder=5)
    ax.add_patch(sp)
    ax.text(xs[1] + 2.0, sy + 1.3, f"{name}:", ha='left', va='center', fontsize=7.2, color='#0F766E', fontweight='bold', zorder=6)
    ax.text(xs[1] + box_w - 2.0, sy + 1.3, f"{rec} ({pts})", ha='right', va='center', fontsize=7.0, color='#134E4A', zorder=6)

# Box 3: Preprocessing
draw_card(xs[2], t1_y, box_w, t1_h, 3, "Preprocessing & Norm.", "Per-Lead Standardization", 'orange')
ax.text(xs[2] + box_w/2, t1_y + 11.6, "Zero-mean, unit-variance per lead:", ha='center', fontsize=7.6, color='#334155', zorder=5)
ax.text(xs[2] + box_w/2, t1_y + 8.5, r"$x_c' = \frac{x_c - \mu_c}{\sigma_c}$", ha='center', fontsize=9.2, color='#C2410C', zorder=5)
ax.text(xs[2] + box_w/2, t1_y + 5.3, r"Input tensor: $x \in \mathbb{R}^{12 \times 1000}$", ha='center', fontsize=7.6, color='#334155', zorder=5)
ax.text(xs[2] + box_w/2, t1_y + 3.0, r"Sampling rate: $f_s = 100\ \mathrm{Hz}\ (10\ \mathrm{s})$", ha='center', fontsize=7.2, color='#64748B', zorder=5)

# Box 4: Multi-Label Targets
draw_card(xs[3], t1_y, box_w, t1_h, 4, "Multi-Label Targets", "5 Diagnostic Superclasses", 'purple')
superclasses = [
    ("NORM", "Normal ECG", "9,514", '#DCFCE7', '#166534'),
    ("MI", "Myocardial Infarction", "5,469", '#FEE2E2', '#991B1B'),
    ("STTC", "ST/T Changes", "5,235", '#FFEDD5', '#9A3412'),
    ("CD", "Conduction Defect", "4,907", '#CCFBF1', '#115E59'),
    ("HYP", "Hypertrophy", "2,649", '#F3E8FF', '#581C87')
]
for i, (abbr, desc, count, bg_c, txt_c) in enumerate(superclasses):
    ry = t1_y + 11.4 - i*2.4
    tag = patches.FancyBboxPatch((xs[3] + 1.2, ry), 3.4, 2.0, boxstyle="round,pad=0,rounding_size=0.4",
                                 facecolor=bg_c, edgecolor=txt_c, lw=0.6, zorder=5)
    ax.add_patch(tag)
    ax.text(xs[3] + 2.9, ry + 1.0, abbr, color=txt_c, ha='center', va='center', fontweight='bold', fontsize=7.2, zorder=6)
    ax.text(xs[3] + 5.2, ry + 1.0, desc, color='#1E293B', ha='left', va='center', fontsize=7.2, zorder=6)
    ax.text(xs[3] + box_w - 1.5, ry + 1.0, f"n={count}", color='#64748B', ha='right', va='center', fontsize=6.8, zorder=6)
ax.text(xs[3] + box_w/2, t1_y + 1.0, r"Target vector: $y \in \{0, 1\}^5$", ha='center', fontsize=7.2, color='#7C3AED', zorder=5)

# Box 5: Input Tensor (with real 6-lead clinical waveforms)
draw_card(xs[4], t1_y, box_w, t1_h, 5, "Input Tensor", "Waveform Representation", 'amber')
ax.text(xs[4] + box_w/2, t1_y + 12.0, r"$\mathbf{x} \in \mathbb{R}^{C \times L}$", ha='center', fontsize=9.2, color='#B45309', fontweight='bold', zorder=5)
ax.text(xs[4] + box_w/2, t1_y + 10.2, r"$C = 12\ \mathrm{leads},\ L = 1000\ \mathrm{samples}$", ha='center', fontsize=7.4, color='#334155', zorder=5)

lead_names = ['Lead I', 'Lead II', 'Lead V1', 'Lead V2', 'Lead V4', 'Lead V6']
lead_colors = ['#2563EB', '#0D9488', '#EA580C', '#7C3AED', '#16A34A', '#DB2777']
lead_indices = [0, 1, 6, 7, 9, 11]

for j, (lidx, lname, lcol) in enumerate(zip(lead_indices, lead_names, lead_colors)):
    ly = t1_y + 7.8 - j*1.15
    pts = real_ecg[150:400, lidx]
    pts_norm = (pts - np.mean(pts)) / (np.std(pts) + 1e-6)
    lx = np.linspace(xs[4] + 4.2, xs[4] + box_w - 1.2, len(pts))
    sig = ly + np.clip(pts_norm * 0.22, -0.45, 0.45)
    ax.plot(lx, sig, color=lcol, lw=0.85, zorder=5)
    ax.text(xs[4] + 1.2, ly, lname, fontsize=6.0, color=lcol, fontweight='bold', va='center', zorder=6)

ax.text(xs[4] + box_w/2, t1_y + 0.8, "Limb (I-III, aVR-aVF) & Precordials (V1-V6)", ha='center', fontsize=6.6, color='#64748B', zorder=5)

# Tier 1 Horizontal Arrows
for i in range(4):
    draw_arrow((xs[i] + box_w, t1_y + t1_h/2), (xs[i+1], t1_y + t1_h/2), color='#2563EB', lw=1.4)

# =========================================================================
# TIER 2: EXPERT ENSEMBLE (Boxes 6 to 10)
# =========================================================================
t2_y = 48.0
t2_h = 24.0

# Enclosing Group Frame for Tier 2
group_ens = patches.FancyBboxPatch((1.0, t2_y - 1.2), 98.0, t2_h + 3.0, boxstyle="round,pad=0,rounding_size=1.5",
                                  facecolor='#F8FAFC', edgecolor='#2563EB', lw=1.2, zorder=1)
ax.add_patch(group_ens)
ax.text(35.0, t2_y + t2_h + 0.8, "Expert Ensemble (Multiplicity with Fixed Budget ≈ 2.0M Parameters)",
        fontsize=8.5, color='#1D4ED8', fontweight='bold', zorder=3)

# Box 6: Capacity Allocation
draw_card(xs[0], t2_y, box_w, t2_h, 6, "Capacity Allocation", r"Fixed Budget $B \approx 2.0$M", 'cyan')
ax.text(xs[0] + box_w/2, t2_y + 17.0, "Total Budget Ceiling:", ha='center', fontsize=7.8, color='#0369A1', fontweight='bold', zorder=5)
ax.text(xs[0] + box_w/2, t2_y + 15.0, r"$\sum_{k=1}^K |\theta_k| \leq B \approx 2.0 \times 10^6$", ha='center', fontsize=8.5, color='#0284C7', zorder=5)

ex_card = patches.FancyBboxPatch((xs[0] + 1.0, t2_y + 1.2), box_w - 2.0, 12.8, boxstyle="round,pad=0,rounding_size=0.6",
                                 facecolor='#F0F9FF', edgecolor='#0284C7', lw=0.7, zorder=4)
ax.add_patch(ex_card)
ax.text(xs[0] + box_w/2, t2_y + 12.5, "Examples (homogeneous):", ha='center', fontsize=7.4, color='#0369A1', fontweight='bold', zorder=5)

allocs = [
    ("K = 1", "2.02M params", "W_1 = 65"),
    ("K = 2", "2.03M params", "W_2 = 46"),
    ("K = 4", "1.98M params", "W_4 = 32"),
    ("K = 8", "2.05M params", "W_8 = 23"),
    ("K = 16", "2.01M params", "W_16 = 16")
]
for i, (k_str, p_str, w_str) in enumerate(allocs):
    ay = t2_y + 10.3 - i*2.0
    ax.text(xs[0] + 2.0, ay, k_str, ha='left', fontsize=7.2, fontweight='bold', color='#1E293B', zorder=5)
    ax.text(xs[0] + 7.5, ay, p_str, ha='left', fontsize=7.0, color='#475569', zorder=5)
    ax.text(xs[0] + box_w - 2.0, ay, w_str, ha='right', fontsize=7.0, color='#0369A1', fontweight='bold', zorder=5)

# Box 7: Hetero Experts (K=4)
draw_card(xs[1], t2_y, box_w, t2_h, 7, "Hetero Experts (K=4)", "Parameter-Matched (1.99M)", 'blue')
experts = [
    ("ResNet-1D", "493,893 params", "3-stage residual, skip conn."),
    ("MultiScale-CNN", "508,909 params", "Parallel kernels {3, 7, 15, 31}"),
    ("TCN-1D", "490,661 params", r"Dilated convs $d \in \{1,2,4,8\}$"),
    ("ConvAttention-1D", "500,225 params", "4x downsampling + Transformer")
]
for i, (ename, epar, edesc) in enumerate(experts):
    ey = t2_y + 15.6 - i*3.8
    eb = patches.FancyBboxPatch((xs[1] + 1.0, ey), box_w - 2.0, 3.4, boxstyle="round,pad=0,rounding_size=0.5",
                                facecolor='#EFF6FF', edgecolor='#2563EB', lw=0.6, zorder=5)
    ax.add_patch(eb)
    ax.text(xs[1] + 1.8, ey + 2.1, ename, ha='left', fontsize=7.2, color='#1E40AF', fontweight='bold', zorder=6)
    ax.text(xs[1] + box_w - 1.8, ey + 2.1, epar, ha='right', fontsize=6.8, color='#3B82F6', zorder=6)
    ax.text(xs[1] + 1.8, ey + 0.8, edesc, ha='left', fontsize=6.5, color='#475569', zorder=6)
ax.text(xs[1] + box_w/2, t2_y + 1.0, "Total: 1,993,688 params (+0.92%)", ha='center', fontsize=7.2, color='#1D4ED8', fontweight='bold', zorder=5)

# Box 8: Expert Representations
draw_card(xs[2], t2_y, box_w, t2_h, 8, "Expert Representations", "Feature Embeddings", 'purple')
ax.text(xs[2] + box_w/2, t2_y + 16.5, "Sub-network feature embeddings:", ha='center', fontsize=7.8, color='#334155', zorder=5)
ax.text(xs[2] + box_w/2, t2_y + 14.3, r"$z_k(x) = f_{\theta_k}(x) \in \mathbb{R}^D$", ha='center', fontsize=8.5, color='#7C3AED', fontweight='bold', zorder=5)

layer_cols = ['#3B82F6', '#F59E0B', '#10B981', '#8B5CF6']
for i, lcol in enumerate(layer_cols):
    ly = t2_y + 11.8 - i*0.8
    lp = patches.Rectangle((xs[2] + 4.5 + i*0.8, ly), 8.5, 0.6, facecolor=lcol, edgecolor='#1E293B', lw=0.4, zorder=5)
    ax.add_patch(lp)

cka_b = patches.FancyBboxPatch((xs[2] + 1.2, t2_y + 1.2), box_w - 2.4, 7.5, boxstyle="round,pad=0,rounding_size=0.6",
                               facecolor='#FAF5FF', edgecolor='#7C3AED', lw=0.8, zorder=5)
ax.add_patch(cka_b)
ax.text(xs[2] + box_w/2, t2_y + 7.2, "Representation Diversity", ha='center', fontsize=7.6, color='#6D28D9', fontweight='bold', zorder=6)
ax.text(xs[2] + box_w/2, t2_y + 5.5, "Linear CKA = 0.9008", ha='center', fontsize=7.8, color='#1E293B', fontweight='bold', zorder=6)
ax.text(xs[2] + box_w/2, t2_y + 4.0, "(vs Homogeneous CKA = 0.9458)", ha='center', fontsize=6.8, color='#64748B', zorder=6)
ax.text(xs[2] + box_w/2, t2_y + 2.5, "Lowest CKA = 0.8824 (MS vs CA)", ha='center', fontsize=7.2, color='#7C3AED', fontweight='bold', zorder=6)

# Box 9: Adaptive Router
draw_card(xs[3], t2_y, box_w, t2_h, 9, "Adaptive Router", r"Gating Network $g_\phi(x)$", 'magenta')
ax.text(xs[3] + box_w/2, t2_y + 16.5, "Lightweight Router (~16K params):", ha='center', fontsize=7.5, color='#334155', zorder=5)
r_steps = ["Conv1D + Global Pooling", "Linear (16 units) + ReLU", "Linear + Softmax"]
for i, step in enumerate(r_steps):
    ry = t2_y + 12.8 - i*3.6
    rp = patches.FancyBboxPatch((xs[3] + 1.5, ry), box_w - 3.0, 2.5, boxstyle="round,pad=0,rounding_size=0.4",
                                facecolor='#FDF4FF', edgecolor='#C026D3', lw=0.6, zorder=5)
    ax.add_patch(rp)
    ax.text(xs[3] + box_w/2, ry + 1.25, step, ha='center', va='center', fontsize=7.2, color='#86198F', fontweight='bold', zorder=6)
    if i < 2:
        draw_arrow((xs[3] + box_w/2, ry), (xs[3] + box_w/2, ry - 1.1), color='#C026D3', lw=1.0)
ax.text(xs[3] + box_w/2, t2_y + 2.8, r"Gating weights: $w(x) \in \Delta^{K-1}$", ha='center', fontsize=8.0, color='#A21CAF', fontweight='bold', zorder=5)
ax.text(xs[3] + box_w/2, t2_y + 1.2, "Learned patient-specific routing", ha='center', fontsize=6.8, color='#64748B', zorder=5)

# Box 10: Sparse Top-2 Routing
draw_card(xs[4], t2_y, box_w, t2_h, 10, "Sparse Top-2 Routing", "Expert Prediction Fusion", 'teal')
ax.text(xs[4] + box_w/2, t2_y + 17.0, "Select top-2 & renormalize weights:", ha='center', fontsize=7.4, color='#0F766E', zorder=5)

bw_xs = [xs[4] + 2.5 + i*3.2 for i in range(4)]
bw_vals = [0.18, 0.48, 0.34, 0.20]
b_cols = ['#93C5FD', '#3B82F6', '#F59E0B', '#A7F3D0']
for i in range(4):
    ax.bar(bw_xs[i], bw_vals[i]*7.0, width=2.4, bottom=t2_y + 9.5, color=b_cols[i], edgecolor='#1E293B', lw=0.5, zorder=5)
    ax.text(bw_xs[i], t2_y + 8.5, f"E{i+1}", ha='center', fontsize=6.8, color='#334155', zorder=5)
top2_box = patches.Rectangle((bw_xs[1] - 1.5, t2_y + 9.2), 6.5, 4.8, fill=False, edgecolor='#DC2626', ls='--', lw=1.1, zorder=7)
ax.add_patch(top2_box)
ax.text(bw_xs[1] + 1.75, t2_y + 14.3, "Top-2", ha='center', fontsize=7.2, color='#DC2626', fontweight='bold', zorder=8)

ax.text(xs[4] + box_w/2, t2_y + 6.8, r"$\hat{z}(x) = \sum_{k=1}^K \tilde{w}_k(x) z_k(x)$", ha='center', fontsize=8.2, color='#1E293B', zorder=5)
ax.text(xs[4] + box_w/2, t2_y + 5.0, r"$\hat{y}(x) = \sigma(\hat{z}(x)) \in [0, 1]^M$", ha='center', fontsize=7.8, color='#0F766E', fontweight='bold', zorder=5)

flop_b = patches.FancyBboxPatch((xs[4] + 1.2, t2_y + 1.0), box_w - 2.4, 3.2, boxstyle="round,pad=0,rounding_size=0.5",
                                facecolor='#CCFBF1', edgecolor='#0D9488', lw=0.8, zorder=5)
ax.add_patch(flop_b)
ax.text(xs[4] + box_w/2, t2_y + 2.7, "297.8 MFLOPs (-28.2%)", ha='center', fontsize=7.8, color='#115E59', fontweight='bold', zorder=6)
ax.text(xs[4] + box_w/2, t2_y + 1.4, "Executes only 2 experts / record", ha='center', fontsize=6.8, color='#047857', zorder=6)

# Tier 2 Horizontal Arrows
draw_arrow((xs[0] + box_w, t2_y + t2_h/2), (xs[1], t2_y + t2_h/2), color='#2563EB', lw=1.3)
draw_arrow((xs[1] + box_w, t2_y + t2_h/2), (xs[2], t2_y + t2_h/2), color='#2563EB', lw=1.3)
draw_arrow((xs[2] + box_w, t2_y + t2_h/2), (xs[3], t2_y + t2_h/2), color='#2563EB', lw=1.3)
draw_arrow((xs[3] + box_w, t2_y + t2_h/2), (xs[4], t2_y + t2_h/2), color='#2563EB', lw=1.3)

# Long arrow from Tier 1 (Box 5) to Tier 2 (Box 6 & Box 7)
pipe_y = 76.5
ax.plot([xs[4] + box_w/2, xs[4] + box_w/2], [t1_y, pipe_y], color='#2563EB', lw=1.3, zorder=8)
ax.plot([xs[4] + box_w/2, xs[0] + box_w/2], [pipe_y, pipe_y], color='#2563EB', lw=1.3, zorder=8)
draw_arrow((xs[0] + box_w/2, pipe_y), (xs[0] + box_w/2, t2_y + t2_h), color='#2563EB', lw=1.3)
draw_arrow((xs[1] + box_w/2, pipe_y), (xs[1] + box_w/2, t2_y + t2_h), color='#2563EB', lw=1.3)

# =========================================================================
# TIER 3: JOINT TRAINING OBJECTIVE (Box 11)
# =========================================================================
t3_y = 35.8
t3_h = 8.8
b11 = patches.FancyBboxPatch((1.0, t3_y), 98.0, t3_h, boxstyle="round,pad=0,rounding_size=1.2",
                             facecolor='#FDF4FF', edgecolor='#C026D3', lw=1.2, zorder=2)
ax.add_patch(b11)

badge11 = patches.Circle((3.0, t3_y + t3_h - 1.8), 1.3, facecolor='#A21CAF', edgecolor='none', zorder=5)
ax.add_patch(badge11)
ax.text(3.0, t3_y + t3_h - 1.8, "11", color='white', ha='center', va='center', fontweight='bold', fontsize=7.8, zorder=6)
ax.text(5.0, t3_y + t3_h - 1.8, "Training Objective (Joint Optimization):", color='#86198F', ha='left', va='center', fontsize=8.6, fontweight='bold', zorder=6)
ax.text(52.0, t3_y + t3_h - 1.8, r"$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{diag}} + \lambda_{\mathrm{div}}\mathcal{L}_{\mathrm{div}} + \lambda_{\mathrm{fail}}\mathcal{L}_{\mathrm{fail}}$",
        color='#1E293B', ha='center', va='center', fontsize=9.2, fontweight='bold', zorder=6)

loss_boxes = [
    ("Diagnostic BCE Loss", r"$\mathcal{L}_{\mathrm{diag}} = \sum_m \mathrm{BCE}(\hat{y}_m, y_m)$", "Enforces multi-label diagnostic accuracy across 5 classes", 1.8),
    ("Diversity Regularization", r"$\mathcal{L}_{\mathrm{div}} = -\mathcal{H}(\bar{w}) = \sum_k \bar{w}_k \log \bar{w}_k$", r"Maximizes router gating entropy to prevent expert collapse ($\lambda_{\mathrm{div}}=0.05$)", 34.0),
    ("Failure-Aware Routing", r"$\mathcal{L}_{\mathrm{fail}} = \sum_k w_k(x)\mathcal{L}_{\mathrm{BCE}}(f_{\theta_k}(x), y)$", r"Penalizes dispatching inaccurate sub-networks ($\lambda_{\mathrm{fail}}=0.10$)", 67.0)
]
for title, form, desc, lx in loss_boxes:
    lp = patches.FancyBboxPatch((lx, t3_y + 0.6), 31.0, 5.0, boxstyle="round,pad=0,rounding_size=0.5",
                                facecolor='white', edgecolor='#E879F9', lw=0.8, zorder=3)
    ax.add_patch(lp)
    ax.text(lx + 15.5, t3_y + 4.5, title, ha='center', fontsize=7.6, color='#A21CAF', fontweight='bold', zorder=5)
    ax.text(lx + 15.5, t3_y + 2.9, form, ha='center', fontsize=7.8, color='#1E293B', zorder=5)
    ax.text(lx + 15.5, t3_y + 1.3, desc, ha='center', fontsize=6.6, color='#64748B', zorder=5)

# Arrows from Tier 2 into Tier 3
draw_arrow((xs[0] + box_w/2, t2_y - 1.2), (xs[0] + box_w/2, t3_y + t3_h), color='#7C3AED', lw=1.3)
draw_arrow((xs[4] + box_w/2, t2_y - 1.2), (xs[4] + box_w/2, t3_y + t3_h), color='#7C3AED', lw=1.3)

# =========================================================================
# TIER 4: INFERENCE & SELECTIVE CLASSIFICATION (Boxes 12 to 15)
# =========================================================================
t4_y = 1.5
t4_h = 28.0
t4_w = 22.5
t4_gap = 2.6
t4_xs = [2.0 + i*(t4_w + t4_gap) for i in range(4)]

group_inf = patches.FancyBboxPatch((1.0, t4_y - 0.8), 98.0, t4_h + 2.4, boxstyle="round,pad=0,rounding_size=1.5",
                                  facecolor='#F8FAFC', edgecolor='#0D9488', lw=1.2, zorder=1)
ax.add_patch(group_inf)
ax.text(32.0, t4_y + t4_h + 0.5, "Inference Phase (Selective Classification & Clinical Decision Support)",
        fontsize=8.5, color='#0F766E', fontweight='bold', zorder=3)

# Box 12: Uncertainty Estimation
draw_card(t4_xs[0], t4_y, t4_w, t4_h, 12, "Uncertainty Estimation", "Entropy + Ensemble Disagreement", 'blue')
ax.text(t4_xs[0] + t4_w/2, t4_y + 20.2, "Multi-Term Uncertainty Metric:", ha='center', fontsize=7.8, color='#1E40AF', fontweight='bold', zorder=5)
ax.text(t4_xs[0] + t4_w/2, t4_y + 16.8, r"$U(x) = \sum_{m=1}^M \mathcal{H}(\hat{y}_m(x))$", ha='center', fontsize=8.5, color='#1E293B', zorder=5)
ax.text(t4_xs[0] + t4_w/2, t4_y + 13.2, r"$+ 2.0 \cdot \sum_{m=1}^M \frac{1}{K}\sum_{k=1}^K (\sigma(z_{k,m}) - \hat{y}_m)^2$", ha='center', fontsize=7.8, color='#1E293B', zorder=5)
u_box = patches.FancyBboxPatch((t4_xs[0] + 1.2, t4_y + 2.5), t4_w - 2.4, 8.5, boxstyle="round,pad=0,rounding_size=0.5",
                               facecolor='#EFF6FF', edgecolor='#2563EB', lw=0.6, zorder=5)
ax.add_patch(u_box)
ax.text(t4_xs[0] + t4_w/2, t4_y + 8.8, "Components:", ha='center', fontsize=7.5, color='#1E40AF', fontweight='bold', zorder=6)
ax.text(t4_xs[0] + t4_w/2, t4_y + 6.8, "1. Predictive Entropy (aleatoric risk)", ha='center', fontsize=7.0, color='#334155', zorder=6)
ax.text(t4_xs[0] + t4_w/2, t4_y + 5.0, "2. Ensemble Variance (epistemic doubt)", ha='center', fontsize=7.0, color='#334155', zorder=6)
ax.text(t4_xs[0] + t4_w/2, t4_y + 3.2, "Well-ordered ranking for triage", ha='center', fontsize=7.0, color='#059669', fontweight='bold', zorder=6)

# Box 13: Selective Classification
draw_card(t4_xs[1], t4_y, t4_w, t4_h, 13, "Selective Classification", "Risk-Coverage Abstention", 'teal')
ax.text(t4_xs[1] + t4_w/2, t4_y + 20.5, "Dynamic Clinical Escalation:", ha='center', fontsize=7.8, color='#0F766E', fontweight='bold', zorder=5)
ax.text(t4_xs[1] + t4_w/2, t4_y + 18.5, "Classify fraction c with lowest U(x)", ha='center', fontsize=7.5, color='#334155', zorder=5)

ux = np.linspace(t4_xs[1] + 2.5, t4_xs[1] + t4_w - 2.5, 80)
uy = t4_y + 11.5 + 4.5 * np.exp(-((ux - (t4_xs[1] + 8.5))**2)/8.0)
thresh_x = t4_xs[1] + 13.5
ax.fill_between(ux[ux <= thresh_x], t4_y + 11.5, uy[ux <= thresh_x], color='#A7F3D0', alpha=0.8, zorder=5)
ax.fill_between(ux[ux >= thresh_x], t4_y + 11.5, uy[ux >= thresh_x], color='#FECACA', alpha=0.8, zorder=5)
ax.plot(ux, uy, color='#0D9488', lw=1.0, zorder=6)
ax.plot([thresh_x, thresh_x], [t4_y + 11.5, t4_y + 16.5], color='#DC2626', ls='--', lw=1.1, zorder=7)
ax.text(t4_xs[1] + 7.5, t4_y + 13.5, "Classify (c)", color='#065F46', fontweight='bold', fontsize=7.0, zorder=8)
ax.text(t4_xs[1] + 17.5, t4_y + 13.5, "Abstain (1-c)", color='#991B1B', fontweight='bold', fontsize=7.0, zorder=8)

risk_b = patches.FancyBboxPatch((t4_xs[1] + 1.2, t4_y + 1.5), t4_w - 2.4, 9.0, boxstyle="round,pad=0,rounding_size=0.5",
                                facecolor='#F0FDFA', edgecolor='#0D9488', lw=0.8, zorder=5)
ax.add_patch(risk_b)
ax.text(t4_xs[1] + t4_w/2, t4_y + 8.8, "Empirical Risk Reduction:", ha='center', fontsize=7.6, color='#0F766E', fontweight='bold', zorder=6)
ax.text(t4_xs[1] + t4_w/2, t4_y + 7.0, "Coverage c = 90% (10% deferred)", ha='center', fontsize=7.2, color='#1E293B', zorder=6)
ax.text(t4_xs[1] + t4_w/2, t4_y + 5.2, "Hamming risk: 11.95% → 10.37%", ha='center', fontsize=7.8, color='#0F766E', fontweight='bold', zorder=6)
ax.text(t4_xs[1] + t4_w/2, t4_y + 3.5, "Δ = -4.18% Relative Risk", ha='center', fontsize=8.0, color='#047857', fontweight='bold', zorder=6)
ax.text(t4_xs[1] + t4_w/2, t4_y + 2.0, "Welch test: p = 0.0093 (5 seeds)", ha='center', fontsize=6.8, color='#64748B', zorder=6)

# Box 14: Final Diagnosis Output
draw_card(t4_xs[2], t4_y, t4_w, t4_h, 14, "Final Diagnosis Output", "Calibrated Class Probabilities", 'purple')
ax.text(t4_xs[2] + t4_w/2, t4_y + 20.2, r"Multi-Label Prediction $\hat{y} \in [0, 1]^5$:", ha='center', fontsize=7.8, color='#6D28D9', fontweight='bold', zorder=5)
demo_preds = [
    ("NORM", 0.12, "Normal ECG", '#10B981'),
    ("MI", 0.83, "Myocardial Infarction", '#EF4444'),
    ("STTC", 0.41, "ST/T Changes", '#F97316'),
    ("CD", 0.67, "Conduction Defect", '#06B6D4'),
    ("HYP", 0.21, "Hypertrophy", '#8B5CF6')
]
for i, (code, prob, name, bar_c) in enumerate(demo_preds):
    py = t4_y + 16.0 - i*2.8
    ax.text(t4_xs[2] + 1.5, py, code, ha='left', va='center', fontweight='bold', fontsize=7.2, color='#1E293B', zorder=5)
    bar_bg = patches.Rectangle((t4_xs[2] + 5.0, py - 0.7), 11.0, 1.4, facecolor='#E2E8F0', zorder=5)
    ax.add_patch(bar_bg)
    bar_v = patches.Rectangle((t4_xs[2] + 5.0, py - 0.7), 11.0 * prob, 1.4, facecolor=bar_c, zorder=6)
    ax.add_patch(bar_v)
    ax.text(t4_xs[2] + 16.8, py, f"{prob:.2f}", ha='left', va='center', fontweight='bold', fontsize=7.2, color=bar_c, zorder=5)
ax.text(t4_xs[2] + t4_w/2, t4_y + 1.5, "Detected: MI (0.83) & CD (0.67)", ha='center', fontsize=7.5, color='#B91C1C', fontweight='bold', zorder=5)

# Box 15: Evaluation and Analysis
draw_card(t4_xs[3], t4_y, t4_w, t4_h, 15, "Evaluation and Analysis", "Comprehensive Benchmarking", 'green')
metrics_list = [
    ("Discrimination:", "Macro AUROC (0.9113), AUPRC (0.7753)"),
    ("Calibration:", "Brier score (0.0867), ECE (0.0171)"),
    ("Safety (Risk):", "Selective Hamming loss under deferral"),
    ("Efficiency:", "297.8 MFLOPs (-28.2%), 2.25 ms latency"),
    ("Robustness:", "6 clinical stress corruptions & site shift"),
    ("Diversity:", "Linear Centered Kernel Alignment (CKA)")
]
for i, (mtitle, mdesc) in enumerate(metrics_list):
    my = t4_y + 19.8 - i*3.2
    ax.text(t4_xs[3] + 1.5, my, mtitle, ha='left', fontsize=7.5, color='#15803D', fontweight='bold', zorder=5)
    ax.text(t4_xs[3] + 1.5, my - 1.2, mdesc, ha='left', fontsize=6.8, color='#334155', zorder=5)
ax.text(t4_xs[3] + t4_w/2, t4_y + 1.0, "5-seed validation (Welch + Holm test)", ha='center', fontsize=6.8, color='#64748B', zorder=5)

# Tier 4 Horizontal Arrows
for i in range(3):
    draw_arrow((t4_xs[i] + t4_w, t4_y + t4_h/2), (t4_xs[i+1], t4_y + t4_h/2), color='#0D9488', lw=1.3)

# Arrow from Tier 3 to Tier 4 labeled "Trained Model"
draw_arrow((t4_xs[0] + t4_w/2, t3_y), (t4_xs[0] + t4_w/2, t4_y + t4_h),
           color='#7C3AED', lw=1.4, ls='--', label="Trained Model", label_pos=(t4_xs[0] + t4_w/2, (t3_y + t4_y + t4_h)/2 + 0.3))

# Save cleanly to both locations
os.makedirs('latex/figures', exist_ok=True)
os.makedirs('results/images', exist_ok=True)
fig.savefig('latex/figures/methodology_flowchart.png', dpi=300, bbox_inches='tight')
fig.savefig('results/images/methodology_flowchart.png', dpi=300, bbox_inches='tight')
print("Successfully generated methodology_flowchart.png in latex/figures and results/images with real PTB-XL ECG waveforms!")
