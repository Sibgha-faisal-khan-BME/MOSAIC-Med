"""
Fix Evidence and Figures (Item 3).
1. Computes real linear CKA between expert embeddings for Homogeneous vs Heterogeneous and saves results/tables/cka_comparison.csv.
2. Fixes mislabeled CKA panel in Figure 2 (Panel A: ResNet #1-#4; Panel B: ResNet, MS, TCN, Attn) -> results/images/cka_representation_diversity.png & .pdf.
3. Fixes hardcoded Stage 6 in ablation waterfall: computes REAL AUROC/AUPRC/Brier on the retained 90% coverage test set -> updates results/tables/ablation_summary.csv and regenerates results/images/ablation_waterfall.png & .pdf.
4. Updates Figure 8 (topline_mosaic_vs_single.png & .pdf) with measured FLOPs (123.9M, 126.8M, 413.7M, 208.0M) instead of estimated values.
Logs runtime to results/tables/RUN_LOG.txt.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scratch.py.data import cache_ptbxl_dataset
from scratch.py.models import create_expert, AdaptiveRouter, compute_predictive_uncertainty
from scratch.py.training import get_dataloader, evaluate_model, free_gpu_memory, DEVICE
from scratch.py.metrics import compute_all_diagnostic_metrics
from scratch.py.experiments import compute_cka_matrix, profile_model_flops_and_params
from scratch.py.plotting import set_ieee_style, save_ieee_fig, COLORS

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

CHECKPOINT_DIR = "scratch/checkpoints"
TABLES_DIR = "results/tables"
IMAGES_DIR = "results/images"

def run_item_3():
    t_start = time.time()
    print("=" * 70)
    print("STARTING ITEM 3: FIX EVIDENCE, MEASURED FLOPS & REGENERATE FIGURES")
    print("=" * 70)

    # Load test data
    data = cache_ptbxl_dataset()
    X_test, y_test = data['X_test'], data['y_test']
    test_loader = get_dataloader(X_test, y_test, batch_size=64, shuffle=False)

    # -------------------------------------------------------------
    # 1. Compute Real Linear CKA Between Expert Embeddings
    # -------------------------------------------------------------
    print("\n>>> 1. Computing Real Linear CKA for Homogeneous and Heterogeneous Experts...")

    # Homogeneous (4x ResNet @ 500k, seed 42)
    homo_feats = []
    for e_idx in range(4):
        ckpt = os.path.join(CHECKPOINT_DIR, f"resnet_K4_500k_s42_e{e_idx}.pt")
        m = create_expert('resnet', '500k')
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        m = m.to(DEVICE)
        _, ft, _, _ = evaluate_model(m, test_loader, return_features=True)
        homo_feats.append(ft)
        del m
        free_gpu_memory()

    cka_homo = compute_cka_matrix(homo_feats)

    # Heterogeneous (ResNet, MultiScale, TCN, ConvAttn @ 500k, seed 42)
    hetero_feats = []
    hetero_models_list = []
    hetero_probs = []
    arches = ['resnet', 'multiscale', 'tcn', 'conv_att']
    for arch in arches:
        ckpt = os.path.join(CHECKPOINT_DIR, f"hetero_{arch}_500k_s42.pt")
        m = create_expert(arch, '500k')
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        m = m.to(DEVICE)
        hetero_models_list.append(m)
        pr, ft, _, _ = evaluate_model(m, test_loader, return_features=True)
        hetero_probs.append(pr)
        hetero_feats.append(ft)

    cka_hetero = compute_cka_matrix(hetero_feats)

    # Save cka_comparison.csv
    cka_rows = []
    homo_labels = ['ResNet #1', 'ResNet #2', 'ResNet #3', 'ResNet #4']
    hetero_labels = ['ResNet-1D', 'MultiScale', 'TCN', 'ConvAttn']

    for i in range(4):
        for j in range(i + 1, 4):
            cka_rows.append({
                'Pair Index': f"({i+1}, {j+1})",
                'Homogeneous Pair': f"{homo_labels[i]} vs {homo_labels[j]}",
                'Homogeneous CKA': cka_homo[i, j],
                'Heterogeneous Pair': f"{hetero_labels[i]} vs {hetero_labels[j]}",
                'Heterogeneous CKA': cka_hetero[i, j],
                'Diversity Delta (Homo - Hetero)': cka_homo[i, j] - cka_hetero[i, j]
            })

    mean_homo_cka = np.mean(cka_homo[np.triu_indices(4, k=1)])
    mean_hetero_cka = np.mean(cka_hetero[np.triu_indices(4, k=1)])
    cka_rows.append({
        'Pair Index': 'Mean Off-Diagonal',
        'Homogeneous Pair': 'All Homogeneous Pairs',
        'Homogeneous CKA': mean_homo_cka,
        'Heterogeneous Pair': 'All Heterogeneous Pairs',
        'Heterogeneous CKA': mean_hetero_cka,
        'Diversity Delta (Homo - Hetero)': mean_homo_cka - mean_hetero_cka
    })

    cka_df = pd.DataFrame(cka_rows)
    cka_csv_path = os.path.join(TABLES_DIR, "cka_comparison.csv")
    cka_df.to_csv(cka_csv_path, index=False)
    print(f"Saved {cka_csv_path}:")
    print(cka_df[['Homogeneous Pair', 'Homogeneous CKA', 'Heterogeneous Pair', 'Heterogeneous CKA']])

    # -------------------------------------------------------------
    # 2. Regenerate Figure 2: CKA Heatmap with Correct Labels
    # -------------------------------------------------------------
    print("\n>>> 2. Regenerating Figure 2 (Fixed CKA labels)...")
    set_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2))

    im1 = axes[0].imshow(cka_homo, cmap='Blues', vmin=0.4, vmax=1.0)
    axes[0].set_title("A. Homogeneous Experts (4x ResNet)")
    axes[0].set_xticks(range(4))
    axes[0].set_yticks(range(4))
    axes[0].set_xticklabels(homo_labels, rotation=30, ha='right')
    axes[0].set_yticklabels(homo_labels)
    for i in range(4):
        for j in range(4):
            val = cka_homo[i, j]
            axes[0].text(j, i, f"{val:.2f}", ha="center", va="center",
                         color="white" if val > 0.75 else "black", fontsize=8.0, fontweight='bold')

    im2 = axes[1].imshow(cka_hetero, cmap='viridis', vmin=0.4, vmax=1.0)
    axes[1].set_title("B. Heterogeneous Experts (ResNet, MS, TCN, Attn)")
    axes[1].set_xticks(range(4))
    axes[1].set_yticks(range(4))
    axes[1].set_xticklabels(hetero_labels, rotation=30, ha='right')
    axes[1].set_yticklabels(hetero_labels)
    for i in range(4):
        for j in range(4):
            val = cka_hetero[i, j]
            axes[1].text(j, i, f"{val:.2f}", ha="center", va="center",
                         color="white" if val > 0.75 else "black", fontsize=8.0, fontweight='bold')

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.20, 0.02, 0.65])
    fig.colorbar(im2, cax=cbar_ax, label='Linear Centered Kernel Alignment (CKA)')
    save_ieee_fig(fig, "cka_representation_diversity")

    # -------------------------------------------------------------
    # 3. Measured FLOPs & Real Stage 6 in Ablation Summary
    # -------------------------------------------------------------
    print("\n>>> 3. Computing Measured FLOPs and Real Stage 6 Performance...")
    # Single large model K=1
    single_ckpt = os.path.join(CHECKPOINT_DIR, "resnet_K1_2m_s42_e0.pt")
    single_model = create_expert('resnet', '2m')
    single_model.load_state_dict(torch.load(single_ckpt, map_location=DEVICE))
    single_model = single_model.to(DEVICE)
    y_prob_single, _, _, _ = evaluate_model(single_model, test_loader)
    m_single = compute_all_diagnostic_metrics(y_test, y_prob_single)

    # Adaptive router & top-2 predictions for seed 42
    router_ckpt = os.path.join(CHECKPOINT_DIR, "adaptive_router.pt")
    router = AdaptiveRouter(in_channels=12, num_experts=4, hidden_dim=64)
    router.load_state_dict(torch.load(router_ckpt, map_location=DEVICE))
    router = router.to(DEVICE)
    router.eval()

    weights_list = []
    dense_list = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(DEVICE)
            with torch.amp.autocast('cuda'):
                w, _ = router(bx)
                weights_list.append(w.cpu().numpy())
                lgs = [exp(bx) for exp in hetero_models_list]
                fused = torch.sum(torch.stack(lgs, dim=1) * w.unsqueeze(-1), dim=1)
                dense_list.append(torch.sigmoid(fused).cpu().numpy())

    all_w = np.concatenate(weights_list, axis=0)
    y_prob_dense = np.concatenate(dense_list, axis=0)
    m_dense = compute_all_diagnostic_metrics(y_test, y_prob_dense)

    top2_w = np.zeros_like(all_w)
    top2_flops_per_sample = []
    # Profile per-expert flops
    m_flps = [profile_model_flops_and_params(exp)[1] for exp in hetero_models_list] # [31.69, 50.72, 245.86, 85.40]
    r_flp = profile_model_flops_and_params(router)[1] # ~1.19 MFLOPs

    for i in range(len(all_w)):
        t2 = np.argsort(all_w[i])[-2:]
        top2_w[i, t2] = all_w[i, t2]
        top2_w[i] /= np.sum(top2_w[i])
        sample_flp = r_flp + m_flps[t2[0]] + m_flps[t2[1]]
        top2_flops_per_sample.append(sample_flp)

    measured_top2_flops = float(np.mean(top2_flops_per_sample))
    print(f"Measured Sparse Top-2 Router Average FLOPs: {measured_top2_flops:.2f} MFLOPs")

    y_prob_top2 = np.zeros_like(y_prob_dense)
    for k in range(4):
        y_prob_top2 += hetero_probs[k] * top2_w[:, k:k+1]
    m_top2 = compute_all_diagnostic_metrics(y_test, y_prob_top2)

    # Real Stage 6: Selective Abstention at 90% coverage
    exp_probs_t = [torch.from_numpy(p) for p in hetero_probs]
    unc_m, dis_m = compute_predictive_uncertainty(torch.from_numpy(y_prob_top2), exp_probs_t)
    total_unc = unc_m.numpy() + 2.0 * dis_m.numpy()

    # Retain lowest 90% uncertainty
    cov_threshold = np.percentile(total_unc, 90.0)
    retained_mask = (total_unc <= cov_threshold)
    y_test_90 = y_test[retained_mask]
    y_prob_top2_90 = y_prob_top2[retained_mask]

    m_stage6 = compute_all_diagnostic_metrics(y_test_90, y_prob_top2_90)
    print(f"Real Stage 6 (90% Coverage): AUROC={m_stage6['auroc']:.4f}, AUPRC={m_stage6['auprc']:.4f}, F1={m_stage6['f1']:.4f}, Brier={m_stage6['brier']:.4f}")

    # Load 5-seed scaling summary and diversity summary for means
    scaling_sum_df = pd.read_csv(os.path.join(TABLES_DIR, "scaling_summary.csv"))
    k1_auroc = float(scaling_sum_df[scaling_sum_df.K == 1]['AUROC'].iloc[0].split('±')[0])
    k1_auprc = float(scaling_sum_df[scaling_sum_df.K == 1]['AUPRC'].iloc[0].split('±')[0])
    k1_f1 = float(scaling_sum_df[scaling_sum_df.K == 1]['F1'].iloc[0].split('±')[0])
    k1_brier = float(scaling_sum_df[scaling_sum_df.K == 1]['Brier Score'].iloc[0].split('±')[0])

    k4_auroc = float(scaling_sum_df[scaling_sum_df.K == 4]['AUROC'].iloc[0].split('±')[0])
    k4_auprc = float(scaling_sum_df[scaling_sum_df.K == 4]['AUPRC'].iloc[0].split('±')[0])
    k4_f1 = float(scaling_sum_df[scaling_sum_df.K == 4]['F1'].iloc[0].split('±')[0])
    k4_brier = float(scaling_sum_df[scaling_sum_df.K == 4]['Brier Score'].iloc[0].split('±')[0])

    div_df = pd.read_csv(os.path.join(TABLES_DIR, "diversity_comparison.csv"))
    het_sub = div_df[div_df.type.str.contains('Hetero')]
    het_auroc = het_sub.auroc.mean()
    het_auprc = het_sub.auprc.mean()
    het_f1 = het_sub.f1.mean()
    het_brier = het_sub.brier.mean()

    # Re-build ablation_summary.csv with measured values
    ablation_records = [
        {'Ablation Stage': '1. Single Large Network (1x2M)', 'AUROC': k1_auroc, 'AUPRC': k1_auprc, 'F1': k1_f1, 'Brier': k1_brier, 'Params': '2.02M', 'FLOPs': '123.9M'},
        {'Ablation Stage': '2. Split-4 Homogeneous (4x500K)', 'AUROC': k4_auroc, 'AUPRC': k4_auprc, 'F1': k4_f1, 'Brier': k4_brier, 'Params': '1.98M', 'FLOPs': '126.8M'},
        {'Ablation Stage': '3. Split-4 Heterogeneous (MS+TCN+Att)', 'AUROC': het_auroc, 'AUPRC': het_auprc, 'F1': het_f1, 'Brier': het_brier, 'Params': '1.99M', 'FLOPs': '413.7M'},
        {'Ablation Stage': '4. + Learned Adaptive Router (Dense)', 'AUROC': m_dense['auroc'], 'AUPRC': m_dense['auprc'], 'F1': m_dense['f1'], 'Brier': m_dense['brier'], 'Params': '2.01M', 'FLOPs': '414.9M'},
        {'Ablation Stage': '5. + Sparse Top-2 Escalation (MOSAIC-Med)', 'AUROC': m_top2['auroc'], 'AUPRC': m_top2['auprc'], 'F1': m_top2['f1'], 'Brier': m_top2['brier'], 'Params': '2.01M', 'FLOPs': f"{measured_top2_flops:.1f}M"},
        {'Ablation Stage': '6. + Selective Abstention (90% Coverage)', 'AUROC': m_stage6['auroc'], 'AUPRC': m_stage6['auprc'], 'F1': m_stage6['f1'], 'Brier': m_stage6['brier'], 'Params': '2.01M', 'FLOPs': f"{measured_top2_flops:.1f}M"}
    ]
    ablation_df = pd.DataFrame(ablation_records)
    ablation_csv_path = os.path.join(TABLES_DIR, "ablation_summary.csv")
    ablation_df.to_csv(ablation_csv_path, index=False)
    print(f"\nUpdated {ablation_csv_path}:")
    print(ablation_df[['Ablation Stage', 'AUROC', 'AUPRC', 'Brier', 'FLOPs']])

    # Regenerate Figure 6: Ablation Waterfall
    print("\n>>> 4. Regenerating Figure 6 (Ablation Waterfall with Real Stage 6)...")
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    stages = ablation_df['Ablation Stage'].apply(lambda x: x.split('.')[1].strip()).tolist()
    aurocs = ablation_df['AUROC'].values
    bar_colors = ['#94A3B8', '#38BDF8', '#0284C7', '#0D9488', '#059669', '#10B981']
    bars = ax.bar(range(len(stages)), aurocs, color=bar_colors, edgecolor='#1E293B', width=0.55)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=22, ha='right', fontsize=7.5)
    ax.set_ylabel('Macro AUROC')
    ax.set_title('Ablation Waterfall: Empirical Progression of Multiplicity, Diversity, Routing & Deferral')
    ax.set_ylim(min(aurocs) - 0.02, max(aurocs) + 0.015)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.0015, f"{h:.4f}", ha='center', fontsize=7.5, fontweight='bold')
    save_ieee_fig(fig, "ablation_waterfall")

    # Regenerate Figure 8: Topline MOSAIC vs Single Large
    print("\n>>> 5. Regenerating Figure 8 (Topline Figure with Measured FLOPs)...")
    from scratch.py.plotting import plot_topline_mosaic_comparison
    plot_topline_mosaic_comparison(
        models=['Single Large (2M)', 'Split-4 Homo', 'Split-4 Hetero', 'MOSAIC-Med Top-2'],
        auroc=[k1_auroc, k4_auroc, het_auroc, m_top2['auroc']],
        auprc=[k1_auprc, k4_auprc, het_auprc, m_top2['auprc']],
        brier=[k1_brier, k4_brier, het_brier, m_top2['brier']],
        flops=[123.9, 126.8, 413.7, measured_top2_flops],
        save_name="topline_mosaic_vs_single"
    )

    elapsed = time.time() - t_start
    log_line = f"Item 3: Fix evidence (measured FLOPs, real CKA table, fixed Fig 2 labels, real stage 6 waterfall, real Fig 8) - completed in {elapsed:.1f}s\n"
    with open(os.path.join(TABLES_DIR, "RUN_LOG.txt"), "a", encoding="utf-8") as f:
        f.write(log_line)
    print(f"\n[RUN_LOG] {log_line.strip()}")

if __name__ == '__main__':
    run_item_3()
