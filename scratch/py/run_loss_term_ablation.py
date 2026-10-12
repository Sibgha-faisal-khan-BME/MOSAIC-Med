"""
Loss-Term Ablation Experiment (Item 4).
Trains 5 variants across 3 seeds (42, 43, 44):
1. Uniform Average (L_diag only, No Router)
2. Router with L_diag only (BCE)
3. Router with L_diag + L_div (Diversity)
4. Router with L_diag + L_fail (Failure-Aware)
5. All Terms (L_diag + L_div + L_fail)

Experts are FROZEN (ResNet, MultiScale, TCN, ConvAttn @ 500k each).
Only the AdaptiveRouter (~16k parameters) is trained using cached expert logits.

Outputs:
- results/tables/loss_ablation_per_seed.csv
- results/tables/loss_ablation_summary.csv (with Holm-corrected Welch tests vs Uniform Avg)
- results/images/loss_ablation_tradeoff.png & .pdf (all values read from CSV)
- Appends to results/tables/RUN_LOG.txt.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from scratch.py.data import cache_ptbxl_dataset
from scratch.py.models import create_expert, AdaptiveRouter
from scratch.py.training import (
    set_seed, get_dataloader, evaluate_model, train_adaptive_router,
    free_gpu_memory, DEVICE
)
from scratch.py.metrics import compute_all_diagnostic_metrics
from scratch.py.stats import holm_bonferroni_correction
from scratch.py.plotting import set_ieee_style, save_ieee_fig, COLORS

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

CHECKPOINT_DIR = "scratch/checkpoints"
TABLES_DIR = "results/tables"
IMAGES_DIR = "results/images"
SEEDS = [42, 43, 44]
HETERO_ARCHES = ['resnet', 'multiscale', 'tcn', 'conv_att']

def get_hetero_expert_path(arch: str, seed: int) -> str:
    path = os.path.join(CHECKPOINT_DIR, f"hetero_{arch}_500k_s{seed}.pt")
    if os.path.exists(path):
        return path
    if arch == 'resnet':
        k4_path = os.path.join(CHECKPOINT_DIR, f"resnet_K4_500k_s{seed}_e0.pt")
        if os.path.exists(k4_path):
            return k4_path
    raise FileNotFoundError(f"Checkpoint for {arch} seed {seed} not found!")

def run_item_4():
    t_start = time.time()
    print("=" * 70)
    print("STARTING ITEM 4: LOSS-TERM ABLATION (5 VARIANTS x 3 SEEDS)")
    print("Note: 4 Hetero Experts are FROZEN; Router trained on cached logits.")
    print("=" * 70)

    data = cache_ptbxl_dataset()
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    train_loader = get_dataloader(X_train, y_train, batch_size=64, shuffle=True)
    val_loader = get_dataloader(X_val, y_val, batch_size=64, shuffle=False)
    test_loader = get_dataloader(X_test, y_test, batch_size=128, shuffle=False)

    variants = [
        ('Uniform Average (L_diag only, No Router)', 0.0, 0.0, False),
        ('Router with L_diag only (BCE)', 0.0, 0.0, True),
        ('Router with L_diag + L_div (Diversity)', 0.05, 0.0, True),
        ('Router with L_diag + L_fail (Failure-Aware)', 0.0, 0.05, True),
        ('All Terms (L_diag + L_div + L_fail)', 0.05, 0.05, True)
    ]

    records = []

    for seed in SEEDS:
        print(f"\n>>> Processing Seed {seed}...")
        set_seed(seed)

        # Load 4 Hetero Experts (FROZEN)
        hetero_models = []
        hetero_probs = []
        for arch in HETERO_ARCHES:
            ckpt = get_hetero_expert_path(arch, seed)
            m = create_expert(arch, '500k')
            m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            m = m.to(DEVICE)
            m.eval()
            hetero_models.append(m)
            pr, _, _, _ = evaluate_model(m, test_loader)
            hetero_probs.append(pr)

        y_prob_uniform = np.mean(hetero_probs, axis=0)

        # Precompute frozen expert logits for train and val loaders ONCE per seed
        print(f"  Precomputing frozen expert logits for seed {seed}...")
        cached_train_logits = []
        with torch.no_grad():
            for bx, _ in train_loader:
                bx = bx.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    lgs = [exp(bx) for exp in hetero_models]
                    cached_train_logits.append(torch.stack(lgs, dim=1).detach())
        cached_val_logits = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    lgs = [exp(bx) for exp in hetero_models]
                    cached_val_logits.append(torch.stack(lgs, dim=1).detach())

        for v_name, l_div, l_fail, use_router in variants:
            print(f"  Variant: {v_name}...")
            if not use_router:
                m_res = compute_all_diagnostic_metrics(y_test, y_prob_uniform)
            else:
                ckpt_router = os.path.join(CHECKPOINT_DIR, f"router_ablation_s{seed}_div{l_div}_fail{l_fail}.pt")
                router = AdaptiveRouter(in_channels=12, num_experts=4, hidden_dim=64)
                
                # Train router on frozen experts with cached logits
                set_seed(seed * 31 + int(l_div * 100) + int(l_fail * 100))
                router = train_adaptive_router(
                    router, hetero_models, train_loader, val_loader,
                    num_epochs=6, lr=1e-3, lambda_div=l_div, lambda_fail=l_fail,
                    checkpoint_path=ckpt_router,
                    cached_train_logits=cached_train_logits,
                    cached_val_logits=cached_val_logits
                )

                # Evaluate router on test set using precomputed expert probabilities
                router.eval()
                router_weights_list = []
                with torch.no_grad():
                    for bx, _ in test_loader:
                        bx = bx.to(DEVICE, non_blocking=True)
                        with torch.amp.autocast('cuda'):
                            w, _ = router(bx)
                            router_weights_list.append(w.cpu().numpy())
                            
                r_w = np.concatenate(router_weights_list, axis=0) # (N, 4)
                y_prob_cfg = np.zeros_like(y_prob_uniform)
                for k in range(4):
                    y_prob_cfg += hetero_probs[k] * r_w[:, k:k+1]
                m_res = compute_all_diagnostic_metrics(y_test, y_prob_cfg)
                del router
                free_gpu_memory()

            print(f"    AUROC: {m_res['auroc']:.4f}, AUPRC: {m_res['auprc']:.4f}, Brier: {m_res['brier']:.4f}, ECE: {m_res['ece']:.4f}")
            records.append({
                'seed': seed,
                'variant': v_name,
                'lambda_div': l_div,
                'lambda_fail': l_fail,
                'router_trained': use_router,
                'experts_frozen': True,
                'auroc': m_res['auroc'],
                'auprc': m_res['auprc'],
                'f1': m_res['f1'],
                'brier': m_res['brier'],
                'ece': m_res['ece']
            })

        for exp in hetero_models:
            del exp
        free_gpu_memory()

    # 1. Save loss_ablation_per_seed.csv
    per_seed_df = pd.DataFrame(records)
    per_seed_path = os.path.join(TABLES_DIR, "loss_ablation_per_seed.csv")
    per_seed_df.to_csv(per_seed_path, index=False)
    print(f"\nSaved {per_seed_path}")

    # 2. Compute Summary with Holm-corrected Welch tests vs Uniform Average
    baseline_variant = 'Uniform Average (L_diag only, No Router)'
    base_aurocs = per_seed_df[per_seed_df.variant == baseline_variant].auroc.values

    raw_p_values = []
    compared_variants = [v[0] for v in variants if v[0] != baseline_variant]
    for v_name in compared_variants:
        v_aurocs = per_seed_df[per_seed_df.variant == v_name].auroc.values
        t_stat, p_val = stats.ttest_ind(v_aurocs, base_aurocs, equal_var=False)
        raw_p_values.append(p_val)

    adj_p_values, reject = holm_bonferroni_correction(raw_p_values, alpha=0.05)
    p_dict = {baseline_variant: ('-', '-')}
    for idx, v_name in enumerate(compared_variants):
        p_dict[v_name] = (f"{raw_p_values[idx]:.4e}", f"{adj_p_values[idx]:.4e}")

    summary_rows = []
    for v_name, l_div, l_fail, use_router in variants:
        sub = per_seed_df[per_seed_df.variant == v_name]
        summary_rows.append({
            'Variant': v_name,
            'Trained Component': 'AdaptiveRouter (~16k params)' if use_router else 'None (Uniform Mean)',
            'Frozen Component': '4 Hetero Experts (2.0M params)',
            'lambda_div': l_div,
            'lambda_fail': l_fail,
            'AUROC': f"{sub.auroc.mean():.4f} ± {sub.auroc.std():.4f}",
            'AUPRC': f"{sub.auprc.mean():.4f} ± {sub.auprc.std():.4f}",
            'F1': f"{sub.f1.mean():.4f} ± {sub.f1.std():.4f}",
            'Brier': f"{sub.brier.mean():.4f} ± {sub.brier.std():.4f}",
            'ECE': f"{sub.ece.mean():.4f} ± {sub.ece.std():.4f}",
            'Raw Welch p vs Baseline': p_dict[v_name][0],
            'Holm p vs Baseline': p_dict[v_name][1]
        })

    summary_df = pd.DataFrame(summary_rows)
    sum_path = os.path.join(TABLES_DIR, "loss_ablation_summary.csv")
    summary_df.to_csv(sum_path, index=False)
    print(f"Saved {sum_path}:")
    print(summary_df[['Variant', 'AUROC', 'AUPRC', 'Brier', 'Holm p vs Baseline']])

    # 3. Save Figure to results/images (every value read strictly from summary CSV)
    print("\n>>> Generating Loss-Term Ablation Figure...")
    sum_df_read = pd.read_csv(sum_path)
    variant_names = sum_df_read['Variant'].apply(lambda x: x.split('(')[0].strip()).tolist()
    auroc_means = [float(v.split('±')[0]) for v in sum_df_read['AUROC']]
    auroc_stds = [float(v.split('±')[1]) for v in sum_df_read['AUROC']]
    brier_means = [float(v.split('±')[0]) for v in sum_df_read['Brier']]
    brier_stds = [float(v.split('±')[1]) for v in sum_df_read['Brier']]

    set_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))

    x = np.arange(len(variant_names))
    axes[0].bar(x, auroc_means, yerr=auroc_stds, capsize=3, color='#0284C7', edgecolor='#0F172A', width=0.55)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(variant_names, rotation=25, ha='right', fontsize=7.0)
    axes[0].set_ylabel('Macro AUROC (Higher = Better)')
    axes[0].set_title('A. Discrimination vs Loss Regularization')
    axes[0].set_ylim(min(auroc_means) - 0.005, max(auroc_means) + 0.005)
    for i, (m, s) in enumerate(zip(auroc_means, auroc_stds)):
        axes[0].text(i, m + s + 0.0003, f"{m:.4f}", ha='center', fontsize=6.8, fontweight='bold')

    axes[1].bar(x, brier_means, yerr=brier_stds, capsize=3, color='#8B5CF6', edgecolor='#0F172A', width=0.55)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(variant_names, rotation=25, ha='right', fontsize=7.0)
    axes[1].set_ylabel('Brier Score (Lower = Better)')
    axes[1].set_title('B. Calibration vs Loss Regularization')
    axes[1].set_ylim(min(brier_means) - 0.003, max(brier_means) + 0.003)
    for i, (m, s) in enumerate(zip(brier_means, brier_stds)):
        axes[1].text(i, m + s + 0.0002, f"{m:.4f}", ha='center', fontsize=6.8, fontweight='bold')

    plt.tight_layout()
    save_ieee_fig(fig, "loss_ablation_tradeoff")

    elapsed = time.time() - t_start
    log_line = f"Item 4: Loss-term ablation (5 variants x 3 seeds, router on frozen experts with cached logits) - completed in {elapsed:.1f}s\n"
    with open(os.path.join(TABLES_DIR, "RUN_LOG.txt"), "a", encoding="utf-8") as f:
        f.write(log_line)
    print(f"\n[RUN_LOG] {log_line.strip()}")

if __name__ == '__main__':
    run_item_4()
