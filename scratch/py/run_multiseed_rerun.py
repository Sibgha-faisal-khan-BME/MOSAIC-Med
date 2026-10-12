"""
Multi-seed rerun for MOSAIC-Med (Item 1).
Evaluates 5 seeds (42, 43, 44, 45, 46) for:
1. Routing comparison (Uniform Average, Dense Router, Sparse Top-2)
2. Robustness benchmark (Clean, HF noise, Baseline drift, Missing V1-V3, Missing I-III, External Site != 0)
3. Selective-risk curve (100% to 80% coverage)
Outputs per-seed CSVs + summary CSVs with Mean ± SD and Holm-corrected Welch t-tests.
Logs runtime to results/tables/RUN_LOG.txt.
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
import shutil

from scratch.py.data import (
    cache_ptbxl_dataset, add_high_frequency_noise, add_baseline_drift, drop_leads, CLASSES
)
from scratch.py.models import (
    create_expert, AdaptiveRouter, compute_predictive_uncertainty
)
from scratch.py.training import (
    set_seed, get_dataloader, evaluate_model, train_adaptive_router,
    free_gpu_memory, DEVICE
)
from scratch.py.metrics import (
    compute_all_diagnostic_metrics, compute_selective_risk_coverage
)
from scratch.py.stats import holm_bonferroni_correction

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

CHECKPOINT_DIR = "scratch/checkpoints"
TABLES_DIR = "results/tables"
IMAGES_DIR = "results/images"
SEEDS = [42, 43, 44, 45, 46]
HETERO_ARCHES = ['resnet', 'multiscale', 'tcn', 'conv_att']

def get_hetero_expert_path(arch: str, seed: int) -> str:
    direct_path = os.path.join(CHECKPOINT_DIR, f"hetero_{arch}_500k_s{seed}.pt")
    if os.path.exists(direct_path):
        return direct_path
    if arch == 'resnet':
        k4_path = os.path.join(CHECKPOINT_DIR, f"resnet_K4_500k_s{seed}_e0.pt")
        if os.path.exists(k4_path):
            shutil.copyfile(k4_path, direct_path)
            return direct_path
    raise FileNotFoundError(f"Checkpoint for {arch} seed {seed} not found!")

def run_item_1():
    t_start_item1 = time.time()
    print("=" * 70)
    print("STARTING ITEM 1: MULTI-SEED RERUN ACROSS ALL 5 SEEDS (42-46)")
    print("=" * 70)

    # 1. Load Data
    data = cache_ptbxl_dataset()
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    meta_test = data['meta_test']

    train_loader = get_dataloader(X_train, y_train, batch_size=64, shuffle=True)
    val_loader = get_dataloader(X_val, y_val, batch_size=64, shuffle=False)
    test_loader = get_dataloader(X_test, y_test, batch_size=64, shuffle=False)

    # Containers for results
    routing_records = []
    robustness_records = []
    selective_records = []

    # Store test predictions per seed for statistical comparisons
    # seed -> {'single': prob, 'uniform': prob, 'dense': prob, 'top2': prob, 'unc_single': u, 'unc_mosaic': u}
    seed_preds = {}

    for seed in SEEDS:
        print(f"\n>>> Processing Seed {seed}...")
        set_seed(seed)

        # -------------------------------------------------------------
        # A. Load Single Large Model (K=1, 2M)
        # -------------------------------------------------------------
        single_ckpt = os.path.join(CHECKPOINT_DIR, f"resnet_K1_2m_s{seed}_e0.pt")
        single_model = create_expert('resnet', target_budget='2m')
        single_model.load_state_dict(torch.load(single_ckpt, map_location=DEVICE))
        single_model = single_model.to(DEVICE)

        y_prob_single, _, _, _ = evaluate_model(single_model, test_loader)
        m_single = compute_all_diagnostic_metrics(y_test, y_prob_single)
        print(f"  [Seed {seed}] Single Large AUROC: {m_single['auroc']:.4f}")

        # -------------------------------------------------------------
        # B. Load 4 Heterogeneous Experts
        # -------------------------------------------------------------
        hetero_models = []
        hetero_probs = []
        for arch in HETERO_ARCHES:
            ckpt = get_hetero_expert_path(arch, seed)
            m = create_expert(arch, target_budget='500k')
            m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            m = m.to(DEVICE)
            hetero_models.append(m)
            pr, _, _, _ = evaluate_model(m, test_loader)
            hetero_probs.append(pr)

        # 1. Uniform Average
        y_prob_uniform = np.mean(hetero_probs, axis=0)
        m_uniform = compute_all_diagnostic_metrics(y_test, y_prob_uniform)
        print(f"  [Seed {seed}] Uniform Avg AUROC: {m_uniform['auroc']:.4f}")

        # -------------------------------------------------------------
        # C. Load or Train Adaptive Router for this seed
        # -------------------------------------------------------------
        router_ckpt = os.path.join(CHECKPOINT_DIR, f"adaptive_router_s{seed}.pt")
        if seed == 42 and os.path.exists(os.path.join(CHECKPOINT_DIR, "adaptive_router.pt")) and not os.path.exists(router_ckpt):
            shutil.copyfile(os.path.join(CHECKPOINT_DIR, "adaptive_router.pt"), router_ckpt)

        router = AdaptiveRouter(in_channels=12, num_experts=4, hidden_dim=64)
        if os.path.exists(router_ckpt):
            print(f"  Loading existing router checkpoint for seed {seed}...")
            router.load_state_dict(torch.load(router_ckpt, map_location=DEVICE))
            router = router.to(DEVICE)
        else:
            print(f"  Training Adaptive Router for seed {seed} (6 epochs)...")
            set_seed(seed * 77)
            router = train_adaptive_router(
                router, hetero_models, train_loader, val_loader,
                num_epochs=6, lr=1e-3, lambda_div=0.05, lambda_fail=0.05,
                checkpoint_path=router_ckpt
            )

        # Router test predictions
        router.eval()
        router = router.to(DEVICE)
        router_weights_list = []
        dense_probs_list = []

        with torch.no_grad():
            for bx, _ in test_loader:
                bx = bx.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    w, _ = router(bx)
                    router_weights_list.append(w.cpu().numpy())
                    exp_lgs = [exp(bx) for exp in hetero_models]
                    stacked_lgs = torch.stack(exp_lgs, dim=1) # (B, 4, 5)
                    fused_lg = torch.sum(stacked_lgs * w.unsqueeze(-1), dim=1)
                    prob = torch.sigmoid(fused_lg)
                    dense_probs_list.append(prob.cpu().numpy())

        router_weights = np.concatenate(router_weights_list, axis=0) # (N, 4)
        y_prob_dense = np.concatenate(dense_probs_list, axis=0) # (N, 5)
        m_dense = compute_all_diagnostic_metrics(y_test, y_prob_dense)
        print(f"  [Seed {seed}] Dense Router AUROC: {m_dense['auroc']:.4f}")

        # -------------------------------------------------------------
        # D. Sparse Top-2 Router
        # -------------------------------------------------------------
        t2_idx = np.argsort(router_weights, axis=-1)[:, -2:]
        top2_weights = np.zeros_like(router_weights)
        np.put_along_axis(top2_weights, t2_idx, np.take_along_axis(router_weights, t2_idx, axis=-1), axis=-1)
        top2_weights = top2_weights / np.sum(top2_weights, axis=-1, keepdims=True)

        y_prob_top2 = np.zeros_like(y_prob_uniform)
        for k in range(4):
            y_prob_top2 += hetero_probs[k] * top2_weights[:, k:k+1]
        m_top2 = compute_all_diagnostic_metrics(y_test, y_prob_top2)
        print(f"  [Seed {seed}] Sparse Top-2 AUROC: {m_top2['auroc']:.4f}")

        # Store routing records
        for strat, m_res, act, flp in [
            ('Uniform Average', m_uniform, 4.0, 413.7),
            ('Dense Router', m_dense, 4.0, 414.9),
            ('MOSAIC-Med Sparse Top-2', m_top2, 2.0, 297.8)
        ]:
            routing_records.append({
                'seed': seed,
                'strategy': strat,
                'active_experts': act,
                'mflops': flp,
                'auroc': m_res['auroc'],
                'auprc': m_res['auprc'],
                'f1': m_res['f1'],
                'brier': m_res['brier'],
                'ece': m_res['ece']
            })

        # Also store Single Large in routing per seed for full comparison
        routing_records.append({
            'seed': seed,
            'strategy': 'Single Large (1x2M)',
            'active_experts': 1.0,
            'mflops': 123.9,
            'auroc': m_single['auroc'],
            'auprc': m_single['auprc'],
            'f1': m_single['f1'],
            'brier': m_single['brier'],
            'ece': m_single['ece']
        })

        # -------------------------------------------------------------
        # E. Robustness Benchmarks for this seed
        # -------------------------------------------------------------
        base_auc_s = m_single['auroc']
        base_auc_m = m_top2['auroc']

        # 1. Clean Test Baseline record directly from clean test evaluations
        robustness_records.append({
            'seed': seed,
            'condition': 'Clean Test Baseline',
            'single_auroc': base_auc_s,
            'single_retention_pct': 100.0,
            'mosaic_auroc': base_auc_m,
            'mosaic_retention_pct': 100.0,
            'delta_auroc': base_auc_m - base_auc_s
        })

        perturbations = {
            'High-Freq EMG Noise (σ=0.15)': lambda x: add_high_frequency_noise(x, snr_sigma=0.15),
            'Low-Freq Baseline Wander': lambda x: add_baseline_drift(x, amplitude=0.6, freq=0.25),
            'Missing Precordial Leads (V1-V3)': lambda x: drop_leads(x, [6, 7, 8]),
            'Missing Limb Leads (I, II, III)': lambda x: drop_leads(x, [0, 1, 2]),
        }

        for p_name, p_fn in perturbations.items():
            X_corrupt = p_fn(X_test)
            loader_c = get_dataloader(X_corrupt, y_test, batch_size=128, shuffle=False)

            pr_s, _, _, _ = evaluate_model(single_model, loader_c)
            m_s = compute_all_diagnostic_metrics(y_test, pr_s)

            # Evaluate router on corrupted data
            c_weights_list = []
            with torch.no_grad():
                for bx, _ in loader_c:
                    bx = bx.to(DEVICE)
                    with torch.amp.autocast('cuda'):
                        w, _ = router(bx)
                        c_weights_list.append(w.cpu().numpy())
            c_weights = np.concatenate(c_weights_list, axis=0)

            c_t2_idx = np.argsort(c_weights, axis=-1)[:, -2:]
            c_top2_w = np.zeros_like(c_weights)
            np.put_along_axis(c_top2_w, c_t2_idx, np.take_along_axis(c_weights, c_t2_idx, axis=-1), axis=-1)
            c_top2_w = c_top2_w / np.sum(c_top2_w, axis=-1, keepdims=True)

            pr_exp_list = [evaluate_model(exp, loader_c)[0] for exp in hetero_models]
            pr_m = np.zeros_like(pr_s)
            for k in range(4):
                pr_m += pr_exp_list[k] * c_top2_w[:, k:k+1]
            m_m = compute_all_diagnostic_metrics(y_test, pr_m)

            robustness_records.append({
                'seed': seed,
                'condition': p_name,
                'single_auroc': m_s['auroc'],
                'single_retention_pct': (m_s['auroc'] / base_auc_s) * 100.0,
                'mosaic_auroc': m_m['auroc'],
                'mosaic_retention_pct': (m_m['auroc'] / base_auc_m) * 100.0,
                'delta_auroc': m_m['auroc'] - m_s['auroc']
            })

        # External acquisition site substitute (site != 0 in test set)
        site_mask = (meta_test['site'] != 0.0).values
        if np.sum(site_mask) > 100:
            y_test_ext = y_test[site_mask]
            m_s_ext = compute_all_diagnostic_metrics(y_test_ext, y_prob_single[site_mask])
            m_m_ext = compute_all_diagnostic_metrics(y_test_ext, y_prob_top2[site_mask])
            robustness_records.append({
                'seed': seed,
                'condition': 'External Site Substitute (Site != 0)',
                'single_auroc': m_s_ext['auroc'],
                'single_retention_pct': (m_s_ext['auroc'] / base_auc_s) * 100.0,
                'mosaic_auroc': m_m_ext['auroc'],
                'mosaic_retention_pct': (m_m_ext['auroc'] / base_auc_m) * 100.0,
                'delta_auroc': m_m_ext['auroc'] - m_s_ext['auroc']
            })

        # -------------------------------------------------------------
        # F. Selective Risk vs Coverage for this seed
        # -------------------------------------------------------------
        unc_s, _ = compute_predictive_uncertainty(torch.from_numpy(y_prob_single))
        exp_probs_t = [torch.from_numpy(p) for p in hetero_probs]
        unc_m, dis_m = compute_predictive_uncertainty(torch.from_numpy(y_prob_top2), exp_probs_t)
        total_unc_m = unc_m.numpy() + 2.0 * dis_m.numpy()

        rc_s = compute_selective_risk_coverage(y_test, y_prob_single, unc_s.numpy())
        rc_m = compute_selective_risk_coverage(y_test, y_prob_top2, total_unc_m)

        for cov in [1.0, 0.95, 0.90, 0.85, 0.80]:
            risk_s = float(np.interp(cov, rc_s['coverages'], rc_s['selective_risks'])) * 100.0
            risk_m = float(np.interp(cov, rc_m['coverages'], rc_m['selective_risks'])) * 100.0
            rel_red = ((risk_s - risk_m) / risk_s) * 100.0 if risk_s > 0 else 0.0
            selective_records.append({
                'seed': seed,
                'coverage': cov,
                'coverage_pct': f"{int(cov*100)}%",
                'single_risk_pct': risk_s,
                'mosaic_risk_pct': risk_m,
                'relative_reduction_pct': rel_red
            })

        # Cleanup memory for seed
        del single_model, router
        for exp in hetero_models:
            del exp
        free_gpu_memory()

    # -------------------------------------------------------------
    # 2. Save Per-Seed CSVs
    # -------------------------------------------------------------
    routing_seed_df = pd.DataFrame(routing_records)
    routing_seed_df.to_csv(os.path.join(TABLES_DIR, "routing_per_seed.csv"), index=False)
    print(f"\nSaved {os.path.join(TABLES_DIR, 'routing_per_seed.csv')}")

    robustness_seed_df = pd.DataFrame(robustness_records)
    robustness_seed_df.to_csv(os.path.join(TABLES_DIR, "robustness_per_seed.csv"), index=False)
    print(f"Saved {os.path.join(TABLES_DIR, 'robustness_per_seed.csv')}")

    selective_seed_df = pd.DataFrame(selective_records)
    selective_seed_df.to_csv(os.path.join(TABLES_DIR, "selective_risk_per_seed.csv"), index=False)
    print(f"Saved {os.path.join(TABLES_DIR, 'selective_risk_per_seed.csv')}")

    # -------------------------------------------------------------
    # 3. Create Multi-Seed Summaries with Holm-corrected Welch t-tests
    # -------------------------------------------------------------
    # A. Routing Summary
    routing_sum_rows = []
    strategies = ['Single Large (1x2M)', 'Uniform Average', 'Dense Router', 'MOSAIC-Med Sparse Top-2']
    single_aurocs = routing_seed_df[routing_seed_df.strategy == 'Single Large (1x2M)'].auroc.values

    # Compute p-values against Single Large
    raw_p_routing = []
    for strat in ['Uniform Average', 'Dense Router', 'MOSAIC-Med Sparse Top-2']:
        strat_aurocs = routing_seed_df[routing_seed_df.strategy == strat].auroc.values
        t_stat, p_val = stats.ttest_ind(strat_aurocs, single_aurocs, equal_var=False)
        raw_p_routing.append(p_val)
    adj_p_routing, _ = holm_bonferroni_correction(raw_p_routing, alpha=0.05)

    p_map = {
        'Single Large (1x2M)': ('-', '-'),
        'Uniform Average': (f"{raw_p_routing[0]:.4e}", f"{adj_p_routing[0]:.4e}"),
        'Dense Router': (f"{raw_p_routing[1]:.4e}", f"{adj_p_routing[1]:.4e}"),
        'MOSAIC-Med Sparse Top-2': (f"{raw_p_routing[2]:.4e}", f"{adj_p_routing[2]:.4e}")
    }

    for strat in strategies:
        sub = routing_seed_df[routing_seed_df.strategy == strat]
        routing_sum_rows.append({
            'Strategy': strat,
            'Active Experts': sub.active_experts.iloc[0],
            'MFLOPs': sub.mflops.iloc[0],
            'AUROC': f"{sub.auroc.mean():.4f} ± {sub.auroc.std():.4f}",
            'AUPRC': f"{sub.auprc.mean():.4f} ± {sub.auprc.std():.4f}",
            'F1': f"{sub.f1.mean():.4f} ± {sub.f1.std():.4f}",
            'Brier': f"{sub.brier.mean():.4f} ± {sub.brier.std():.4f}",
            'ECE': f"{sub.ece.mean():.4f} ± {sub.ece.std():.4f}",
            'Welch p vs Single': p_map[strat][0],
            'Holm p vs Single': p_map[strat][1]
        })
    routing_sum_df = pd.DataFrame(routing_sum_rows)
    routing_sum_df.to_csv(os.path.join(TABLES_DIR, "routing_multiseed_summary.csv"), index=False)
    print("\n--- Routing Multi-Seed Summary ---")
    print(routing_sum_df[['Strategy', 'AUROC', 'AUPRC', 'Brier', 'Welch p vs Single', 'Holm p vs Single']])

    # B. Robustness Summary
    robustness_conditions = robustness_seed_df.condition.unique()
    raw_p_robust = []
    condition_order = []
    for cond in robustness_conditions:
        sub = robustness_seed_df[robustness_seed_df.condition == cond]
        t_stat, p_val = stats.ttest_ind(sub.mosaic_auroc.values, sub.single_auroc.values, equal_var=False)
        raw_p_robust.append(p_val)
        condition_order.append(cond)
    adj_p_robust, _ = holm_bonferroni_correction(raw_p_robust, alpha=0.05)

    robust_sum_rows = []
    for idx, cond in enumerate(condition_order):
        sub = robustness_seed_df[robustness_seed_df.condition == cond]
        robust_sum_rows.append({
            'Condition': cond,
            'Single Large AUROC': f"{sub.single_auroc.mean():.4f} ± {sub.single_auroc.std():.4f}",
            'Single Retention (%)': f"{sub.single_retention_pct.mean():.2f}% ± {sub.single_retention_pct.std():.2f}%",
            'MOSAIC-Med AUROC': f"{sub.mosaic_auroc.mean():.4f} ± {sub.mosaic_auroc.std():.4f}",
            'MOSAIC Retention (%)': f"{sub.mosaic_retention_pct.mean():.2f}% ± {sub.mosaic_retention_pct.std():.2f}%",
            'Delta AUROC (M - S)': f"{sub.delta_auroc.mean():.4f} ± {sub.delta_auroc.std():.4f}",
            'Welch p-value': f"{raw_p_robust[idx]:.4e}",
            'Holm p-value': f"{adj_p_robust[idx]:.4e}",
            'Significant (α=0.05)': adj_p_robust[idx] < 0.05
        })
    robust_sum_df = pd.DataFrame(robust_sum_rows)
    robust_sum_df.to_csv(os.path.join(TABLES_DIR, "robustness_multiseed_summary.csv"), index=False)
    print("\n--- Robustness Multi-Seed Summary ---")
    print(robust_sum_df[['Condition', 'Single Large AUROC', 'MOSAIC-Med AUROC', 'Delta AUROC (M - S)', 'Holm p-value']])

    # C. Selective Risk Summary
    coverages = [1.0, 0.95, 0.90, 0.85, 0.80]
    raw_p_risk = []
    for cov in coverages:
        sub = selective_seed_df[selective_seed_df.coverage == cov]
        t_stat, p_val = stats.ttest_ind(sub.mosaic_risk_pct.values, sub.single_risk_pct.values, equal_var=False)
        raw_p_risk.append(p_val)
    adj_p_risk, _ = holm_bonferroni_correction(raw_p_risk, alpha=0.05)

    selective_sum_rows = []
    for idx, cov in enumerate(coverages):
        sub = selective_seed_df[selective_seed_df.coverage == cov]
        selective_sum_rows.append({
            'Coverage': f"{int(cov*100)}%",
            'Abstention Rate': f"{int((1-cov)*100)}%",
            'Single Large Risk (%)': f"{sub.single_risk_pct.mean():.2f}% ± {sub.single_risk_pct.std():.2f}%",
            'MOSAIC-Med Risk (%)': f"{sub.mosaic_risk_pct.mean():.2f}% ± {sub.mosaic_risk_pct.std():.2f}%",
            'Relative Reduction (%)': f"{sub.relative_reduction_pct.mean():.2f}% ± {sub.relative_reduction_pct.std():.2f}%",
            'Welch p-value': f"{raw_p_risk[idx]:.4e}",
            'Holm p-value': f"{adj_p_risk[idx]:.4e}",
            'Significant (α=0.05)': adj_p_risk[idx] < 0.05
        })
    selective_sum_df = pd.DataFrame(selective_sum_rows)
    selective_sum_df.to_csv(os.path.join(TABLES_DIR, "selective_risk_multiseed_summary.csv"), index=False)
    print("\n--- Selective Risk Multi-Seed Summary ---")
    print(selective_sum_df[['Coverage', 'Single Large Risk (%)', 'MOSAIC-Med Risk (%)', 'Relative Reduction (%)', 'Holm p-value']])

    elapsed_item1 = time.time() - t_start_item1
    log_line = f"Item 1: Multi-seed rerun (routing, robustness, selective-risk across 5 seeds 42-46) - completed in {elapsed_item1:.1f}s\n"
    with open(os.path.join(TABLES_DIR, "RUN_LOG.txt"), "a", encoding="utf-8") as f:
        f.write(log_line)
    print(f"\n[RUN_LOG] {log_line.strip()}")

if __name__ == '__main__':
    run_item_1()
