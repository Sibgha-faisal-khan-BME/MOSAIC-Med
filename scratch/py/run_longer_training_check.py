"""
Longer Training Check Experiment (Item 5).
Trains Single Large (K=1, 2M) and 4-Hetero (500k each) for 20 epochs across 3 seeds (42, 43, 44)
to evaluate whether the ensemble multiplicity/heterogeneity gain persists under extended convergence.
Outputs results/tables/longer_training_check.csv and results/tables/longer_training_summary.csv.
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
from scipy import stats

from scratch.py.data import cache_ptbxl_dataset
from scratch.py.models import create_expert
from scratch.py.training import (
    set_seed, get_dataloader, train_single_expert, evaluate_model,
    free_gpu_memory, DEVICE
)
from scratch.py.metrics import compute_all_diagnostic_metrics

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

CHECKPOINT_DIR = "scratch/checkpoints"
TABLES_DIR = "results/tables"
SEEDS = [42, 43, 44]
HETERO_ARCHES = ['resnet', 'multiscale', 'tcn', 'conv_att']

def run_item_5():
    t_start = time.time()
    print("=" * 70)
    print("STARTING ITEM 5: LONGER TRAINING CHECK (20 EPOCHS, 3 SEEDS)")
    print("=" * 70)

    data = cache_ptbxl_dataset()
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    train_loader = get_dataloader(X_train, y_train, batch_size=64, shuffle=True)
    val_loader = get_dataloader(X_val, y_val, batch_size=64, shuffle=False)
    test_loader = get_dataloader(X_test, y_test, batch_size=64, shuffle=False)

    records = []

    for seed in SEEDS:
        print(f"\n>>> Running Seed {seed} for 20 Epochs...")
        set_seed(seed)

        # -------------------------------------------------------------
        # 1. Single Large (K=1, 2M) for 20 Epochs
        # -------------------------------------------------------------
        single_ckpt = os.path.join(CHECKPOINT_DIR, f"longer_single_20ep_s{seed}.pt")
        single_model = create_expert('resnet', '2m')

        if os.path.exists(single_ckpt):
            print(f"  Loading existing checkpoint: {single_ckpt}")
            single_model.load_state_dict(torch.load(single_ckpt, map_location=DEVICE))
            single_model = single_model.to(DEVICE)
        else:
            print(f"  Training Single Large for 20 epochs (seed {seed})...")
            train_res = train_single_expert(
                single_model, train_loader, val_loader,
                num_epochs=20, lr=1e-3, checkpoint_path=single_ckpt, verbose=False
            )
            single_model = train_res['model']

        pr_single, _, _, _ = evaluate_model(single_model, test_loader)
        m_single = compute_all_diagnostic_metrics(y_test, pr_single)
        print(f"  [Seed {seed}] Single Large (20 ep): AUROC={m_single['auroc']:.4f}, AUPRC={m_single['auprc']:.4f}, Brier={m_single['brier']:.4f}")

        records.append({
            'seed': seed,
            'model': 'Single Large (1x2M)',
            'epochs': 20,
            'auroc': m_single['auroc'],
            'auprc': m_single['auprc'],
            'f1': m_single['f1'],
            'brier': m_single['brier'],
            'ece': m_single['ece']
        })

        del single_model
        free_gpu_memory()

        # -------------------------------------------------------------
        # 2. Heterogeneous 4-Expert Ensemble for 20 Epochs
        # -------------------------------------------------------------
        hetero_probs = []
        for a_idx, arch in enumerate(HETERO_ARCHES):
            h_ckpt = os.path.join(CHECKPOINT_DIR, f"longer_hetero_{arch}_20ep_s{seed}.pt")
            m_h = create_expert(arch, '500k')

            if os.path.exists(h_ckpt):
                print(f"  Loading existing checkpoint: {h_ckpt}")
                m_h.load_state_dict(torch.load(h_ckpt, map_location=DEVICE))
                m_h = m_h.to(DEVICE)
            else:
                print(f"  Training {arch} (500k) for 20 epochs (seed {seed})...")
                set_seed(seed * 20 + a_idx)
                train_res = train_single_expert(
                    m_h, train_loader, val_loader,
                    num_epochs=20, lr=1e-3, checkpoint_path=h_ckpt, verbose=False
                )
                m_h = train_res['model']

            pr, _, _, _ = evaluate_model(m_h, test_loader)
            hetero_probs.append(pr)
            del m_h
            free_gpu_memory()

        y_prob_hetero = np.mean(hetero_probs, axis=0)
        m_hetero = compute_all_diagnostic_metrics(y_test, y_prob_hetero)
        print(f"  [Seed {seed}] 4-Hetero Ensemble (20 ep): AUROC={m_hetero['auroc']:.4f}, AUPRC={m_hetero['auprc']:.4f}, Brier={m_hetero['brier']:.4f}")

        records.append({
            'seed': seed,
            'model': 'Heterogeneous 4-Expert Ensemble',
            'epochs': 20,
            'auroc': m_hetero['auroc'],
            'auprc': m_hetero['auprc'],
            'f1': m_hetero['f1'],
            'brier': m_hetero['brier'],
            'ece': m_hetero['ece']
        })

    # Save detailed runs
    df_runs = pd.DataFrame(records)
    out_csv = os.path.join(TABLES_DIR, "longer_training_check.csv")
    df_runs.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    # Summary table
    s_sub = df_runs[df_runs.model == 'Single Large (1x2M)']
    h_sub = df_runs[df_runs.model == 'Heterogeneous 4-Expert Ensemble']

    metrics = ['auroc', 'auprc', 'f1', 'brier', 'ece']
    metric_names = {
        'auroc': 'Macro AUROC',
        'auprc': 'Macro AUPRC',
        'f1': 'Macro F1',
        'brier': 'Brier Score (Calibration)',
        'ece': 'Expected Calibration Error'
    }

    sum_rows = []
    for m in metrics:
        s_vals = s_sub[m].values
        h_vals = h_sub[m].values
        t_res = stats.ttest_rel(h_vals, s_vals)

        sum_rows.append({
            'Metric': metric_names[m],
            'Single Large (20 ep Mean ± SD)': f"{np.mean(s_vals):.4f} ± {np.std(s_vals):.4f}",
            '4-Hetero (20 ep Mean ± SD)': f"{np.mean(h_vals):.4f} ± {np.std(h_vals):.4f}",
            'Delta (Hetero - Single)': f"{np.mean(h_vals) - np.mean(s_vals):+.4f}",
            'Paired t-statistic': f"{t_res.statistic:.3f}",
            'p-value': f"{t_res.pvalue:.4e}",
            'Ensemble Gain Survives': (np.mean(h_vals) > np.mean(s_vals)) if m != 'brier' and m != 'ece' else (np.mean(h_vals) < np.mean(s_vals))
        })

    sum_df = pd.DataFrame(sum_rows)
    sum_csv = os.path.join(TABLES_DIR, "longer_training_summary.csv")
    sum_df.to_csv(sum_csv, index=False)
    print(f"\nSaved {sum_csv}:")
    print(sum_df[['Metric', 'Single Large (20 ep Mean ± SD)', '4-Hetero (20 ep Mean ± SD)', 'Delta (Hetero - Single)', 'p-value', 'Ensemble Gain Survives']])

    elapsed = time.time() - t_start
    log_line = f"Item 5: Longer training check (Single Large vs 4-Hetero for 20 epochs, 3 seeds) - completed in {elapsed:.1f}s\n"
    with open(os.path.join(TABLES_DIR, "RUN_LOG.txt"), "a", encoding="utf-8") as f:
        f.write(log_line)
    print(f"\n[RUN_LOG] {log_line.strip()}")

if __name__ == '__main__':
    run_item_5()
