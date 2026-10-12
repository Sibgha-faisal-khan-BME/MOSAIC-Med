"""
Train remaining heterogeneous experts for seeds 45 and 46 to complete the 5-seed evaluation.
Updates results/tables/diversity_comparison.csv and results/tables/ablation_summary.csv.
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
HETERO_ARCHES = ['resnet', 'multiscale', 'tcn', 'conv_att']

def train_hetero_seeds():
    print("=" * 60)
    print("Training Heterogeneous Models for Seeds 45 & 46 (Completing 5 Seeds)")
    print("=" * 60)

    data = cache_ptbxl_dataset()
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    train_loader = get_dataloader(X_train, y_train, batch_size=64, shuffle=True)
    val_loader = get_dataloader(X_val, y_val, batch_size=64, shuffle=False)
    test_loader = get_dataloader(X_test, y_test, batch_size=64, shuffle=False)

    all_seeds = [42, 43, 44, 45, 46]
    hetero_records = []

    for s_idx, seed in enumerate(all_seeds):
        set_seed(seed)
        probs_h = []

        for a_idx, arch in enumerate(HETERO_ARCHES):
            ckpt = os.path.join(CHECKPOINT_DIR, f"hetero_{arch}_500k_s{seed}.pt")
            
            # If ResNet, check if K4 checkpoint exists
            if arch == 'resnet' and not os.path.exists(ckpt):
                k4_ckpt = os.path.join(CHECKPOINT_DIR, f"resnet_K4_500k_s{seed}_e0.pt")
                if os.path.exists(k4_ckpt):
                    ckpt = k4_ckpt

            m_arch = create_expert(arch, target_budget='500k')

            if os.path.exists(ckpt):
                print(f"Loading checkpoint for {arch} seed {seed} from {ckpt}")
                m_arch.load_state_dict(torch.load(ckpt, map_location=DEVICE))
                m_arch = m_arch.to(DEVICE)
            else:
                print(f"Training {arch} for seed {seed}...")
                set_seed(seed * 10 + a_idx)
                train_res = train_single_expert(
                    m_arch, train_loader, val_loader,
                    num_epochs=5, lr=1e-3,
                    checkpoint_path=ckpt, verbose=False
                )
                m_arch = train_res['model']

            pr, _, _, _ = evaluate_model(m_arch, test_loader)
            probs_h.append(pr)
            del m_arch
            free_gpu_memory()

        y_prob_hetero = np.mean(probs_h, axis=0)
        m_h = compute_all_diagnostic_metrics(y_test, y_prob_hetero)
        print(f"Heterogeneous Seed {seed}: AUROC={m_h['auroc']:.4f}, AUPRC={m_h['auprc']:.4f}, F1={m_h['f1']:.4f}, Brier={m_h['brier']:.4f}")

        hetero_records.append({
            'type': 'Heterogeneous (ResNet+MS+TCN+Att)',
            'seed': seed,
            'auroc': m_h['auroc'],
            'auprc': m_h['auprc'],
            'f1': m_h['f1'],
            'brier': m_h['brier'],
            'ece': m_h['ece']
        })

    # Load homogeneous K=4 records from scaling_results.csv
    scaling_df = pd.read_csv(os.path.join(TABLES_DIR, "scaling_results.csv"))
    homo_records = []
    for seed in all_seeds:
        sub_h = scaling_df[(scaling_df.K == 4) & (scaling_df.seed == seed)].iloc[0]
        homo_records.append({
            'type': 'Homogeneous (4x ResNet)',
            'seed': seed,
            'auroc': sub_h['auroc'],
            'auprc': sub_h['auprc'],
            'f1': sub_h['f1'],
            'brier': sub_h['brier'],
            'ece': sub_h['ece']
        })

    full_diversity_df = pd.DataFrame(homo_records + hetero_records)
    full_diversity_df.to_csv(os.path.join(TABLES_DIR, "diversity_comparison.csv"), index=False)
    print("\nUpdated diversity comparison table saved to results/tables/diversity_comparison.csv:")
    print(full_diversity_df.groupby('type')[['auroc', 'auprc', 'f1', 'brier', 'ece']].agg(['mean', 'std']))

    # Update ablation summary table
    ablation_df = pd.read_csv(os.path.join(TABLES_DIR, "ablation_summary.csv"))
    hetero_mean_auc = full_diversity_df[full_diversity_df.type.str.contains('Hetero')].auroc.mean()
    hetero_mean_prc = full_diversity_df[full_diversity_df.type.str.contains('Hetero')].auprc.mean()
    hetero_mean_f1 = full_diversity_df[full_diversity_df.type.str.contains('Hetero')].f1.mean()
    hetero_mean_brier = full_diversity_df[full_diversity_df.type.str.contains('Hetero')].brier.mean()

    ablation_df.loc[ablation_df['Ablation Stage'].str.contains('Heterogeneous'), ['AUROC', 'AUPRC', 'F1', 'Brier']] = [
        hetero_mean_auc, hetero_mean_prc, hetero_mean_f1, hetero_mean_brier
    ]
    ablation_df.to_csv(os.path.join(TABLES_DIR, "ablation_summary.csv"), index=False)
    print("\nUpdated ablation summary table:")
    print(ablation_df)

if __name__ == '__main__':
    train_hetero_seeds()
