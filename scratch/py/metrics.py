"""
Clinical evaluation metrics for multi-label 12-lead ECG diagnosis:
- AUROC (macro & per-class)
- AUPRC (macro & per-class)
- Sensitivity, Specificity, Balanced Accuracy, F1
- Calibration: Brier score, Expected Calibration Error (ECE), reliability diagrams
- Selective Classification: Risk-Coverage curve, Area Under Risk-Coverage Curve (AURC)
- Efficiency: Params, FLOPs, Latency, Peak VRAM, Active Experts
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, brier_score_loss
from typing import Dict, List, Tuple, Any

CLASSES = ['CD', 'HYP', 'MI', 'NORM', 'STTC']

def compute_all_diagnostic_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Computes comprehensive clinical metrics for multi-label binary predictions.
    y_true: (N, 5) binary targets {0, 1}
    y_prob: (N, 5) predicted probabilities in [0, 1]
    """
    N, num_classes = y_true.shape
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
    y_prob = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    y_pred = (y_prob >= threshold).astype(int)
    
    auroc_list = []
    auprc_list = []
    sens_list = []
    spec_list = []
    f1_list = []
    brier_list = []
    ece_list = []
    
    per_class = {}
    
    for i, c in enumerate(CLASSES):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        ybin = y_pred[:, i]
        
        # AUROC & AUPRC
        try:
            auc = roc_auc_score(yt, yp)
        except Exception:
            auc = np.nan
        try:
            prc = average_precision_score(yt, yp)
        except Exception:
            prc = np.nan
            
        auroc_list.append(auc)
        auprc_list.append(prc)
        
        # Confusion matrix for binary class
        tn, fp, fn, tp = confusion_matrix(yt, ybin, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(yt, ybin, zero_division=0)
        brier = brier_score_loss(yt, yp)
        
        # Expected Calibration Error (10 bins)
        ece = compute_binary_ece(yt, yp, num_bins=10)
        
        sens_list.append(sens)
        spec_list.append(spec)
        f1_list.append(f1)
        brier_list.append(brier)
        ece_list.append(ece)
        
        per_class[c] = {
            'auroc': float(auc),
            'auprc': float(prc),
            'sensitivity': float(sens),
            'specificity': float(spec),
            'f1': float(f1),
            'brier': float(brier),
            'ece': float(ece)
        }
        
    macro_auroc = float(np.nanmean(auroc_list))
    macro_auprc = float(np.nanmean(auprc_list))
    macro_sens = float(np.mean(sens_list))
    macro_spec = float(np.mean(spec_list))
    macro_f1 = float(np.mean(f1_list))
    macro_brier = float(np.mean(brier_list))
    macro_ece = float(np.mean(ece_list))
    balanced_acc = float((macro_sens + macro_spec) / 2.0)
    
    return {
        'auroc': macro_auroc,
        'auprc': macro_auprc,
        'sensitivity': macro_sens,
        'specificity': macro_spec,
        'balanced_acc': balanced_acc,
        'f1': macro_f1,
        'brier': macro_brier,
        'ece': macro_ece,
        'per_class': per_class
    }

def compute_binary_ece(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) with equal-width bins."""
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    ece = 0.0
    n = len(y_true)
    for b in range(num_bins):
        mask = (bin_indices == b)
        if np.sum(mask) > 0:
            bin_conf = np.mean(y_prob[mask])
            bin_acc = np.mean(y_true[mask])
            ece += (np.sum(mask) / n) * np.abs(bin_acc - bin_conf)
    return float(ece)

def compute_reliability_diagram_data(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10) -> Dict[str, np.ndarray]:
    """Generates calibration curve points (confidence vs accuracy) across all classes."""
    y_true_flat = y_true.flatten()
    y_prob_flat = y_prob.flatten()
    
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_indices = np.digitize(y_prob_flat, bins) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    accs = []
    confs = []
    counts = []
    
    for b in range(num_bins):
        mask = (bin_indices == b)
        if np.sum(mask) > 0:
            accs.append(np.mean(y_true_flat[mask]))
            confs.append(np.mean(y_prob_flat[mask]))
            counts.append(np.sum(mask))
        else:
            accs.append(np.nan)
            confs.append(bin_centers[b])
            counts.append(0)
            
    return {
        'bin_centers': bin_centers,
        'bin_confs': np.array(confs),
        'bin_accs': np.array(accs),
        'counts': np.array(counts)
    }

def compute_selective_risk_coverage(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty_scores: np.ndarray,
    num_thresholds: int = 50
) -> Dict[str, Any]:
    """
    Computes selective classification curve:
    Sorts samples by confidence (negative uncertainty), retains top fraction (coverage),
    and evaluates multi-label Hamming error (risk) on retained samples.
    """
    confidence = - uncertainty_scores # higher confidence = lower uncertainty
    N = len(confidence)
    order = np.argsort(confidence)[::-1] # descending confidence
    
    # Hamming loss per sample: fraction of incorrect class labels
    y_pred = (y_prob >= 0.5).astype(int)
    sample_risk = np.mean(y_true != y_pred, axis=1) # (N,)
    
    ordered_risk = sample_risk[order]
    
    coverages = np.linspace(0.1, 1.0, num_thresholds)
    selective_risks = []
    
    for cov in coverages:
        k = int(np.ceil(cov * N))
        k = max(1, min(k, N))
        risk_at_cov = np.mean(ordered_risk[:k])
        selective_risks.append(risk_at_cov)
        
    selective_risks = np.array(selective_risks)
    # Area under risk-coverage curve (AURC)
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    aurc = float(trapz_fn(selective_risks, coverages))
    
    return {
        'coverages': coverages,
        'selective_risks': selective_risks,
        'aurc': aurc,
        'risk_at_80': float(np.interp(0.8, coverages, selective_risks)),
        'risk_at_90': float(np.interp(0.9, coverages, selective_risks)),
        'risk_at_95': float(np.interp(0.95, coverages, selective_risks)),
        'full_risk': float(selective_risks[-1])
    }
