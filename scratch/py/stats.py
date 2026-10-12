"""
Statistical testing suite for paired clinical comparisons:
- DeLong's test for correlated ROC curves
- McNemar's test for paired diagnostic decisions
- Stratified patient-level bootstrap confidence intervals (2,000+ resamples)
- Holm-Bonferroni multiple testing correction
- Paired effect size estimation (Cohen's d)
"""

import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
from typing import Dict, List, Tuple, Any, Optional

def compute_delong_auc_ci_and_pvalue(y_true: np.ndarray, y_prob_a: np.ndarray, y_prob_b: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Computes DeLong variance for AUROC, and if y_prob_b is provided, computes
    the DeLong test for comparing two correlated ROC curves on the same cohort.
    Multi-label: average covariance across classes.
    """
    # Binary DeLong calculation per class
    num_classes = y_true.shape[1] if y_true.ndim > 1 else 1
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_prob_a = y_prob_a[:, None]
        if y_prob_b is not None:
            y_prob_b = y_prob_b[:, None]
            
    diffs = []
    z_scores = []
    p_values = []
    auc_a_list = []
    auc_b_list = []
    
    for c in range(num_classes):
        yt = y_true[:, c]
        pa = y_prob_a[:, c]
        
        pos_idx = np.where(yt == 1)[0]
        neg_idx = np.where(yt == 0)[0]
        m = len(pos_idx)
        n = len(neg_idx)
        
        if m == 0 or n == 0:
            continue
            
        auc_a = roc_auc_score(yt, pa)
        auc_a_list.append(auc_a)
        
        # Structural components for model A
        # V10: pos against all neg
        v10_a = np.array([np.mean(pa[i] > pa[neg_idx]) + 0.5 * np.mean(pa[i] == pa[neg_idx]) for i in pos_idx])
        # V01: neg against all pos
        v01_a = np.array([np.mean(pa[pos_idx] > pa[j]) + 0.5 * np.mean(pa[pos_idx] == pa[j]) for j in neg_idx])
        
        s10_a = np.var(v10_a, ddof=1) if len(v10_a) > 1 else 0
        s01_a = np.var(v01_a, ddof=1) if len(v01_a) > 1 else 0
        var_a = (s10_a / m) + (s01_a / n)
        
        if y_prob_b is not None:
            pb = y_prob_b[:, c]
            auc_b = roc_auc_score(yt, pb)
            auc_b_list.append(auc_b)
            
            v10_b = np.array([np.mean(pb[i] > pb[neg_idx]) + 0.5 * np.mean(pb[i] == pb[neg_idx]) for i in pos_idx])
            v01_b = np.array([np.mean(pb[pos_idx] > pb[j]) + 0.5 * np.mean(pb[pos_idx] == pb[j]) for j in neg_idx])
            
            s10_b = np.var(v10_b, ddof=1) if len(v10_b) > 1 else 0
            s01_b = np.var(v01_b, ddof=1) if len(v01_b) > 1 else 0
            var_b = (s10_b / m) + (s01_b / n)
            
            # Covariance
            cov10 = np.cov(v10_a, v10_b)[0, 1] if len(v10_a) > 1 else 0
            cov01 = np.cov(v01_a, v01_b)[0, 1] if len(v01_a) > 1 else 0
            cov_ab = (cov10 / m) + (cov01 / n)
            
            var_diff = max(var_a + var_b - 2 * cov_ab, 1e-12)
            z = (auc_a - auc_b) / np.sqrt(var_diff)
            p = 2 * (1.0 - stats.norm.cdf(abs(z)))
            
            diffs.append(auc_a - auc_b)
            z_scores.append(z)
            p_values.append(p)
            
    if y_prob_b is not None:
        mean_diff = float(np.mean(diffs))
        mean_z = float(np.mean(z_scores))
        combined_p = float(stats.combine_pvalues(p_values, method='fisher')[1]) if p_values else 1.0
        return {
            'auc_a': float(np.mean(auc_a_list)),
            'auc_b': float(np.mean(auc_b_list)),
            'diff': mean_diff,
            'z_score': mean_z,
            'p_value': combined_p,
            'per_class_p': p_values
        }
    else:
        return {
            'auc_a': float(np.mean(auc_a_list))
        }

def compute_mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict[str, Any]:
    """
    Computes McNemar's paired test for classification disagreements.
    b: A correct, B wrong
    c: A wrong, B correct
    Continuity-corrected test statistic: (|b - c| - 1)^2 / (b + c)
    """
    # Sample-level correctness (exact match or mean correctness across classes)
    correct_a = (y_pred_a == y_true).all(axis=-1)
    correct_b = (y_pred_b == y_true).all(axis=-1)
    
    # Contingency counts
    n00 = np.sum(~correct_a & ~correct_b)
    n01 = np.sum(~correct_a & correct_b) # b: A wrong, B correct
    n10 = np.sum(correct_a & ~correct_b) # c: A correct, B wrong
    n11 = np.sum(correct_a & correct_b)
    
    b = n10
    c = n01
    
    if (b + c) > 0:
        stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
        p_val = 1.0 - stats.chi2.cdf(stat, df=1)
    else:
        stat = 0.0
        p_val = 1.0
        
    return {
        'contingency_table': [[int(n11), int(n10)], [int(n01), int(n00)]],
        'model_a_superior_counts': int(b),
        'model_b_superior_counts': int(c),
        'chi2_stat': float(stat),
        'p_value': float(p_val)
    }

def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: Optional[np.ndarray] = None,
    num_resamples: int = 2000,
    seed: int = 42
) -> Dict[str, Tuple[float, float, float]]:
    """
    Stratified patient-level bootstrap confidence intervals (2,000 resamples).
    Computes 95% CIs for AUROC, AUPRC, F1, Brier score, and paired differences.
    Returns {metric: (mean, lower_95, upper_95)}.
    """
    rng = np.random.default_rng(seed)
    N = len(y_true)
    
    boot_auroc_a, boot_auroc_b, boot_diff_auroc = [], [], []
    boot_auprc_a, boot_auprc_b, boot_diff_auprc = [], [], []
    boot_f1_a, boot_f1_b, boot_diff_f1 = [], [], []
    boot_brier_a, boot_brier_b, boot_diff_brier = [], [], []
    
    pred_a = (y_prob_a >= 0.5).astype(int)
    pred_b = (y_prob_b >= 0.5).astype(int) if y_prob_b is not None else None
    
    for _ in range(num_resamples):
        idx = rng.choice(N, size=N, replace=True)
        yt = y_true[idx]
        
        # Avoid all-zero columns in resample
        if np.any(yt.sum(axis=0) == 0):
            continue
            
        pa = y_prob_a[idx]
        auc_a = roc_auc_score(yt, pa, average='macro')
        prc_a = average_precision_score(yt, pa, average='macro')
        f1_a = f1_score(yt, pred_a[idx], average='macro', zero_division=0)
        brier_a = np.mean([brier_score_loss(yt[:, c], pa[:, c]) for c in range(5)])
        
        boot_auroc_a.append(auc_a)
        boot_auprc_a.append(prc_a)
        boot_f1_a.append(f1_a)
        boot_brier_a.append(brier_a)
        
        if y_prob_b is not None:
            pb = y_prob_b[idx]
            auc_b = roc_auc_score(yt, pb, average='macro')
            prc_b = average_precision_score(yt, pb, average='macro')
            f1_b = f1_score(yt, pred_b[idx], average='macro', zero_division=0)
            brier_b = np.mean([brier_score_loss(yt[:, c], pb[:, c]) for c in range(5)])
            
            boot_auroc_b.append(auc_b)
            boot_auprc_b.append(prc_b)
            boot_f1_b.append(f1_b)
            boot_brier_b.append(brier_b)
            
            boot_diff_auroc.append(auc_a - auc_b)
            boot_diff_auprc.append(prc_a - prc_b)
            boot_diff_f1.append(f1_a - f1_b)
            boot_diff_brier.append(brier_a - brier_b)
            
    def get_summary(arr):
        return (float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
        
    res = {
        'auroc_a': get_summary(boot_auroc_a),
        'auprc_a': get_summary(boot_auprc_a),
        'f1_a': get_summary(boot_f1_a),
        'brier_a': get_summary(boot_brier_a)
    }
    
    if y_prob_b is not None:
        res['auroc_b'] = get_summary(boot_auroc_b)
        res['auprc_b'] = get_summary(boot_auprc_b)
        res['f1_b'] = get_summary(boot_f1_b)
        res['brier_b'] = get_summary(boot_brier_b)
        
        res['diff_auroc'] = get_summary(boot_diff_auroc)
        res['diff_auprc'] = get_summary(boot_diff_auprc)
        res['diff_f1'] = get_summary(boot_diff_f1)
        res['diff_brier'] = get_summary(boot_diff_brier)
        
    return res

def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """
    Applies Holm-Bonferroni step-down procedure to control Family-Wise Error Rate (FWER).
    Returns (adjusted_p_values, reject_null_flags).
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    adj_p = np.zeros(n)
    reject = np.zeros(n, dtype=bool)
    
    cum_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        p = p_values[idx]
        adj = min((n - rank) * p, 1.0)
        adj = max(adj, cum_max)
        cum_max = adj
        adj_p[idx] = adj
        reject[idx] = (adj < alpha)
        
    return adj_p.tolist(), reject.tolist()

def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Computes paired Cohen's d effect size."""
    diff = x - y
    sd = np.std(diff, ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(diff) / sd)
