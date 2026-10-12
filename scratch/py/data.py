"""
Data loading, preprocessing, caching, and corruption utilities for PTB-XL.
"""

import os
import ast
import time
import numpy as np
import pandas as pd
import wfdb
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Dict, List, Optional

CLASSES = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
CLASS_NAMES = {
    'CD': 'Conduction Disturbance',
    'HYP': 'Hypertrophy',
    'MI': 'Myocardial Infarction',
    'NORM': 'Normal ECG',
    'STTC': 'ST/T Changes'
}

def load_ptbxl_metadata(ptbxl_dir: str = 'ptbxl') -> pd.DataFrame:
    """Load and parse PTB-XL database metadata."""
    csv_path = os.path.join(ptbxl_dir, 'ptbxl_database.csv')
    df = pd.read_csv(csv_path, index_col='ecg_id')
    df['scp_codes_dict'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Load scp_statements.csv for diagnostic aggregation
    statements_path = os.path.join(ptbxl_dir, 'scp_statements.csv')
    agg_df = pd.read_csv(statements_path, index_col=0)
    diag_agg = agg_df[agg_df.diagnostic == 1]
    
    def aggregate_diagnostic(scp_dict):
        classes = set()
        for key in scp_dict.keys():
            if key in diag_agg.index:
                dclass = diag_agg.loc[key].diagnostic_class
                if isinstance(dclass, str) and dclass in CLASSES:
                    classes.add(dclass)
        return sorted(list(classes))
    
    df['diagnostic_superclass'] = df['scp_codes_dict'].apply(aggregate_diagnostic)
    
    # One-hot / multi-label matrix (5 classes)
    for c in CLASSES:
        df[f'label_{c}'] = df['diagnostic_superclass'].apply(lambda x: 1 if c in x else 0)
    
    return df

def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    """
    Standardize 12-lead ECG signal:
    - Replace NaNs/Infs with 0
    - Per-lead zero mean and unit variance normalization
    - Clip extreme artifacts to [-10, 10]
    Signal shape: (1000, 12) -> return (12, 1000) float32
    """
    sig = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    # Mean and std across time per lead
    mean = np.mean(sig, axis=0, keepdims=True)
    std = np.std(sig, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    sig = (sig - mean) / std
    sig = np.clip(sig, -10.0, 10.0)
    # Transpose to (leads, time) -> (12, 1000)
    return sig.T.astype(np.float32)

def cache_ptbxl_dataset(
    ptbxl_dir: str = 'ptbxl',
    cache_dir: str = 'scratch/cache',
    max_workers: int = 8,
    force_recompute: bool = False
) -> Dict[str, np.ndarray]:
    """
    Load raw ECG records from records100/, preprocess them, and cache to disk as .npy.
    Returns dictionary with loaded arrays.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_files = {
        'X_train': os.path.join(cache_dir, 'X_train.npy'),
        'y_train': os.path.join(cache_dir, 'y_train.npy'),
        'X_val': os.path.join(cache_dir, 'X_val.npy'),
        'y_val': os.path.join(cache_dir, 'y_val.npy'),
        'X_test': os.path.join(cache_dir, 'X_test.npy'),
        'y_test': os.path.join(cache_dir, 'y_test.npy'),
        'meta_test': os.path.join(cache_dir, 'meta_test.csv')
    }
    
    # Check if cache exists
    if not force_recompute and all(os.path.exists(p) for p in cache_files.values()):
        print(f"Loading cached arrays from {cache_dir}...")
        return {
            'X_train': np.load(cache_files['X_train']),
            'y_train': np.load(cache_files['y_train']),
            'X_val': np.load(cache_files['X_val']),
            'y_val': np.load(cache_files['y_val']),
            'X_test': np.load(cache_files['X_test']),
            'y_test': np.load(cache_files['y_test']),
            'meta_test': pd.read_csv(cache_files['meta_test'])
        }
    
    print("Reading PTB-XL metadata and loading signals...")
    t0 = time.time()
    df = load_ptbxl_metadata(ptbxl_dir)
    
    # Read files in parallel
    filenames = df['filename_lr'].tolist()
    def load_single_file(f):
        full_path = os.path.join(ptbxl_dir, f)
        record, _ = wfdb.rdsamp(full_path)
        return preprocess_signal(record)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        all_signals = list(executor.map(load_single_file, filenames))
    
    X_all = np.stack(all_signals, axis=0) # shape: (N, 12, 1000)
    label_cols = [f'label_{c}' for c in CLASSES]
    y_all = df[label_cols].values.astype(np.float32) # shape: (N, 5)
    
    # Folds split: 1-8 train, 9 val, 10 test
    train_mask = (df['strat_fold'] <= 8).values
    val_mask = (df['strat_fold'] == 9).values
    test_mask = (df['strat_fold'] == 10).values
    
    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    
    meta_test = df[test_mask].copy()
    
    # Save cache
    np.save(cache_files['X_train'], X_train)
    np.save(cache_files['y_train'], y_train)
    np.save(cache_files['X_val'], X_val)
    np.save(cache_files['y_val'], y_val)
    np.save(cache_files['X_test'], X_test)
    np.save(cache_files['y_test'], y_test)
    meta_test.to_csv(cache_files['meta_test'], index=True)
    
    print(f"Caching complete in {time.time()-t0:.1f}s.")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'meta_test': meta_test
    }

# Noise and Corruption Perturbations for Robustness Stress Testing
def add_high_frequency_noise(X: np.ndarray, snr_sigma: float = 0.1, seed: int = 42) -> np.ndarray:
    """Add zero-mean Gaussian noise simulating EMG / muscle tremor artifact."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, snr_sigma, size=X.shape).astype(np.float32)
    return X + noise

def add_baseline_drift(X: np.ndarray, amplitude: float = 0.5, freq: float = 0.2, fs: float = 100.0, seed: int = 42) -> np.ndarray:
    """Add low-frequency sinusoidal baseline wander simulating respiration/electrode movement."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, X.shape[-1] / fs, X.shape[-1], endpoint=False, dtype=np.float32)
    phase = rng.uniform(0, 2 * np.pi, size=(X.shape[0], X.shape[1], 1)).astype(np.float32)
    drift = amplitude * np.sin(2 * np.pi * freq * t + phase)
    return X + drift

def drop_leads(X: np.ndarray, lead_indices: List[int]) -> np.ndarray:
    """Simulate missing lead / disconnected electrode artifact by zeroing selected channels."""
    X_corrupted = X.copy()
    X_corrupted[:, lead_indices, :] = 0.0
    return X_corrupted
