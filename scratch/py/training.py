"""
Sequential expert training and evaluation engine with AMP, early stopping,
checkpoint saving, and GPU memory cleanup.
"""

import os
import gc
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from .models import create_expert, AdaptiveRouter, compute_representation_diversity_loss, compute_failure_aware_loss
from .metrics import compute_all_diagnostic_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False # Native CUDA 1D conv kernels avoid Turing half-precision cuDNN instability

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    t_X = torch.from_numpy(X).float()
    t_y = torch.from_numpy(y).float()
    ds = TensorDataset(t_X, t_y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

def train_single_expert(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 12,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    checkpoint_path: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Trains a single expert using AdamW, CosineAnnealingLR, BCEWithLogitsLoss and AMP fp16.
    Saves best checkpoint based on validation loss.
    """
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')
    best_state = None
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE, non_blocking=True), by.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                logits = model(bx)
                loss = criterion(logits, by)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            n_batches += 1
            
        scheduler.step()
        train_loss /= max(n_batches, 1)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE, non_blocking=True), by.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    logits = model(bx)
                    loss = criterion(logits, by)
                val_loss += loss.item()
                val_batches += 1
        val_loss /= max(val_batches, 1)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if verbose and (epoch + 1) % 4 == 0:
            print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(best_state, checkpoint_path)
                
    if best_state is not None:
        model.load_state_dict(best_state)
        
    return {
        'model': model,
        'best_val_loss': best_val_loss,
        'history': history
    }

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    return_features: bool = False
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Evaluates model on test loader.
    Returns: (probabilities, features, avg_latency_ms_per_sample, peak_vram_mb)
    """
    model.eval()
    model = model.to(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    
    probs_list = []
    feats_list = []
    
    t0 = time.time()
    total_samples = 0
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(DEVICE, non_blocking=True)
            total_samples += len(bx)
            with torch.amp.autocast('cuda'):
                if return_features:
                    logits, feat = model(bx, return_feat=True)
                    feats_list.append(feat.cpu().numpy())
                else:
                    logits = model(bx)
            prob = torch.sigmoid(logits).cpu().numpy()
            prob = np.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)
            probs_list.append(prob)
            
    torch.cuda.synchronize()
    latency_ms = ((time.time() - t0) * 1000.0) / max(total_samples, 1)
    peak_vram_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)
    
    all_probs = np.concatenate(probs_list, axis=0)
    all_feats = np.concatenate(feats_list, axis=0) if return_features else None
    
    return all_probs, all_feats, latency_ms, peak_vram_mb

def train_adaptive_router(
    router: nn.Module,
    trained_experts: List[nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 8,
    lr: float = 1e-3,
    lambda_div: float = 0.05,
    lambda_fail: float = 0.05,
    checkpoint_path: Optional[str] = None,
    cached_train_logits: Optional[List[torch.Tensor]] = None,
    cached_val_logits: Optional[List[torch.Tensor]] = None
) -> nn.Module:
    """
    Trains the adaptive router on frozen or joint expert representations with diversity & failure loss.
    """
    router = router.to(DEVICE)
    for exp in trained_experts:
        exp.eval()
        exp.to(DEVICE)
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')
    best_state = None
    
    if cached_train_logits is None:
        cached_train_logits = []
    if cached_val_logits is None:
        cached_val_logits = []
    has_precomputed_train = (len(cached_train_logits) > 0)
    has_precomputed_val = (len(cached_val_logits) > 0)
    
    for epoch in range(num_epochs):
        router.train()
        train_loss = 0.0
        n_b = 0
        for b_idx, (bx, by) in enumerate(train_loader):
            bx, by = bx.to(DEVICE, non_blocking=True), by.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                weights, _ = router(bx) # (B, K)
                
                # Get expert predictions (cached after epoch 0 for speed)
                if epoch == 0 and not has_precomputed_train:
                    expert_logits = []
                    for exp in trained_experts:
                        with torch.no_grad():
                            lg = exp(bx)
                        expert_logits.append(lg)
                    stacked_logits = torch.stack(expert_logits, dim=1) # (B, K, num_classes)
                    cached_train_logits.append(stacked_logits.detach())
                else:
                    stacked_logits = cached_train_logits[b_idx]
                    
                # Weighted mixture
                fused_logits = torch.sum(stacked_logits * weights.unsqueeze(-1), dim=1)
                loss_diag = criterion(fused_logits, by)
                
                # Failure-aware penalty: penalize routing to expert k when expert k has error |p_k - y|
                if lambda_fail > 0:
                    with torch.no_grad():
                        exp_errors = torch.abs(torch.sigmoid(stacked_logits) - by.unsqueeze(1)).mean(dim=-1) # (B, K)
                    loss_fail = torch.mean(torch.sum(weights * exp_errors, dim=-1))
                else:
                    loss_fail = torch.tensor(0.0, device=DEVICE)
                    
                # Diversity penalty: encourage uniform expert exploration / entropy
                if lambda_div > 0:
                    eps = 1e-7
                    w_clamped = torch.clamp(weights, eps, 1.0)
                    entropy = -torch.sum(w_clamped * torch.log(w_clamped), dim=-1).mean()
                    loss_div = -entropy
                else:
                    loss_div = torch.tensor(0.0, device=DEVICE)
                    
                total_loss = loss_diag + lambda_fail * loss_fail + lambda_div * loss_div
                
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
            n_b += 1
            
        # Validation check
        router.eval()
        val_loss = 0.0
        v_b = 0
        with torch.no_grad():
            for vb_idx, (bx, by) in enumerate(val_loader):
                bx, by = bx.to(DEVICE, non_blocking=True), by.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    weights, _ = router(bx)
                    if epoch == 0 and not has_precomputed_val:
                        expert_logits = [exp(bx) for exp in trained_experts]
                        stacked_logits = torch.stack(expert_logits, dim=1)
                        cached_val_logits.append(stacked_logits.detach())
                    else:
                        stacked_logits = cached_val_logits[vb_idx]
                    fused_logits = torch.sum(stacked_logits * weights.unsqueeze(-1), dim=1)
                    vloss = criterion(fused_logits, by)
                val_loss += vloss.item()
                v_b += 1
        val_loss /= max(v_b, 1)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in router.state_dict().items()}
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(best_state, checkpoint_path)
                
    if best_state is not None:
        router.load_state_dict(best_state)
    return router

def free_gpu_memory(models_list: Optional[List[Any]] = None):
    """Frees models and flushes CUDA VRAM cache."""
    if models_list:
        for m in models_list:
            del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
