"""
Architectures for MOSAIC-Med:
- ResNet1D
- MultiScaleCNN
- TCN1D
- ConvAttention1D
- AdaptiveRouter
- MOSAICMedEnsemble (with escalation and abstention logic)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

# -------------------------------------------------------------
# 1. 1D ResNet Architecture
# -------------------------------------------------------------
class ResBlock1D(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(out_c)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))

class ResNet1D(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 5, base_filters: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
        bf = base_filters
        self.layer1 = ResBlock1D(bf, bf, stride=1)
        self.layer2 = ResBlock1D(bf, bf * 2, stride=2)
        self.layer3 = ResBlock1D(bf * 2, bf * 4, stride=2)
        self.layer4 = ResBlock1D(bf * 4, bf * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.feat_dim = bf * 4
        self.fc = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x: torch.Tensor, return_feat: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat = self.pool(x).squeeze(-1)
        logits = self.fc(feat)
        if return_feat:
            return logits, feat
        return logits

# -------------------------------------------------------------
# 2. MultiScale / Inception 1D Architecture
# -------------------------------------------------------------
class InceptionBlock1D(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        c = out_c // 4
        self.b1 = nn.Sequential(nn.Conv1d(in_c, c, kernel_size=5, stride=stride, padding=2, bias=False), nn.BatchNorm1d(c), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv1d(in_c, c, kernel_size=11, stride=stride, padding=5, bias=False), nn.BatchNorm1d(c), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(nn.Conv1d(in_c, c, kernel_size=23, stride=stride, padding=11, bias=False), nn.BatchNorm1d(c), nn.ReLU(inplace=True))
        self.b4 = nn.Sequential(nn.MaxPool1d(3, stride=stride, padding=1), nn.Conv1d(in_c, c, 1, bias=False), nn.BatchNorm1d(c), nn.ReLU(inplace=True))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 5, width: int = 44):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True)
        )
        self.b1 = InceptionBlock1D(width, width * 2, stride=2)
        self.b2 = InceptionBlock1D(width * 2, width * 4, stride=2)
        self.b3 = InceptionBlock1D(width * 4, width * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.feat_dim = width * 4
        self.fc = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x: torch.Tensor, return_feat: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        feat = self.pool(x).squeeze(-1)
        logits = self.fc(feat)
        if return_feat:
            return logits, feat
        return logits

# -------------------------------------------------------------
# 3. Temporal Convolutional Network (TCN)
# -------------------------------------------------------------
class TCNBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, dilation: int):
        super().__init__()
        pad = 2 * dilation
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=5, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, kernel_size=5, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_c)
        )
        self.skip = nn.Conv1d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.skip(x))

class TCN1D(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 5, channels: int = 48):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        self.b1 = TCNBlock(channels, channels, dilation=1)
        self.b2 = TCNBlock(channels, channels * 2, dilation=2)
        self.b3 = TCNBlock(channels * 2, channels * 2, dilation=4)
        self.b4 = TCNBlock(channels * 2, channels * 4, dilation=8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.feat_dim = channels * 4
        self.fc = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x: torch.Tensor, return_feat: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        feat = self.pool(x).squeeze(-1)
        logits = self.fc(feat)
        if return_feat:
            return logits, feat
        return logits

# -------------------------------------------------------------
# 4. Conv-Attention / Transformer 1D
# -------------------------------------------------------------
class ConvAttention1D(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 5, d_model: int = 140, num_layers: int = 2, nhead: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.feat_dim = d_model
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, return_feat: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.stem(x) # (B, d_model, 250)
        x = x.permute(0, 2, 1) # (B, 250, d_model)
        out = self.transformer(x)
        feat = out.mean(dim=1) # (B, d_model)
        logits = self.fc(feat)
        if return_feat:
            return logits, feat
        return logits

# -------------------------------------------------------------
# Model Factory for Target Budgets
# Total Budget = ~2.0M parameters
# -------------------------------------------------------------
def create_expert(arch: str, target_budget: str = '500k', num_classes: int = 5) -> nn.Module:
    """
    Creates an expert calibrated to target parameters:
    - '2m': ~2,000,000 params (for K=1)
    - '1m': ~1,000,000 params (for K=2)
    - '500k': ~500,000 params (for K=4)
    - '250k': ~250,000 params (for K=8)
    - '125k': ~125,000 params (for K=16)
    """
    if arch == 'resnet':
        filter_map = {'2m': 65, '1m': 46, '500k': 32, '250k': 23, '125k': 16}
        return ResNet1D(in_channels=12, num_classes=num_classes, base_filters=filter_map[target_budget])
    elif arch == 'multiscale':
        width_map = {'1m': 62, '500k': 44, '250k': 31, '125k': 22}
        return MultiScaleCNN(in_channels=12, num_classes=num_classes, width=width_map[target_budget])
    elif arch == 'tcn':
        ch_map = {'1m': 68, '500k': 48, '250k': 34, '125k': 24}
        return TCN1D(in_channels=12, num_classes=num_classes, channels=ch_map[target_budget])
    elif arch == 'conv_att':
        dm_map = {'1m': 200, '500k': 140, '250k': 96, '125k': 68}
        return ConvAttention1D(in_channels=12, num_classes=num_classes, d_model=dm_map[target_budget])
    else:
        raise ValueError(f"Unknown architecture {arch}")

# -------------------------------------------------------------
# 5. MOSAIC-Med Adaptive Router
# -------------------------------------------------------------
class AdaptiveRouter(nn.Module):
    """
    Learned patient-specific routing network.
    Uses summary statistics & pooled 1D features from the 12 leads to generate expert weights.
    Parameter count: ~15K (minimal overhead).
    """
    def __init__(self, in_channels: int = 12, num_experts: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.num_experts = num_experts
        # Compact 1D conv pooling front-end
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=15, stride=10, padding=7),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x: torch.Tensor, top_k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        logits = self.fc(feat)
        weights = F.softmax(logits, dim=-1) # (B, num_experts)
        
        if top_k is not None and top_k < self.num_experts:
            # Top-k sparse gating
            top_vals, top_indices = torch.topk(weights, k=top_k, dim=-1)
            sparse_weights = torch.zeros_like(weights).scatter_(-1, top_indices, top_vals)
            # Re-normalize across active experts
            sparse_weights = sparse_weights / (sparse_weights.sum(dim=-1, keepdim=True) + 1e-8)
            return sparse_weights, logits
            
        return weights, logits

# -------------------------------------------------------------
# 6. Diversity Regularization & Failure-Aware Loss Functions
# -------------------------------------------------------------
def compute_representation_diversity_loss(representations: List[torch.Tensor]) -> torch.Tensor:
    """
    Computes pairwise representation diversity / kernel alignment loss between penultimate expert representations.
    Dimension-invariant: computes patient-patient cosine Gram matrices G_i, G_j in (B, B) and penalizes correlation.
    Minimizing L_div enforces representation orthogonality across heterogeneous architectures.
    """
    K = len(representations)
    if K <= 1:
        return torch.tensor(0.0, device=representations[0].device)
    loss = torch.tensor(0.0, device=representations[0].device)
    count = 0
    for i in range(K):
        zi = F.normalize(representations[i], p=2, dim=-1) # (B, Di)
        gi = torch.mm(zi, zi.t()) # (B, B)
        norm_gi = torch.norm(gi, p='fro') + 1e-8
        gi = gi / norm_gi
        for j in range(i + 1, K):
            zj = F.normalize(representations[j], p=2, dim=-1) # (B, Dj)
            gj = torch.mm(zj, zj.t()) # (B, B)
            norm_gj = torch.norm(gj, p='fro') + 1e-8
            gj = gj / norm_gj
            sim = torch.sum(gi * gj)
            loss = loss + torch.clamp(sim, min=0.0)
            count += 1
    return loss / max(count, 1)

def compute_failure_aware_loss(expert_logits: List[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    """
    Computes negative correlation penalty / failure-aware loss:
    Encourages complementary error patterns rather than correlated errors.
    L_fail = 1/K sum_i (p_i - y) * (p_i - p_mean)
    """
    K = len(expert_logits)
    if K <= 1:
        return torch.tensor(0.0, device=expert_logits[0].device)
    probs = [torch.sigmoid(lg) for lg in expert_logits]
    p_mean = torch.mean(torch.stack(probs, dim=0), dim=0) # (B, num_classes)
    loss = torch.tensor(0.0, device=expert_logits[0].device)
    for pi in probs:
        loss = loss + torch.mean((pi - targets) * (pi - p_mean))
    return loss / K

# -------------------------------------------------------------
# 7. Uncertainty Quantification, Escalation, and Abstention
# -------------------------------------------------------------
def compute_predictive_uncertainty(probs: torch.Tensor, expert_probs_list: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes multi-label predictive uncertainty:
    1. Binary entropy across 5 classes: H(p) = - sum_c [p log p + (1-p) log (1-p)]
    2. Variance / disagreement across expert predictions (epistemic uncertainty)
    Returns: (mean_entropy, expert_disagreement)
    """
    eps = 1e-7
    p_clamped = torch.clamp(probs, eps, 1.0 - eps)
    entropy = - (p_clamped * torch.log(p_clamped) + (1.0 - p_clamped) * torch.log(1.0 - p_clamped))
    total_entropy = entropy.sum(dim=-1) # (B,)
    
    if expert_probs_list is not None and len(expert_probs_list) > 1:
        # Stack shape: (K, B, num_classes)
        stack_probs = torch.stack(expert_probs_list, dim=0)
        disagreement = torch.var(stack_probs, dim=0).sum(dim=-1) # (B,)
    else:
        disagreement = torch.zeros_like(total_entropy)
        
    return total_entropy, disagreement
