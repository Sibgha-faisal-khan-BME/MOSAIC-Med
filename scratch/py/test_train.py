import sys
sys.path.append('.')
import time
import torch
from scratch.py.data import cache_ptbxl_dataset
from scratch.py.models import create_expert
from scratch.py.training import train_single_expert, evaluate_model, get_dataloader
from scratch.py.metrics import compute_all_diagnostic_metrics

data = cache_ptbxl_dataset()
train_loader = get_dataloader(data['X_train'], data['y_train'], batch_size=64, shuffle=True)
val_loader = get_dataloader(data['X_val'], data['y_val'], batch_size=64, shuffle=False)
test_loader = get_dataloader(data['X_test'], data['y_test'], batch_size=64, shuffle=False)

print('Testing 500k ResNet expert training for 5 epochs...')
model = create_expert('resnet', target_budget='500k')
t0 = time.time()
res = train_single_expert(model, train_loader, val_loader, num_epochs=5, lr=1e-3, verbose=True)
train_time = time.time() - t0

y_prob, _, lat, vram = evaluate_model(res['model'], test_loader)
m = compute_all_diagnostic_metrics(data['y_test'], y_prob)

print(f"Training time: {train_time:.1f}s")
print(f"Test Macro AUROC: {m['auroc']:.4f}, AUPRC: {m['auprc']:.4f}, F1: {m['f1']:.4f}, Brier: {m['brier']:.4f}")
