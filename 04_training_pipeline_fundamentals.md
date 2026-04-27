# Module 04: Training Pipeline Fundamentals

## Learning Objectives
By the end of this module you will be able to:
- Build production-quality `Dataset` and `DataLoader` classes
- Implement a complete training loop with validation and checkpointing
- Choose and configure the right loss function for any task
- Select and tune optimizers: SGD, Adam, AdamW, and their variants
- Use learning rate schedulers effectively
- Track experiments with TensorBoard and logging
- Debug common training failures (loss not decreasing, NaN, overfitting)

---

## 4.1 Dataset & DataLoader

PyTorch's data pipeline uses two abstractions:
- **`Dataset`** — knows how to load one sample
- **`DataLoader`** — batches samples, shuffles, and runs workers in parallel

### Custom Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

class TabularDataset(Dataset):
    """
    Generic tabular dataset.
    Assumes X is float, y is long (for classification) or float (regression).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class ImageFolderDataset(Dataset):
    """
    Image classification dataset. Directory layout:
        root/
            class_a/img1.jpg, img2.jpg, ...
            class_b/img1.jpg, ...
    """

    def __init__(self, root: str, transform=None):
        self.root      = Path(root)
        self.transform = transform
        self.classes   = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            for img_path in (self.root / cls).glob("*.jpg"):
                self.samples.append((img_path, self.class_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── DataLoaders ──────────────────────────────────────────────────────────────
train_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = ImageFolderDataset("data/train", transform=train_transform)
val_ds   = ImageFolderDataset("data/val",   transform=val_transform)

train_dl = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    num_workers=4,        # parallel data loading (0 = main process)
    pin_memory=True,      # faster host→GPU transfer
    drop_last=True,       # drop the last incomplete batch (good for BatchNorm)
    persistent_workers=True,  # keep workers alive between epochs
    prefetch_factor=2,    # batches to prefetch per worker
)
val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
```

### Weighted Sampling (handling class imbalance)

```python
from torch.utils.data import WeightedRandomSampler

def make_weighted_sampler(labels: list) -> WeightedRandomSampler:
    labels = torch.tensor(labels)
    class_counts = torch.bincount(labels)
    # weight per class: inverse of frequency
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

sampler = make_weighted_sampler(train_ds.targets)
balanced_dl = DataLoader(train_ds, batch_size=64, sampler=sampler)
# NOTE: sampler and shuffle=True are mutually exclusive
```

---

## 4.2 Loss Functions

### Classification Losses

```python
import torch
import torch.nn as nn

# ── Cross-Entropy Loss ────────────────────────────────────────────────────────
# CE(y, ŷ) = -Σᵢ yᵢ log(ŷᵢ)
# In PyTorch: takes RAW LOGITS (not softmax output) — more numerically stable
criterion = nn.CrossEntropyLoss()
logits = torch.randn(8, 10)    # (batch, num_classes)
targets = torch.randint(0, 10, (8,))
loss = criterion(logits, targets)

# With class weights (for imbalanced datasets)
weights = torch.tensor([1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
criterion_weighted = nn.CrossEntropyLoss(weight=weights)

# With label smoothing (prevents overconfidence)
criterion_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)

# ── Binary Cross-Entropy ──────────────────────────────────────────────────────
# Takes LOGITS (not sigmoid), numerically safe
bce = nn.BCEWithLogitsLoss()
logits_binary = torch.randn(8, 1).squeeze()   # (8,)
targets_binary = torch.randint(0, 2, (8,)).float()
loss = bce(logits_binary, targets_binary)

# Multi-label classification (each sample can have multiple classes)
multilabel_logits = torch.randn(8, 5)
multilabel_targets = torch.randint(0, 2, (8, 5)).float()
loss = bce(multilabel_logits, multilabel_targets)
```

### Regression Losses

```python
# ── Mean Squared Error: L = (1/N) Σ (y - ŷ)² ────────────────────────────────
mse = nn.MSELoss()
preds   = torch.randn(16)
targets = torch.randn(16)
loss = mse(preds, targets)

# ── Mean Absolute Error: L = (1/N) Σ |y - ŷ| ────────────────────────────────
# More robust to outliers than MSE
mae = nn.L1Loss()

# ── Huber Loss: quadratic for small errors, linear for large ──────────────────
# Combines robustness of MAE with smoothness of MSE
huber = nn.HuberLoss(delta=1.0)

# ── Cosine Embedding Loss: for metric learning ────────────────────────────────
cos = nn.CosineEmbeddingLoss()
x1 = torch.randn(8, 128)
x2 = torch.randn(8, 128)
y  = torch.ones(8)        # 1 for similar, -1 for dissimilar
loss = cos(x1, x2, y)

# ── Triplet Margin Loss: anchor-positive-negative ─────────────────────────────
triplet = nn.TripletMarginLoss(margin=1.0)
anchor   = torch.randn(8, 128)
positive = torch.randn(8, 128)
negative = torch.randn(8, 128)
loss = triplet(anchor, positive, negative)
```

### Custom Loss

```python
class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy examples to focus on hard ones.
    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
    Reference: Lin et al., 2017 — RetinaNet
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        probs    = torch.sigmoid(logits)
        p_t      = probs * targets + (1 - probs) * (1 - targets)
        alpha_t  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        fl       = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        return fl.mean()
```

---

## 4.3 Optimizers

Optimizers update model parameters to minimise the loss.

### Core Mathematics

**SGD with momentum:**
```
v_t = β·v_{t-1} + ∇L(θ_t)
θ_{t+1} = θ_t − η·v_t
```

**Adam:**
```
m_t = β₁·m_{t-1} + (1−β₁)·g_t          (1st moment: mean)
v_t = β₂·v_{t-1} + (1−β₂)·g_t²         (2nd moment: variance)
m̂_t = m_t / (1−β₁ᵗ)                    (bias correction)
v̂_t = v_t / (1−β₂ᵗ)
θ_{t+1} = θ_t − η · m̂_t / (√v̂_t + ε)
```

**AdamW** = Adam + decoupled weight decay:
```
θ_{t+1} = (1 − η·λ)·θ_t − η · m̂_t / (√v̂_t + ε)
```
Weight decay in Adam incorrectly applies to the adaptive learning rate; AdamW fixes this.

```python
import torch.optim as optim

model = MLP(784, [256], 10)  # from module 03

# ── SGD ──────────────────────────────────────────────────────────────────────
sgd = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,   # L2 regularisation
    nesterov=True,       # look-ahead gradient
)

# ── Adam ─────────────────────────────────────────────────────────────────────
adam = optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),  # (β₁, β₂)
    eps=1e-8,
    weight_decay=0,      # L2 is INCORRECTLY applied in Adam — use AdamW instead
)

# ── AdamW (preferred for deep learning) ──────────────────────────────────────
adamw = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.01,   # properly decoupled weight decay
)

# ── Layer-wise learning rates ──────────────────────────────────────────────
# Different lrs for different parts of the model (common in fine-tuning)
optimizer = optim.AdamW([
    {"params": model.net[0].parameters(), "lr": 1e-4},   # frozen-ish backbone
    {"params": model.net[-1].parameters(), "lr": 1e-3},  # head trains faster
], lr=1e-3, weight_decay=0.01)

# ── Optimizer state_dict (save/restore) ────────────────────────────────────
state = optimizer.state_dict()
optimizer.load_state_dict(state)
```

---

## 4.4 Learning Rate Schedulers

The learning rate schedule is often as important as the optimizer choice.

```python
from torch.optim import lr_scheduler

optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# ── StepLR: multiply lr by γ every step_size epochs ─────────────────────────
sched_step = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# ── CosineAnnealingLR: cosine decay from lr_max to eta_min ───────────────────
# η_t = η_min + 0.5(η_max − η_min)(1 + cos(πt/T))
sched_cos = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# ── OneCycleLR: warmup → cosine decay (fast convergence, used w/ SGD/Adam) ──
# Often the best single-run scheduler
sched_onecycle = lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-2,
    steps_per_epoch=len(train_dl),
    epochs=30,
    pct_start=0.3,      # 30% of training is warmup
    anneal_strategy="cos",
)

# ── LinearWarmupCosine (manual implementation) ───────────────────────────────
def warmup_cosine_schedule(step, warmup_steps, total_steps, min_lr=0.0, max_lr=1.0):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))

sched_lambda = lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: warmup_cosine_schedule(step, 500, 5000),
)

# ── ReduceLROnPlateau: reduce when metric stops improving ────────────────────
sched_plateau = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
)

# ── Usage in training loop ────────────────────────────────────────────────────
for epoch in range(num_epochs):
    train(model, train_dl, optimizer, ...)
    val_loss = validate(model, val_dl, ...)

    sched_step.step()               # epoch-based
    # sched_onecycle.step()         # call after EACH BATCH for OneCycleLR
    # sched_plateau.step(val_loss)  # metric-based

    print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
```

---

## 4.5 The Complete Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional
import time

class Trainer:
    """
    A reusable training harness with:
    - Train + validation loop
    - Checkpointing (best model + latest)
    - Early stopping
    - Gradient clipping
    - TensorBoard logging
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler=None,
        device: torch.device = torch.device("cpu"),
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        patience: int = 10,
    ):
        self.model          = model.to(device)
        self.optimizer      = optimizer
        self.criterion      = criterion
        self.scheduler      = scheduler
        self.device         = device
        self.max_grad_norm  = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.patience       = patience

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ── single epoch ─────────────────────────────────────────────────────────
    def _train_epoch(self, loader: DataLoader) -> tuple:
        self.model.train()
        total_loss = correct = total = 0

        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()        # 1. zero gradients
            logits = self.model(x)            # 2. forward
            loss   = self.criterion(logits, y) # 3. loss
            loss.backward()                   # 4. backward

            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)  # 5. clip

            self.optimizer.step()             # 6. update

            if self.scheduler and hasattr(self.scheduler, "step_batch"):
                self.scheduler.step()         # batch-level scheduler (OneCycleLR)

            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)

        return total_loss / total, correct / total

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> tuple:
        self.model.eval()
        total_loss = correct = total = 0

        for x, y in loader:
            x, y   = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            logits  = self.model(x)
            loss    = self.criterion(logits, y)
            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)

        return total_loss / total, correct / total

    # ── checkpoint ───────────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        state = {
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss":        val_loss,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
        }
        path = self.checkpoint_dir / "latest.pt"
        torch.save(state, path)
        if is_best:
            torch.save(state, self.checkpoint_dir / "best.pt")
            print(f"  ✔ Saved best model (val_loss={val_loss:.4f})")

    def load_checkpoint(self, path: str = "best"):
        full_path = self.checkpoint_dir / f"{path}.pt"
        ckpt = torch.load(full_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.scheduler and ckpt.get("scheduler_state"):
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
        return ckpt["epoch"]

    # ── main fit loop ─────────────────────────────────────────────────────────
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        resume: bool = False,
    ) -> dict:
        start_epoch = 0
        if resume:
            start_epoch = self.load_checkpoint("latest") + 1

        for epoch in range(start_epoch, epochs):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss,   val_acc   = self._val_epoch(val_loader)

            if self.scheduler and not hasattr(self.scheduler, "step_batch"):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss    = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            self._save_checkpoint(epoch, val_loss, is_best)

            for k, v in [("train_loss", train_loss), ("val_loss", val_loss),
                          ("train_acc",  train_acc),  ("val_acc",  val_acc)]:
                self.history[k].append(v)

            lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch+1:3d}/{epochs} "
                f"| train_loss={train_loss:.4f} acc={train_acc:.4f} "
                f"| val_loss={val_loss:.4f} acc={val_acc:.4f} "
                f"| lr={lr:.2e} | {elapsed:.1f}s"
            )

            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping after {self.patience} epochs without improvement.")
                break

        return self.history
```

---

## 4.6 TensorBoard Logging

```python
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

writer = SummaryWriter(log_dir="runs/experiment_01")

# ── Scalars ──────────────────────────────────────────────────────────────────
for epoch in range(50):
    train_loss = np.random.rand()
    val_loss   = np.random.rand() + 0.05
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val",   val_loss,   epoch)
    writer.add_scalar("LR",         optimizer.param_groups[0]["lr"], epoch)

# ── Histograms: monitor weight distributions ──────────────────────────────────
for name, param in model.named_parameters():
    writer.add_histogram(f"weights/{name}", param, epoch)
    if param.grad is not None:
        writer.add_histogram(f"gradients/{name}", param.grad, epoch)

# ── Images ───────────────────────────────────────────────────────────────────
imgs = torch.rand(16, 3, 32, 32)   # a grid of images
writer.add_images("input_samples", imgs, epoch)

# ── Graph: visualise model architecture ──────────────────────────────────────
dummy_input = torch.randn(1, 784)
writer.add_graph(model, dummy_input)

writer.close()

# Launch: tensorboard --logdir=runs
```

---

## 4.7 Debugging Training Failures

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss is NaN immediately | Learning rate too high, bad init, NaN in data | Reduce lr, check data, add `torch.isnan` assertions |
| Loss not decreasing | LR too low, model too small, data bug | Overfit one batch first to verify model works |
| Loss decreases then plateaus | LR decay needed, learning rate too high | Use scheduler, reduce LR |
| Train loss low, val loss high | Overfitting | More data, dropout, weight decay, early stopping |
| GPU memory OOM | Batch size too large, activations not freed | Reduce batch, gradient checkpointing, `torch.cuda.empty_cache()` |
| Gradients all zero | Activation kills gradient (dead ReLU), wrong loss | Check activation outputs, verify loss isn't constant |
| Gradients exploding | No clipping, LR too high | `clip_grad_norm_`, reduce LR |

### Overfit One Batch (Sanity Check)

```python
def overfit_one_batch(model, loader, optimizer, criterion, device, n_steps=100):
    """
    If a model can't overfit a single batch, something is fundamentally wrong.
    The loss should reach near-zero in < 100 steps for most models.
    """
    model.train()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    for step in range(n_steps):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"Step {step:3d}: loss={loss.item():.6f}")

    if loss.item() < 0.01:
        print("Sanity check PASSED — model can overfit a batch.")
    else:
        print("WARNING: model failed to overfit — check architecture and loss.")
```

---

## Exercises

**Exercise 4.1** Build a `CelebA` attribute prediction dataset: binary attributes (e.g. "Smiling", "Eyeglasses") loaded from CSV + image paths. Use a `WeightedRandomSampler` to handle class imbalance.

**Exercise 4.2** Implement `LabelSmoothingCrossEntropy` from scratch. Verify it matches `nn.CrossEntropyLoss(label_smoothing=0.1)`.

**Exercise 4.3** Add gradient norm logging to the `Trainer` class. Plot the gradient norms across epochs to detect exploding/vanishing gradients.

**Exercise 4.4** Implement the warmup + cosine annealing scheduler from scratch as a `LambdaLR` and verify it matches the expected learning rate curve by plotting it.

---

## Module Summary

| Concept | Key Points |
|---------|-----------|
| Dataset/DataLoader | `__len__` + `__getitem__`; `num_workers`, `pin_memory`, `prefetch_factor` |
| Loss functions | CE for classification (logits!); MSE/Huber for regression |
| Optimizers | AdamW is default; SGD+momentum+OneCycleLR for best accuracy |
| Schedulers | OneCycleLR (fast); CosineAnnealingLR (stable); ReduceLROnPlateau (safe) |
| Training loop | zero_grad → forward → loss → backward → clip → step → scheduler |
| Checkpointing | Save model + optimizer + scheduler state_dict |
| Early stopping | Stop when val_loss doesn't improve for N epochs |

---

## Quiz

1. What does `pin_memory=True` do and when is it beneficial?
2. Why should you use `BCEWithLogitsLoss` instead of `BCELoss + sigmoid`?
3. What is the difference between Adam and AdamW?
4. When would you call `scheduler.step()` per batch vs per epoch?
5. Why does `drop_last=True` in DataLoader help with BatchNorm?
6. What does it mean if loss is NaN at step 0?
7. Why is the "overfit one batch" test a valuable sanity check?

---

*Next: [Module 05 — Convolutional Neural Networks](./05_convolutional_neural_networks.md)*
