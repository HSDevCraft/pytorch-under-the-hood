# Module 09: Advanced Training Techniques

## Learning Objectives
By the end of this module you will be able to:
- Implement gradient accumulation for large effective batch sizes
- Use mixed precision training (AMP) with `torch.cuda.amp`
- Apply regularisation: weight decay, dropout variants, stochastic depth
- Implement learning rate warm-up and cosine annealing with restarts
- Use exponential moving average (EMA) of model weights
- Apply gradient checkpointing to train large models with limited memory
- Debug training instabilities: loss spikes, NaN gradients, oscillation

---

## 9.1 Gradient Accumulation

Simulates a larger batch size by accumulating gradients over multiple micro-steps before updating weights. Useful when GPU memory limits batch size.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_with_accumulation(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    accumulate_steps: int = 4,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Accumulate over `accumulate_steps` micro-batches before each optimizer.step().
    Effective batch size = loader.batch_size * accumulate_steps.
    """
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    step = 0

    for micro_step, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Scale loss so gradients are equivalent to a single large batch
        loss = criterion(model(x), y) / accumulate_steps
        loss.backward()
        total_loss += loss.item() * accumulate_steps

        is_last_micro = (micro_step + 1) % accumulate_steps == 0

        if is_last_micro or (micro_step + 1) == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

    return total_loss / len(loader)
```

---

## 9.2 Automatic Mixed Precision (AMP)

FP16 computation is 2–8× faster on modern GPUs and halves memory usage. The `GradScaler` handles the numerical instability of FP16 gradients.

```python
import torch
from torch.cuda.amp import GradScaler, autocast

def train_amp(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    accumulate_steps: int = 1,
) -> float:
    """
    AMP training with gradient accumulation support.
    """
    scaler = GradScaler()  # scales loss to prevent FP16 underflow
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Forward pass in FP16
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss   = criterion(logits, y) / accumulate_steps

        # Backward pass with scaled loss (prevents FP16 underflow)
        scaler.scale(loss).backward()

        if (step + 1) % accumulate_steps == 0:
            # Unscale gradients before clipping (operates in FP32)
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # If gradients are finite: update; else: skip step
            scaler.step(optimizer)

            # Update scaler (may increase/decrease scale factor)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulate_steps

    return total_loss / len(loader)


# ── BF16: preferred on Ampere+ GPUs (A100, RTX 3090+) ───────────────────────
# BF16 has the same range as FP32 (8 exponent bits) but lower precision
# Safer than FP16 — no loss spikes from overflow
with autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(x)

# No GradScaler needed with BF16!
```

---

## 9.3 Stochastic Depth (DropPath)

Randomly drops entire residual branches during training. Originally from "Deep Networks with Stochastic Depth" (Huang et al., 2016); now used in DeiT, Swin, ConvNeXt.

```python
import torch
import torch.nn as nn

class DropPath(nn.Module):
    """
    Drops the entire residual contribution with probability `drop_prob`.
    Applied per-sample within a batch.
    Identity during evaluation.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        # Bernoulli mask: 1 = keep, 0 = drop (shape: batch, 1, 1, ...)
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
        return x * mask / keep_prob   # scale up to maintain expected value


def build_stochastic_depth_rates(n_layers: int, max_drop_rate: float = 0.1) -> list:
    """
    Linearly increase drop rate from 0 to max_drop_rate.
    Standard practice: first layer is never dropped.
    """
    return [max_drop_rate * i / (n_layers - 1) for i in range(n_layers)]


# Usage in a block
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, drop_path_rate: float = 0.0):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm    = nn.LayerNorm(dim)
        self.pw1     = nn.Linear(dim, 4 * dim)
        self.pw2     = nn.Linear(4 * dim, dim)
        self.act     = nn.GELU()
        self.drop    = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)    # NCHW → NHWC for LayerNorm
        x = self.pw2(self.act(self.pw1(self.norm(x))))
        x = x.permute(0, 3, 1, 2)    # NHWC → NCHW
        return residual + self.drop(x)
```

---

## 9.4 Exponential Moving Average (EMA)

EMA of model weights often gives better test performance than the last checkpoint:

```
θ_ema ← α · θ_ema + (1 - α) · θ_model
```

Used in: DeiT, YOLOv8, Stable Diffusion, Whisper.

```python
import torch
import torch.nn as nn
from copy import deepcopy

class EMA:
    """
    Maintains an exponential moving average of model parameters.
    Typical decay: 0.9999 for large models, 0.999 for smaller ones.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay    = decay
        self.model    = deepcopy(model).eval()   # shadow model
        # Disable gradient on shadow model
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Call after every optimizer.step()."""
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def eval_model(self) -> nn.Module:
        """Returns the EMA model for evaluation."""
        return self.model

    def state_dict(self):
        return {"model": self.model.state_dict(), "decay": self.decay}

    def load_state_dict(self, state: dict):
        self.model.load_state_dict(state["model"])
        self.decay = state["decay"]


# Usage in training loop
model = resnet50()
ema   = EMA(model, decay=0.9999)
optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(100):
    for x, y in train_dl:
        optim.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optim.step()
        ema.update(model)   # ← update EMA after each step

    # Evaluate using EMA model
    ema_model = ema.eval_model()
    val_acc = evaluate(ema_model, val_dl, device)
```

---

## 9.5 Gradient Checkpointing

Trades compute for memory: instead of storing all intermediate activations for backprop, recompute them during the backward pass. Enables training models ~3–4× larger.

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

# ── Method 1: checkpoint_sequential for sequential models ─────────────────────
model = nn.Sequential(*[nn.Linear(512, 512) for _ in range(20)])

def forward_with_checkpointing(model, x):
    # Splits model into 4 segments; only 1/4 of activations stored at a time
    return checkpoint_sequential(model, segments=4, input=x)

# ── Method 2: checkpoint individual forward calls ─────────────────────────────
class CheckpointedResidualBlock(nn.Module):
    def __init__(self, block: nn.Module, use_checkpoint: bool = True):
        super().__init__()
        self.block = block
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            # Recompute block during backward instead of storing activations
            return checkpoint(self.block, x, use_reentrant=False)
        return self.block(x)

# ── Method 3: for transformers ────────────────────────────────────────────────
class CheckpointedTransformerBlock(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x, mask=None):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        if self.training:
            return checkpoint(create_custom_forward(self.block), x, mask, use_reentrant=False)
        return self.block(x, mask)
```

---

## 9.6 Learning Rate Warm-Up Strategies

```python
import torch
import numpy as np

class WarmupCosineSchedule:
    """
    Linear warmup followed by cosine annealing.
    Used in BERT, GPT, ViT training.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer     = optimizer
        self.warmup_steps  = warmup_steps
        self.total_steps   = total_steps
        self.min_lr_ratio  = min_lr_ratio
        self.base_lrs      = [g["lr"] for g in optimizer.param_groups]
        self.current_step  = 0

    def step(self):
        self.current_step += 1
        lrs = self._get_lrs()
        for lr, group in zip(lrs, self.optimizer.param_groups):
            group["lr"] = lr

    def _get_lrs(self) -> list:
        step = self.current_step
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale    = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
        return [base * scale for base in self.base_lrs]


class CosineAnnealingWithRestarts:
    """
    SGDR: Cosine Annealing with Warm Restarts (Loshchilov & Hutter, 2017).
    Periodically resets LR to max then re-decays — allows escaping local minima.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int = 100,       # initial cycle length
        T_mult: int = 2,      # multiply cycle length after each restart
        eta_min: float = 1e-6,
    ):
        self.sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )

    def step(self, epoch: float):
        self.sched.step(epoch)
```

---

## 9.7 Label Smoothing & Mixup in Training

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing.
    Replaces hard targets y ∈ {0,1} with smooth y = (1-ε)·y + ε/K
    where K is the number of classes and ε is the smoothing factor.
    Prevents overconfidence and improves calibration.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_cls = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Hard targets → one-hot
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n_cls - 1))
            smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MixupAugmentation:
    """
    Mixup (Zhang et al., 2018): interpolate two samples and their labels.
    Forces the model to behave linearly between training examples.
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.dist  = torch.distributions.Beta(alpha, alpha)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        if not self.training:
            return x, y, y, 1.0

        lam  = self.dist.sample().item()
        idx  = torch.randperm(x.size(0), device=x.device)
        x_mix = lam * x + (1 - lam) * x[idx]
        return x_mix, y, y[idx], lam

    def criterion(self, pred, y_a, y_b, lam, criterion):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

---

## 9.8 Complete Advanced Training Loop

```python
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

def advanced_train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    scheduler,
    ema: EMA,
    device: torch.device,
    accumulate_steps: int = 1,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
) -> dict:
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    n_correct  = 0
    n_total    = 0
    opt_steps  = 0

    for micro_step, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        ctx = autocast(device_type="cuda", dtype=torch.float16) if use_amp else torch.no_grad.__class__()

        with (autocast(device_type="cuda") if use_amp else contextlib.nullcontext()):
            logits = model(x)
            loss   = criterion(logits, y) / accumulate_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * accumulate_steps
        n_correct  += (logits.argmax(-1) == y).sum().item()
        n_total    += len(y)

        is_update_step = (micro_step + 1) % accumulate_steps == 0 or (micro_step + 1) == len(loader)

        if is_update_step:
            if use_amp:
                scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            ema.update(model)

            if hasattr(scheduler, "step_batch"):
                scheduler.step()

            opt_steps += 1

    return {
        "loss":     total_loss / len(loader),
        "acc":      n_correct / n_total,
        "opt_steps": opt_steps,
    }
```

---

## 9.9 Diagnosing Training Instabilities

```python
import torch
import torch.nn as nn
from typing import Dict

class TrainingMonitor:
    """Attach to a model to track gradient health during training."""

    def __init__(self, model: nn.Module, log_every: int = 100):
        self.model     = model
        self.log_every = log_every
        self.step      = 0
        self._hooks    = []
        self._grad_norms: Dict[str, list] = {}
        self._attach_hooks()

    def _attach_hooks(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                handle = param.register_hook(self._make_hook(name))
                self._hooks.append(handle)

    def _make_hook(self, name: str):
        def hook(grad):
            if grad is None:
                return
            norm = grad.norm().item()
            self._grad_norms.setdefault(name, []).append(norm)
        return hook

    def log(self) -> dict:
        self.step += 1
        if self.step % self.log_every != 0:
            return {}

        report = {}
        for name, norms in self._grad_norms.items():
            if norms:
                report[name] = {
                    "mean": sum(norms) / len(norms),
                    "max":  max(norms),
                    "min":  min(norms),
                    "has_nan": any(n != n for n in norms),
                }
        self._grad_norms.clear()

        # Check for dead neurons (zero-norm layers)
        dead = [k for k, v in report.items() if v["max"] < 1e-7]
        if dead:
            print(f"  ⚠ Possible dead gradients in: {dead}")

        # Check for exploding gradients
        exploding = [k for k, v in report.items() if v["max"] > 100]
        if exploding:
            print(f"  ⚠ Large gradients in: {exploding}")

        return report

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
```

---

## 9.10 Best Practices Checklist

| Technique | When to Apply | Expected Benefit |
|-----------|--------------|-----------------|
| AMP (FP16/BF16) | Always on GPU | 2–3× throughput, 2× memory |
| Gradient accumulation | When batch size is limited by memory | Stable training at large effective batch |
| EMA weights | Most tasks | +0.5–1% accuracy on evaluation |
| Gradient checkpointing | Transformer / very deep models | Train 3–4× larger model same memory |
| Stochastic depth | Deep CNNs and transformers (>12 layers) | Better regularisation, faster training |
| Label smoothing | All classification tasks | Better calibration, prevents overconfidence |
| Mixup/CutMix | Image classification | +1–2% on ImageNet-scale tasks |
| Warmup LR | Transformers especially | Prevents early divergence |
| Cosine with restarts (SGDR) | Long training runs | Can find better minima |
| `torch.compile` | PyTorch 2.0+, all models | 20–50% speedup with one line of code |

```python
# One-line speedup: torch.compile (PyTorch 2.0+)
import torch
model = MyModel()
model = torch.compile(model)   # traces and compiles the computation graph
# Works with autograd, AMP, DDP; best on A100/H100
```

---

## Exercises

**Exercise 9.1** Benchmark AMP vs full FP32 training on ResNet-50 + CIFAR-10. Report: throughput (samples/s), peak GPU memory, and final validation accuracy.

**Exercise 9.2** Implement `CutMix` from scratch. Verify that the expected mixing ratio matches the actual pixel-level mixing ratio. Combine with `MixUp` in a 50/50 split strategy as used in DeiT.

**Exercise 9.3** Add gradient checkpointing to `MiniGPT` from Module 08. Measure the maximum sequence length you can train with 8GB GPU memory with and without checkpointing.

---

## Module Summary

| Technique | Core Idea |
|-----------|----------|
| Gradient accumulation | Divide loss by N, accumulate N times, then step |
| AMP | Forward in FP16, loss scale + backward in FP32 |
| GradScaler | Multiply loss so FP16 grads don't underflow |
| EMA | Shadow model = α·shadow + (1-α)·model after each step |
| Gradient checkpointing | Recompute activations in backward instead of storing |
| DropPath | Drop whole residual branches randomly per sample |
| Label smoothing | Replace hard 0/1 targets with soft (1-ε)/ε/K targets |

---

## Quiz

1. Why must you divide the loss by `accumulate_steps` when doing gradient accumulation?
2. What does `GradScaler` do and why is it needed for FP16 but not BF16?
3. When would you use gradient checkpointing and what is the memory-compute tradeoff?
4. Why does EMA give better test accuracy than using the raw model weights?
5. What is the difference between DropPath and Dropout?
6. Why use warmup before cosine annealing in transformer training?

---

*Next: [Module 10 — GPU Performance & Mixed Precision](./10_gpu_performance_and_mixed_precision.md)*
