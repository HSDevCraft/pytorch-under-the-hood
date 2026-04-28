# Module 09: Advanced Training Techniques — Training Bigger, Faster, Better

> **Goal:** Master the techniques that separate hobbyist training from production-grade training — the tricks used to train GPT, BERT, and modern SOTA models.

---

## Learning Objectives

By the end of this module, you will:
- **Implement** gradient accumulation to simulate large batches
- **Use** Automatic Mixed Precision (AMP) for 2-4× speed without accuracy loss
- **Apply** Exponential Moving Average (EMA) of weights for better generalization
- **Use** gradient checkpointing to train models larger than GPU memory
- **Design** learning rate schedules: warmup, cosine annealing, OneCycle
- **Apply** advanced regularization: DropPath, Mixup, Label Smoothing

---

## Part 1: Gradient Accumulation — Simulating Large Batches

### 1.1 Why Large Batches Matter

Large batch training:
- More stable gradient estimates (less noise)
- Better utilization of parallel hardware
- Often leads to better final models

Problem: a batch of 1024 samples might not fit in GPU memory.
Solution: accumulate gradients over multiple smaller steps before updating.

```python
import torch
import torch.nn as nn

# ── The key insight ────────────────────────────────────────────────────────────
# Running 4 steps of batch_size=64 with gradient accumulation
# = mathematically equivalent to 1 step of batch_size=256
# (assuming you scale the loss correctly)

def train_with_gradient_accumulation(
    model: nn.Module,
    train_loader,
    optimizer,
    criterion,
    accumulate_steps: int = 4,  # Effective batch = loader_batch * accumulate_steps
    device: str = 'cuda'
):
    """
    Gradient accumulation training loop.
    
    The trick: divide loss by accumulate_steps BEFORE calling backward().
    This ensures the accumulated gradient = gradient of the full virtual batch.
    
    Without scaling: accumulated grad = 4 * true_grad (too large!)
    With scaling:    accumulated grad = true_grad (correct!)
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)  # Start fresh
    
    for step, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        
        # Forward pass
        logits = model(x)
        
        # Scale loss by number of accumulation steps
        # This ensures gradient magnitude is consistent regardless of accumulate_steps
        loss = criterion(logits, y) / accumulate_steps
        
        # Backward (accumulate gradients — no optimizer.step() yet!)
        loss.backward()
        
        # Only update parameters every `accumulate_steps` steps
        if (step + 1) % accumulate_steps == 0:
            # Optional: gradient clipping (clip before step!)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # Reset for next accumulation
        
        if step % 100 == 0:
            print(f"Step {step}: loss={loss.item() * accumulate_steps:.4f}")
            # Multiply by accumulate_steps to see the un-scaled loss
```

---

## Part 2: Automatic Mixed Precision (AMP)

### 2.1 What Is Mixed Precision?

Modern GPUs have **Tensor Cores** that perform matrix multiplications in FP16 much faster than FP32:
- A100: 312 TFLOPS FP16 vs 77 TFLOPS FP32 → **4× faster matmuls!**
- Memory: FP16 uses 2 bytes vs FP32's 4 bytes → **2× more data in GPU cache**

Problem with pure FP16:
- Small gradients **underflow** to 0 (FP16 min normal: ~6×10⁻⁵)
- Model diverges

Solution: **Mixed precision** — compute in FP16, store weights/gradients in FP32.

```python
from torch.cuda.amp import GradScaler, autocast

def train_with_amp(model, train_loader, optimizer, criterion, device='cuda'):
    """
    AMP training with GradScaler.
    
    GradScaler solves the underflow problem:
    1. Multiply loss by large scale factor (e.g., 2^16)
    2. Backprop with scaled loss → gradients are also scaled up
    3. Unscale before optimizer step → gradients return to correct magnitude
    4. If gradients overflow (NaN/Inf): skip this step, reduce scale
    5. Periodically increase scale to maximize numerical range
    """
    model.train()
    
    # GradScaler manages dynamic loss scaling
    scaler = GradScaler(
        init_scale=2**16,      # Initial scale factor (start high)
        growth_factor=2.0,     # Double scale if no overflow for growth_interval steps
        backoff_factor=0.5,    # Halve scale if overflow detected
        growth_interval=2000,  # Steps between scale increases
    )
    
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        # ── autocast: automatically chooses FP16 or FP32 per operation ────────
        # Operations in FP16: matmul, conv2d (fast on Tensor Cores)
        # Operations in FP32: softmax, loss, batchnorm (numerical precision)
        with autocast(device_type='cuda', dtype=torch.float16):
            logits = model(x)      # Computed in FP16
            loss = criterion(logits, y)  # Computed in FP32
        
        # Scale the loss before backward (avoids FP16 underflow)
        scaler.scale(loss).backward()
        
        # Unscale gradients (restores to true magnitude)
        scaler.unscale_(optimizer)
        
        # Clip gradients AFTER unscaling (otherwise clips wrong magnitude)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Step optimizer (skips if gradients contain NaN/Inf)
        scaler.step(optimizer)
        
        # Update scale for next iteration
        scaler.update()
    
    return scaler.get_scale()  # Useful for monitoring


# BF16: Better format on Ampere+ GPUs (no scaler needed!)
# BF16 has same exponent range as FP32 → no overflow/underflow
# 8 exponent bits (like FP32) but only 7 mantissa bits
def train_with_bf16(model, train_loader, optimizer, criterion, device='cuda'):
    """
    BF16 training — cleaner than FP16 (no scaler needed).
    Requires: Ampere GPU (A100, A10, RTX 3090+) or newer.
    """
    model.train()
    
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits, y)
        
        # No scaler needed for BF16!
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

---

## Part 3: Exponential Moving Average (EMA)

### 3.1 What Is EMA and Why It Helps

EMA maintains a "smoothed" version of model weights:
```
ema_weights = decay * ema_weights + (1 - decay) * current_weights
```

Training with SGD/Adam is **noisy** — weights oscillate around the optimum. The EMA tracks the "average position" over many steps, landing in a flatter, more generalizable part of the loss landscape.

**Empirical result:** EMA weights almost always outperform raw weights by 0.1–1.0% accuracy.

```python
import copy
from torch import nn

class ExponentialMovingAverage:
    """
    Maintains an exponential moving average of model parameters.
    
    Typical decay values:
    - 0.999 for long training runs (slow accumulation)
    - 0.9999 for very long runs (even slower)
    - 0.99 for short training (faster adaptation)
    
    Higher decay = smoother/slower EMA = more conservative updates
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        
        # Create a deep copy of the model to store EMA weights
        # This copy is NOT used for training — only for evaluation
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()  # Always in eval mode
        
        # Disable gradient tracking for EMA model (never trained directly)
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update EMA weights after each training step.
        ema_param = decay * ema_param + (1-decay) * model_param
        """
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            # in-place update: more memory efficient
            ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
    
    def get_model(self) -> nn.Module:
        """Return the EMA model for evaluation/inference"""
        return self.ema_model


# Usage in training loop
model = nn.Linear(10, 5).cuda()
ema = ExponentialMovingAverage(model, decay=0.9999)
optimizer = torch.optim.AdamW(model.parameters())

for epoch in range(100):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        
        # Update EMA after EVERY optimizer step
        ema.update(model)
    
    # Evaluate using EMA model
    ema_model = ema.get_model()
    ema_model.eval()
    with torch.no_grad():
        val_acc = evaluate(ema_model, val_loader)
    print(f"Epoch {epoch}: EMA val acc = {val_acc:.4f}")
```

---

## Part 4: Gradient Checkpointing

### 4.1 The Memory Problem in Deep Networks

During backpropagation, PyTorch stores **all intermediate activations** from the forward pass. For a 1B parameter model, this can require hundreds of GB of GPU memory.

**Gradient checkpointing** trades compute for memory:
- Don't store intermediate activations
- During backward pass, recompute activations on-the-fly
- Memory: O(√N) instead of O(N)
- Speed: ~33% slower (one extra forward pass)

```python
import torch.utils.checkpoint as checkpoint

class MemoryEfficientTransformer(nn.Module):
    """
    Transformer with gradient checkpointing for large models.
    """
    
    def __init__(self, d_model=768, n_heads=12, n_layers=12):
        super().__init__()
        from module_08 import GPTDecoderBlock  # reuse from previous module
        self.blocks = nn.ModuleList([
            GPTDecoderBlock(d_model, n_heads, d_model*4)
            for _ in range(n_layers)
        ])
    
    def forward(self, x):
        for block in self.blocks:
            # checkpoint.checkpoint: recomputes activations during backward
            # instead of storing them → saves ~60-70% activation memory
            x = checkpoint.checkpoint(
                block,           # The module to checkpoint
                x,               # Input argument
                use_reentrant=False  # New API (more flexible)
            )
        return x

# Memory comparison (rough estimate for 12-layer transformer):
# Without checkpointing: stores activations for ALL 12 layers
# With checkpointing: stores activations for SQRT(12) ≈ 4 layers
# Memory savings: ~60-70%
# Speed cost: ~33% (one extra forward pass for recomputation)
```

---

## Part 5: Learning Rate Schedules

### 5.1 Warmup + Cosine Annealing

```python
import math

def get_cosine_with_warmup_schedule(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    """
    Linear warmup then cosine decay.
    Used by: BERT, GPT-2, most modern transformers.
    
    Phase 1 (warmup): LR linearly increases from 0 → peak
        Reason: AdamW needs several steps to build reliable gradient statistics.
        Starting with large LR can push weights to bad regions.
    
    Phase 2 (cosine): LR decreases from peak → min following cosine curve
        Reason: Slow annealing helps find flatter minima that generalize better.
    
    Formula:
    Warmup: lr = peak_lr * step / warmup_steps
    Cosine: lr = min_lr + 0.5*(peak_lr - min_lr) * (1 + cos(π * progress))
    """
    def lr_lambda(step):
        if step < num_warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Visualize the schedule
optimizer = torch.optim.AdamW([torch.zeros(1)], lr=1e-3)
scheduler = get_cosine_with_warmup_schedule(
    optimizer, num_warmup_steps=500, num_training_steps=10000
)

lrs = []
for step in range(10000):
    scheduler.step()
    lrs.append(optimizer.param_groups[0]['lr'])

# LR pattern:
# 0-500 steps:   0 → 1e-3 (warmup)
# 500-10000:     1e-3 → 1e-4 (cosine decay)
```

---

## Part 6: Advanced Regularization

### 6.1 Label Smoothing

```python
# Standard cross-entropy uses hard labels: [0, 0, 1, 0, 0]
# Model trained to output 100% confidence → overconfident, poor calibration
#
# Label smoothing: replace hard label with soft label
# [0, 0, 1, 0, 0] → [0.025, 0.025, 0.9, 0.025, 0.025]
# = 0.9 probability on true class, 0.025/4 on others
#
# Effect: prevents overconfidence, improves calibration, slight regularization

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# That's it! Built into PyTorch.

# Understanding what it does:
# Hard CE:   -log(p_true)
# Smooth CE: -[ε/K * Σlog(p_j)] + [(1-ε) * (-log(p_true))]
# ε = smoothing parameter (0.1 is standard)
# K = number of classes
```

### 6.2 DropPath (Stochastic Depth)

```python
class DropPath(nn.Module):
    """
    Stochastic Depth: randomly drop entire residual blocks during training.
    
    For each sample in the batch, with probability `drop_prob`, 
    skip the entire residual computation (just return the identity x).
    
    This is different from Dropout:
    - Dropout: drop individual neurons
    - DropPath: drop entire block computations
    
    Effect: acts like training an ensemble of shallower networks.
    Used in: DeiT, Swin Transformer, ConvNeXt.
    Typical drop_prob: 0.1 to 0.2, linearly increasing from 0 for deeper layers.
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x  # No dropping during eval
        
        # Random mask per sample in the batch
        keep_prob = 1.0 - self.drop_prob
        # Shape: (batch, 1, 1, 1) for 4D tensors — broadcast over spatial dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        
        # Bernoulli mask: 1 with prob keep_prob, 0 with prob drop_prob
        random_tensor = torch.rand(shape, device=x.device) < keep_prob
        
        # Scale retained samples to maintain expected value
        # E[output] = keep_prob * x/keep_prob + drop_prob * 0 = x
        return x * random_tensor.float() / keep_prob

# Apply with linearly increasing drop rate per layer (deeper = more dropout)
n_layers = 12
drop_probs = [i * 0.2 / (n_layers - 1) for i in range(n_layers)]
# Layer 0: 0.0, Layer 6: 0.1, Layer 11: 0.2
```

### 6.3 Complete Advanced Training Loop

```python
def advanced_training_loop(
    model: nn.Module,
    train_loader,
    val_loader,
    n_epochs: int = 100,
    device: str = 'cuda',
    base_lr: float = 3e-4,
    weight_decay: float = 0.05,
    gradient_accumulate: int = 4,
    label_smoothing: float = 0.1,
    ema_decay: float = 0.9999,
    clip_grad_norm: float = 1.0,
):
    model = model.to(device)
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # AdamW optimizer (separate weight decay from adaptive learning rate)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Calculate total steps for scheduler
    steps_per_epoch = len(train_loader) // gradient_accumulate
    total_steps = n_epochs * steps_per_epoch
    
    # Cosine schedule with warmup
    scheduler = get_cosine_with_warmup_schedule(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),  # 5% warmup
        num_training_steps=total_steps,
    )
    
    # AMP scaler
    scaler = GradScaler()
    
    # EMA
    ema = ExponentialMovingAverage(model, decay=ema_decay)
    
    best_val_acc = 0.0
    global_step = 0
    
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # AMP forward pass
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits, y) / gradient_accumulate
            
            # AMP backward
            scaler.scale(loss).backward()
            
            # Update every gradient_accumulate steps
            if (step + 1) % gradient_accumulate == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # Update EMA after each optimizer step
                ema.update(model)
                
                # Step scheduler
                scheduler.step()
                global_step += 1
            
            running_loss += loss.item() * gradient_accumulate
        
        # Validation with EMA model
        ema_model = ema.get_model()
        ema_model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = ema_model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += len(y)
        
        val_acc = correct / total
        current_lr = optimizer.param_groups[0]['lr']
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'ema': ema.ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_checkpoint.pt')
        
        print(f"Epoch {epoch+1:3d} | Loss: {running_loss/len(train_loader):.4f} "
              f"| Val Acc: {val_acc:.4f} | LR: {current_lr:.2e}")
    
    print(f"Best val acc: {best_val_acc:.4f}")
```

---

## Key Takeaways

| Technique | Benefit | Cost |
|-----------|---------|------|
| **Gradient Accumulation** | Simulate large batches | Slower training |
| **AMP (FP16/BF16)** | 2-4× speed, 2× memory | Slight complexity |
| **EMA weights** | +0.1–1.0% accuracy | Extra memory (2× model) |
| **Gradient Checkpointing** | 60% memory reduction | 33% slower |
| **Warmup + Cosine** | Better convergence | Need to tune warmup steps |
| **Label Smoothing** | Better calibration | Marginal compute |
| **DropPath** | Regularization | Slower convergence |

---

## Quiz

1. **Why must you divide the loss by `accumulate_steps` in gradient accumulation?**
   - Answer: Ensures accumulated gradients equal the gradient of the full virtual batch

2. **What does GradScaler do?**
   - Answer: Scales loss up before backward to prevent FP16 gradient underflow, then unscales before optimizer step

3. **Why is BF16 simpler than FP16 for training?**
   - Answer: BF16 has same exponent range as FP32 → no underflow, no GradScaler needed

4. **What is the EMA update rule?**
   - Answer: ema = decay * ema + (1-decay) * current_weights

5. **What is the memory-compute tradeoff in gradient checkpointing?**
   - Answer: Saves ~60% activation memory at the cost of ~33% extra compute (recomputes during backward)

6. **Why use warmup at the start of training?**
   - Answer: Lets Adam accumulate reliable gradient statistics before taking large steps; large early LR can push to bad regions

7. **What does label smoothing do to the target distribution?**
   - Answer: Distributes a small probability mass (ε) uniformly across all classes, reducing overconfidence

8. **What is DropPath and how does it differ from Dropout?**
   - Answer: DropPath randomly drops entire residual blocks; Dropout randomly zeros individual neurons

9. **What is the `unscale_` step in AMP training for?**
   - Answer: Restores gradients to correct magnitude before gradient clipping and optimizer step

10. **How do you implement gradient accumulation with DDP (multi-GPU)?**
    - Answer: Use `model.no_sync()` context manager during accumulation micro-steps to avoid all-reduce on every backward
