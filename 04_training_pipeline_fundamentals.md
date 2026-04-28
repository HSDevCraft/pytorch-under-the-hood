# Module 04: Training Pipeline Fundamentals — From Raw Data to a Trained Model

> **Goal:** Understand the complete training loop deeply — every component, every line of code, and *why* it exists. This is the engine that drives all of deep learning.

---

## Learning Objectives

By the end of this module, you will:
- **Design** custom `Dataset` and `DataLoader` classes for any data source
- **Understand** loss functions mathematically and know which to use when
- **Master** optimizers: SGD, Adam, AdamW — their differences and use cases
- **Use** learning rate schedulers to improve convergence
- **Write** a complete, production-grade training loop with validation
- **Debug** common training failures (loss not decreasing, NaN, overfitting)

---

## Part 1: Dataset and DataLoader

### 1.1 The Dataset Abstraction

PyTorch's `Dataset` is a simple interface: given an index, return a sample. This clean design works for any data source — files, databases, URLs, memory.

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# The three methods you MUST implement:
# __len__  → how many samples?
# __getitem__ → return the i-th sample
# __init__  → set up data loading

class TitanicDataset(Dataset):
    """
    Example: Titanic survival prediction dataset
    Features: [pclass, age, fare, sex_encoded, embarked_encoded]
    Label: survived (0 or 1)
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, transform=None):
        """
        Store the data. Transforms are applied lazily (in __getitem__).
        
        Args:
            features: np.ndarray of shape (N, n_features)
            labels:   np.ndarray of shape (N,)
            transform: optional callable applied to each sample
        """
        # Convert to tensors once during init (faster than in __getitem__)
        self.features = torch.FloatTensor(features)  # float32 for model input
        self.labels   = torch.LongTensor(labels)      # int64 for CrossEntropyLoss
        self.transform = transform
        
    def __len__(self) -> int:
        """Returns total number of samples"""
        return len(self.features)
    
    def __getitem__(self, idx: int):
        """
        Returns the i-th sample as (features, label) tuple.
        idx can be an int or a tensor (DataLoader passes tensor slices).
        """
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)  # Apply augmentation/preprocessing
        
        return x, y

# Create dummy data
np.random.seed(42)
n_samples = 1000
n_features = 5
features = np.random.randn(n_samples, n_features).astype(np.float32)
labels = np.random.randint(0, 2, n_samples)

# Create dataset
dataset = TitanicDataset(features, labels)
print(f"Dataset size: {len(dataset)}")  # 1000

# Access individual samples
x_sample, y_sample = dataset[0]
print(f"Sample feature shape: {x_sample.shape}")  # (5,)
print(f"Sample label: {y_sample}")                 # tensor(0) or tensor(1)
```

### 1.2 DataLoader — Efficient Batching

```python
# DataLoader handles: batching, shuffling, parallel loading, memory pinning
# These options can dramatically affect training speed

train_loader = DataLoader(
    dataset,
    
    batch_size=32,       # Samples per batch
                          # ↑ larger = more stable gradients, more memory
                          # ↓ smaller = noisier gradients, faster updates, less memory
    
    shuffle=True,         # Shuffle at each epoch
                          # ALWAYS True for training to prevent ordering bias
                          # ALWAYS False for validation/test (reproducibility)
    
    num_workers=4,        # Parallel data loading workers (CPU processes)
                          # 0 = main process (slow for heavy preprocessing)
                          # 4–8 = good for most cases
                          # Rule of thumb: num_workers = num_CPU_cores / 4
    
    pin_memory=True,      # Keep batches in pinned (page-locked) RAM
                          # Enables faster CPU→GPU transfers via DMA
                          # Set True when training on GPU
    
    drop_last=False,      # Drop last batch if smaller than batch_size
                          # Set True with BatchNorm (needs ≥2 samples per batch)
    
    prefetch_factor=2,    # Prefetch 2 batches per worker in advance
)

# Iterate over batches
for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
    print(f"Batch {batch_idx}: x={x_batch.shape}, y={y_batch.shape}")
    if batch_idx == 2:
        break
# Batch 0: x=torch.Size([32, 5]), y=torch.Size([32])
# Batch 1: x=torch.Size([32, 5]), y=torch.Size([32])
# Batch 2: x=torch.Size([32, 5]), y=torch.Size([32])
```

### 1.3 Train/Val/Test Splits

```python
from torch.utils.data import random_split

# Option 1: random_split (simple, built-in)
n_total = len(dataset)  # 1000
n_train = int(0.7 * n_total)   # 700
n_val   = int(0.15 * n_total)  # 150
n_test  = n_total - n_train - n_val  # 150

train_ds, val_ds, test_ds = random_split(
    dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)  # Reproducible split
)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=4)
```

---

## Part 2: Loss Functions — Measuring How Wrong We Are

### 2.1 The Role of a Loss Function

The loss function measures the **disagreement** between predictions and ground truth. It's the signal that tells gradients which direction to flow.

Requirements for a good loss function:
- **Differentiable** (almost everywhere) — so gradients can be computed
- **Lower = better prediction** — minimum at perfect predictions
- **Appropriate for the task** — classification vs regression use different losses

### 2.2 Cross-Entropy Loss for Classification

```python
import torch
import torch.nn as nn

# For MULTI-CLASS classification (most common)
# 
# Mathematical formula:
# CE = -Σ y_true_i * log(softmax(logit_i))
# For one-hot labels, this simplifies to:
# CE = -log(softmax(logit_correct_class))
#
# Intuition: penalize low confidence on the correct class.
# If model assigns 1.0 to correct class → loss = 0
# If model assigns 0.01 to correct class → loss = -log(0.01) ≈ 4.6 (high!)

criterion = nn.CrossEntropyLoss()

# IMPORTANT: CrossEntropyLoss expects RAW LOGITS (not softmax probabilities!)
# It computes softmax + log + negation internally for numerical stability

logits = torch.tensor([[2.0, 1.0, 0.1],   # Sample 1: confident class 0
                        [0.1, 2.0, 0.5]])   # Sample 2: confident class 1
labels = torch.tensor([0, 1])              # True classes

loss = criterion(logits, labels)
print(f"CE Loss: {loss.item():.4f}")  # Low loss (predictions match labels)

# Wrong predictions → higher loss
logits_wrong = torch.tensor([[0.1, 2.0, 1.0],   # Wrong: predicted class 1
                              [2.0, 0.1, 0.5]])   # Wrong: predicted class 0
loss_wrong = criterion(logits_wrong, labels)
print(f"CE Loss (wrong): {loss_wrong.item():.4f}")  # Higher than above

# Binary classification: use BCEWithLogitsLoss
# Mathematically: -[y*log(σ(x)) + (1-y)*log(1-σ(x))]
# σ = sigmoid function
bce_criterion = nn.BCEWithLogitsLoss()

binary_logits = torch.tensor([2.5, -1.0, 0.3])   # Raw outputs
binary_labels = torch.tensor([1.0, 0.0, 1.0])    # True labels (float!)
bce_loss = bce_criterion(binary_logits, binary_labels)
print(f"BCE Loss: {bce_loss.item():.4f}")
```

### 2.3 Regression Losses

```python
# MSE Loss: Mean Squared Error
# Formula: (1/N) Σ (y_pred - y_true)²
# Sensitive to outliers (error is squared)
mse_loss = nn.MSELoss()

predictions = torch.tensor([2.5, 0.0, 2.0, 8.0])
targets     = torch.tensor([3.0, -0.5, 2.0, 7.5])

loss_mse = mse_loss(predictions, targets)
print(f"MSE Loss: {loss_mse.item():.4f}")  # 0.1875

# MAE Loss: Mean Absolute Error
# Formula: (1/N) Σ |y_pred - y_true|
# Less sensitive to outliers than MSE
mae_loss = nn.L1Loss()
loss_mae = mae_loss(predictions, targets)
print(f"MAE Loss: {loss_mae.item():.4f}")  # 0.375

# Huber Loss: combines MSE (for small errors) and MAE (for large errors)
# Formula: 0.5*(y_pred-y_true)² for |error| < δ
#          δ*|y_pred-y_true| - 0.5*δ² for |error| ≥ δ
# Great when data has outliers but you want smooth gradients near zero
huber_loss = nn.HuberLoss(delta=1.0)
loss_huber = huber_loss(predictions, targets)
print(f"Huber Loss: {loss_huber.item():.4f}")

# Choosing the right loss:
# Binary classification → BCEWithLogitsLoss
# Multi-class classification → CrossEntropyLoss
# Regression (no outliers) → MSELoss
# Regression (with outliers) → HuberLoss or L1Loss
```

---

## Part 3: Optimizers — Updating Parameters

### 3.1 Gradient Descent: The Core Idea

```python
# The simplest possible optimizer: vanilla gradient descent
# θ = θ - learning_rate * gradient

# Manual implementation
def manual_gradient_descent(params, grads, lr=0.01):
    with torch.no_grad():  # Don't track these updates
        for param, grad in zip(params, grads):
            param -= lr * grad

# PyTorch optimizers automate this across all model parameters
```

### 3.2 SGD — Stochastic Gradient Descent

```python
model = nn.Linear(10, 5)

# Vanilla SGD
optimizer_sgd = torch.optim.SGD(
    model.parameters(),
    lr=0.01,       # Learning rate: step size in weight space
                    # Too large → diverge; too small → slow
    momentum=0.9,  # Accumulates velocity in consistent gradient direction
                    # Helps escape local minima, accelerates convergence
                    # Formula: v = momentum*v - lr*grad; θ = θ + v
    weight_decay=1e-4,  # L2 regularization: adds ‖W‖² to loss
                         # Prevents weights from growing too large
    nesterov=True,  # Nesterov momentum: "look ahead" before computing gradient
                     # Slightly better than standard momentum in practice
)
```

### 3.3 Adam — Adaptive Moment Estimation

```python
# Adam tracks TWO quantities for each parameter:
# m: 1st moment (exponential moving average of gradients) → direction
# v: 2nd moment (exponential moving average of squared gradients) → scale
#
# Update rule:
# m = β₁*m + (1-β₁)*grad        (gradient direction)
# v = β₂*v + (1-β₂)*grad²       (gradient magnitude)
# m̂ = m/(1-β₁^t)                (bias correction)
# v̂ = v/(1-β₂^t)                (bias correction)
# θ = θ - lr * m̂/(√v̂ + ε)      (adaptive step)
#
# Intuition: parameters with consistently large gradients get SMALLER steps
#            parameters with consistently small gradients get LARGER steps

optimizer_adam = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,    # Default: 1e-3 is a good starting point
    betas=(0.9, 0.999),  # β₁, β₂ — almost always leave at defaults
    eps=1e-8,   # ε — numerical stability, avoid division by zero
    weight_decay=0.0,  # L2 reg (note: in Adam this is slightly wrong theoretically)
)
```

### 3.4 AdamW — The Better Adam

```python
# Problem with Adam + weight_decay:
# Adam scales the weight decay by 1/√v̂, which makes the effective
# weight decay different for each parameter (not what we want!)
#
# AdamW fixes this by applying weight decay SEPARATELY from the gradient step:
# θ = θ * (1 - lr*weight_decay) - lr * m̂/(√v̂ + ε)
# This is the "correct" implementation of L2 regularization with Adam

optimizer_adamw = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,  # Standard value; now works correctly
)

# When to use which optimizer:
# SGD + momentum: CNNs with carefully tuned LR schedule
#                 Often achieves better final accuracy than Adam
# Adam: NLP/Transformers, quick prototyping, when you need to train fast
# AdamW: Transformers (BERT, GPT use this), any time you use Adam with weight_decay
```

---

## Part 4: Learning Rate Schedulers

### 4.1 Why Learning Rate Scheduling Matters

A **fixed learning rate** is almost never optimal:
- **Too large early** → unstable training, loss oscillates
- **Too small late** → stuck in plateau, slow convergence

Schedulers **anneal** (reduce) the learning rate over time.

```python
import torch.optim.lr_scheduler as lr_scheduler

model = nn.Linear(10, 5)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ── StepLR: reduce by factor every N epochs ────────────────────────────────
scheduler_step = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# LR: 1e-3 → 5e-4 (epoch 10) → 2.5e-4 (epoch 20) → ...

# ── CosineAnnealingLR: smooth cosine decay to minimum ──────────────────────
# Formula: lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π*t/T))
# Creates a smooth curve from lr_max to lr_min over T_max epochs
scheduler_cosine = lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)
# LR follows a cosine curve from 1e-3 down to 1e-6 over 100 epochs

# ── OneCycleLR: warmup + anneal in one cycle ───────────────────────────────
# 3 phases: warmup (linear up), cosine anneal (down), final cooldown
# Best for: fast training, often achieves great results in 1/3 the epochs
scheduler_onecycle = lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-2,          # Peak LR (often 10x the base LR)
    total_steps=1000,      # Total training steps (not epochs!)
    pct_start=0.3,         # Fraction of steps for warmup phase (30%)
    anneal_strategy='cos', # Cosine annealing
)

# ── ReduceLROnPlateau: reduce when metric stops improving ─────────────────
scheduler_plateau = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',       # Reduce when monitored metric STOPS decreasing
    factor=0.5,       # Multiply LR by 0.5 when triggered
    patience=5,       # Wait 5 epochs with no improvement before reducing
    min_lr=1e-6,      # Never go below this LR
)

# This one requires manual metric passing:
# scheduler_plateau.step(val_loss)
```

---

## Part 5: The Complete Training Loop

### 5.1 Every Line Explained

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 20,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Complete training loop with validation, timing, and best model saving.
    """
    
    # ── Setup ────────────────────────────────────────────────────────────────
    model = model.to(device)  # Move model weights to GPU (if available)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5
    )
    
    # Track metrics for plotting/analysis
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        
        # ── TRAINING PHASE ──────────────────────────────────────────────────
        model.train()  # Enable Dropout, use batch stats in BatchNorm
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            # Step 1: Move data to same device as model
            x_batch = x_batch.to(device, non_blocking=True)  # non_blocking for async GPU transfer
            y_batch = y_batch.to(device, non_blocking=True)
            
            # Step 2: Clear gradients from previous batch
            # MUST do this! Otherwise gradients accumulate
            optimizer.zero_grad(set_to_none=True)  
            # set_to_none=True is faster than zero_grad() — sets .grad to None
            # instead of filling with zeros (saves a memset operation)
            
            # Step 3: Forward pass — compute predictions
            logits = model(x_batch)  # Shape: (batch_size, n_classes)
            
            # Step 4: Compute loss
            loss = criterion(logits, y_batch)
            
            # Step 5: Backward pass — compute gradients
            loss.backward()  
            # Computes d(loss)/d(every_param) via chain rule
            # Gradients stored in param.grad for each parameter
            
            # Step 6: (Optional but recommended) Gradient clipping
            # Prevents exploding gradients in deep networks / RNNs
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step 7: Update parameters using computed gradients
            optimizer.step()
            # Each parameter: param = param - lr * param.grad (simplified)
            
            # Accumulate metrics
            train_loss += loss.item() * len(y_batch)  # loss.item() converts tensor to float
            preds = logits.argmax(dim=1)               # Predicted class (highest logit)
            train_correct += (preds == y_batch).sum().item()
            train_total += len(y_batch)
        
        # Average training metrics over all batches
        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Step 8: Update learning rate scheduler (once per epoch)
        scheduler.step()
        
        # ── VALIDATION PHASE ─────────────────────────────────────────────────
        model.eval()  # Disable Dropout, use running stats in BatchNorm
        
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # Disable gradient computation → faster + less memory
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                
                val_loss += loss.item() * len(y_batch)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)
        
        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        print(
            f"Epoch {epoch+1:3d}/{n_epochs} "
            f"| Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} "
            f"| Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f} "
            f"| LR: {current_lr:.6f} "
            f"| Time: {epoch_time:.1f}s"
        )
    
    return history
```

---

## Part 6: Diagnosing Training Problems

### 6.1 Loss Not Decreasing

```python
# Check 1: Can the model overfit ONE batch? (Sanity check)
def overfit_one_batch(model, x, y, n_steps=100):
    """
    If a model can't overfit 1 batch, something is fundamentally broken:
    - wrong loss function
    - learning rate too small
    - model has a bug
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for step in range(n_steps):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            acc = (logits.argmax(1) == y).float().mean()
            print(f"Step {step:3d}: loss={loss.item():.4f}, acc={acc.item():.4f}")

# If accuracy doesn't reach ~1.0 after 100 steps on 1 batch: bug!
```

### 6.2 Learning Rate Finding

```python
def lr_finder(model, train_loader, start_lr=1e-7, end_lr=10, n_steps=200):
    """
    LR Range Test: increases LR exponentially, plots loss vs LR.
    Choose LR just before loss starts rising steeply.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)
    criterion = nn.CrossEntropyLoss()
    
    lr_mult = (end_lr / start_lr) ** (1 / n_steps)
    
    lrs, losses = [], []
    
    for step, (x, y) in enumerate(train_loader):
        if step >= n_steps:
            break
        
        # Set current LR
        current_lr = start_lr * (lr_mult ** step)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr
        
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        
        lrs.append(current_lr)
        losses.append(loss.item())
    
    # Plot: good LR is where loss is steepest downward slope
    # (not the minimum — that LR is too high!)
    return lrs, losses
```

---

## Key Takeaways

| Component | Purpose | Key Choices |
|-----------|---------|-------------|
| **Dataset** | Encapsulates data access | Custom `__getitem__` for any data source |
| **DataLoader** | Batches + parallelizes | `shuffle=True` for train, `num_workers` for speed |
| **Loss Function** | Measures error | CE for classification, MSE/Huber for regression |
| **Optimizer** | Updates weights | AdamW for transformers, SGD for CNNs |
| **Scheduler** | Adapts learning rate | Cosine for most, ReduceLROnPlateau for unknowns |
| **Training Loop** | Ties it all together | Forward → loss → backward → step |

---

## Quiz

1. **What is the order of operations in one training step?**
   - Answer: zero_grad → forward → loss → backward → clip_grad_norm → step

2. **Why must `optimizer.zero_grad()` be called before `loss.backward()`?**
   - Answer: Gradients accumulate by default; not zeroing causes wrong updates

3. **What is the difference between Adam and AdamW?**
   - Answer: AdamW decouples weight decay from the gradient update; Adam's weight_decay is scaled by adaptive terms

4. **When should you set `shuffle=False` in DataLoader?**
   - Answer: Always for validation and test sets (consistency); never for training

5. **What does `loss.item()` do?**
   - Answer: Converts a scalar tensor to a Python float (detaches from computation graph)

6. **Why use `torch.no_grad()` during validation?**
   - Answer: Disables gradient tracking → saves memory + runs faster

7. **What is the "overfit one batch" test for?**
   - Answer: Sanity check — if model can't overfit 1 batch, there's a bug in the code/model

8. **What does CosineAnnealingLR do?**
   - Answer: Smoothly reduces LR from max to min following a cosine curve over T_max epochs

9. **What is gradient clipping and why is it used?**
   - Answer: Limits gradient norm to max_norm; prevents exploding gradients in RNNs/deep nets

10. **What does `non_blocking=True` in `.to(device)` do?**
    - Answer: Allows async CPU→GPU transfers; CPU continues while GPU loads data
