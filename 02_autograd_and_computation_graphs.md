# Module 02: Autograd & Computation Graphs

## Learning Objectives
By the end of this module you will be able to:
- Explain what a computation graph is and how PyTorch builds it
- Use `requires_grad`, `backward()`, and `.grad` to compute derivatives
- Apply the chain rule manually and verify it with autograd
- Use `torch.no_grad()` and `detach()` to control gradient flow
- Implement gradient clipping and understand gradient accumulation
- Build custom autograd functions with `torch.autograd.Function`
- Debug gradient problems: vanishing, exploding, and None gradients

---

## 2.1 Gradient Descent: The Mathematical Foundation

Neural network training minimises a loss function L(θ) over parameters θ. The update rule for gradient descent is:

```
θ ← θ − η · ∇_θ L
```

where η (eta) is the learning rate and ∇_θ L is the gradient of L with respect to θ.

**Chain Rule (the engine of backprop):**

For a composition L = f₃(f₂(f₁(x, θ))):

```
∂L/∂θ = (∂L/∂f₃)(∂f₃/∂f₂)(∂f₂/∂f₁)(∂f₁/∂θ)
```

This is computed by backpropagation: start from the loss, multiply Jacobians layer by layer, propagating gradients backward through the graph.

---

## 2.2 The Computation Graph

PyTorch builds a **dynamic directed acyclic graph (DAG)** as operations execute. Each node represents an operation (or tensor); edges carry the data flow.

```
x ──(requires_grad=True)──┐
                           ├──[mul]──→ z = x * w ──[add]──→ out = z + b ──[sum]──→ L
w ──(requires_grad=True)──┘                                               ↑
b ──(requires_grad=True)────────────────────────────────────────────────────
```

When you call `L.backward()`, PyTorch:
1. Traverses the graph in reverse (from L back to x, w, b)
2. Calls the stored backward function at each node
3. Accumulates `∂L/∂leaf` into each leaf tensor's `.grad`

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Forward pass — graph is built here
z   = x * w            # z = 6.0
out = z + b            # out = 7.0

# Backward pass — gradients computed
out.backward()

print(x.grad)  # ∂out/∂x = w = 3.0
print(w.grad)  # ∂out/∂w = x = 2.0
print(b.grad)  # ∂out/∂b = 1.0

# Inspect the graph
print(out.grad_fn)  # <AddBackward0>
print(z.grad_fn)    # <MulBackward0>
print(x.grad_fn)    # None — leaf tensor, no history
```

### Leaf vs Non-Leaf Tensors

- **Leaf:** created by the user (not from an operation); `.requires_grad` can be set
- **Non-leaf:** result of an operation on tensors with `requires_grad=True`; `.grad` is *not* retained by default (unless `.retain_grad()` is called)

```python
a = torch.randn(3, requires_grad=True)   # leaf
b = a * 2                                # non-leaf
c = b.sum()

c.backward()
print(a.grad)  # accumulated: [2, 2, 2]
print(b.grad)  # None (non-leaf, not retained)

# Retain grad for intermediate nodes (debugging/research)
b.retain_grad()
c.backward()   # must call again after retain_grad
print(b.grad)  # [1, 1, 1]
```

---

## 2.3 The `backward()` Call

```python
import torch

# ── Scalar loss: simple .backward() ─────────────────────────────────────────
x = torch.randn(3, requires_grad=True)
L = (x ** 2).sum()   # L = x₀² + x₁² + x₂²
L.backward()
print(x.grad)        # [2*x₀, 2*x₁, 2*x₂]

# ── Non-scalar output: requires gradient argument ─────────────────────────────
# backward(gradient) computes v^T * J (vector-Jacobian product, VJP)
x = torch.randn(3, requires_grad=True)
y = x ** 2           # y is (3,) — non-scalar!
v = torch.ones(3)    # the "incoming gradient" (same shape as y)
y.backward(v)        # computes v^T @ J = [2*x₀, 2*x₁, 2*x₂]
print(x.grad)

# ── Retain graph for multiple backward passes ────────────────────────────────
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3           # y = 8.0

y.backward(retain_graph=True)   # first backward
print(x.grad)  # 3 * 2² = 12

x.grad.zero_()                   # MUST zero grad before next backward
y.backward()                     # second backward (graph consumed now)
print(x.grad)  # 12 again

# ── Gradient accumulation ────────────────────────────────────────────────────
# PyTorch ACCUMULATES gradients by default (adds to .grad)
# You MUST zero gradients between training steps
x = torch.tensor(1.0, requires_grad=True)
for _ in range(3):
    loss = x ** 2
    loss.backward()
print(x.grad)  # 6.0 (accumulated: 2+2+2), not 2.0!

# Correct pattern: zero before each backward
x = torch.tensor(1.0, requires_grad=True)
for _ in range(3):
    if x.grad is not None:
        x.grad.zero_()
    loss = x ** 2
    loss.backward()
    print(x.grad)   # always 2.0
```

---

## 2.4 Controlling Gradient Flow

```python
# ── torch.no_grad(): disables gradient tracking ────────────────────────────
model = torch.nn.Linear(10, 5)
x = torch.randn(2, 10)

with torch.no_grad():
    out = model(x)    # no graph built, faster and memory-efficient
    # use this for: inference, evaluation, weight updates in custom training loops

print(out.requires_grad)  # False

# ── @torch.no_grad() decorator ───────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, x):
    return model(x)

# ── torch.inference_mode(): even faster (ops can't escape the context) ───────
with torch.inference_mode():
    out = model(x)    # preferred for inference over no_grad

# ── detach(): cuts a tensor from the graph ─────────────────────────────────
x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.detach()        # z is a new tensor sharing y's data but with no grad_fn
print(z.requires_grad)  # False

# Use case: target/discriminator in GANs, stop-gradient, teacher models
fake_detached = generator(z_noise).detach()  # don't backprop into generator
disc_loss = discriminator(fake_detached)

# ── requires_grad_(): toggle in-place ─────────────────────────────────────
model.eval()
for param in model.parameters():
    param.requires_grad_(False)   # freeze all parameters
```

---

## 2.5 Computing Higher-Order Gradients

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3    # y = x³

# First derivative
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
print(dy_dx)  # 3 * 2² = 12.0

# Second derivative (d²y/dx²)
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(d2y_dx2)  # 6 * 2 = 12.0 — wait, d²/dx²(x³) = 6x = 12

# Jacobian computation
def f(x):
    return torch.stack([x[0]**2 + x[1], x[0]*x[1]])

x = torch.randn(2, requires_grad=True)
J = torch.autograd.functional.jacobian(f, x)
print(J.shape)   # (2, 2) — the full Jacobian matrix

# Hessian (second-order: useful for meta-learning, curvature analysis)
def scalar_loss(x):
    return (x ** 2).sum()

H = torch.autograd.functional.hessian(scalar_loss, x)
print(H)  # should be 2*I (identity scaled by 2)
```

---

## 2.6 Custom Autograd Functions

When you need an operation with a non-standard derivative (e.g., a custom CUDA kernel, a straight-through estimator for discrete variables), subclass `torch.autograd.Function`:

```python
import torch
from torch.autograd import Function

class StraightThroughEstimator(Function):
    """
    Forward: hard quantize to {-1, +1}
    Backward: pass gradient through unchanged (straight-through)
    This is used in binary neural networks.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.sign()                    # {-1, 0, +1}

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output                 # identity gradient

# Usage
quantize = StraightThroughEstimator.apply

x = torch.randn(4, requires_grad=True)
y = quantize(x)
loss = y.sum()
loss.backward()
print(x.grad)   # [1, 1, 1, 1] — straight-through!

# ── More realistic: custom sigmoid with saved tensors ───────────────────────
class MySigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        s = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(s)          # save output for backward
        return s

    @staticmethod
    def backward(ctx, grad_output):
        (s,) = ctx.saved_tensors
        return grad_output * s * (1 - s)  # sigmoid derivative

# Verify correctness against torch.sigmoid
x = torch.randn(4, requires_grad=True)
y_custom  = MySigmoid.apply(x)
y_torch   = torch.sigmoid(x)
assert torch.allclose(y_custom, y_torch)

# Check gradients with finite differences
from torch.autograd import gradcheck
x_double = torch.randn(4, dtype=torch.float64, requires_grad=True)
assert gradcheck(MySigmoid.apply, x_double, eps=1e-6, atol=1e-4)
print("Gradient check passed!")
```

---

## 2.7 Gradient Diagnostics: Hooks

```python
import torch
import torch.nn as nn
from typing import Dict, List

# ── Register backward hooks for gradient inspection ──────────────────────────
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)

grad_log: Dict[str, List] = {}

def make_hook(name):
    def hook(grad):
        norm = grad.norm().item()
        grad_log.setdefault(name, []).append(norm)
        return grad  # optionally modify the gradient here
    return hook

for name, param in model.named_parameters():
    param.register_hook(make_hook(name))

# Run a forward/backward pass
x = torch.randn(16, 4)
loss = model(x).sum()
loss.backward()

print("\nGradient norms per parameter:")
for name, norms in grad_log.items():
    print(f"  {name}: {norms[-1]:.4f}")
```

---

## 2.8 Gradient Clipping

Large gradients ("exploding gradients") can destabilize training, especially in RNNs and deep networks.

```python
import torch
import torch.nn as nn

model   = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
optim   = torch.optim.Adam(model.parameters(), lr=1e-3)

x = torch.randn(32, 20, 64)
out, _ = model(x)
loss = out.sum()

optim.zero_grad()
loss.backward()

# Clip before step — prevents exploding gradient
# This rescales the entire gradient vector to have norm ≤ max_norm
total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"Gradient norm before clipping: {total_norm:.4f}")

optim.step()

# Clip by value (alternative — clips each gradient element independently)
nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

---

## 2.9 Understanding Vanishing & Exploding Gradients

```python
import torch
import matplotlib.pyplot as plt

# Simulate gradient flow through n layers with weight W and activation f
def simulate_gradient_flow(n_layers=20, weight_scale=0.9, activation="tanh"):
    grad = torch.tensor(1.0)
    grad_history = [grad.item()]

    for _ in range(n_layers):
        if activation == "tanh":
            # derivative of tanh(x) ≈ (1 - tanh²(x)) ≤ 1
            # with small weight: gradient shrinks each layer
            grad = grad * weight_scale * 0.5   # tanh deriv ≈ 0.5 at x=0
        elif activation == "relu":
            grad = grad * weight_scale * (1.0 if torch.rand(1) > 0.1 else 0.0)
        grad_history.append(grad.item())

    return grad_history

vanishing = simulate_gradient_flow(weight_scale=0.5)
exploding = simulate_gradient_flow(weight_scale=1.5)

print("Final gradient (vanishing):", vanishing[-1])   # near 0
print("Final gradient (exploding):", exploding[-1])   # enormous

# Solutions:
# 1. Residual connections (skip connections)
# 2. Gradient clipping
# 3. Better weight initialisation (Kaiming/He for ReLU, Xavier/Glorot for sigmoid/tanh)
# 4. Batch/Layer normalisation
# 5. LSTM/GRU gating (for RNNs)
```

---

## 2.10 Real-World Example: Manual Training Step

```python
import torch
import torch.nn as nn
from typing import Tuple

def manual_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    max_grad_norm: float = 1.0,
    grad_accumulate_steps: int = 1,
    step: int = 0,
) -> dict:
    """
    A single training step. Returns a dict of metrics.
    Supports gradient accumulation for large-batch training.
    """
    x, y = batch

    # ── Forward ─────────────────────────────────────────────────────────────
    logits = model(x)
    loss   = criterion(logits, y)
    loss   = loss / grad_accumulate_steps   # scale for accumulation

    # ── Backward ────────────────────────────────────────────────────────────
    loss.backward()   # gradients ACCUMULATE in .grad buffers

    # ── Optimizer step (only every N micro-steps) ───────────────────────────
    if (step + 1) % grad_accumulate_steps == 0:
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()   # IMPORTANT: zero after step
    else:
        grad_norm = 0.0

    return {
        "loss": loss.item() * grad_accumulate_steps,   # unscaled for logging
        "grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
    }
```

---

## 2.11 Best Practices

| Issue | Symptom | Fix |
|-------|---------|-----|
| Gradient accumulation bug | loss diverges | Always `zero_grad()` before or after `optimizer.step()` |
| Memory leak | GPU memory grows every step | Don't store tensors with grad_fn in lists; use `.item()` or `.detach()` |
| Vanishing gradient | early layers don't learn | Residual connections, good init, normalisation |
| Exploding gradient | loss becomes NaN | Gradient clipping, lower learning rate |
| Non-leaf .grad is None | can't debug intermediates | Call `.retain_grad()` before backward |
| Second backward fails | `RuntimeError: graph freed` | `retain_graph=True` on first backward |
| Slow inference | | Use `torch.no_grad()` or `torch.inference_mode()` |

---

## Exercises

**Exercise 2.1** Manually compute `∂L/∂w` and `∂L/∂b` for `L = ((w*x + b) - y)² / 2` where `x=3, y=5, w=1, b=0`. Then verify using PyTorch autograd.

**Exercise 2.2** Implement a custom autograd function `ClampGrad` that clamps the gradient to [-1, 1] in the backward pass while letting the forward pass be identity.

**Exercise 2.3** Build a simple 3-layer MLP, run a forward+backward pass, and use gradient hooks to log the L2 norm of gradients for all parameters. Identify which layer has the smallest gradient norm.

**Exercise 2.4** Demonstrate gradient accumulation: achieve equivalent gradient updates using 4 micro-steps of batch-size 8 vs a single step of batch-size 32.

<details>
<summary>Solutions</summary>

```python
# 2.1
import torch

x_val, y_val = 3.0, 5.0
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
x = torch.tensor(x_val)
y = torch.tensor(y_val)

pred = w * x + b
L    = 0.5 * (pred - y) ** 2
L.backward()

print(f"w.grad = {w.grad:.4f}")  # (pred-y)*x = (3-5)*3 = -6
print(f"b.grad = {b.grad:.4f}")  # (pred-y)*1 = -2

# Manual:
pred_val = w.item() * x_val + b.item()
print(f"Manual ∂L/∂w = {(pred_val - y_val) * x_val:.4f}")  # -6
print(f"Manual ∂L/∂b = {(pred_val - y_val):.4f}")          # -2

# 2.2
from torch.autograd import Function

class ClampGrad(Function):
    @staticmethod
    def forward(ctx, x):
        return x  # identity

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp(-1, 1)

# Verify
x = torch.randn(4, requires_grad=True) * 10  # large values
y = ClampGrad.apply(x)
y.sum().backward()
print(x.grad)  # all values in [-1, 1]

# 2.4 — gradient accumulation equivalence
torch.manual_seed(0)
model_single = torch.nn.Linear(4, 2)
model_accum  = torch.nn.Linear(4, 2)

# Copy weights
model_accum.load_state_dict(model_single.state_dict())

data = torch.randn(32, 4)
target = torch.randint(0, 2, (32,))
criterion = torch.nn.CrossEntropyLoss()

# Single step, batch=32
out = model_single(data)
loss = criterion(out, target)
loss.backward()
single_grads = {n: p.grad.clone() for n, p in model_single.named_parameters()}

# Accumulated, 4 steps of batch=8
for i in range(4):
    batch_x = data[i*8:(i+1)*8]
    batch_y = target[i*8:(i+1)*8]
    out  = model_accum(batch_x)
    loss = criterion(out, batch_y) / 4  # scale by accumulation steps
    loss.backward()

# Gradients should match
for name, param in model_accum.named_parameters():
    assert torch.allclose(param.grad, single_grads[name], atol=1e-5), f"Mismatch at {name}"
print("Gradient accumulation matches single-batch gradient!")
```

</details>

---

## Module Summary

| Concept | Key Points |
|---------|-----------|
| Computation graph | Dynamic DAG; built on forward, consumed on backward |
| `requires_grad` | Set on leaf parameters; propagates to outputs |
| `backward()` | Traverses graph; accumulates `∂L/∂leaf` into `.grad` |
| Gradient accumulation | `.grad` adds up — always zero before stepping |
| `no_grad / inference_mode` | Disables graph; mandatory for inference |
| `detach()` | Cuts tensor from graph; shares data |
| Custom Function | Override `forward`/`backward` via `autograd.Function` |
| Clipping | `clip_grad_norm_` prevents exploding gradients |

---

## Quiz

1. What does `retain_graph=True` do and when do you need it?
2. Why must you call `optimizer.zero_grad()` before each backward pass?
3. What is the difference between `detach()` and `torch.no_grad()`?
4. What is a VJP (vector-Jacobian product) and why does `backward()` compute it?
5. What happens if you call `.backward()` on a non-scalar without providing a gradient argument?
6. How does `gradcheck` verify a custom autograd function?
7. When would you use `retain_grad()` on a non-leaf tensor?

---

*Next: [Module 03 — Neural Networks with nn.Module](./03_neural_networks_with_nn_module.md)*
