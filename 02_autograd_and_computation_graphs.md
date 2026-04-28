# Module 02: Autograd & Computation Graphs — How Neural Networks Learn

> **Goal:** Understand automatic differentiation deeply. This is the *heart* of PyTorch—how it computes gradients automatically for backpropagation.

---

## Learning Objectives

By the end of this module, you will:
- **Understand** what automatic differentiation is and why it's revolutionary
- **Grasp** how PyTorch builds and traverses computational graphs
- **Implement** custom autograd functions for specialized operations
- **Control** gradient computation with `requires_grad`, `no_grad()`, and `detach()`
- **Debug** gradient-related issues (NaN, vanishing/exploding gradients)
- **Know** the difference between `.backward()` and `.grad`

---

## Part 1: The Problem We're Solving

### 1.1 Manual Gradient Computation (The Old Way)

Before automatic differentiation, computing gradients was manual and error-prone.

```python
# Suppose we have a simple function: loss = (y_pred - y_true)²
# We want to compute: d(loss)/d(weights)

# Manual approach (tedious and error-prone):
def forward(x, w):
    """Compute prediction: y = w * x"""
    return w * x

def loss_fn(y_pred, y_true):
    """Compute loss: (y_pred - y_true)²"""
    return (y_pred - y_true) ** 2

# Manual gradient computation
x = 2.0
y_true = 3.0
w = 1.0

# Forward pass
y_pred = forward(x, w)
loss = loss_fn(y_pred, y_true)

# Manual gradient computation (using chain rule)
# loss = (w*x - y_true)²
# d(loss)/d(w) = 2 * (w*x - y_true) * x
grad_w_manual = 2 * (y_pred - y_true) * x
print(f"Manual gradient: {grad_w_manual}")  # 2 * (2 - 3) * 2 = -4

# For complex networks with millions of parameters, this is impossible!
```

### 1.2 Automatic Differentiation (The PyTorch Way)

PyTorch computes gradients automatically using computational graphs.

```python
import torch

# Same computation, but with automatic differentiation
x = torch.tensor(2.0)
y_true = torch.tensor(3.0)
w = torch.tensor(1.0, requires_grad=True)  # Mark as requiring gradients

# Forward pass (PyTorch records operations)
y_pred = w * x
loss = (y_pred - y_true) ** 2

# Backward pass (automatic gradient computation)
loss.backward()

# Gradients are computed automatically!
print(f"Automatic gradient: {w.grad}")  # -4.0
print(f"Match manual? {abs(w.grad.item() - grad_w_manual) < 1e-6}")  # True
```

**Key insight:** PyTorch automatically computes gradients using the chain rule. We just call `.backward()` and it handles everything!

---

## Part 2: Computational Graphs

### 2.1 What Is a Computational Graph?

A **computational graph** is a directed acyclic graph (DAG) where:
- **Nodes** represent tensors or operations
- **Edges** represent data flow
- **Leaf nodes** are input tensors (created by user)
- **Intermediate nodes** are results of operations

```python
import torch

# Create a simple computational graph
x = torch.tensor(2.0, requires_grad=True)  # Leaf node
y = torch.tensor(3.0, requires_grad=True)  # Leaf node

# Operations create intermediate nodes
z = x + y  # Intermediate node (result of addition)
w = z * 2  # Intermediate node (result of multiplication)
loss = w.sum()  # Intermediate node (result of sum)

# Computational graph:
#     x ──┐
#         ├──> (+) ──> z ──┐
#     y ──┘                ├──> (*2) ──> w ──> sum ──> loss
#                          │
#                          └─────────────────────────┘

# Inspect the graph
print(f"x.requires_grad: {x.requires_grad}")  # True
print(f"z.requires_grad: {z.requires_grad}")  # True (inherited from x, y)
print(f"z.grad_fn: {z.grad_fn}")  # <AddBackward0> (operation that created z)
print(f"w.grad_fn: {w.grad_fn}")  # <MulBackward0>
print(f"loss.grad_fn: {loss.grad_fn}")  # <SumBackward0>
```

### 2.2 Leaf Nodes vs Intermediate Nodes

```python
x = torch.tensor(2.0, requires_grad=True)
y = x + 3  # Intermediate node

print(f"x.is_leaf: {x.is_leaf}")  # True (created by user)
print(f"y.is_leaf: {y.is_leaf}")  # False (result of operation)

# Only leaf nodes have .grad populated after backward()
# Intermediate nodes have .grad_fn but not .grad
print(f"x.grad_fn: {x.grad_fn}")  # None (leaf node)
print(f"y.grad_fn: {y.grad_fn}")  # <AddBackward0> (intermediate)
```

### 2.3 Building Graphs Dynamically

PyTorch builds graphs **dynamically** as you execute code. This is different from static graphs (TensorFlow 1.x).

```python
import torch

def dynamic_computation(x, use_branch_a):
    """Different computation based on input value"""
    if x.item() > 0:  # Dynamic control flow
        return x ** 2
    else:
        return x * 3

x = torch.tensor(2.0, requires_grad=True)
y = dynamic_computation(x, True)
y.backward()
print(f"Gradient (x > 0): {x.grad}")  # 2*x = 4

# Clear gradient
x.grad = None

x = torch.tensor(-2.0, requires_grad=True)
y = dynamic_computation(x, True)
y.backward()
print(f"Gradient (x < 0): {x.grad}")  # 3
```

**Intuition:** The graph is built as you run the code. Different inputs can create different graphs. This flexibility is powerful but requires careful handling.

---

## Part 3: The Backward Pass (Backpropagation)

### 3.1 Understanding Backward

```python
import torch

# Create a simple function: loss = (x - 2)²
x = torch.tensor(3.0, requires_grad=True)
y = (x - 2) ** 2

# Backward pass
y.backward()

# Gradient: d(loss)/d(x) = 2(x - 2) = 2(3 - 2) = 2
print(f"x.grad: {x.grad}")  # 2.0

# Intuition: "How much does loss change if we change x by a tiny amount?"
# Answer: by approximately 2 times that amount
```

### 3.2 The Chain Rule in Action

```python
# Composite function: loss = ((x + 1) * 2) ** 2
x = torch.tensor(1.0, requires_grad=True)

# Forward pass (PyTorch records each operation)
a = x + 1  # a = 2
b = a * 2  # b = 4
loss = b ** 2  # loss = 16

# Backward pass (applies chain rule)
loss.backward()

# Chain rule: d(loss)/d(x) = d(loss)/d(b) * d(b)/d(a) * d(a)/d(x)
#                          = 2*b * 2 * 1
#                          = 2*4 * 2 * 1
#                          = 16
print(f"x.grad: {x.grad}")  # 16.0

# Verify with numerical gradient
h = 1e-5
x_plus = 1.0 + h
a_plus = x_plus + 1
b_plus = a_plus * 2
loss_plus = b_plus ** 2

x_minus = 1.0 - h
a_minus = x_minus + 1
b_minus = a_minus * 2
loss_minus = b_minus ** 2

numerical_grad = (loss_plus - loss_minus) / (2 * h)
print(f"Numerical gradient: {numerical_grad:.4f}")  # ~16.0
```

### 3.3 Gradients for Multiple Outputs

```python
# When loss is a vector, we need to specify which scalar to backprop
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# y is a vector [1, 4, 9], not a scalar!
# We can't call y.backward() directly (which scalar to backprop?)

# Option 1: Sum the outputs
loss = y.sum()
loss.backward()
print(f"x.grad (sum): {x.grad}")  # [2, 4, 6] (d(sum(y))/d(x))

# Clear gradient
x.grad = None

# Option 2: Specify gradient weights
y.backward(torch.tensor([1.0, 2.0, 3.0]))  # Weight each output
print(f"x.grad (weighted): {x.grad}")  # [2, 8, 18]
```

---

## Part 4: Controlling Gradient Computation

### 4.1 `requires_grad` Control

```python
# Tensors created without requires_grad don't track operations
x = torch.tensor(2.0)  # requires_grad=False by default
y = x + 3
print(f"y.requires_grad: {y.requires_grad}")  # False
print(f"y.grad_fn: {y.grad_fn}")  # None (no graph built)

# Enable gradient tracking
x = torch.tensor(2.0, requires_grad=True)
y = x + 3
print(f"y.requires_grad: {y.requires_grad}")  # True
print(f"y.grad_fn: {y.grad_fn}")  # <AddBackward0>

# Toggle requires_grad
x.requires_grad_(False)  # Disable (in-place)
y = x + 3
print(f"y.requires_grad: {y.requires_grad}")  # False
```

### 4.2 `torch.no_grad()` Context

Use `torch.no_grad()` to disable gradient tracking for a block of code.

```python
x = torch.tensor(2.0, requires_grad=True)

# With gradient tracking
y = x ** 2
print(f"y.requires_grad: {y.requires_grad}")  # True

# Without gradient tracking
with torch.no_grad():
    y = x ** 2
    print(f"y.requires_grad: {y.requires_grad}")  # False

# Common use case: inference (no need to compute gradients)
model = torch.nn.Linear(10, 5)
x = torch.randn(32, 10)

with torch.no_grad():
    predictions = model(x)  # Faster, uses less memory
```

### 4.3 `detach()` Method

`detach()` creates a tensor that shares data but doesn't track operations.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

# Detach y (stops gradient flow)
y_detached = y.detach()

print(f"y.requires_grad: {y.requires_grad}")  # True
print(f"y_detached.requires_grad: {y_detached.requires_grad}")  # False

# Modifying detached tensor doesn't affect gradients
loss = y_detached.sum()
loss.backward()  # Error! y_detached has no grad_fn

# Common use case: use intermediate results without gradients
x = torch.randn(100, requires_grad=True)
x_normalized = (x - x.mean()) / x.std()  # Normalize
x_normalized = x_normalized.detach()  # Don't backprop through normalization
y = x_normalized ** 2
y.sum().backward()  # Only backprop through squaring
```

---

## Part 5: Custom Autograd Functions

### 5.1 When Do You Need Custom Functions?

Sometimes you need operations that aren't in PyTorch, or you want to optimize gradient computation.

```python
import torch

# Example: custom operation with special gradient
class CustomReLU(torch.autograd.Function):
    """Custom ReLU with modified gradient"""
    
    @staticmethod
    def forward(ctx, x):
        """Forward pass: return max(0, x)"""
        ctx.save_for_backward(x)
        return torch.clamp(x, min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: gradient is 1 if x > 0, else 0"""
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0
        return grad_input

# Use the custom function
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
y = CustomReLU.apply(x)
loss = y.sum()
loss.backward()

print(f"x.grad: {x.grad}")  # [0, 0, 0, 1, 1]
```

### 5.2 Gradient Checking

Always verify custom gradients with numerical gradients.

```python
def numerical_gradient(f, x, h=1e-5):
    """Compute numerical gradient using finite differences"""
    grad = torch.zeros_like(x)
    for i in range(x.numel()):
        x_plus = x.clone()
        x_plus.view(-1)[i] += h
        x_minus = x.clone()
        x_minus.view(-1)[i] -= h
        
        grad.view(-1)[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# Test custom function
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

# Analytical gradient
y = CustomReLU.apply(x)
loss = y.sum()
loss.backward()
analytical_grad = x.grad.clone()

# Numerical gradient
x.grad = None
x.requires_grad = True
numerical_grad = numerical_gradient(lambda t: CustomReLU.apply(t).sum(), x)

# Compare
print(f"Analytical: {analytical_grad}")
print(f"Numerical: {numerical_grad}")
print(f"Max difference: {(analytical_grad - numerical_grad).abs().max()}")
```

---

## Part 6: Common Gradient Issues

### 6.1 Accumulating Gradients

By default, gradients accumulate. You must zero them between batches.

```python
x = torch.tensor(2.0, requires_grad=True)

# First backward pass
y = x ** 2
y.backward()
print(f"After first backward: {x.grad}")  # 4.0

# Second backward pass (without zeroing)
y = x ** 2
y.backward()
print(f"After second backward: {x.grad}")  # 8.0 (accumulated!)

# Correct approach: zero gradients
x.grad = None  # or x.grad.zero_()
y = x ** 2
y.backward()
print(f"After zeroing: {x.grad}")  # 4.0
```

### 6.2 Vanishing and Exploding Gradients

```python
# Vanishing gradients: gradients become very small
x = torch.tensor(1.0, requires_grad=True)
y = x
for _ in range(100):
    y = y * 0.99  # Multiply by 0.99 repeatedly

y.backward()
print(f"Vanishing gradient: {x.grad}")  # Very small number

# Exploding gradients: gradients become very large
x = torch.tensor(1.0, requires_grad=True)
y = x
for _ in range(100):
    y = y * 1.01  # Multiply by 1.01 repeatedly

y.backward()
print(f"Exploding gradient: {x.grad}")  # Very large number

# Solution: gradient clipping
x = torch.tensor(1.0, requires_grad=True)
y = x
for _ in range(100):
    y = y * 1.01

y.backward()
x.grad.clamp_(-1, 1)  # Clip gradient to [-1, 1]
print(f"Clipped gradient: {x.grad}")
```

### 6.3 NaN Gradients

```python
# NaN can occur from invalid operations
x = torch.tensor(0.0, requires_grad=True)
y = torch.log(x)  # log(0) = -inf
z = y + 1  # -inf + 1 = -inf
z.backward()
print(f"Gradient: {x.grad}")  # nan

# Debug: check for NaN
if torch.isnan(x.grad).any():
    print("NaN detected in gradients!")
```

---

## Part 7: Practical Example: Training a Simple Model

```python
import torch
import torch.nn as nn

# Create a simple model
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training data
x = torch.randn(100, 10)
y = torch.randn(100, 1)

# Training loop
for epoch in range(5):
    # Forward pass
    predictions = model(x)
    loss = criterion(predictions, y)
    
    # Backward pass
    optimizer.zero_grad()  # Zero gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

---

## Key Takeaways

| Concept | Why It Matters |
|---------|----------------|
| **Computational Graph** | Records operations for automatic gradient computation |
| **Backward Pass** | Applies chain rule to compute gradients automatically |
| **requires_grad** | Controls which tensors need gradients |
| **no_grad()** | Disables gradient tracking for efficiency |
| **detach()** | Stops gradient flow while sharing data |
| **Custom Functions** | Implement specialized operations with custom gradients |
| **Gradient Accumulation** | Must zero gradients between batches |
| **Vanishing/Exploding** | Common issues requiring careful initialization and clipping |

---

## Exercises

1. Create a computational graph for `loss = ((x + 2) * 3) ** 2` and compute gradients manually and automatically.
2. Implement a custom autograd function for `f(x) = x³` and verify with numerical gradients.
3. Demonstrate gradient accumulation and show how to fix it.
4. Create a simple neural network and train it for one epoch, printing gradients at each step.
5. Show the difference between `detach()` and `no_grad()`.

---

## Quiz

1. **What is a computational graph?**
   - Answer: A directed acyclic graph recording all operations performed on tensors

2. **What does `.backward()` compute?**
   - Answer: Gradients using the chain rule, starting from the loss and working backward

3. **What is the difference between `requires_grad=True` and `requires_grad=False`?**
   - Answer: True builds a computational graph; False skips graph building

4. **Why must you call `optimizer.zero_grad()` in training loops?**
   - Answer: Gradients accumulate by default; zeroing prevents incorrect updates

5. **What does `detach()` do?**
   - Answer: Creates a tensor that shares data but stops gradient flow

6. **What is the chain rule?**
   - Answer: A method for computing derivatives of composite functions

7. **What causes vanishing gradients?**
   - Answer: Repeated multiplication by values < 1, making gradients exponentially small

8. **How do you disable gradient computation for a block of code?**
   - Answer: Use `with torch.no_grad():`

9. **What is a leaf node in a computational graph?**
   - Answer: A tensor created by the user (not from an operation)

10. **How do you verify a custom autograd function?**
    - Answer: Compare analytical gradients with numerical gradients using finite differences
