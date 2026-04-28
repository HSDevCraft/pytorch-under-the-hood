# Module 03: Neural Networks with nn.Module — Building Intelligence Layer by Layer

> **Goal:** Understand how neural networks are constructed in PyTorch — from individual neurons to deep architectures — and *why* each design choice matters.

---

## Learning Objectives

By the end of this module, you will:
- **Understand** what a neuron is mathematically and biologically
- **Build** neural networks using `nn.Module` the right way
- **Use** all major built-in layers: Linear, Conv2d, BatchNorm, Dropout, etc.
- **Choose** appropriate activation functions and understand their mathematical properties
- **Initialize** weights properly to prevent vanishing/exploding gradients
- **Compose** complex architectures using Sequential, ModuleList, ModuleDict
- **Inspect** and debug models: count parameters, visualize architecture

---

## Part 1: The Neuron — Where It All Begins

### 1.1 The Biological Inspiration

A biological neuron:
1. Receives signals from many inputs (dendrites)
2. Combines them (cell body)
3. Fires a signal if combined input exceeds a threshold (axon)

An artificial neuron mirrors this:
1. Takes multiple inputs x₁, x₂, ..., xₙ
2. Computes a weighted sum: z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
3. Applies an activation function: output = f(z)

```python
import torch
import torch.nn as nn

# A single artificial neuron (manually implemented)
def single_neuron(x, weights, bias):
    """
    x: input vector, shape (n_inputs,)
    weights: learned weights, shape (n_inputs,)
    bias: learned bias, scalar
    
    Returns scalar output after activation
    """
    # Step 1: weighted sum (linear combination)
    # This measures "how aligned" the input is with the learned weights
    z = torch.dot(weights, x) + bias  # z = Σ(wᵢxᵢ) + b
    
    # Step 2: activation function (introduces non-linearity)
    # Without this, the entire network would be a single linear function
    output = torch.relu(z)  # max(0, z)
    
    return output

# Example: a neuron that detects "high temperature" from features
# Features: [temperature, humidity, wind_speed]
x = torch.tensor([35.0, 60.0, 10.0])    # Hot, humid, calm day

# Learned weights (after training):
# High weight on temperature → temperature matters most
weights = torch.tensor([0.8, -0.1, 0.2])  
bias = torch.tensor(-20.0)  # Threshold offset

output = single_neuron(x, weights, bias)
print(f"Neuron output: {output.item():.4f}")
# z = 0.8*35 + (-0.1)*60 + 0.2*10 - 20 = 28 + (-6) + 2 - 20 = 4
# relu(4) = 4.0 (neuron fires)
```

### 1.2 From One Neuron to a Layer

A **layer** is a group of neurons that all receive the same input. Each neuron learns different features.

```
Input [x₁, x₂, x₃]
    │    │    │
    ├────┼────┤  ← Neuron 1 (w₁, b₁) → o₁
    ├────┼────┤  ← Neuron 2 (w₂, b₂) → o₂
    ├────┼────┤  ← Neuron 3 (w₃, b₃) → o₃
    └────┴────┘  ← Neuron 4 (w₄, b₄) → o₄
```

Mathematically, a layer computes: **output = activation(X @ W.T + b)**
- X: input matrix (batch_size × n_inputs)
- W: weight matrix (n_outputs × n_inputs)
- b: bias vector (n_outputs,)

```python
# Manual implementation of a fully connected layer
def fc_layer(x, weight, bias):
    """
    x: input, shape (batch_size, n_inputs)
    weight: shape (n_outputs, n_inputs)
    bias: shape (n_outputs,)
    
    Returns: shape (batch_size, n_outputs)
    """
    # Matrix multiplication applies all neurons at once
    # Each row of weight corresponds to one neuron's weights
    z = x @ weight.T + bias   # (batch, n_inputs) @ (n_inputs, n_outputs) → (batch, n_outputs)
    return torch.relu(z)

# Example: 5 samples, each with 3 features → 4 neurons
batch_size = 5
n_inputs = 3
n_outputs = 4

x = torch.randn(batch_size, n_inputs)
weight = torch.randn(n_outputs, n_inputs)
bias = torch.zeros(n_outputs)

output = fc_layer(x, weight, bias)
print(f"Input shape:  {x.shape}")    # (5, 3)
print(f"Output shape: {output.shape}")  # (5, 4)
```

---

## Part 2: nn.Module — PyTorch's Building Block

### 2.1 Why nn.Module Exists

`nn.Module` is the base class for ALL neural network components in PyTorch. It provides:
- **Automatic parameter tracking** (all `nn.Parameter` objects are registered)
- **`state_dict()` / `load_state_dict()`** for saving and loading
- **`.parameters()`** iterator for optimizers
- **`.train()` / `.eval()` modes** for Dropout, BatchNorm, etc.
- **Recursive nesting** (modules can contain other modules)

```python
import torch
import torch.nn as nn

# Every custom network MUST inherit from nn.Module
class SingleLayerNet(nn.Module):
    
    def __init__(self, n_inputs: int, n_outputs: int):
        """
        __init__ defines the STRUCTURE (what layers/params exist)
        Always call super().__init__() first — this sets up internal bookkeeping
        """
        super().__init__()  # NEVER forget this line!
        
        # nn.Linear(in_features, out_features) creates a layer:
        # weight: (out_features, in_features)
        # bias:   (out_features,)
        # It automatically registers these as parameters
        self.linear = nn.Linear(n_inputs, n_outputs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward() defines the COMPUTATION (how data flows through layers)
        This is called when you do: output = model(x)
        """
        return self.linear(x)  # Applies: x @ W.T + b

# Create the model
model = SingleLayerNet(n_inputs=10, n_outputs=5)

# Inspect it
print(model)
# SingleLayerNet(
#   (linear): Linear(in_features=10, out_features=5, bias=True)
# )

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")  # 10*5 + 5 = 55

# Forward pass
x = torch.randn(32, 10)  # Batch of 32 samples with 10 features each
output = model(x)
print(f"Output shape: {output.shape}")  # (32, 5)
```

### 2.2 Building a Multi-Layer Network (MLP)

```python
class MLP(nn.Module):
    """
    Multi-Layer Perceptron with:
    - Input layer
    - 2 hidden layers with ReLU activation
    - Output layer
    
    Architecture:
    Input(784) → Linear(256) → ReLU → Linear(128) → ReLU → Linear(10) → Output
    """
    
    def __init__(self, n_inputs: int, hidden_sizes: list, n_outputs: int):
        super().__init__()
        
        # Build layers programmatically
        self.hidden1 = nn.Linear(n_inputs, hidden_sizes[0])
        self.hidden2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.output = nn.Linear(hidden_sizes[1], n_outputs)
        
        # Activation function (shared, stateless — no parameters)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1: Linear + ReLU
        x = self.relu(self.hidden1(x))   # (batch, 784) → (batch, 256)
        
        # Layer 2: Linear + ReLU
        x = self.relu(self.hidden2(x))   # (batch, 256) → (batch, 128)
        
        # Output: Linear (no activation — the loss function handles it)
        x = self.output(x)                # (batch, 128) → (batch, 10)
        
        return x

# MNIST-scale model: 28×28 images (784 pixels) → 10 digit classes
model = MLP(n_inputs=784, hidden_sizes=[256, 128], n_outputs=10)
print(model)

# Count parameters per layer
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, {param.numel()} params")

# Test forward pass
x = torch.randn(64, 784)  # Batch of 64 images
logits = model(x)
print(f"Logits shape: {logits.shape}")  # (64, 10)
```

---

## Part 3: Activation Functions — Adding Non-Linearity

### 3.1 Why Activation Functions Are Critical

**Without activation functions**, stacking linear layers is mathematically equivalent to a single linear layer:

```
Linear₁(Linear₂(x)) = Ax + b  (still linear!)
```

Activation functions break this linearity, allowing networks to learn **any function** (Universal Approximation Theorem).

```python
# Demonstrate why non-linearity is essential
# If we stack two linear layers without activation:
W1 = torch.randn(4, 3)
W2 = torch.randn(5, 4)
# W2 @ (W1 @ x) = (W2 @ W1) @ x  → same as one layer W_combined
W_combined = W2 @ W1  # shape (5, 3) — equivalent single transformation!

# With activation, this collapsing doesn't happen:
# relu(W2 @ relu(W1 @ x)) ≠ W_combined @ x
# The relu makes each layer genuinely useful
```

### 3.2 The Major Activation Functions

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 100)  # Input range for visualization

# ── ReLU: Rectified Linear Unit ───────────────────────────────────────────────
# Formula: max(0, x)
# Gradient: 1 if x > 0, else 0
# 
# PROS: Fast to compute, no vanishing gradient for positive x
# CONS: "Dead neurons" — if x always < 0, gradient is always 0 (neuron never learns)
# USE WHEN: Hidden layers of most standard networks (CNNs, MLPs)
relu = torch.relu(x)

# ── Leaky ReLU: fixes dead neurons ────────────────────────────────────────────
# Formula: x if x > 0, else 0.01*x
# Gradient: 1 if x > 0, else 0.01 (never truly zero)
# USE WHEN: When dead neurons are a problem
leaky_relu = F.leaky_relu(x, negative_slope=0.01)

# ── ELU: Exponential Linear Unit ─────────────────────────────────────────────
# Formula: x if x > 0, else α(exp(x) - 1)
# Gradient: 1 if x > 0, else α*exp(x)
# PROS: Smooth, centered around zero → faster convergence
# USE WHEN: Deep networks where batch norm is not used
elu = F.elu(x, alpha=1.0)

# ── GELU: Gaussian Error Linear Unit ──────────────────────────────────────────
# Formula: x * Φ(x), where Φ is the cumulative normal distribution
# ≈ x * sigmoid(1.702 * x) (approximation)
# PROS: Smooth, probabilistic interpretation, empirically best for transformers
# USE WHEN: Transformers, BERT, GPT — all modern LLMs use GELU
gelu = F.gelu(x)

# ── Sigmoid ────────────────────────────────────────────────────────────────────
# Formula: 1 / (1 + exp(-x))
# Range: (0, 1) — perfect for binary probability output
# CONS: Saturates at extremes → vanishing gradients for deep nets
# USE WHEN: Binary classification output (not hidden layers)
sigmoid = torch.sigmoid(x)

# ── Tanh: Hyperbolic Tangent ──────────────────────────────────────────────────
# Formula: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
# Range: (-1, 1) — zero-centered (better than sigmoid)
# USE WHEN: RNN hidden states, when zero-centering matters
tanh = torch.tanh(x)

# ── Softmax: Multi-class probabilities ────────────────────────────────────────
# Formula: exp(xᵢ) / Σ exp(xⱼ)
# Output: probability distribution (sums to 1)
# USE WHEN: Final output for multi-class classification
# NOTE: In PyTorch, use nn.CrossEntropyLoss which has softmax built in!
logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=0)
print(f"Softmax output: {probs}")  # [0.659, 0.242, 0.099]  sums to 1
print(f"Sum: {probs.sum():.4f}")  # 1.0
```

### 3.3 Choosing the Right Activation

| Location | Recommended | Avoid |
|----------|-------------|-------|
| Hidden layers (MLP/CNN) | ReLU, GELU, ELU | Sigmoid, Tanh |
| Transformer hidden | GELU | Sigmoid, Tanh |
| RNN hidden states | Tanh, ReLU | Sigmoid |
| Binary output | Sigmoid | Softmax |
| Multi-class output | Softmax (via CrossEntropyLoss) | Sigmoid |
| Regression output | None | Sigmoid, Softmax |

---

## Part 4: Built-in Layers Deep Dive

### 4.1 Linear Layers

```python
# nn.Linear: applies y = xA^T + b
# - in_features: number of input features
# - out_features: number of output neurons
# - bias: whether to add bias (default True)

layer = nn.Linear(in_features=128, out_features=64, bias=True)
print(f"Weight shape: {layer.weight.shape}")  # (64, 128)
print(f"Bias shape:   {layer.bias.shape}")    # (64,)

x = torch.randn(32, 128)  # Batch of 32 samples
out = layer(x)
print(f"Output shape: {out.shape}")  # (32, 64)
```

### 4.2 Batch Normalization — Stabilizing Training

**Batch Normalization** normalizes the inputs of each layer to have zero mean and unit variance. This solves the **internal covariate shift** problem.

**Intuition:** Imagine training with very different input scales at each layer. BatchNorm ensures each layer always sees "nicely scaled" inputs, making training much more stable.

```python
# Without BatchNorm: activations can become very large or very small
# over time as weights change, slowing training or causing instability.
#
# With BatchNorm: for each mini-batch, normalize to μ=0, σ²=1,
# then apply learnable scale (γ) and shift (β)
#
# Formula: y = γ * (x - μ_batch) / √(σ²_batch + ε) + β
# During training: uses batch statistics
# During eval: uses running (exponential moving average) statistics

# 1D BatchNorm (for Linear layers)
bn1d = nn.BatchNorm1d(num_features=64)
x = torch.randn(32, 64)  # (batch=32, features=64)
out = bn1d(x)
print(f"BatchNorm1d output: {out.shape}")  # (32, 64)
print(f"Mean before: {x.mean():.4f}, after: {out.mean():.4f}")  # ≈ 0
print(f"Std before:  {x.std():.4f}, after: {out.std():.4f}")    # ≈ 1

# 2D BatchNorm (for Conv2d layers — most common)
bn2d = nn.BatchNorm2d(num_features=64)  # 64 channels
x = torch.randn(32, 64, 28, 28)  # (batch, channels, H, W)
out = bn2d(x)
print(f"BatchNorm2d output: {out.shape}")  # (32, 64, 28, 28)

# Important: set train/eval modes correctly!
bn2d.train()   # Uses batch statistics
bn2d.eval()    # Uses running statistics (for inference)
```

### 4.3 Dropout — Regularization through Random Deactivation

**Dropout** randomly sets neurons to zero during training. This prevents **co-adaptation** of neurons — neurons can't rely on specific other neurons, forcing them to learn more robust features.

```python
# Dropout(p=0.5): zeroes each element with probability p during training
# During eval mode: passes all elements (no dropout, but scales outputs)
# The scaling: during training, remaining neurons are divided by (1-p)
# so expected values are the same at test time (inverted dropout)

dropout = nn.Dropout(p=0.3)  # Drop 30% of neurons

x = torch.ones(5, 10)  # All ones for clarity

dropout.train()  # Training mode
out_train = dropout(x)
print(f"Train mode (some zeros): {out_train[0]}")

dropout.eval()   # Eval mode
out_eval = dropout(x)
print(f"Eval mode (all pass): {out_eval[0]}")  # All ones

# Common placement: after each hidden layer BEFORE activation, or after activation
class MLPWithRegularization(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),  # Normalize
            nn.ReLU(),            # Activate
            nn.Dropout(0.3),      # Regularize
            nn.Linear(256, 10),
        )
    
    def forward(self, x):
        return self.net(x)
```

### 4.4 Normalization Variants

```python
# LayerNorm: normalizes across features (not batch)
# Used in: Transformers, NLP models
# Formula: normalize over last `normalized_shape` dimensions
ln = nn.LayerNorm(normalized_shape=512)
x = torch.randn(32, 10, 512)  # (batch, seq_len, d_model)
out = ln(x)
print(f"LayerNorm output: {out.shape}")  # (32, 10, 512)

# GroupNorm: divides channels into groups, normalizes within each group
# Used when: batch size is small (e.g., detection, segmentation)
gn = nn.GroupNorm(num_groups=8, num_channels=64)
x = torch.randn(4, 64, 56, 56)  # Small batch
out = gn(x)
print(f"GroupNorm output: {out.shape}")  # (4, 64, 56, 56)

# When to use which:
# BatchNorm: large batch classification (CNNs) → best empirical results
# LayerNorm: transformers, sequences, small/variable batches
# GroupNorm: detection/segmentation with small batches
```

---

## Part 5: Weight Initialization

### 5.1 Why Initialization Matters

Poor initialization → vanishing or exploding activations at the start of training → extremely slow convergence or complete failure.

```python
# The problem: if weights are too large, activations explode
#              if weights are too small, activations vanish

def demonstrate_initialization_problem():
    n_layers = 50  # Deep network
    
    # Bad initialization: weights drawn from N(0, 1)
    x = torch.randn(100, 256)
    for i in range(n_layers):
        W = torch.randn(256, 256)  # Too large!
        x = torch.relu(x @ W)
    print(f"After {n_layers} layers (bad init): mean={x.mean():.4f}, std={x.std():.4f}")
    # → NaN or inf (exploded)
    
    # Good initialization: He/Kaiming (designed for ReLU)
    # Var(W) = 2/fan_in ensures variance ≈ 1 after ReLU
    x = torch.randn(100, 256)
    for i in range(n_layers):
        W = torch.randn(256, 256) * (2.0 / 256) ** 0.5  # He initialization
        x = torch.relu(x @ W)
    print(f"After {n_layers} layers (He init):  mean={x.mean():.4f}, std={x.std():.4f}")
    # → reasonable values!

demonstrate_initialization_problem()
```

### 5.2 Initialization Methods in PyTorch

```python
layer = nn.Linear(256, 128)

# ── Kaiming / He initialization (for ReLU activations) ──────────────────────
# Var(W) = 2/fan_in — accounts for ReLU zeroing half the activations
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.zeros_(layer.bias)  # Bias usually initialized to 0

# ── Xavier / Glorot initialization (for tanh/sigmoid activations) ─────────
# Var(W) = 2/(fan_in + fan_out) — keeps variance stable for symmetric activations
nn.init.xavier_normal_(layer.weight)

# ── Orthogonal initialization (for RNNs) ──────────────────────────────────
# Creates orthogonal matrices → preserves gradient norms
nn.init.orthogonal_(layer.weight, gain=1.0)

# ── Constant/Zero initialization ─────────────────────────────────────────
nn.init.zeros_(layer.bias)    # Bias → 0
nn.init.ones_(layer.weight)   # Weight → 1 (rarely used alone)
nn.init.constant_(layer.bias, val=0.01)

# Apply proper initialization to a full model:
def init_weights(module):
    """Apply to model via model.apply(init_weights)"""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.ones_(module.weight)   # Scale γ = 1
        nn.init.zeros_(module.bias)    # Shift β = 0

model = MLP(784, [256, 128], 10)
model.apply(init_weights)  # Recursively applies to all submodules
```

---

## Part 6: Composing Architectures

### 6.1 nn.Sequential — Linear Stacking

```python
# nn.Sequential: passes output of each module to the next
# Simple, clean, but limited (no branches, no skips)

model = nn.Sequential(
    nn.Linear(784, 256),    # Layer 1
    nn.BatchNorm1d(256),    # Normalize
    nn.ReLU(),              # Activate
    nn.Dropout(0.3),        # Regularize
    nn.Linear(256, 128),    # Layer 2
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 10),     # Output
)

x = torch.randn(32, 784)
logits = model(x)
print(f"Output: {logits.shape}")  # (32, 10)
```

### 6.2 nn.ModuleList — Dynamic Layer Containers

```python
# ModuleList: like a Python list, but properly registers modules as children
# USE WHEN: number of layers varies, or you need to index into layers

class VariableDepthMLP(nn.Module):
    def __init__(self, layer_sizes: list):
        super().__init__()
        # Build layers dynamically
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes) - 1)
        ])
        self.activation = nn.ReLU()
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation on last layer
                x = self.activation(x)
        return x

# Create networks of different depths
shallow = VariableDepthMLP([784, 128, 10])
deep    = VariableDepthMLP([784, 512, 256, 128, 64, 10])

x = torch.randn(32, 784)
print(f"Shallow output: {shallow(x).shape}")   # (32, 10)
print(f"Deep output:    {deep(x).shape}")      # (32, 10)
```

### 6.3 nn.ModuleDict — Named Layer Containers

```python
class MultiHeadNet(nn.Module):
    """Network with shared backbone + multiple output heads"""
    
    def __init__(self):
        super().__init__()
        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        # Multiple task-specific heads
        self.heads = nn.ModuleDict({
            'digit': nn.Linear(128, 10),       # Digit classification
            'even_odd': nn.Linear(128, 2),     # Even/odd prediction
            'image_quality': nn.Linear(128, 1) # Quality regression
        })
    
    def forward(self, x, task: str):
        features = self.backbone(x)
        return self.heads[task](features)

model = MultiHeadNet()
x = torch.randn(32, 784)
print(model(x, 'digit').shape)         # (32, 10)
print(model(x, 'even_odd').shape)      # (32, 2)
print(model(x, 'image_quality').shape) # (32, 1)
```

---

## Part 7: Residual Connections — How Very Deep Networks Train

### 7.1 The Problem: Degradation

Adding more layers to a network should improve it (more capacity). But in practice, very deep networks performed **worse** than shallower ones — not due to overfitting, but due to **optimization difficulty**.

**Key insight:** If a deeper network contains the shallower one as a sub-network (remaining layers = identity), it should perform at least as well. But optimizers struggle to learn identity mappings.

### 7.2 The Solution: Residual Block

```python
class ResidualBlock(nn.Module):
    """
    Instead of learning: output = F(x)
    Learn:              output = F(x) + x   ← residual connection
    
    If the optimal transform is the identity, 
    the network just needs to learn F(x) = 0 (much easier!)
    
    Architecture:
    x ──────────────────────────────────────────┐
    │                                            │ (skip connection)
    └──> Conv → BN → ReLU → Conv → BN ──────(+)──> ReLU ──> output
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x  # Save input for skip connection
        
        # Main path: two conv layers
        out = self.relu(self.bn1(self.conv1(x)))  # Conv1 → BN → ReLU
        out = self.bn2(self.conv2(out))            # Conv2 → BN (no ReLU yet!)
        
        # Skip connection: add identity to output
        out = out + identity  # The "residual" connection!
        out = self.relu(out)  # ReLU after addition
        
        return out

# Test residual block
block = ResidualBlock(channels=64)
x = torch.randn(32, 64, 28, 28)  # (batch, channels, H, W)
out = block(x)
print(f"Input shape:  {x.shape}")   # (32, 64, 28, 28)
print(f"Output shape: {out.shape}") # (32, 64, 28, 28) — same!
```

---

## Part 8: Inspecting and Debugging Models

### 8.1 Model Summary and Parameter Counting

```python
model = MLP(784, [256, 128], 10)

# Count total parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params:     {total:,}")
print(f"Trainable params: {trainable:,}")

# Per-layer breakdown
print("\nLayer-wise parameter count:")
for name, param in model.named_parameters():
    print(f"  {name}: shape={list(param.shape)}, params={param.numel():,}")
```

### 8.2 Saving and Loading Models

```python
model = MLP(784, [256, 128], 10)

# ── Save and load state_dict (RECOMMENDED) ──────────────────────────────────
# state_dict: OrderedDict mapping parameter names → tensors
torch.save(model.state_dict(), 'model_weights.pt')

# Load into same architecture
new_model = MLP(784, [256, 128], 10)
new_model.load_state_dict(torch.load('model_weights.pt'))
new_model.eval()  # Always set to eval before inference!

# ── Save entire model (less recommended — fragile across code changes) ────────
torch.save(model, 'full_model.pt')
loaded_model = torch.load('full_model.pt')
```

### 8.3 Train vs Eval Mode

```python
model = MLPWithRegularization()

# Train mode: Dropout active, BatchNorm uses batch statistics
model.train()
out_train = model(x)

# Eval mode: Dropout disabled, BatchNorm uses running statistics
model.eval()
with torch.no_grad():  # Also disable gradient tracking for inference
    out_eval = model(x)

# Note: always call model.train() before training loop
#       always call model.eval() before inference/validation
```

---

## Key Takeaways

| Concept | Why It Matters |
|---------|----------------|
| **nn.Module** | Base class providing parameter tracking, save/load, train/eval modes |
| **Activation Functions** | Without them, deep = shallow (stacked linear = one linear) |
| **BatchNorm** | Normalizes layer inputs → faster training, larger learning rates |
| **Dropout** | Prevents co-adaptation → reduces overfitting |
| **Weight Init** | Bad init → vanishing/exploding gradients → training failure |
| **Residual Connections** | Allows training of very deep networks (100+ layers) |
| **ModuleList/Dict** | Register sub-modules properly so parameters are tracked |

---

## Exercises

1. Build a 4-layer MLP for MNIST (784→512→256→128→10) with BatchNorm and Dropout.
2. Compare accuracy with/without BatchNorm after 5 epochs of training.
3. Implement a ResidualBlock and stack 4 of them in a small network.
4. Write `init_weights` for a network and verify the initial activation statistics are reasonable.
5. Build a multi-task network with shared backbone and three heads (classification, regression, binary).

---

## Quiz

1. **What does `super().__init__()` do in nn.Module?**
   - Answer: Initializes the parent nn.Module, enabling parameter tracking and other functionality

2. **Why is `nn.ModuleList` preferred over a Python list for storing layers?**
   - Answer: ModuleList registers sub-modules so their parameters appear in `model.parameters()`

3. **What happens to Dropout in eval mode?**
   - Answer: It's disabled—all neurons pass through unchanged

4. **What problem does Batch Normalization solve?**
   - Answer: Internal covariate shift—normalizes inputs so each layer always sees similar distributions

5. **What is the Kaiming initialization formula and for which activation is it designed?**
   - Answer: Var(W) = 2/fan_in; designed for ReLU (compensates for ReLU zeroing ~half its inputs)

6. **Why do residual connections help very deep networks?**
   - Answer: They make it easy to learn identity mappings; gradients flow directly through skip connections

7. **What does `model.eval()` change?**
   - Answer: Disables Dropout; BatchNorm switches from batch to running statistics

8. **What is the Universal Approximation Theorem?**
   - Answer: A network with one hidden layer of sufficient width and non-linear activations can approximate any continuous function

9. **Why should the output layer rarely have an activation function?**
   - Answer: Loss functions like CrossEntropyLoss include softmax internally; adding it separately causes double-application

10. **How do you save only model weights (not the entire model)?**
    - Answer: `torch.save(model.state_dict(), 'weights.pt')`
