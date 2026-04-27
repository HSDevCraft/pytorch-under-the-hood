# Module 03: Neural Networks with nn.Module

## Learning Objectives
By the end of this module you will be able to:
- Build any neural network architecture using `nn.Module`
- Use all major built-in layers: Linear, Conv, Normalization, Dropout, Embedding
- Apply activation functions correctly and understand their mathematics
- Initialise weights using the right strategy for each activation
- Inspect and debug network structure with `torchinfo`
- Implement custom layers and parameterised modules
- Use `nn.Sequential`, `nn.ModuleList`, and `nn.ModuleDict` effectively

---

## 3.1 The Neuron and the Layer

A single artificial neuron computes:

```
y = f(w · x + b)
```

where **w** ∈ ℝⁿ is a weight vector, b ∈ ℝ is a bias scalar, and f is an activation function.

A **fully connected layer** (linear layer) applies this to all neurons simultaneously:

```
Y = f(XW^T + b)
```

where **X** ∈ ℝ^(B×n) (batch × input), **W** ∈ ℝ^(m×n) (output × input), **b** ∈ ℝᵐ.

---

## 3.2 nn.Module: The Fundamental Building Block

Every model, layer, and component in PyTorch inherits from `nn.Module`.

**Three responsibilities of nn.Module:**
1. Hold parameters (`nn.Parameter`) and sub-modules
2. Define `forward()` — the computation to perform
3. Provide utility methods: `.parameters()`, `.state_dict()`, `.to()`, `.train()`/`.eval()`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleNeuron(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        # nn.Parameter: a tensor that is registered as a parameter
        # automatically included in .parameters() and .state_dict()
        self.weight = nn.Parameter(torch.randn(in_features))
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x @ self.weight + self.bias)

neuron = SingleNeuron(in_features=4)
x = torch.randn(8, 4)            # batch of 8 samples
out = neuron(x)                  # calls neuron.forward(x)
print(out.shape)                  # (8,) — one output per sample

# Inspect parameters
for name, param in neuron.named_parameters():
    print(name, param.shape)
# weight  torch.Size([4])
# bias    torch.Size([1])
```

---

## 3.3 Built-In Layers

### Linear (Fully Connected) Layer

```python
# nn.Linear(in_features, out_features, bias=True)
# weight: (out, in), bias: (out,)
# Kaiming-uniform init by default

linear = nn.Linear(128, 64)
x = torch.randn(32, 128)
out = linear(x)                  # (32, 64)

print(linear.weight.shape)       # (64, 128)
print(linear.bias.shape)         # (64,)
```

### Convolutional Layers

```python
# 2D Conv (images): (N, C_in, H, W) → (N, C_out, H_out, W_out)
conv = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,       # 'same' padding to preserve spatial size
    bias=False,      # often False when followed by BatchNorm
)

# Output size formula:
# H_out = (H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1

x = torch.randn(8, 3, 224, 224)
out = conv(x)                    # (8, 64, 224, 224)

# Depthwise-separable convolution (MobileNet style)
depthwise   = nn.Conv2d(64, 64, 3, padding=1, groups=64)   # per-channel
pointwise   = nn.Conv2d(64, 128, 1)                        # 1×1 mixing
```

### Normalisation Layers

```python
# Batch Normalisation: normalise over (N, H, W), one stat per channel
# Best for CNNs with large batch sizes
bn = nn.BatchNorm2d(num_features=64)

# Layer Normalisation: normalise over the last D dimensions, per sample
# Best for transformers and RNNs
ln = nn.LayerNorm(normalized_shape=512)

# Instance Normalisation: normalise per (N, C); style transfer
inst = nn.InstanceNorm2d(num_features=64)

# Group Normalisation: normalise within groups of channels; small batches
gn = nn.GroupNorm(num_groups=8, num_channels=64)

# Forward behaviour:
# BatchNorm uses RUNNING stats during .eval()
# LayerNorm, InstanceNorm, GroupNorm: no difference between train/eval
x_img = torch.randn(4, 64, 28, 28)
print(bn(x_img).shape)       # (4, 64, 28, 28)

x_seq = torch.randn(4, 128, 512)
print(ln(x_seq).shape)       # (4, 128, 512)
```

### Dropout & Regularisation

```python
dropout   = nn.Dropout(p=0.5)    # zero random elements w/ prob p
dropout2d = nn.Dropout2d(p=0.2)  # zero random CHANNELS (for CNNs)

# IMPORTANT: behaves differently in train vs eval mode
model.train()   # dropout active
model.eval()    # dropout disabled (pass-through)
```

### Pooling

```python
# Max pooling — takes the max in each window
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)  # halves H and W

# Average pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# Adaptive: specify OUTPUT size, not window size
gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))    # Global Average Pooling
x = torch.randn(8, 512, 7, 7)
print(gap(x).shape)           # (8, 512, 1, 1)
print(gap(x).flatten(1).shape) # (8, 512) — common head pattern
```

### Embedding Layer

```python
# Lookup table: integer indices → dense vectors
# Used for word/token embeddings
emb = nn.Embedding(num_embeddings=10000, embedding_dim=256)

token_ids = torch.randint(0, 10000, (8, 50))  # (batch, seq_len)
vectors = emb(token_ids)                       # (8, 50, 256)

# Pretrained initialisation
emb.weight.data.copy_(pretrained_weights)
emb.weight.requires_grad = False              # optionally freeze
```

---

## 3.4 Activation Functions

Activations introduce **non-linearity**. Without them, a deep network collapses to a single linear transformation.

```python
import torch
import torch.nn.functional as F

x = torch.randn(4, 8)

# ── ReLU: max(0, x) ──────────────────────────────────────────────────────────
# ∂/∂x = 1 if x > 0, else 0
# Problem: "dead ReLU" — neurons stuck at 0 if weights too negative
print(F.relu(x).shape)

# ── Leaky ReLU: max(αx, x) where α is small (e.g. 0.01) ──────────────────────
print(F.leaky_relu(x, negative_slope=0.01))

# ── GELU: x * Φ(x) where Φ is standard normal CDF ────────────────────────────
# Smooth approximation used in transformers (BERT, GPT)
# Mathematically: GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
print(F.gelu(x))

# ── Sigmoid: 1/(1+e^{-x}), output ∈ (0,1) ────────────────────────────────────
# ∂/∂x = σ(x)(1-σ(x)), max ≈ 0.25 at x=0 → vanishing gradients in deep nets
print(torch.sigmoid(x))

# ── Tanh: (e^x - e^{-x})/(e^x + e^{-x}), output ∈ (-1, 1) ──────────────────
# ∂/∂x = 1 - tanh²(x), max 1 at x=0 → better than sigmoid but still vanishes
print(torch.tanh(x))

# ── Softmax: exp(xᵢ)/Σ exp(xⱼ), used as output layer for classification ─────
logits = torch.randn(4, 10)
probs  = F.softmax(logits, dim=-1)    # (4, 10), sum along dim=-1 equals 1
log_probs = F.log_softmax(logits, dim=-1)  # numerically stable log

# ── SiLU / Swish: x * sigmoid(x) ─────────────────────────────────────────────
# Used in EfficientNet, modern CNNs
print(F.silu(x))

# ── Mish: x * tanh(softplus(x)) ──────────────────────────────────────────────
print(F.mish(x))

# ── When to use which ─────────────────────────────────────────────────────────
# Hidden layers: ReLU (default), GELU (transformers), SiLU (CNNs)
# Output for binary: Sigmoid (or just use BCEWithLogitsLoss)
# Output for multi-class: Softmax (or just use CrossEntropyLoss)
# RNNs: Tanh (gates), Sigmoid (forget gate)
```

---

## 3.5 Weight Initialisation

Bad initialisation causes vanishing/exploding gradients from the very first forward pass.

```python
import torch.nn as nn
import torch.nn.init as init

def reset_parameters(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Xavier/Glorot: designed for sigmoid/tanh
            # Var(W) = 2 / (fan_in + fan_out)
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)

        elif isinstance(module, nn.Conv2d):
            # Kaiming/He: designed for ReLU
            # Var(W) = 2 / fan_in
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)

        elif isinstance(module, nn.BatchNorm2d):
            # Standard: weight=1 (scale), bias=0 (shift)
            init.ones_(module.weight)
            init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            # Small normal distribution
            init.normal_(module.weight, mean=0.0, std=0.02)
```

---

## 3.6 Building Real Architectures

### Multi-Layer Perceptron (MLP)

```python
class MLP(nn.Module):
    """
    A configurable multi-layer perceptron.
    Architecture: Linear → [BN → ReLU → Dropout] × (n_layers-1) → Linear
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: list,           # e.g. [256, 128, 64]
        out_features: int,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        acts = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}

        layers = []
        prev = in_features
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(acts[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = hidden

        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Usage
mlp = MLP(in_features=784, hidden_dims=[512, 256], out_features=10, dropout=0.3)
x = torch.randn(32, 784)
print(mlp(x).shape)   # (32, 10)
```

### Residual Block (ResNet-style)

```python
class ResidualBlock(nn.Module):
    """
    Pre-activation residual block: BN → ReLU → Conv → BN → ReLU → Conv + skip
    """

    def __init__(self, channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1,      padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.act   = nn.ReLU(inplace=True)

        # If stride > 1, the skip connection must match spatial size
        self.skip = nn.Identity() if stride == 1 else nn.Sequential(
            nn.Conv2d(channels, channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)   # ← skip connection
        return out
```

### Using nn.ModuleList and nn.ModuleDict

```python
class DynamicMLP(nn.Module):
    """MLP using ModuleList — layers are tracked as parameters."""

    def __init__(self, layer_dims: list):
        super().__init__()
        # ModuleList: use when you need to iterate over modules
        self.layers = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i+1])
            for i in range(len(layer_dims) - 1)
        ])
        # WARNING: plain Python lists do NOT register submodules!
        # self.layers = [nn.Linear(...)]  ← WRONG: params not tracked

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class MultiTaskHead(nn.Module):
    """ModuleDict: when modules are accessed by name."""

    def __init__(self, in_features: int, tasks: dict):
        super().__init__()
        # tasks = {"classify": 10, "regress": 1}
        self.heads = nn.ModuleDict({
            task: nn.Linear(in_features, n_out)
            for task, n_out in tasks.items()
        })

    def forward(self, x, task: str):
        return self.heads[task](x)
```

---

## 3.7 Model Introspection

```python
import torch
import torch.nn as nn
from torchinfo import summary

model = MLP(784, [512, 256], 10, dropout=0.3)

# ── torchinfo: detailed summary ───────────────────────────────────────────────
summary(model, input_size=(32, 784), col_names=["input_size", "output_size", "num_params"])

# ── Manual parameter counting ─────────────────────────────────────────────────
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable:    {trainable_params:,}")

# ── Iterate modules ───────────────────────────────────────────────────────────
for name, module in model.named_modules():
    print(f"{name}: {module.__class__.__name__}")

# ── state_dict: model weights for saving/loading ──────────────────────────────
sd = model.state_dict()          # OrderedDict of {name: tensor}
print(list(sd.keys()))

# Save and reload
torch.save(sd, "model_weights.pt")
model2 = MLP(784, [512, 256], 10)
model2.load_state_dict(torch.load("model_weights.pt"))

# ── Freezing and unfreezing parameters ───────────────────────────────────────
for param in model.net[0].parameters():  # freeze first linear layer
    param.requires_grad = False

# Check which params are trainable
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  trainable: {name}")
```

---

## 3.8 Train / Eval Mode

```python
model = MLP(784, [256, 128], 10, dropout=0.5, use_batchnorm=True)

# ── Training mode: dropout ON, BatchNorm uses batch statistics ───────────────
model.train()
out_train = model(torch.randn(32, 784))

# ── Eval mode: dropout OFF, BatchNorm uses running statistics ────────────────
model.eval()
with torch.no_grad():           # also disable gradient tracking for speed
    out_eval = model(torch.randn(32, 784))

# The distributions will differ! Always call model.eval() before inference.

# ── Context manager pattern ────────────────────────────────────────────────
from contextlib import contextmanager

@contextmanager
def eval_mode(model: nn.Module):
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        if was_training:
            model.train()

with eval_mode(model) as m:
    pred = m(x)
```

---

## 3.9 Custom Parameterised Module: Scaled Dot-Product Attention (Preview)

```python
class ScaledLinear(nn.Module):
    """
    A linear layer whose output is divided by sqrt(d_out).
    Used in attention mechanisms (preview of Module 08).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.scale  = out_features ** -0.5   # 1/√d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.scale
```

---

## 3.10 Real-World Case Study: MNIST Classifier

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Model ────────────────────────────────────────────────────────────────────
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(self.flatten(x))

# ── Data ─────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_ds = datasets.MNIST("./data", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST("./data", train=False, download=True, transform=transform)
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2)
test_dl  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2)

# ── Training Loop ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MNISTClassifier().to(device)
optim  = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
crit   = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    total_loss = correct = total = 0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss   = crit(logits, y)
        loss.backward()
        optim.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)

    print(f"Epoch {epoch+1}: loss={total_loss/total:.4f}, acc={correct/total:.4f}")

    model.eval()
    test_correct = test_total = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            test_correct += (preds == y).sum().item()
            test_total   += len(y)
    print(f"  Test acc: {test_correct/test_total:.4f}")
```

---

## Exercises

**Exercise 3.1** Implement a `MultiLayerMLP` from scratch (no `nn.Sequential`), using `nn.ModuleList`. Add `forward_with_intermediates()` that returns the output of each hidden layer.

**Exercise 3.2** Build a `BottleneckBlock` as used in ResNet-50: 1×1 conv (reduce channels) → 3×3 conv → 1×1 conv (restore channels), with a skip projection when needed.

**Exercise 3.3** Implement `ChannelAttention` (Squeeze-and-Excitation block): global average pool → FC → ReLU → FC → Sigmoid → channel-wise scaling. Test on a `(4, 64, 28, 28)` tensor.

---

## Module Summary

| Concept | Key Points |
|---------|-----------|
| `nn.Module` | Base class for all models; registers params, enables `.to()`, `.state_dict()` |
| `nn.Parameter` | Tensor that becomes part of `parameters()` |
| Built-in layers | Linear, Conv2d, BatchNorm, Dropout, Embedding, Pooling |
| Activations | ReLU (default), GELU (transformers), Sigmoid/Tanh (gates) |
| Init | Kaiming for ReLU; Xavier for sigmoid/tanh; zeros for biases |
| `ModuleList/Dict` | Track sub-modules that aren't direct attributes |
| `train()`/`eval()` | Switch dropout and BatchNorm behaviour |
| `state_dict` | Model weights as an OrderedDict; save/load |

---

## Quiz

1. What is the difference between `nn.ReLU()` and `F.relu()`?
2. Why should `bias=False` be used before BatchNorm?
3. What happens if you store sub-modules in a plain Python `list` instead of `nn.ModuleList`?
4. What is the dead ReLU problem and how is it mitigated?
5. Why should you call `model.eval()` before inference?
6. What does Kaiming initialisation set as the variance of weights, and why?
7. What is the skip connection's role in a residual block?

---

*Next: [Module 04 — Training Pipeline Fundamentals](./04_training_pipeline_fundamentals.md)*
