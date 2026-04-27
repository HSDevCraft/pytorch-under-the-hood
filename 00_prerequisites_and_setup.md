# Module 00: Prerequisites & Environment Setup

## Learning Objectives
By the end of this module you will be able to:
- Set up a reproducible PyTorch development environment
- Verify GPU availability and understand device management
- Recall the mathematical prerequisites needed for the course
- Write idiomatic Python (list comprehensions, generators, type hints) at a level sufficient for PyTorch code
- Understand NumPy arrays and how they relate to PyTorch tensors

---

## 0.1 Mathematical Prerequisites

### Linear Algebra
You must be comfortable with:

**Vectors and Matrices**
- A vector **x** ∈ ℝⁿ is an ordered list of n real numbers
- A matrix **A** ∈ ℝ^(m×n) has m rows and n columns
- Matrix–vector product: **y** = **Ax**, where **y** ∈ ℝᵐ

**Key operations:**
- Dot product: **u** · **v** = Σᵢ uᵢvᵢ = ‖u‖‖v‖cos θ
- Matrix multiplication: (**AB**)ᵢⱼ = Σₖ AᵢₖBₖⱼ
- Transpose: (**Aᵀ**)ᵢⱼ = Aⱼᵢ
- Inverse: **AA**⁻¹ = **I** (only for square, non-singular matrices)
- Eigendecomposition: **Av** = λ**v** where λ is an eigenvalue and **v** is an eigenvector

**Norms:**
- L1: ‖x‖₁ = Σ|xᵢ|
- L2: ‖x‖₂ = √(Σxᵢ²)
- Frobenius (matrices): ‖A‖_F = √(Σᵢⱼ Aᵢⱼ²)

### Calculus & Differentiation

**Derivatives and Gradients**
- Scalar derivative: df/dx = lim_{h→0} (f(x+h) − f(x)) / h
- Gradient of f: ℝⁿ → ℝ: ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
- The gradient points in the direction of steepest ascent

**Chain Rule** (essential for backpropagation):
If y = f(g(x)), then dy/dx = (df/dg)(dg/dx)

**Partial Derivatives:**
For f(x, y): ∂f/∂x treats y as constant; ∂f/∂y treats x as constant

**Jacobian Matrix:**
For f: ℝⁿ → ℝᵐ, the Jacobian **J** ∈ ℝ^(m×n) where Jᵢⱼ = ∂fᵢ/∂xⱼ

### Probability & Statistics

**Core concepts:**
- Probability distribution: P(X = x) ≥ 0, Σ P(X = x) = 1
- Expectation: E[X] = Σ x·P(X = x)
- Variance: Var(X) = E[(X − E[X])²]
- Gaussian distribution: p(x) = (1/√(2πσ²)) · exp(−(x−μ)²/(2σ²))
- Bayes' theorem: P(A|B) = P(B|A)·P(A) / P(B)

---

## 0.2 Python Prerequisites

```python
# ── Type hints ──────────────────────────────────────────────────────────────
from typing import List, Tuple, Dict, Optional, Union

def process_batch(data: List[float], scale: float = 1.0) -> List[float]:
    return [x * scale for x in data]

# ── List comprehensions ──────────────────────────────────────────────────────
squares = [x**2 for x in range(10)]
evens   = [x for x in range(20) if x % 2 == 0]

# ── Generators (memory-efficient iteration) ─────────────────────────────────
def infinite_counter(start: int = 0):
    n = start
    while True:
        yield n
        n += 1

# ── Context managers ────────────────────────────────────────────────────────
class Timer:
    import time
    def __enter__(self):
        self.start = self.time.time()
        return self
    def __exit__(self, *args):
        self.elapsed = self.time.time() - self.start

with Timer() as t:
    result = sum(range(1_000_000))
# t.elapsed contains the wall-clock time

# ── Dataclasses ─────────────────────────────────────────────────────────────
from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 10
    device: str = "cuda"
    grad_clip: Optional[float] = 1.0
    tags: List[str] = field(default_factory=list)

# ── OOP essentials ──────────────────────────────────────────────────────────
class BaseModel:
    def __init__(self, name: str):
        self.name = name

    def forward(self, x):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"
```

### NumPy Refresher

```python
import numpy as np

# Array creation
a = np.array([1.0, 2.0, 3.0])           # 1D array, shape (3,)
A = np.zeros((3, 4))                      # shape (3, 4)
B = np.random.randn(2, 3)                # standard normal

# Indexing and slicing
print(B[0])          # first row
print(B[:, 1])       # second column
print(B[0:2, 1:3])   # submatrix

# Broadcasting: shapes (3,) and (3, 3) → (3, 3)
v = np.array([1, 2, 3])
M = np.ones((3, 3))
print(M + v)         # adds v to each row

# Vectorized operations (no Python loops needed)
x = np.linspace(-3, 3, 100)
sigmoid = 1 / (1 + np.exp(-x))          # element-wise

# Axis operations
data = np.random.randn(100, 10)
col_means = data.mean(axis=0)           # shape (10,)
row_norms = np.linalg.norm(data, axis=1) # shape (100,)

# Matrix operations
W = np.random.randn(10, 5)
b = np.random.randn(5)
out = data @ W + b                      # (100, 5)
```

---

## 0.3 Environment Setup

### Option A: Conda (Recommended)

```bash
# Install Miniconda from https://docs.conda.io/en/latest/miniconda.html

# Create environment
conda create -n pytorch-course python=3.11 -y
conda activate pytorch-course

# PyTorch — choose ONE of the following based on your hardware:

# CPU only
pip install torch torchvision torchaudio

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Mac (Apple Silicon MPS)
pip install torch torchvision torchaudio
```

### Option B: pip + venv

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
```

### Full Dependency Installation

```bash
pip install \
    numpy pandas matplotlib seaborn \
    scikit-learn scipy \
    transformers datasets accelerate \
    torchmetrics torchinfo \
    jupyter jupyterlab ipywidgets \
    onnx onnxruntime \
    fastapi uvicorn pydantic \
    tensorboard wandb \
    tqdm rich
```

### Verify Your Installation

```python
# run_this: verify_setup.py
import sys
import torch
import torchvision
import numpy as np

print(f"Python:       {sys.version}")
print(f"PyTorch:      {torch.__version__}")
print(f"TorchVision:  {torchvision.__version__}")
print(f"NumPy:        {np.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count:    {torch.cuda.device_count()}")
    print(f"GPU name:     {torch.cuda.get_device_name(0)}")
    print(f"GPU memory:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Check MPS (Apple Silicon)
if hasattr(torch.backends, "mps"):
    print(f"MPS available: {torch.backends.mps.is_available()}")

# Quick sanity check — a tensor operation on GPU
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
x = torch.randn(1000, 1000, device=device)
y = x @ x.T
print(f"\nDevice: {device} — matrix multiply OK, output shape: {y.shape}")
```

---

## 0.4 IDE & Tooling Setup

### VS Code Extensions
- **Python** (Microsoft)
- **Pylance** — type checking
- **Jupyter** — notebook support in VS Code
- **GitHub Copilot** (optional but useful)

### Jupyter Lab Configuration

```bash
jupyter lab --generate-config
# Edit ~/.jupyter/jupyter_lab_config.py
# c.ServerApp.open_browser = True
# c.ServerApp.port = 8888
```

### Useful IPython Magic Commands

```python
# In Jupyter notebooks
%timeit torch.randn(1000, 1000).cuda()  # benchmark an expression
%matplotlib inline                      # render plots inline
%load_ext tensorboard                   # load TensorBoard extension

# Profile a cell
%%timeit
model(batch)
```

---

## 0.5 PyTorch Core Philosophy

PyTorch is built on three core ideas:

1. **Dynamic computation graphs (define-by-run):** The graph is built as code executes, making debugging feel like regular Python.
2. **NumPy-compatible tensors:** PyTorch tensors are the GPU-accelerated, differentiable counterpart of NumPy arrays.
3. **Python-first:** PyTorch is designed to feel like Python, not like a framework you fight against.

```
┌─────────────────────────────────────────────────┐
│                  Your Python Code                │
│                                                 │
│  x = torch.randn(...)   ← Tensor creation       │
│  y = model(x)           ← Forward pass          │
│  loss = criterion(y, t) ← Loss computation      │
│  loss.backward()        ← Autograd computes ∇   │
│  optimizer.step()       ← Weight update         │
└─────────────────────────────────────────────────┘
         ↓ dispatched to
┌─────────────────────────────────────────────────┐
│          C++ / CUDA / cuDNN backend              │
│      (fast numerics, GPU kernels)                │
└─────────────────────────────────────────────────┘
```

---

## Module Summary

| Concept | Key Takeaway |
|---------|-------------|
| Linear Algebra | Vectors, matrices, dot products, eigenvalues are the language of ML |
| Calculus | Gradients and the chain rule power backpropagation |
| Python | List comps, generators, type hints, dataclasses, OOP |
| NumPy | Vectorized ops, broadcasting, no Python loops |
| PyTorch Install | One command, CUDA-aware; verify with `torch.cuda.is_available()` |
| Core Philosophy | Dynamic graph, Python-first, GPU-accelerated tensors |

---

## Quiz

1. What does `np.random.randn(3, 4)` produce? What is its shape?
2. If **A** ∈ ℝ^(4×3) and **B** ∈ ℝ^(3×5), what is the shape of **AB**?
3. What does the chain rule state? Write it symbolically.
4. How do you verify PyTorch can see your GPU?
5. What is the difference between a Python generator and a list comprehension in terms of memory usage?
6. What does `axis=0` mean in `np.mean(data, axis=0)` for a shape (100, 10) array?

---

*Next: [Module 01 — Tensors & Fundamental Operations](./01_tensors_and_fundamentals.md)*
