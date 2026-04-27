# Module 01: Tensors & Fundamental Operations

## Learning Objectives
By the end of this module you will be able to:
- Create tensors using every major factory function
- Understand the relationship between dtype, device, and shape
- Perform element-wise, reduction, and matrix operations efficiently
- Exploit broadcasting rules to write loop-free code
- Move tensors between CPU and GPU
- Interoperate seamlessly between PyTorch and NumPy
- Avoid common memory pitfalls (in-place ops, views vs copies)

---

## 1.1 What Is a Tensor?

A **tensor** is a generalization of scalars, vectors, and matrices to arbitrary dimensions (called *rank* or *ndim*):

| Rank | Name | Shape Example | Real-world Example |
|------|------|--------------|-------------------|
| 0 | Scalar | `()` | a single loss value |
| 1 | Vector | `(128,)` | an embedding |
| 2 | Matrix | `(32, 512)` | a batch of embeddings |
| 3 | 3-D tensor | `(32, 3, 224)` | a batch of 1-D signals with 3 channels |
| 4 | 4-D tensor | `(32, 3, 224, 224)` | a batch of RGB images |
| 5 | 5-D tensor | `(8, 4, 3, 224, 224)` | a batch of video clips |

Internally, PyTorch stores tensor data as a contiguous (or strided) block of typed memory, described by:
- **`dtype`** — the element type (float32, int64, bool, ...)
- **`device`** — where the data lives (cpu, cuda:0, mps)
- **`shape`** — sizes along each dimension
- **`stride`** — how many elements to skip in memory to advance one step along each dimension

---

## 1.2 Creating Tensors

```python
import torch
import numpy as np

# ── From Python data ─────────────────────────────────────────────────────────
scalar = torch.tensor(3.14)                    # 0-D, float32
vector = torch.tensor([1, 2, 3])               # 1-D, int64 (inferred)
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) # 2-D, float32

# ── Factory functions ────────────────────────────────────────────────────────
zeros   = torch.zeros(3, 4)           # all zeros, shape (3, 4)
ones    = torch.ones(2, 5)            # all ones
full    = torch.full((2, 3), fill_value=7.0)  # filled with 7
eye     = torch.eye(4)                # 4×4 identity matrix
arange  = torch.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
linspace= torch.linspace(0, 1, steps=5)  # [0.0, 0.25, 0.5, 0.75, 1.0]

# ── Random tensors ───────────────────────────────────────────────────────────
torch.manual_seed(42)                 # reproducibility
uniform = torch.rand(3, 4)            # U[0, 1)
normal  = torch.randn(3, 4)           # N(0, 1)
randint = torch.randint(0, 10, (3, 4)) # integers in [0, 10)
perm    = torch.randperm(10)          # random permutation of 0..9

# ── Like-other-tensor factories ──────────────────────────────────────────────
x = torch.randn(3, 4)
z = torch.zeros_like(x)   # same shape, dtype, device; filled with 0
o = torch.ones_like(x)
r = torch.rand_like(x)

# ── Specifying dtype and device ──────────────────────────────────────────────
f16 = torch.zeros(3, 4, dtype=torch.float16)
i32 = torch.zeros(3, 4, dtype=torch.int32)
gpu = torch.ones(2, 2, device="cuda")   # directly on GPU

# ── From NumPy (zero-copy when dtype matches) ─────────────────────────────────
arr = np.array([1.0, 2.0, 3.0])
t   = torch.from_numpy(arr)             # shares memory!
t2  = torch.tensor(arr)                 # COPIES data

# Danger: shared memory
arr[0] = 99.0
print(t[0])   # 99.0 — they share the same buffer
print(t2[0])  # 1.0  — independent copy

# ── Back to NumPy ────────────────────────────────────────────────────────────
back = t.numpy()                        # zero-copy, CPU tensors only
```

### dtype Reference

| PyTorch dtype | NumPy equivalent | Bits | Notes |
|--------------|-----------------|------|-------|
| `torch.float32` | `float32` | 32 | Default for model parameters |
| `torch.float64` | `float64` | 64 | High-precision math |
| `torch.float16` | `float16` | 16 | Mixed precision (GPU) |
| `torch.bfloat16` | — | 16 | Better range than fp16 (TPU/Ampere+) |
| `torch.int64` | `int64` | 64 | Default for integer tensors; class indices |
| `torch.int32` | `int32` | 32 | |
| `torch.bool` | `bool` | 1 | Masks, conditions |

---

## 1.3 Inspecting Tensors

```python
x = torch.randn(4, 3, 32, 32)

print(x.shape)         # torch.Size([4, 3, 32, 32])
print(x.size())        # same as shape
print(x.ndim)          # 4 (number of dimensions)
print(x.dtype)         # torch.float32
print(x.device)        # device(type='cpu')
print(x.requires_grad) # False
print(x.is_contiguous())  # True
print(x.numel())       # 4 * 3 * 32 * 32 = 12288 elements
print(x.element_size()) # 4 bytes (float32)
print(x.nbytes)        # 12288 * 4 = 49152 bytes

# Memory layout (strides)
print(x.stride())      # (3072, 1024, 32, 1)
# To advance 1 step along dim-0: skip 3*32*32 = 3072 elements in memory
```

---

## 1.4 Indexing & Slicing

```python
x = torch.arange(24).reshape(4, 6)  # shape (4, 6)
# tensor([[ 0,  1,  2,  3,  4,  5],
#         [ 6,  7,  8,  9, 10, 11],
#         [12, 13, 14, 15, 16, 17],
#         [18, 19, 20, 21, 22, 23]])

# Basic slicing (returns a VIEW, not a copy)
print(x[0])           # first row: tensor([0, 1, 2, 3, 4, 5])
print(x[:, 0])        # first column: tensor([0, 6, 12, 18])
print(x[1:3, 2:5])    # submatrix rows 1-2, cols 2-4

# Negative indexing
print(x[-1])          # last row
print(x[:, -2:])      # last 2 columns

# Step slicing
print(x[::2])         # every other row (rows 0, 2)
print(x[:, ::3])      # every 3rd column (cols 0, 3)

# ── Integer / Fancy indexing (returns a COPY) ────────────────────────────────
rows = torch.tensor([0, 2])
cols = torch.tensor([1, 4])
print(x[rows, cols])  # elements (0,1) and (2,4): tensor([1, 16])

# ── Boolean mask indexing ────────────────────────────────────────────────────
mask = x > 10
print(x[mask])        # 1-D tensor of all elements > 10

# ── torch.where ─────────────────────────────────────────────────────────────
result = torch.where(x % 2 == 0, x, torch.zeros_like(x))  # even or 0
```

---

## 1.5 Reshaping Operations

```python
x = torch.arange(12)  # shape (12,)

# reshape: returns a view when possible
y = x.reshape(3, 4)         # (3, 4)
z = x.reshape(2, 2, 3)      # (2, 2, 3)
flat = y.reshape(-1)         # -1 is inferred: (12,)
auto = y.reshape(6, -1)      # -1 inferred: (6, 2)

# view: ALWAYS returns a view; fails if non-contiguous
w = x.view(4, 3)

# contiguous + view pattern
t = x.T                      # transpose — non-contiguous
t_contig = t.contiguous()    # makes a fresh copy that IS contiguous
v = t_contig.view(-1)        # now safe

# squeeze / unsqueeze
a = torch.randn(1, 3, 1, 4)
print(a.squeeze().shape)     # (3, 4) — removes all size-1 dims
print(a.squeeze(0).shape)    # (3, 1, 4) — removes dim 0 only
b = torch.randn(3, 4)
print(b.unsqueeze(0).shape)  # (1, 3, 4) — inserts dim at position 0
print(b.unsqueeze(-1).shape) # (3, 4, 1)

# flatten (equivalent to reshape(-1) but more explicit)
img = torch.randn(32, 3, 28, 28)
flat = img.flatten(start_dim=1)  # (32, 3*28*28) = (32, 2352)

# permute: reorder dimensions
x = torch.randn(32, 224, 224, 3)     # NHWC format
x_chw = x.permute(0, 3, 1, 2)        # NCHW: (32, 3, 224, 224)
```

---

## 1.6 Arithmetic Operations

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Element-wise ops — functional style
print(torch.add(a, b))          # [5, 7, 9]
print(torch.sub(a, b))          # [-3, -3, -3]
print(torch.mul(a, b))          # [4, 10, 18]
print(torch.div(a, b))          # [0.25, 0.4, 0.5]
print(torch.pow(a, 2))          # [1, 4, 9]

# Operator overloads (identical behavior)
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** 2)

# In-place operations (suffix _)
a.add_(b)         # modifies a IN PLACE; equivalent to a += b
a.mul_(2.0)       # a *= 2.0 in place

# WARNING: avoid in-place on tensors that have requires_grad=True
# or that are part of a computation graph — it can corrupt gradients.

# Reduction operations
x = torch.randn(4, 6)
print(x.sum())              # scalar sum of all elements
print(x.sum(dim=0))         # sum along rows → shape (6,)
print(x.sum(dim=1))         # sum along cols → shape (4,)
print(x.sum(dim=1, keepdim=True))  # shape (4, 1)  ← keepdim!

print(x.mean())
print(x.std())
print(x.max())
print(x.min())
print(x.argmax(dim=1))      # index of max along dim 1
print(x.topk(3, dim=1))     # top 3 values and indices along dim 1

# Cumulative ops
v = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(v.cumsum(dim=0))      # [1, 3, 6, 10]
print(v.cumprod(dim=0))     # [1, 2, 6, 24]
```

---

## 1.7 Matrix Operations

```python
A = torch.randn(4, 3)
B = torch.randn(3, 5)

# Matrix multiplication
C = torch.mm(A, B)           # (4, 3) @ (3, 5) → (4, 5)
C = A @ B                    # same, using @ operator (preferred)

# Batched matrix multiply
batch_A = torch.randn(32, 4, 3)  # 32 matrices of shape (4, 3)
batch_B = torch.randn(32, 3, 5)
batch_C = torch.bmm(batch_A, batch_B)  # (32, 4, 5)
# or equivalently with broadcasting:
batch_C = batch_A @ batch_B

# Einstein summation notation (extremely expressive)
# Equivalent to mm:
C = torch.einsum("ij,jk->ik", A, B)
# Batched matrix multiply:
D = torch.einsum("bij,bjk->bik", batch_A, batch_B)
# Dot product:
dot = torch.einsum("i,i->", a, b)
# Outer product:
outer = torch.einsum("i,j->ij", a, b)

# Transpose
At = A.T        # or A.transpose(0, 1)
At = A.mT       # batch-aware transpose (transposes last 2 dims)

# Useful matrix operations
sym = A.T @ A                        # symmetric matrix (3, 3)
vals, vecs = torch.linalg.eig(sym.float())   # eigendecomposition
inv  = torch.linalg.inv(sym)         # matrix inverse
det  = torch.linalg.det(sym)         # determinant
norm = torch.linalg.norm(A, ord="fro")  # Frobenius norm
U, S, Vh = torch.linalg.svd(A, full_matrices=False)  # SVD
```

---

## 1.8 Broadcasting

Broadcasting lets you do arithmetic on tensors with different shapes **without copying data**. PyTorch follows NumPy's broadcasting rules:

**Rules (applied right-to-left on dimensions):**
1. If tensors don't have the same number of dims, prepend 1s to the smaller shape.
2. Dimensions of size 1 are expanded to match the other tensor.
3. If sizes differ and neither is 1, it's an error.

```python
# Example 1: add bias to every row of a matrix
x = torch.randn(32, 512)   # (32, 512)
b = torch.randn(512)        # (512,)  → broadcast to (32, 512)
out = x + b                 # (32, 512) — no copy of b!

# Example 2: scale each channel in a NCHW image batch
imgs = torch.randn(8, 3, 224, 224)  # (8, 3, 224, 224)
scale = torch.tensor([0.229, 0.224, 0.225])  # (3,)
# need shape (1, 3, 1, 1) for broadcasting
scale = scale.view(1, 3, 1, 1)
normalized = imgs / scale

# Example 3: pairwise distance matrix (n points in d dims)
X = torch.randn(100, 64)   # (100, 64)
# ‖X_i - X_j‖² = ‖X_i‖² - 2 X_i·X_j + ‖X_j‖²
sq_norms = (X * X).sum(dim=1, keepdim=True)  # (100, 1)
dists = sq_norms + sq_norms.T - 2 * (X @ X.T)  # (100, 100)

# Visualize broadcasting shapes
import torch
a = torch.randn(4, 1, 6)
b = torch.randn(   3, 6)
c = a + b   # shapes: (4, 1, 6) + (1, 3, 6) → (4, 3, 6)
print(c.shape)  # torch.Size([4, 3, 6])
```

---

## 1.9 Device Management

```python
# ── Select the best available device ─────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print(device)  # e.g. device(type='cuda')

# ── Moving tensors ────────────────────────────────────────────────────────────
x = torch.randn(3, 4)           # on CPU
x_gpu = x.to(device)            # move to GPU
x_gpu = x.cuda()                # shorthand (CUDA only)
x_back = x_gpu.cpu()            # move back to CPU
x_fp16 = x.to(device, dtype=torch.float16)  # move and cast in one call

# ── Multi-GPU: specify device index ──────────────────────────────────────────
if torch.cuda.device_count() > 1:
    x_gpu1 = x.to("cuda:1")

# ── Pin memory for fast host→device transfer ──────────────────────────────────
# Used in DataLoaders
x_pinned = x.pin_memory()       # page-locked CPU memory

# ── CUDA memory inspection ────────────────────────────────────────────────────
if torch.cuda.is_available():
    print(torch.cuda.memory_allocated() / 1e6, "MB allocated")
    print(torch.cuda.memory_reserved() / 1e6,  "MB reserved")
    torch.cuda.empty_cache()    # release cached but unused memory

# ── Best practice: always use .to(device) not .cuda() ────────────────────────
# This makes your code hardware-agnostic
model = MyModel().to(device)
batch = batch.to(device)
```

---

## 1.10 Views vs Copies

Understanding when PyTorch shares memory is critical:

```python
x = torch.arange(12).reshape(3, 4)

# These SHARE memory (views):
v1 = x[0]            # first row — view
v2 = x[:, :2]        # column slice — view
v3 = x.view(2, 6)    # reshape — view
v4 = x.t()           # transpose — view (non-contiguous!)

# These CREATE a copy:
c1 = x.clone()              # always copies, preserves grad_fn
c2 = x.detach().clone()     # copies, detached from computation graph
c3 = x[torch.tensor([0,2])] # fancy indexing — copy

# Verify sharing
v1[0] = 999
print(x[0, 0])  # 999 — they share data

# Check if a tensor is a view
print(x.storage().data_ptr() == v1.storage().data_ptr())  # True

# is_leaf check (important for autograd)
w = torch.randn(3, 3, requires_grad=True)
print(w.is_leaf)   # True — created by user
y = w * 2
print(y.is_leaf)   # False — result of an operation
```

---

## 1.11 Concatenating & Stacking

```python
a = torch.randn(3, 4)
b = torch.randn(3, 4)

# cat: concatenate along an EXISTING dimension
h = torch.cat([a, b], dim=0)   # (6, 4) — stack vertically
w = torch.cat([a, b], dim=1)   # (3, 8) — stack horizontally

# stack: creates a NEW dimension
s = torch.stack([a, b], dim=0)  # (2, 3, 4)
s = torch.stack([a, b], dim=1)  # (3, 2, 4)

# chunk / split: the inverse of cat
parts = torch.chunk(h, chunks=3, dim=0)   # 3 tensors of shape (2, 4)
p1, p2 = torch.split(h, split_size_or_sections=3, dim=0)  # (3,4) each
```

---

## 1.12 Real-World Use Case: Image Preprocessing Pipeline

```python
import torch
import torchvision.transforms.functional as TF

def preprocess_image_batch(
    images: list,           # list of PIL images
    target_size: tuple = (224, 224),
    mean: tuple = (0.485, 0.456, 0.406),  # ImageNet stats
    std:  tuple = (0.229, 0.224, 0.225),
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Convert a list of PIL images to a normalized tensor batch.
    Returns: (N, C, H, W) float32 tensor on `device`.
    """
    tensors = []
    for img in images:
        # Resize → to tensor (scales to [0,1]) → normalize
        img = TF.resize(img, list(target_size))
        t   = TF.to_tensor(img)            # (3, H, W), float32, [0,1]
        t   = TF.normalize(t, mean, std)   # (3, H, W), zero-centred
        tensors.append(t)

    batch = torch.stack(tensors, dim=0)    # (N, 3, H, W)
    return batch.to(device, non_blocking=True)

# Inverse: denormalize for visualization
def denormalize(
    tensor: torch.Tensor,
    mean=(0.485, 0.456, 0.406),
    std =(0.229, 0.224, 0.225),
) -> torch.Tensor:
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t  = torch.tensor(std).view(1, 3, 1, 1)
    return (tensor * std_t + mean_t).clamp(0, 1)
```

---

## 1.13 Best Practices

| Practice | Why |
|----------|-----|
| Use `torch.manual_seed(seed)` + `torch.cuda.manual_seed_all(seed)` | Reproducibility |
| Prefer `x.to(device)` over `x.cuda()` | Hardware-agnostic code |
| Use `keepdim=True` in reductions when you need broadcasting downstream | Avoid shape bugs |
| Avoid Python loops over tensor elements; use vectorized ops | 100–1000× faster |
| Use `torch.no_grad()` for inference | Saves memory, disables grad tracking |
| Call `.detach()` before `.numpy()` on grad-tracked tensors | Prevents errors |
| Use `.clone()` when you need an independent copy | Prevents accidental aliasing |
| Prefer `@` over `torch.mm` | Works for any number of batch dims |

---

## Exercises

**Exercise 1.1** Create a tensor of shape `(5, 5)` where element `[i, j]` = `i * j`. Do this without Python loops.

**Exercise 1.2** Write a function `normalize(x)` that standardizes a 2-D tensor (zero mean, unit variance per column) using only PyTorch tensor ops.

**Exercise 1.3** Given `x = torch.randn(100, 64)`, compute the pairwise cosine similarity matrix of shape `(100, 100)` using broadcasting (no loops).

**Exercise 1.4** Implement batch-wise softmax (`dim=-1`) from scratch using only `torch.exp`, `torch.sum`, and broadcasting. Verify against `torch.softmax`.

**Exercise 1.5** Write a function that takes a 4-D image batch `(N, C, H, W)` and applies random horizontal flipping to each image independently using tensor ops.

<details>
<summary>Solutions</summary>

```python
# 1.1
i = torch.arange(5).unsqueeze(1)  # (5, 1)
j = torch.arange(5).unsqueeze(0)  # (1, 5)
result = i * j                     # (5, 5) via broadcasting

# 1.2
def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)   # (1, F)
    std  = x.std(dim=0, keepdim=True)    # (1, F)
    return (x - mean) / (std + 1e-8)

# 1.3
def cosine_similarity_matrix(X: torch.Tensor) -> torch.Tensor:
    norms = X.norm(dim=1, keepdim=True)          # (100, 1)
    X_norm = X / (norms + 1e-8)                  # (100, 64)
    return X_norm @ X_norm.T                      # (100, 100)

# 1.4
def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_shifted = x - x.max(dim=dim, keepdim=True).values  # numerical stability
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

# Verify
x = torch.randn(4, 10)
assert torch.allclose(my_softmax(x), torch.softmax(x, dim=-1), atol=1e-6)

# 1.5
def random_hflip(batch: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    N = batch.shape[0]
    flip_mask = torch.rand(N) < p         # (N,)  bool
    flipped = batch.clone()
    flipped[flip_mask] = batch[flip_mask].flip(dims=[-1])  # flip W dim
    return flipped
```

</details>

---

## Module Summary

| Concept | Key Points |
|---------|-----------|
| Tensor creation | `torch.tensor`, `zeros/ones/rand/randn`, `arange/linspace`, `*_like` |
| Shape manipulation | `reshape`, `view`, `squeeze/unsqueeze`, `permute`, `flatten` |
| Indexing | slices → views; fancy/bool indexing → copies |
| Broadcasting | align shapes from the right; size-1 dims expand |
| Matrix ops | `@`, `bmm`, `einsum`, `torch.linalg.*` |
| Device mgmt | `.to(device)` is the idiomatic way to move tensors |
| Memory | views share storage; `.clone()` for independent copies |

---

## Quiz

1. What is the shape of `torch.randn(2, 3).unsqueeze(1)`?
2. Why does `x.T.view(-1)` sometimes raise an error?
3. What is the difference between `torch.cat` and `torch.stack`?
4. Given shapes `(3, 1, 5)` and `(1, 4, 5)`, what is the broadcast result shape?
5. What does `keepdim=True` do in `x.sum(dim=1, keepdim=True)`?
6. Why is `torch.from_numpy(arr)` potentially dangerous?
7. What is a stride in the context of tensor memory layout?
8. When would you use `einsum` over `@`?

---

*Next: [Module 02 — Autograd & Computation Graphs](./02_autograd_and_computation_graphs.md)*
