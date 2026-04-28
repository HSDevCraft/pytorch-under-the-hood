# Module 01: Tensors & Fundamentals — The Building Blocks of PyTorch

> **Goal:** Understand PyTorch tensors deeply—not just how to use them, but *why* they're designed the way they are and how they differ from NumPy arrays.

---

## Learning Objectives

By the end of this module, you will:
- **Understand** what PyTorch tensors are and how they differ from NumPy arrays
- **Create** tensors in multiple ways and understand tensor properties (shape, dtype, device)
- **Perform** tensor operations: indexing, slicing, reshaping, broadcasting
- **Grasp** the concept of **computational graphs** (tensors track operations for backprop)
- **Know** when to use CPU vs GPU and how to move tensors between devices
- **Understand** views vs copies and why it matters for memory efficiency

---

## Part 1: What Are PyTorch Tensors?

### 1.1 Tensors vs NumPy Arrays

A **PyTorch tensor** is similar to a NumPy array but with crucial differences:

| Feature | NumPy Array | PyTorch Tensor |
|---------|-------------|----------------|
| **GPU Support** | CPU only | CPU or GPU |
| **Automatic Differentiation** | No | Yes (tracks operations) |
| **Computational Graph** | No | Yes (for backprop) |
| **Speed** | Good for CPU | Excellent for GPU |
| **Deep Learning** | Not designed for it | Built for it |

**Intuition:** Think of a NumPy array as a static container of numbers. A PyTorch tensor is a container that also remembers how it was created—this memory is crucial for computing gradients during backpropagation.

```python
import torch
import numpy as np

# NumPy array
np_array = np.array([1, 2, 3])
print(f"NumPy array: {np_array}")
print(f"Type: {type(np_array)}")

# PyTorch tensor
torch_tensor = torch.tensor([1, 2, 3])
print(f"PyTorch tensor: {torch_tensor}")
print(f"Type: {type(torch_tensor)}")

# Convert between them
np_to_torch = torch.from_numpy(np_array)
torch_to_np = torch_tensor.numpy()
```

### 1.2 Creating Tensors

```python
# ── From Python lists ──────────────────────────────────────────────────────────
t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([[1, 2], [3, 4]])  # 2D tensor

# ── From NumPy arrays ─────────────────────────────────────────────────────────
np_arr = np.array([1, 2, 3])
t3 = torch.from_numpy(np_arr)  # Shares memory with NumPy array!

# ── Common initializations ────────────────────────────────────────────────────
zeros = torch.zeros(3, 4)      # 3×4 matrix of zeros
ones = torch.ones(2, 5)        # 2×5 matrix of ones
random = torch.rand(3, 3)      # Random values in [0, 1)
randn = torch.randn(3, 3)      # Random from normal distribution
arange = torch.arange(0, 10, 2) # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5) # [0, 0.25, 0.5, 0.75, 1]

# ── Specify data type ──────────────────────────────────────────────────────────
t_int = torch.tensor([1, 2, 3], dtype=torch.int32)
t_float = torch.tensor([1, 2, 3], dtype=torch.float32)
t_double = torch.tensor([1, 2, 3], dtype=torch.float64)

# ── Create on GPU (if available) ───────────────────────────────────────────────
if torch.cuda.is_available():
    t_gpu = torch.tensor([1, 2, 3], device='cuda')
    print(f"Tensor device: {t_gpu.device}")
```

### 1.3 Tensor Properties

```python
t = torch.randn(2, 3, 4)  # 3D tensor

# Shape: dimensions of the tensor
print(f"Shape: {t.shape}")  # torch.Size([2, 3, 4])
print(f"Shape[0]: {t.shape[0]}")  # 2

# Size: same as shape
print(f"Size: {t.size()}")  # torch.Size([2, 3, 4])

# Number of dimensions
print(f"Ndim: {t.ndim}")  # 3

# Total number of elements
print(f"Numel: {t.numel()}")  # 2*3*4 = 24

# Data type
print(f"Dtype: {t.dtype}")  # torch.float32

# Device (CPU or GPU)
print(f"Device: {t.device}")  # cpu

# Requires gradient? (for backprop)
print(f"Requires grad: {t.requires_grad}")  # False
```

---

## Part 2: Tensor Operations

### 2.1 Indexing and Slicing

```python
t = torch.arange(12).reshape(3, 4)
# [[0, 1, 2, 3],
#  [4, 5, 6, 7],
#  [8, 9, 10, 11]]

# ── Single element ─────────────────────────────────────────────────────────────
print(t[0, 1])  # 1 (row 0, column 1)
print(t[2, 3])  # 11

# ── Rows and columns ───────────────────────────────────────────────────────────
print(t[0])     # [0, 1, 2, 3] (first row)
print(t[:, 0])  # [0, 4, 8] (first column)

# ── Slicing ────────────────────────────────────────────────────────────────────
print(t[1:3])   # Rows 1-2 (row 3 excluded)
print(t[:, 1:3]) # Columns 1-2

# ── Negative indexing ──────────────────────────────────────────────────────────
print(t[-1])    # Last row
print(t[-1, -1]) # Last element

# ── Step slicing ───────────────────────────────────────────────────────────────
print(t[::2])   # Every 2nd row
print(t[:, ::2]) # Every 2nd column
```

### 2.2 Reshaping and Viewing

```python
t = torch.arange(12)  # [0, 1, 2, ..., 11]

# ── Reshape: change shape, same data ───────────────────────────────────────────
t_reshaped = t.reshape(3, 4)  # 3×4 matrix
print(t_reshaped.shape)  # torch.Size([3, 4])

# Reshape to 1D
t_flat = t_reshaped.reshape(-1)  # -1 means "infer this dimension"
print(t_flat.shape)  # torch.Size([12])

# ── View: create a view (shares memory) ────────────────────────────────────────
t_view = t.view(3, 4)  # Same as reshape, but requires contiguous memory
print(t_view.shape)  # torch.Size([3, 4])

# ── Squeeze: remove dimensions of size 1 ───────────────────────────────────────
t_1d = torch.tensor([1, 2, 3])
t_unsqueezed = t_1d.unsqueeze(0)  # Add dimension at position 0
print(t_unsqueezed.shape)  # torch.Size([1, 3])

t_squeezed = t_unsqueezed.squeeze()  # Remove dimensions of size 1
print(t_squeezed.shape)  # torch.Size([3])

# ── Transpose: swap dimensions ────────────────────────────────────────────────
t_2d = torch.arange(6).reshape(2, 3)
t_transposed = t_2d.T  # or t_2d.transpose(0, 1)
print(t_transposed.shape)  # torch.Size([3, 2])
```

### 2.3 Arithmetic Operations

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# ── Element-wise operations ────────────────────────────────────────────────────
add = a + b  # [5, 7, 9]
sub = a - b  # [-3, -3, -3]
mul = a * b  # [4, 10, 18]
div = a / b  # [0.25, 0.4, 0.5]
pow = a ** 2  # [1, 4, 9]

# ── In-place operations (modify tensor in place) ────────────────────────────────
a_copy = a.clone()
a_copy += 10  # a_copy becomes [11, 12, 13]
a_copy *= 2   # a_copy becomes [22, 24, 26]

# ── Dot product and matrix multiplication ──────────────────────────────────────
dot = torch.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32

A = torch.arange(6).reshape(2, 3)  # [[0, 1, 2], [3, 4, 5]]
B = torch.arange(6).reshape(3, 2)  # [[0, 1], [2, 3], [4, 5]]
C = A @ B  # Matrix multiplication
# [[0*0+1*2+2*4, 0*1+1*3+2*5],
#  [3*0+4*2+5*4, 3*1+4*3+5*5]]
# = [[10, 13], [28, 40]]
```

### 2.4 Broadcasting

**Broadcasting** is the automatic expansion of tensor shapes to make operations compatible.

```python
# ── Broadcasting rules ────────────────────────────────────────────────────────
# When operating on two tensors, PyTorch compares shapes element-wise.
# Two dimensions are compatible when:
#   1. They are equal, or
#   2. One of them is 1 (the 1 is "broadcast" to match the other)

# Example 1: Scalar with tensor
a = torch.tensor([1, 2, 3])
b = 2
c = a + b  # [3, 4, 5] (scalar 2 is broadcast to [2, 2, 2])

# Example 2: Different shapes
a = torch.arange(6).reshape(2, 3)  # shape (2, 3)
b = torch.arange(3)  # shape (3,)
c = a + b  # b is broadcast to (2, 3)
# [[0, 1, 2],    [[0, 1, 2],
#  [3, 4, 5]]  +  [0, 1, 2]]  = [[0, 2, 4], [3, 5, 7]]

# Example 3: Column vector with row vector
col = torch.arange(3).reshape(3, 1)  # shape (3, 1)
row = torch.arange(4)  # shape (4,)
result = col + row  # Broadcast to (3, 4)
# [[0],      [[0, 1, 2, 3],
#  [1],   +   [0, 1, 2, 3]]  = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]
#  [2]]
```

---

## Part 3: Device Management (CPU vs GPU)

### 3.1 Understanding Devices

```python
# Check if GPU is available
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")

# Create tensors on different devices
t_cpu = torch.tensor([1, 2, 3], device='cpu')
if torch.cuda.is_available():
    t_gpu = torch.tensor([1, 2, 3], device='cuda')
    print(f"CPU tensor device: {t_cpu.device}")
    print(f"GPU tensor device: {t_gpu.device}")
```

### 3.2 Moving Tensors Between Devices

```python
t = torch.tensor([1, 2, 3])

# Move to GPU
if torch.cuda.is_available():
    t_gpu = t.to('cuda')
    print(f"Tensor on GPU: {t_gpu.device}")

    # Move back to CPU
    t_cpu = t_gpu.to('cpu')
    print(f"Tensor on CPU: {t_cpu.device}")

# Alternative syntax
if torch.cuda.is_available():
    t_gpu = t.cuda()  # Move to GPU
    t_cpu = t_gpu.cpu()  # Move to CPU
```

### 3.3 Why GPU Matters

```python
import time

# Create large tensors
size = 10000
a = torch.randn(size, size)
b = torch.randn(size, size)

# CPU computation
start = time.time()
c_cpu = a @ b
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.4f}s")

# GPU computation (if available)
if torch.cuda.is_available():
    a_gpu = a.cuda()
    b_gpu = b.cuda()
    
    # Warm up GPU
    _ = a_gpu @ b_gpu
    
    # Time GPU computation
    torch.cuda.synchronize()  # Wait for GPU to finish
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU time: {gpu_time:.4f}s")
    print(f"Speedup: {cpu_time / gpu_time:.1f}x")
```

---

## Part 4: Views vs Copies

### 4.1 Understanding Memory Sharing

```python
# ── View: shares memory ────────────────────────────────────────────────────────
t = torch.arange(12)
t_view = t.view(3, 4)

# Modify the view
t_view[0, 0] = 999

# Original tensor is also modified!
print(t[0])  # 999 (first element changed)

# ── Clone: creates a copy ────────────────────────────────────────────────────────
t = torch.arange(12)
t_clone = t.clone()

# Modify the clone
t_clone[0] = 999

# Original tensor is NOT modified
print(t[0])  # 0 (unchanged)
```

### 4.2 Contiguity

```python
# Views require contiguous memory
t = torch.arange(12).reshape(3, 4)
print(f"Is contiguous: {t.is_contiguous()}")  # True

# Transpose creates non-contiguous tensor
t_transposed = t.T
print(f"Is contiguous: {t_transposed.is_contiguous()}")  # False

# Can't view a non-contiguous tensor
# t_transposed.view(-1)  # Error!

# Solution: make it contiguous first
t_contiguous = t_transposed.contiguous()
t_flat = t_contiguous.view(-1)  # Now works
```

---

## Part 5: Computational Graphs (Preview)

### 5.1 Tensors Track Operations

This is the key difference from NumPy: PyTorch tensors can track how they were created.

```python
# Create tensors that require gradients
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0], requires_grad=True)

# Perform operations
z = x + y
w = z * 2

print(f"z: {z}")
print(f"w: {w}")

# The computational graph is:
# x → (+) → z → (*2) → w
# y →(+)↗

# We can compute gradients (more on this in Module 02)
loss = w.sum()
loss.backward()

print(f"Gradient of x: {x.grad}")
print(f"Gradient of y: {y.grad}")
```

**Intuition:** PyTorch remembers the chain of operations. When you call `.backward()`, it uses the chain rule to compute gradients automatically.

---

## Part 6: Common Tensor Operations

### 6.1 Aggregation Functions

```python
t = torch.arange(12).reshape(3, 4)

# Sum all elements
total = t.sum()  # 66

# Sum along axis
row_sums = t.sum(dim=1)  # Sum each row
col_sums = t.sum(dim=0)  # Sum each column

# Mean, std, min, max
mean = t.mean()
std = t.std()
min_val = t.min()
max_val = t.max()

# Argmax: index of maximum value
max_idx = t.argmax()  # 11 (index of value 11)
max_idx_per_row = t.argmax(dim=1)  # [3, 3, 3] (max index in each row)
```

### 6.2 Useful Functions

```python
t = torch.tensor([1.5, -2.3, 0.0, 3.7])

# Absolute value
abs_t = torch.abs(t)

# Square root
sqrt_t = torch.sqrt(torch.abs(t))

# Exponential and logarithm
exp_t = torch.exp(t)
log_t = torch.log(torch.abs(t) + 1e-8)  # Add small value to avoid log(0)

# Trigonometric
sin_t = torch.sin(t)
cos_t = torch.cos(t)

# Activation functions (more on these in Module 03)
relu_t = torch.relu(t)  # max(0, x)
sigmoid_t = torch.sigmoid(t)  # 1 / (1 + exp(-x))
tanh_t = torch.tanh(t)  # (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

---

## Key Takeaways

| Concept | Why It Matters |
|---------|----------------|
| **Tensors vs Arrays** | Tensors track operations for automatic differentiation |
| **Shape & Size** | Understanding tensor dimensions is crucial for debugging |
| **Indexing & Slicing** | Core operations for data manipulation |
| **Broadcasting** | Enables efficient operations on different-shaped tensors |
| **Device Management** | GPU acceleration requires moving tensors to GPU |
| **Views vs Copies** | Views save memory but share data; copies are independent |
| **Computational Graphs** | Foundation for backpropagation and automatic differentiation |

---

## Exercises

1. Create a 4×5 tensor of random values. Extract the middle 2×3 submatrix.
2. Create two tensors of shape (3, 4) and (4, 5). Multiply them using matrix multiplication.
3. Create a tensor and verify that transposing twice gives the original tensor.
4. Demonstrate broadcasting by adding a (3, 1) tensor to a (1, 4) tensor.
5. Create a tensor on GPU (if available) and move it back to CPU.

---

## Quiz

1. **What is the main difference between a PyTorch tensor and a NumPy array?**
   - Answer: PyTorch tensors support automatic differentiation and GPU acceleration

2. **How do you create a 3×4 tensor of zeros?**
   - Answer: `torch.zeros(3, 4)`

3. **What does `t.shape` return?**
   - Answer: A `torch.Size` object containing the dimensions of the tensor

4. **What is broadcasting?**
   - Answer: Automatic expansion of tensor shapes to make operations compatible

5. **How do you move a tensor to GPU?**
   - Answer: `t.to('cuda')` or `t.cuda()`

6. **What is the difference between `.view()` and `.reshape()`?**
   - Answer: `.view()` requires contiguous memory; `.reshape()` handles non-contiguous tensors

7. **What does `requires_grad=True` do?**
   - Answer: Tells PyTorch to track operations on this tensor for gradient computation

8. **How do you get the maximum value in a tensor?**
   - Answer: `t.max()`

9. **What is a computational graph?**
   - Answer: A record of all operations performed on tensors, used for backpropagation

10. **How do you clone a tensor?**
    - Answer: `t.clone()`
