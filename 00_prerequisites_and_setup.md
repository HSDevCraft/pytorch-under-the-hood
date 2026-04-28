# Module 00: Prerequisites & Setup — Mathematical & Conceptual Foundations

> **Goal:** Build the mathematical intuition and practical skills needed to understand PyTorch deeply. This module is not about memorizing formulas—it's about understanding *why* these concepts matter in deep learning.

---

## Learning Objectives

By the end of this module, you will:
- **Understand** (not just memorize) linear algebra: vectors, matrices, norms, and operations
- **Grasp** calculus fundamentals: derivatives, partial derivatives, gradients, and the chain rule
- **Recognize** probability distributions and their role in neural networks
- **Master** NumPy arrays and operations (the foundation for PyTorch)
- **Set up** a professional Python environment with PyTorch
- **Know why** these concepts matter in deep learning (the intuition, not just the mechanics)

---

## Part 1: Linear Algebra — The Language of Deep Learning

### 1.1 Vectors: Ordered Lists with Meaning

A **vector** is an ordered list of numbers. In deep learning, vectors represent:
- **Data points** (e.g., pixel intensities in an image, features of a person)
- **Weights** (parameters the model learns)
- **Gradients** (directions to update weights during training)

**Intuition:** Think of a vector as a **point in space**.
- 1D vector [5] → a point on a number line
- 2D vector [3, 4] → a point in a 2D plane
- 3D vector [1, 2, 3] → a point in 3D space
- 100D vector → a point in 100-dimensional space (hard to visualize, but mathematically the same)

```python
import numpy as np

# Create vectors
v1 = np.array([1, 2, 3])  # 1D vector (shape: (3,))
v2 = np.array([4, 5, 6])

print(f"v1 shape: {v1.shape}")  # (3,)
print(f"v1 length: {len(v1)}")  # 3
print(f"v1 type: {type(v1)}")   # <class 'numpy.ndarray'>
```

**Vector Operations and Their Intuitions:**

```python
# ── ADDITION: Move a point by another vector ────────────────────────────────────
# [1,2,3] + [4,5,6] = [5,7,9]
# Intuition: If you're at point (1,2,3) and move by vector (4,5,6), you end up at (5,7,9)
v_sum = v1 + v2  # [5, 7, 9]
print(f"v1 + v2 = {v_sum}")

# ── SCALAR MULTIPLICATION: Scale (stretch or shrink) a vector ───────────────────
# 2 * [1,2,3] = [2,4,6]
# Intuition: Multiply the vector by 2 to make it twice as long, pointing same direction
v_scaled = 2 * v1  # [2, 4, 6]
print(f"2 * v1 = {v_scaled}")

# ── DOT PRODUCT: Measure "alignment" between vectors ──────────────────────────
# [1,2,3] · [4,5,6] = 1*4 + 2*5 + 3*6 = 32
# Intuition:
#   - If vectors point in SAME direction → large POSITIVE value
#   - If vectors are PERPENDICULAR → 0
#   - If vectors point in OPPOSITE directions → large NEGATIVE value
# This is CRUCIAL in neural networks: dot products measure similarity!
dot_product = np.dot(v1, v2)  # 32
print(f"v1 · v2 = {dot_product}")

# Example: perpendicular vectors
v_perp1 = np.array([1, 0])
v_perp2 = np.array([0, 1])
print(f"Perpendicular dot product: {np.dot(v_perp1, v_perp2)}")  # 0

# ── VECTOR NORM (MAGNITUDE): Distance from origin to the point ────────────────
# ‖[1,2,3]‖ = √(1² + 2² + 3²) = √14 ≈ 3.74
# Intuition: How far is the point from the origin?
# In neural networks: norms measure the "size" of weights (used in regularization)
norm = np.linalg.norm(v1)  # ~3.74
print(f"‖v1‖ = {norm:.4f}")

# L1 norm: sum of absolute values (used in L1 regularization)
l1_norm = np.sum(np.abs(v1))  # |1| + |2| + |3| = 6
print(f"L1 norm: {l1_norm}")

# L2 norm: Euclidean distance (most common)
l2_norm = np.sqrt(np.sum(v1 ** 2))  # √(1² + 2² + 3²)
print(f"L2 norm: {l2_norm:.4f}")
```

**Why vectors matter in neural networks:**
- Input data is represented as vectors
- Weights are vectors (or matrices, which are stacks of vectors)
- Dot products compute how much each neuron "fires" based on input
- Gradients are vectors showing which direction to update weights

### 1.2 Matrices: 2D Arrays with Structure

A **matrix** is a 2D array of numbers. In deep learning:
- **Weight matrices** store the parameters of layers
- **Data matrices** stack multiple data points (rows = samples, columns = features)
- **Covariance matrices** describe relationships between variables

**Intuition:** A matrix is a **table of numbers**. Each row is a vector.

```python
# Create a matrix (2 rows, 3 columns)
M = np.array([
    [1, 2, 3],    # Row 1 (a vector)
    [4, 5, 6]     # Row 2 (another vector)
])
print(f"Shape: {M.shape}")  # (2, 3)

# Access elements
print(M[0, 1])  # Row 0, Column 1 → 2
print(M[1, :])  # Row 1, all columns → [4, 5, 6]
print(M[:, 0])  # All rows, column 0 → [1, 4]
```

**Matrix Multiplication: The Most Important Operation**

Matrix multiplication is **NOT** element-wise. It's a special operation that combines rows and columns.

```python
# A: (2, 3) matrix, B: (3, 2) matrix
# Result: (2, 2) matrix
A = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
B = np.array([[7, 8], [9, 10], [11, 12]])  # (3, 2)

# C[i, j] = dot product of A[i, :] (row i) and B[:, j] (column j)
# C[0, 0] = [1,2,3] · [7,9,11] = 1*7 + 2*9 + 3*11 = 58
# C[0, 1] = [1,2,3] · [8,10,12] = 1*8 + 2*10 + 3*12 = 64
# C[1, 0] = [4,5,6] · [7,9,11] = 4*7 + 5*9 + 6*11 = 139
# C[1, 1] = [4,5,6] · [8,10,12] = 4*8 + 5*10 + 6*12 = 154
C = A @ B  # (2, 2)
print(C)
# [[58, 64],
#  [139, 154]]

# Intuition: Each element C[i,j] is the dot product of row i of A with column j of B
# This is how neural network layers work!
```

**Why matrix multiplication matters:**
- In a neural network layer: `output = input @ weights.T + bias`
- This single operation applies the learned transformation to all data points at once
- It's the core computation in deep learning

### 1.3 Tensors: N-Dimensional Generalisation

A **tensor** is the generalisation of vectors and matrices to any number of dimensions.

```python
# 1D tensor (vector): shape (3,)
t1 = np.array([1, 2, 3])

# 2D tensor (matrix): shape (2, 3)
t2 = np.array([[1, 2, 3], [4, 5, 6]])

# 3D tensor: shape (2, 3, 4)
t3 = np.random.randn(2, 3, 4)

# 4D tensor: shape (batch_size, channels, height, width)
# This is how images are stored in deep learning!
# batch_size=32, RGB channels=3, height=224, width=224
images = np.random.randn(32, 3, 224, 224)
print(f"Image batch shape: {images.shape}")
```

**Intuition for 4D tensors (images):**
- **Dimension 0 (batch):** 32 different images
- **Dimension 1 (channels):** 3 color channels (R, G, B)
- **Dimension 2 (height):** 224 pixels tall
- **Dimension 3 (width):** 224 pixels wide

**Indexing tensors:**

```python
images = np.random.randn(32, 3, 224, 224)

# Get first image (all channels, all pixels)
first_image = images[0]  # shape: (3, 224, 224)

# Get red channel of first image
red_channel = images[0, 0]  # shape: (224, 224)

# Get a single pixel from red channel
pixel = images[0, 0, 100, 150]  # shape: () (scalar)

# Get all red channels from all images
all_red = images[:, 0]  # shape: (32, 224, 224)
```

---

## Part 2: Calculus — How Models Learn

### 2.1 Derivatives: Rate of Change

A **derivative** measures how much a function changes when you change its input.

**Intuition:** Imagine driving a car. The derivative is your speed (how much distance changes per unit time).

```python
# Function: f(x) = x²
# Derivative: f'(x) = 2x
# At x=3: f'(3) = 6 (steep slope)
# At x=0: f'(0) = 0 (flat)

def f(x):
    return x ** 2

# Numerical derivative (approximate)
# f'(x) ≈ (f(x+h) - f(x-h)) / (2h) for small h
# This is how computers compute derivatives!
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

x_values = [-2, -1, 0, 1, 2, 3]
print("x\t| f'(x) approx\t| f'(x) actual")
for x in x_values:
    deriv = numerical_derivative(f, x)
    actual = 2 * x  # Analytical derivative
    print(f"{x}\t| {deriv:.4f}\t\t| {actual}")
    # x=-2: f'(-2) ≈ -4.0000, actual: -4
    # x=-1: f'(-1) ≈ -2.0000, actual: -2
    # x=0:  f'(0)  ≈ 0.0000,  actual: 0
    # x=1:  f'(1)  ≈ 2.0000,  actual: 2
    # x=2:  f'(2)  ≈ 4.0000,  actual: 4
    # x=3:  f'(3)  ≈ 6.0000,  actual: 6
```

**Why derivatives matter in deep learning:**
- We want to **minimize loss** (error)
- The derivative tells us which direction to move the weights
- **Negative derivative** → decrease the weight
- **Positive derivative** → increase the weight
- **Zero derivative** → we're at a critical point (minimum or maximum)

### 2.2 Partial Derivatives: Multiple Inputs

When a function has multiple inputs, we compute **partial derivatives** — the derivative with respect to one input, holding others constant.

```python
# Function: f(x, y) = x² + 2xy + y²
# ∂f/∂x = 2x + 2y (derivative w.r.t. x, treating y as constant)
# ∂f/∂y = 2x + 2y (derivative w.r.t. y, treating x as constant)

def f(x, y):
    return x**2 + 2*x*y + y**2

# Partial derivative w.r.t. x
def partial_x(f, x, y, h=1e-5):
    return (f(x+h, y) - f(x-h, y)) / (2*h)

# Partial derivative w.r.t. y
def partial_y(f, x, y, h=1e-5):
    return (f(x, y+h) - f(x, y-h)) / (2*h)

x, y = 1, 2
px = partial_x(f, x, y)
py = partial_y(f, x, y)
print(f"∂f/∂x at (1,2): {px:.4f}, analytical: {2*1 + 2*2} = 6")
print(f"∂f/∂y at (1,2): {py:.4f}, analytical: {2*1 + 2*2} = 6")
```

**Intuition:** Imagine standing on a hill. The partial derivative w.r.t. x tells you the slope if you move east-west. The partial derivative w.r.t. y tells you the slope if you move north-south.

### 2.3 Gradient: Vector of Partial Derivatives

The **gradient** is a vector containing all partial derivatives. It points in the direction of **steepest increase**.

```python
# For f(x, y) = x² + 2xy + y²
# Gradient: ∇f = [∂f/∂x, ∂f/∂y] = [2x + 2y, 2x + 2y]

def gradient(f, x, y, h=1e-5):
    grad_x = (f(x+h, y) - f(x-h, y)) / (2*h)
    grad_y = (f(x, y+h) - f(x, y-h)) / (2*h)
    return np.array([grad_x, grad_y])

x, y = 1, 2
grad = gradient(f, x, y)
print(f"Gradient at (1,2): {grad}")  # [6, 6]

# Interpretation:
# - Gradient points in direction of STEEPEST INCREASE
# - To minimize, move in OPPOSITE direction: (x, y) - learning_rate * gradient
# - This is exactly what gradient descent does!

# Example: gradient descent step
learning_rate = 0.01
new_x = x - learning_rate * grad[0]
new_y = y - learning_rate * grad[1]
print(f"After one GD step: ({new_x:.4f}, {new_y:.4f})")
print(f"Loss decreased from {f(x, y):.4f} to {f(new_x, new_y):.4f}")
```

**Intuition:** Imagine you're lost in fog on a hill. You can't see the valley, but you can feel the slope under your feet. The gradient tells you the steepest downward direction. Gradient descent is like taking small steps downhill until you reach the valley.

### 2.4 Chain Rule: Composing Functions

The **chain rule** tells us how to compute derivatives of composite functions.

**Formula:** If y = f(g(x)), then dy/dx = (df/dg) × (dg/dx)

**Intuition:** If you have a chain of transformations, multiply the derivatives at each step.

```python
# Example: y = (x²)² = x⁴
# Let u = x², then y = u²
# dy/dx = (dy/du) × (du/dx) = 2u × 2x = 2(x²) × 2x = 4x³

def outer(u):
    return u ** 2

def inner(x):
    return x ** 2

def composite(x):
    return outer(inner(x))

# Numerical derivatives
x = 2
h = 1e-5

# dy/du at u=x²
u = inner(x)
dy_du = (outer(u+h) - outer(u-h)) / (2*h)
print(f"dy/du at u={u}: {dy_du:.4f}")

# du/dx at x
du_dx = (inner(x+h) - inner(x-h)) / (2*h)
print(f"du/dx at x={x}: {du_dx:.4f}")

# Chain rule: dy/dx = dy/du × du/dx
dy_dx_chain = dy_du * du_dx
print(f"dy/dx (chain rule): {dy_dx_chain:.4f}")

# Direct computation
dy_dx_direct = (composite(x+h) - composite(x-h)) / (2*h)
print(f"dy/dx (direct): {dy_dx_direct:.4f}")
# Both should be ≈ 4x³ = 4*8 = 32
```

**Why chain rule is crucial:**
- Neural networks are **chains of transformations**: input → layer1 → layer2 → ... → output
- **Backpropagation** uses the chain rule to compute gradients through all layers
- Without it, we couldn't train deep networks
- This is the mathematical foundation of how neural networks learn!

---

## Part 3: Probability — Understanding Randomness

### 3.1 Normal Distribution

The **normal (Gaussian) distribution** is the most important distribution in deep learning.

**Intuition:** Bell curve. Most values cluster around the mean, fewer at extremes.

```python
import matplotlib.pyplot as plt

# Generate samples from normal distribution
# mean=0, std=1 (standard normal)
samples = np.random.normal(loc=0, scale=1, size=10000)

# Plot histogram
plt.hist(samples, bins=50, density=True, alpha=0.7, label='Samples')
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.title("Normal Distribution (μ=0, σ=1)")
plt.legend()
plt.show()

# Key properties
print(f"Mean: {np.mean(samples):.4f}")  # ≈ 0
print(f"Std: {np.std(samples):.4f}")   # ≈ 1
print(f"Min: {np.min(samples):.4f}")   # ≈ -3 to -4
print(f"Max: {np.max(samples):.4f}")   # ≈ 3 to 4
```

**Why normal distribution matters:**
- **Weight initialization:** Initialize weights from normal distribution (prevents vanishing/exploding gradients)
- **Noise modeling:** Many real-world phenomena follow normal distributions
- **Theoretical foundation:** Central Limit Theorem says sums of random variables approach normal distribution

### 3.2 Uniform Distribution

The **uniform distribution** assigns equal probability to all values in a range.

```python
# Generate samples from uniform distribution [0, 1)
samples = np.random.uniform(0, 1, size=10000)

plt.hist(samples, bins=50, density=True, alpha=0.7, label='Samples')
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.title("Uniform Distribution [0, 1)")
plt.legend()
plt.show()

print(f"Mean: {np.mean(samples):.4f}")  # ≈ 0.5
print(f"Min: {np.min(samples):.4f}")    # ≈ 0
print(f"Max: {np.max(samples):.4f}")    # ≈ 1
```

---

## Part 4: NumPy Mastery — Foundation for PyTorch

NumPy is the **essential prerequisite** for PyTorch. PyTorch tensors work almost identically to NumPy arrays, but with GPU acceleration and automatic differentiation.

### 4.1 Creating Arrays

```python
# From Python list
arr1 = np.array([1, 2, 3])

# Zeros (useful for initializing)
zeros = np.zeros((3, 4))  # 3 rows, 4 columns of zeros

# Ones
ones = np.ones((2, 5))

# Random values from [0, 1)
rand = np.random.rand(3, 3)

# Random values from normal distribution
randn = np.random.randn(3, 3)  # mean=0, std=1

# Range (like Python's range)
arange = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# Linspace (evenly spaced values)
linspace = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]

# Identity matrix (1s on diagonal, 0s elsewhere)
identity = np.eye(3)
```

### 4.2 Array Operations

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise operations
add = a + b  # [5, 7, 9]
sub = a - b  # [-3, -3, -3]
mul = a * b  # [4, 10, 18]
div = a / b  # [0.25, 0.4, 0.5]

# Broadcasting: automatic expansion of shapes
scalar = 2
result = a * scalar  # [2, 4, 6]

# Dot product (inner product)
dot = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])  # (2, 2)
B = np.array([[5, 6], [7, 8]])  # (2, 2)
C = A @ B  # Matrix multiplication
```

### 4.3 Indexing and Slicing

```python
arr = np.array([0, 1, 2, 3, 4, 5])

# Single element
print(arr[0])   # 0
print(arr[-1])  # 5 (last element)

# Slicing [start:stop:step]
print(arr[1:4])    # [1, 2, 3] (stop is exclusive)
print(arr[::2])    # [0, 2, 4] (every 2nd element)
print(arr[::-1])   # [5, 4, 3, 2, 1, 0] (reverse)

# 2D indexing
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix[0, 1])   # 2 (row 0, column 1)
print(matrix[1, :])   # [4, 5, 6] (row 1, all columns)
print(matrix[:, 0])   # [1, 4, 7] (all rows, column 0)
print(matrix[1:, 1:]) # [[5, 6], [8, 9]] (rows 1+, cols 1+)
```

### 4.4 Aggregation Functions

```python
arr = np.array([1, 2, 3, 4, 5])

# Sum all elements
total = np.sum(arr)  # 15

# Mean (average)
mean = np.mean(arr)  # 3

# Standard deviation (spread)
std = np.std(arr)  # ~1.41

# Min and max
min_val = np.min(arr)  # 1
max_val = np.max(arr)  # 5

# Along specific axis (for 2D arrays)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
row_sums = np.sum(matrix, axis=1)  # [6, 15] (sum each row)
col_sums = np.sum(matrix, axis=0)  # [5, 7, 9] (sum each column)
```

---

## Part 5: Professional Environment Setup

### 5.1 Install Python

1. Download from [python.org](https://www.python.org/downloads/) (Python 3.10+)
2. Verify: `python --version`

### 5.2 Create Virtual Environment

Virtual environments isolate project dependencies, preventing conflicts.

```bash
# Create virtual environment
python -m venv pytorch_env

# Activate (Linux/macOS)
source pytorch_env/bin/activate

# Activate (Windows)
pytorch_env\Scripts\activate

# Verify activation (prompt should show (pytorch_env))
```

### 5.3 Install PyTorch

```bash
# CPU version (for development)
pip install torch torchvision torchaudio

# GPU version (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# GPU version (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 5.4 Verify Installation

```python
import torch
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Create a simple tensor
x = torch.tensor([1, 2, 3])
print(f"Tensor: {x}")
```

---

## Key Takeaways

| Concept | Why It Matters | Example |
|---------|----------------|---------|
| **Vectors** | Represent data and weights | Image pixels, model parameters |
| **Matrices** | Store transformations | Weight matrices in layers |
| **Tensors** | Generalize to any dimension | Batches of images (4D) |
| **Derivatives** | Measure rate of change | How to update weights |
| **Gradients** | Direction of steepest increase | Backpropagation direction |
| **Chain Rule** | Compose derivatives | Backprop through layers |
| **Normal Distribution** | Initialize weights properly | Prevent vanishing/exploding gradients |
| **NumPy** | Foundation for PyTorch | Master before moving to PyTorch |

---

## Exercises

**Exercise 1:** Create a 3×3 matrix and compute its transpose. Verify that (A^T)^T = A.

**Exercise 2:** Compute the dot product of [1, 2, 3] and [4, 5, 6] manually, then verify with NumPy.

**Exercise 3:** Write a function that computes the numerical gradient of f(x) = x³ at x = 2. Compare with analytical gradient (3x²).

**Exercise 4:** Generate 1000 samples from a normal distribution with mean=5, std=2. Plot histogram and verify mean/std.

**Exercise 5:** Create a (100, 50) matrix of random values. Compute row-wise and column-wise means.

---

## Quiz

1. **What is the shape of a matrix with 5 rows and 3 columns?**
   - Answer: (5, 3)

2. **What does the chain rule compute?**
   - Answer: The derivative of a composite function by multiplying derivatives at each step

3. **What is the dot product of [2, 3] and [4, 5]?**
   - Answer: 2×4 + 3×5 = 8 + 15 = 23

4. **Why do we initialize neural network weights from a normal distribution?**
   - Answer: To prevent vanishing/exploding gradients and ensure stable training

5. **What is the difference between element-wise multiplication and matrix multiplication?**
   - Answer: Element-wise multiplies corresponding elements; matrix multiplication combines rows and columns

6. **What does the gradient vector point towards?**
   - Answer: The direction of steepest increase of the function

7. **How do you create a NumPy array of 10 zeros?**
   - Answer: `np.zeros(10)`

8. **What is the purpose of a virtual environment?**
   - Answer: Isolate project dependencies to prevent conflicts between projects

9. **What is the standard normal distribution?**
   - Answer: Normal distribution with mean=0 and standard deviation=1

10. **How do you verify that PyTorch can access your GPU?**
    - Answer: `torch.cuda.is_available()` should return `True`
