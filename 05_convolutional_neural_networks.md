# Module 05: Convolutional Neural Networks — Seeing with Deep Learning

> **Goal:** Understand convolutions from first principles — the mathematics, the intuition, and how modern architectures (ResNet, EfficientNet) are built from these primitives.

---

## Learning Objectives

By the end of this module, you will:
- **Understand** the convolution operation mathematically and intuitively
- **Calculate** output dimensions with any padding/stride/dilation combination
- **Build** LeNet, VGG-style, and ResNet-style networks from scratch
- **Apply** modern training techniques: data augmentation, Mixup, CutMix
- **Use** depthwise-separable and dilated convolutions for efficiency
- **Debug** common CNN issues: gradient flow, dead filters, overfitting

---

## Part 1: Why Convolutions for Images?

### 1.1 The Problem with Fully Connected Layers on Images

Imagine a 224×224 RGB image (standard ImageNet size):
- Pixels: 224 × 224 × 3 = **150,528 input features**
- If first hidden layer has 1,024 neurons: **154 MILLION weights** just for layer 1!

Problems:
1. **Too many parameters** → overfitting, massive memory
2. **No spatial structure** exploited → every pixel treated independently
3. **Not translation invariant** → a cat at top-left ≠ cat at bottom-right

Convolutions solve ALL three problems.

### 1.2 The Key Insight: Shared Weights + Local Connectivity

A convolution filter (kernel) is a **small, reusable pattern detector**:
- A 3×3 filter has only 9 weights (+ 1 bias) — **shared across all spatial locations**
- It slides across the image, detecting the same pattern everywhere
- This gives **translation equivariance**: if the cat moves, the feature map moves too

```
Input image (5×5):       3×3 filter:        Output (3×3):
┌─────────────────┐      ┌─────────┐        ┌───────────┐
│ 1  2  3  4  5  │      │ 1  0 -1 │        │ -8  -8  -8│
│ 2  3  4  5  6  │      │ 1  0 -1 │  →     │ -8  -8  -8│
│ 3  4  5  6  7  │      │ 1  0 -1 │        │ -8  -8  -8│
│ 4  5  6  7  8  │      └─────────┘        └───────────┘
│ 5  6  7  8  9  │       (Sobel x-edge)     (vertical edges)
└─────────────────┘
```

The filter slides across (convolution), computing a dot product at each position.

---

## Part 2: Convolution Mathematics

### 2.1 The 2D Convolution Operation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Manual convolution (for understanding, not speed!)
def manual_conv2d(input_2d, kernel):
    """
    Demonstrates what Conv2d does under the hood.
    
    input_2d: (H, W) — single channel input
    kernel:   (kH, kW) — single filter
    output:   (H-kH+1, W-kW+1) — with no padding
    """
    H, W = input_2d.shape
    kH, kW = kernel.shape
    out_H = H - kH + 1
    out_W = W - kW + 1
    
    output = torch.zeros(out_H, out_W)
    
    for i in range(out_H):       # Slide vertically
        for j in range(out_W):   # Slide horizontally
            # Extract the patch at position (i, j)
            patch = input_2d[i:i+kH, j:j+kW]
            # Dot product with kernel (element-wise multiply + sum)
            output[i, j] = (patch * kernel).sum()
    
    return output

# Example: edge detection
image = torch.arange(25, dtype=torch.float32).reshape(5, 5)
edge_kernel = torch.tensor([[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]])  # Sobel-x filter

result = manual_conv2d(image, edge_kernel)
print(f"Input:\n{image}")
print(f"Sobel-x output:\n{result}")
```

### 2.2 Output Size Formula

```python
# This formula tells you EXACTLY what shape the output will be.
# Memorize it — you'll use it constantly.
#
# output_size = floor((input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
#
# For standard case (dilation=1):
# output_size = floor((input_size + 2*padding - kernel_size) / stride + 1)

def conv_output_size(input_size, kernel_size, padding=0, stride=1, dilation=1):
    return (input_size + 2*padding - dilation*(kernel_size-1) - 1) // stride + 1

# Examples:
print(conv_output_size(28, 3, padding=0, stride=1))  # (28+0-3)/1+1 = 26
print(conv_output_size(28, 3, padding=1, stride=1))  # (28+2-3)/1+1 = 28 (same size!)
print(conv_output_size(28, 3, padding=1, stride=2))  # (28+2-3)/2+1 = 14 (halved!)
print(conv_output_size(224, 7, padding=3, stride=2)) # (224+6-7)/2+1 = 112 (ResNet stem)
```

### 2.3 Padding, Stride, and Dilation

```python
# Create a sample 4D input: (batch=1, channels=1, H=8, W=8)
x = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)

# ── PADDING: add zeros around the input ────────────────────────────────────
# padding=0: output is smaller than input (H-k+1 × W-k+1)
# padding=1: "same" padding — output same size as input (for k=3, s=1)
# Purpose: control output size; protect border pixels from being undersampled

conv_no_pad  = nn.Conv2d(1, 1, kernel_size=3, padding=0)  # 8×8 → 6×6
conv_same_pad = nn.Conv2d(1, 1, kernel_size=3, padding=1)  # 8×8 → 8×8

with torch.no_grad():
    print(f"No padding:   {conv_no_pad(x).shape}")   # (1, 1, 6, 6)
    print(f"Same padding: {conv_same_pad(x).shape}") # (1, 1, 8, 8)

# ── STRIDE: step size when sliding the kernel ───────────────────────────────
# stride=1: slide one pixel at a time (default)
# stride=2: slide 2 pixels — halves the spatial size
# Purpose: downsampling (replace pooling, computationally efficient)

conv_stride1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1) # 8×8 → 8×8
conv_stride2 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2) # 8×8 → 4×4

with torch.no_grad():
    print(f"Stride 1: {conv_stride1(x).shape}")  # (1, 1, 8, 8)
    print(f"Stride 2: {conv_stride2(x).shape}")  # (1, 1, 4, 4)

# ── DILATION: spread kernel elements apart ─────────────────────────────────
# dilation=1: standard kernel (elements adjacent)
# dilation=2: kernel elements 2 pixels apart → larger receptive field with same params!
# Purpose: capture multi-scale context without increasing parameters

conv_dilated = nn.Conv2d(1, 1, kernel_size=3, padding=2, dilation=2)
# Effective kernel size = dilation*(k-1)+1 = 2*(3-1)+1 = 5
with torch.no_grad():
    print(f"Dilated:  {conv_dilated(x).shape}")  # (1, 1, 8, 8)
```

---

## Part 3: Building CNN Architectures

### 3.1 A CNN Block — The Standard Pattern

```python
def conv_block(in_channels, out_channels, kernel_size=3, stride=1):
    """
    Standard CNN building block: Conv → BN → ReLU
    
    Note: bias=False when followed by BatchNorm!
    BatchNorm has its own learnable bias (β), making conv bias redundant.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=kernel_size // 2,  # 'same' padding
                  bias=False),               # No bias needed before BN
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),  # inplace=True saves memory
    )
```

### 3.2 LeNet-5 — The Pioneer (1998)

```python
class LeNet5(nn.Module):
    """
    The network that started it all.
    Designed for 32×32 grayscale images (MNIST).
    
    Architecture:
    Input(1,32,32) → Conv(6,5×5) → AvgPool → Conv(16,5×5) → AvgPool → FC(120) → FC(84) → FC(10)
    """
    
    def __init__(self, n_classes=10):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            # Block 1: 1×32×32 → 6×28×28 → 6×14×14
            nn.Conv2d(1, 6, kernel_size=5),  # 32→28 (no padding)
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 28→14
            
            # Block 2: 6×14×14 → 16×10×10 → 16×5×5
            nn.Conv2d(6, 16, kernel_size=5),  # 14→10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 10→5
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),           # 16×5×5 = 400
            nn.Linear(400, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, n_classes),
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

# Test
model = LeNet5(10)
x = torch.randn(32, 1, 32, 32)  # (batch, 1 channel, 32×32)
print(f"Output: {model(x).shape}")  # (32, 10)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")  # ~60K
```

### 3.3 ResNet Block — Modern Deep Learning (2015)

```python
class ResNetBlock(nn.Module):
    """
    The residual block that enabled training 100+ layer networks.
    
    Key innovations:
    1. Skip connection bypasses 2 conv layers
    2. Gradient flows directly through skip connection
    3. If optimal output = identity, model learns F(x) = 0 (easy!)
    
    Architecture:
    x ─────────────────────────────────────> (+) ─> ReLU ─> output
    │                                         ↑
    └> Conv → BN → ReLU → Conv → BN ─────────┘
    
    When stride=2 or channels change, the skip connection needs a 1×1 conv:
    x ──────────────────────────────> Conv(1×1) ─> BN ─> (+) ─> ReLU
    │                                                      ↑
    └> Conv(3×3) → BN → ReLU → Conv(3×3) → BN ───────────┘
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        
        # Projection shortcut: needed when size or channels change
        self.shortcut = nn.Sequential()  # Identity by default
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x):
        identity = self.shortcut(x)  # May be identity or projected
        
        out = self.relu(self.bn1(self.conv1(x)))  # Conv1 → BN → ReLU
        out = self.bn2(self.conv2(out))            # Conv2 → BN
        
        out = out + identity  # Add skip connection (BEFORE final ReLU)
        out = self.relu(out)  # Final ReLU
        
        return out


class ResNet(nn.Module):
    """
    Simplified ResNet for CIFAR-10 (32×32 images).
    Architecture follows ResNet-20.
    """
    
    def __init__(self, n_classes=10):
        super().__init__()
        
        # Stem: initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        
        # Three stages, each halving spatial size
        # Stage 1: 16 channels, 32×32
        self.layer1 = nn.Sequential(
            ResNetBlock(16, 16),
            ResNetBlock(16, 16),
            ResNetBlock(16, 16),
        )
        # Stage 2: 32 channels, 16×16 (stride=2 for first block)
        self.layer2 = nn.Sequential(
            ResNetBlock(16, 32, stride=2),
            ResNetBlock(32, 32),
            ResNetBlock(32, 32),
        )
        # Stage 3: 64 channels, 8×8 (stride=2 for first block)
        self.layer3 = nn.Sequential(
            ResNetBlock(32, 64, stride=2),
            ResNetBlock(64, 64),
            ResNetBlock(64, 64),
        )
        
        # Global average pooling + classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Any spatial size → 1×1
        self.classifier = nn.Linear(64, n_classes)
    
    def forward(self, x):
        x = self.stem(x)        # (B, 3, 32, 32) → (B, 16, 32, 32)
        x = self.layer1(x)      # (B, 16, 32, 32) → (B, 16, 32, 32)
        x = self.layer2(x)      # (B, 16, 32, 32) → (B, 32, 16, 16)
        x = self.layer3(x)      # (B, 32, 16, 16) → (B, 64, 8, 8)
        x = self.pool(x)        # (B, 64, 8, 8) → (B, 64, 1, 1)
        x = x.flatten(1)        # (B, 64, 1, 1) → (B, 64)
        return self.classifier(x)  # (B, 64) → (B, 10)

model = ResNet()
x = torch.randn(32, 3, 32, 32)
print(f"Output: {model(x).shape}")  # (32, 10)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## Part 4: Efficient Convolutions

### 4.1 Depthwise Separable Convolution

Standard conv: each filter sees all input channels simultaneously.
Depthwise separable: split into depthwise (spatial) + pointwise (channel mixing).

```python
# Standard 3×3 conv: in_ch filters applied to all in_ch channels
# Parameters: k*k * in_ch * out_ch = 9 * 64 * 128 = 73,728

# Depthwise separable: two steps
# Step 1 - Depthwise conv: one filter per input channel (spatial only)
# Parameters: k*k * in_ch = 9 * 64 = 576
# Step 2 - Pointwise conv: 1×1 conv mixes channels
# Parameters: in_ch * out_ch = 64 * 128 = 8,192
# Total: 576 + 8,192 = 8,768  ← ~8.4x fewer parameters!

class DepthwiseSeparableConv(nn.Module):
    """
    Used in MobileNet, EfficientNet for mobile-friendly models.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,  # Same in and out channels
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels,  # KEY: one filter per channel
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                    kernel_size=1, bias=False)  # 1×1 conv
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)   # Spatial filtering
        x = self.pointwise(x)   # Channel mixing
        x = self.relu(self.bn(x))
        return x

# Compare parameter counts
standard = nn.Conv2d(64, 128, 3, padding=1)
dw_sep = DepthwiseSeparableConv(64, 128)

std_params = sum(p.numel() for p in standard.parameters())
dws_params = sum(p.numel() for p in dw_sep.parameters())
print(f"Standard conv params:         {std_params:,}")  # 73,728
print(f"Depthwise-separable params:   {dws_params:,}")  # ~8,832
print(f"Reduction: {std_params/dws_params:.1f}x")       # ~8.4×
```

---

## Part 5: Data Augmentation

### 5.1 Why Augmentation?

Augmentation artificially increases training set diversity. If a model should recognize a cat regardless of flip/rotation/brightness, training with these variations improves generalization.

```python
import torchvision.transforms as T

# Training augmentation: aggressive (increases diversity)
train_transform = T.Compose([
    T.RandomCrop(32, padding=4),        # Randomly crop (with padding)
    T.RandomHorizontalFlip(p=0.5),      # 50% chance to flip
    T.ColorJitter(brightness=0.4,       # Random brightness/contrast
                  contrast=0.4,
                  saturation=0.4,
                  hue=0.1),
    T.RandomRotation(degrees=15),        # ±15° rotation
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                std=[0.229, 0.224, 0.225]),
])

# Validation transform: minimal (just normalize)
val_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# Modern augmentation: RandAugment (auto-selects augmentations)
rand_augment = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandAugment(num_ops=2, magnitude=9),  # Randomly pick 2 augmentations
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

### 5.2 Mixup — Training on Blended Images

```python
def mixup_batch(x, y, n_classes, alpha=0.2):
    """
    Mixup: create new samples by linearly interpolating two existing samples.
    
    Both the images AND labels are mixed — this teaches the model to
    predict in-between probabilities, improving calibration.
    
    Args:
        alpha: controls Beta distribution shape. Higher α → more mixing.
    """
    # Sample mixing coefficient from Beta distribution
    # Beta(0.2, 0.2) mostly gives values near 0 or 1, occasionally 0.5
    lam = torch.distributions.Beta(alpha, alpha).sample()
    
    batch_size = x.shape[0]
    
    # Random permutation to get pairs
    idx = torch.randperm(batch_size)
    
    # Mix images: λ * img_a + (1-λ) * img_b
    x_mixed = lam * x + (1 - lam) * x[idx]
    
    # Mix one-hot labels
    y_onehot = torch.zeros(batch_size, n_classes).scatter_(1, y.unsqueeze(1), 1)
    y_onehot_b = y_onehot[idx]
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot_b
    
    return x_mixed, y_mixed

# Training with Mixup
for x_batch, y_batch in train_loader:
    x_mixed, y_mixed = mixup_batch(x_batch, y_batch, n_classes=10)
    logits = model(x_mixed)
    # Use soft cross-entropy (labels are soft now, not one-hot)
    loss = -(y_mixed * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
```

---

## Key Takeaways

| Concept | Why It Matters |
|---------|----------------|
| **Weight Sharing** | Detectors work everywhere; 9 params detect edges in any location |
| **Output Size Formula** | Know your tensor shapes at every layer |
| **Padding='same'** | Preserve spatial dimensions through conv layers |
| **Stride=2** | Efficient downsampling (replaces MaxPool) |
| **Dilation** | Wider receptive field without more parameters |
| **Residual Connections** | Enable very deep networks (100+ layers) |
| **Global Avg Pool** | Fixed-size output regardless of input spatial size |
| **Depthwise Separable** | 8x fewer parameters, similar accuracy |
| **Augmentation** | Critical for generalization on limited data |

---

## Quiz

1. **What is the output size of a Conv2d(3, 64, 3, padding=1) on a 224×224 input?**
   - Answer: 224×224 (same-size padding)

2. **Why is `bias=False` used before BatchNorm?**
   - Answer: BN has learnable β that serves as bias; two biases is redundant

3. **What does `groups=in_channels` in Conv2d do?**
   - Answer: Creates a depthwise convolution (one filter per channel)

4. **What problem do residual connections solve?**
   - Answer: Vanishing gradients in very deep networks; enable training 100+ layers

5. **Why use `AdaptiveAvgPool2d((1,1))` instead of flattening?**
   - Answer: Works for any input spatial size; removes positional dependence

6. **What is Mixup augmentation?**
   - Answer: Linear interpolation of two training images and their labels

7. **What does dilation do to a conv layer?**
   - Answer: Spreads kernel elements apart, increasing receptive field without more parameters

8. **What is the stride=2 equivalent to in older architectures?**
   - Answer: MaxPool2d(2, 2) — both halve spatial dimensions

9. **What is the formula for conv output size?**
   - Answer: (H + 2p - k) / s + 1 (simplified for dilation=1)

10. **Why does training transform differ from validation transform?**
    - Answer: Training uses augmentation for diversity; validation uses minimal transforms for consistent, reproducible evaluation
