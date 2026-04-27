# Module 05: Convolutional Neural Networks

## Learning Objectives
By the end of this module you will be able to:
- Explain convolution mathematically and implement it from scratch
- Build complete CNN architectures: LeNet, VGG, ResNet, EfficientNet-style
- Understand and apply depthwise-separable convolutions, dilated convolutions, and 1×1 convs
- Implement common vision tasks: image classification, object detection (conceptually)
- Use data augmentation strategies for improved generalisation
- Apply batch normalisation correctly within CNN pipelines
- Profile and reduce CNN memory and compute footprint

---

## 5.1 Convolution: Mathematical Foundation

The discrete 2D convolution of input **X** ∈ ℝ^(H×W) with kernel **K** ∈ ℝ^(k×k) is:

```
(X ⋆ K)[i, j] = Σ_m Σ_n X[i+m, j+n] · K[m, n]
```

In deep learning we use **cross-correlation** (same formula, no flip of the kernel), but call it convolution by convention.

**Key parameters:**
- **Kernel size k:** spatial footprint of the filter
- **Stride s:** step between filter applications → H_out = ⌊(H_in − k + 2p) / s⌋ + 1
- **Padding p:** zeros around the border; `padding=(k−1)//2` gives "same" padding
- **Dilation d:** inserts gaps in the kernel; effective kernel size = d(k−1)+1
- **Groups g:** splits channels into g independent groups (g=C_in → depthwise conv)

**Output channel count:**
- Each output channel is produced by one kernel of shape (C_in/g, k, k)
- With `out_channels=C_out` kernels we get `C_out` output channels

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implement 2D cross-correlation manually (for 1 sample, 1 input channel)
def cross_correlate2d(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """X: (H, W), K: (kH, kW) → output: (H-kH+1, W-kW+1)"""
    kH, kW = K.shape
    H, W   = X.shape
    out_H, out_W = H - kH + 1, W - kW + 1
    out = torch.zeros(out_H, out_W)
    for i in range(out_H):
        for j in range(out_W):
            out[i, j] = (X[i:i+kH, j:j+kW] * K).sum()
    return out

# Compare with nn.Conv2d
X = torch.randn(1, 1, 8, 8)
conv = nn.Conv2d(1, 1, kernel_size=3, padding=0, bias=False)
manual = cross_correlate2d(X[0, 0], conv.weight[0, 0])
torch_out = conv(X)[0, 0]
print(torch.allclose(manual, torch_out, atol=1e-5))  # True
```

---

## 5.2 Convolutional Layer Anatomy

```python
# ── Basic Conv block: Conv → BN → ReLU ───────────────────────────────────────
class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        use_bn: bool = True,
        activation: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=not use_bn,
        )
        self.bn  = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

# ── Depthwise-Separable Convolution ──────────────────────────────────────────
# Used in MobileNet: reduces params by ~8–9× for a 3×3 conv
class DepthwiseSeparableConv(nn.Module):
    """
    Params: C_in * 1 * k * k  (depthwise) + C_in * C_out * 1 * 1 (pointwise)
    vs standard: C_in * C_out * k * k
    Reduction factor ≈ 1/C_out + 1/k²
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = ConvBNReLU(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels,
        )
        self.pointwise = ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))

# ── Dilated Convolution ───────────────────────────────────────────────────────
# Expands receptive field without increasing parameters
# Used in segmentation (DeepLab, ASPP)
dilated_conv = nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2)
# Effective kernel size: 2*(3-1)+1 = 5×5 but only 3×3=9 parameters
x = torch.randn(1, 32, 64, 64)
print(dilated_conv(x).shape)   # (1, 32, 64, 64) — same spatial size

# ── 1×1 Convolution (Pointwise) ───────────────────────────────────────────────
# Channel mixing without spatial aggregation
# Used for bottleneck layers, channel adjustment
bottleneck_in  = nn.Conv2d(256, 64, kernel_size=1)   # reduce channels
bottleneck_out = nn.Conv2d(64, 256, kernel_size=1)   # restore channels
```

---

## 5.3 Classic CNN Architectures

### LeNet-5 (1998 — the original)

```python
class LeNet5(nn.Module):
    """
    Original LeNet-5 for MNIST (32×32 grayscale).
    5 layers: 2 conv + 3 FC.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),      # (1,32,32) → (6,28,28)
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2),           # → (6,14,14)
            nn.Conv2d(6, 16, kernel_size=5),     # → (16,10,10)
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2),           # → (16,5,5)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

### VGG Block (2014 — depth beats complexity)

```python
def make_vgg_block(in_channels: int, out_channels: int, n_convs: int) -> nn.Sequential:
    """VGG block: n × (3×3 conv → BN → ReLU) → MaxPool."""
    layers = []
    for _ in range(n_convs):
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        in_channels = out_channels
    layers.append(nn.MaxPool2d(2, stride=2))
    return nn.Sequential(*layers)

class VGG16(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            make_vgg_block(3,   64,  2),   # → (64, 112, 112)
            make_vgg_block(64,  128, 2),   # → (128, 56, 56)
            make_vgg_block(128, 256, 3),   # → (256, 28, 28)
            make_vgg_block(256, 512, 3),   # → (512, 14, 14)
            make_vgg_block(512, 512, 3),   # → (512, 7, 7)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(4096,        4096), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

### ResNet (2015 — skip connections solve depth)

```python
class BasicBlock(nn.Module):
    """ResNet-18/34 block. expansion=1."""
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,      padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act   = nn.ReLU(inplace=True)
        self.skip  = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.skip(x))


class Bottleneck(nn.Module):
    """ResNet-50/101/152 block. expansion=4."""
    expansion = 4

    def __init__(self, in_ch: int, mid_ch: int, stride: int = 1):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch,   mid_ch, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch,  mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch,  out_ch, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)
        self.act   = nn.ReLU(inplace=True)
        self.skip  = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.act(out + self.skip(x))


class ResNet(nn.Module):
    def __init__(self, block, layers: list, num_classes: int = 1000):
        super().__init__()
        self.in_ch = 64
        self.stem  = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * block.expansion, num_classes),
        )
        self._init_weights()

    def _make_layer(self, block, mid_ch: int, n_blocks: int, stride: int):
        layers = [block(self.in_ch, mid_ch, stride=stride)]
        self.in_ch = mid_ch * block.expansion
        for _ in range(1, n_blocks):
            layers.append(block(self.in_ch, mid_ch))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.head(x)

# Factory functions
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

# Verify
model = resnet50()
x = torch.randn(2, 3, 224, 224)
print(model(x).shape)   # (2, 1000)
```

---

## 5.4 Squeeze-and-Excitation (Channel Attention)

```python
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block (Hu et al., 2018).
    'Squeezes' spatial information via GAP, then 'excites'
    channels via an FC bottleneck: channels → r channels → channels → sigmoid.
    Multiplicatively reweights each channel.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.gap(x).view(b, c)  # (B, C)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w                # channel-wise scaling
```

---

## 5.5 Data Augmentation for Vision

```python
import torchvision.transforms as T
import torchvision.transforms.v2 as T2   # modern API

# ── Standard augmentations ────────────────────────────────────────────────────
train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Mixup ─────────────────────────────────────────────────────────────────────
# Interpolates two images and their labels
def mixup_batch(x, y, alpha=0.2):
    lam = torch.distributions.Beta(alpha, alpha).sample()
    idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_mix, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ── CutMix ────────────────────────────────────────────────────────────────────
def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = (1 - lam) ** 0.5
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = torch.randint(W, (1,)).item(), torch.randint(H, (1,)).item()
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2

def cutmix_batch(x, y, alpha=1.0):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0))
    x_mix = x.clone()
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    x_mix[:, :, x1:x2, y1:y2] = x[idx, :, x1:x2, y1:y2]
    lam = 1 - (x2-x1)*(y2-y1) / (x.size(-1)*x.size(-2))  # actual lam
    return x_mix, y, y[idx], lam

# ── RandAugment (automated augmentation) ─────────────────────────────────────
rand_aug = T.RandAugment(num_ops=2, magnitude=9)
```

---

## 5.6 Receptive Field Analysis

The **effective receptive field** determines how large a region of the input influences each output neuron. Critical for understanding CNN capacity.

```
Stack of k × k convolutions:
  After 1 layer: RF = k
  After n layers (stride=1): RF = n*(k-1) + 1
  With stride s: RF grows much faster
```

```python
def compute_receptive_field(layers: list) -> int:
    """
    Compute effective receptive field given a list of (kernel_size, stride, dilation).
    layers: [(k, s, d), ...]
    """
    rf, stride_product = 1, 1
    for k, s, d in layers:
        eff_k = d * (k - 1) + 1           # dilation-adjusted kernel
        rf    = rf + (eff_k - 1) * stride_product
        stride_product *= s
    return rf

# ResNet-50 stem (7×7 s=2 + maxpool 3×3 s=2) then 4 stages
stem_rf = compute_receptive_field([(7,2,1), (3,2,1)])
print(f"After stem: RF = {stem_rf}")   # 11
```

---

## 5.7 CNN Case Study: CIFAR-10 Classification

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# ── Model: a modern small CNN ─────────────────────────────────────────────────
class CIFAR10Net(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3,  32, 3, padding=1),
            ConvBNReLU(32, 32, 3, padding=1),
            nn.MaxPool2d(2),                        # 32→16
            ConvBNReLU(32, 64, 3, padding=1),
            ConvBNReLU(64, 64, 3, padding=1),
            nn.MaxPool2d(2),                        # 16→8
            ConvBNReLU(64, 128, 3, padding=1),
            ConvBNReLU(128, 128, 3, padding=1),
            nn.AdaptiveAvgPool2d((4, 4)),            # 8→4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Training ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_ds = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=transform_train)
test_ds  = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

model = CIFAR10Net().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Expected accuracy: ~85–88% on CIFAR-10 in 100 epochs
```

---

## 5.8 Best Practices for CNNs

| Practice | Reason |
|----------|--------|
| `bias=False` before BatchNorm | BN already has a learnable shift (β) |
| Use Global Average Pooling before FC | Fewer parameters than flattening; spatial invariance |
| Kaiming init for conv weights | Preserves gradient magnitude through ReLU layers |
| Residual connections in deep nets | Prevent vanishing gradients; allow gradient highway |
| BN before activation (original) or after (modern pre-activation ResNets) | Both work; pre-activation is often slightly better |
| Aggressive data augmentation | CNNs overfit image statistics quickly; augmentation is regularisation |
| Mixup/CutMix | +1–2% accuracy on ImageNet classification tasks |

---

## Exercises

**Exercise 5.1** Implement a function `count_flops(model, input_shape)` that counts the multiply-accumulate operations (MACs) for a CNN forward pass.

**Exercise 5.2** Build `MobileNetV1` using `DepthwiseSeparableConv` blocks. Compare its parameter count and inference speed against VGG-16.

**Exercise 5.3** Add `SEBlock` to the `BasicBlock` in ResNet-18 to create `SE-ResNet-18`. Measure the accuracy improvement on CIFAR-10.

---

## Module Summary

| Concept | Key Points |
|---------|-----------|
| Convolution | Cross-correlation; output_size = (in + 2p − k) / s + 1 |
| Depthwise-separable | Factor: 1/C_out + 1/k²; used in MobileNet |
| Dilated | Expands RF without extra params; used in segmentation |
| ResNet | Skip connections solve depth; `BasicBlock` (18/34), `Bottleneck` (50+) |
| SE Block | Channel attention: GAP → FC → sigmoid → scale |
| Data augmentation | Mixup, CutMix, RandAugment are most impactful |
| BN placement | Before ReLU in standard blocks |

---

## Quiz

1. What is the output size of a conv with H=28, k=5, p=0, s=1?
2. How many parameters does a 3×3 depthwise-separable conv have vs standard conv (C=64)?
3. Why does dilation expand the receptive field without adding parameters?
4. What problem do skip connections solve, and how?
5. Why is `nn.AdaptiveAvgPool2d((1,1))` preferred over flattening the full feature map?
6. What is label smoothing and why does it help?
7. Why is `bias=False` common when followed by BatchNorm?

---

*Next: [Module 06 — Transfer Learning & Fine-Tuning](./06_transfer_learning_and_fine_tuning.md)*
