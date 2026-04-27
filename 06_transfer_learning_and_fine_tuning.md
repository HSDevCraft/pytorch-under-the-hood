# Module 06: Transfer Learning & Fine-Tuning

## Learning Objectives
By the end of this module you will be able to:
- Explain why and when transfer learning outperforms training from scratch
- Load and adapt pretrained models from `torchvision` and `timm`
- Apply the three fine-tuning strategies: feature extraction, partial fine-tuning, full fine-tuning
- Implement discriminative learning rates (lower LR for backbone, higher for head)
- Use techniques like linear probing, gradual unfreezing, and layer-wise LR decay
- Adapt pretrained models to custom tasks and non-standard input sizes
- Fine-tune large vision models (ViT, DeiT) for downstream tasks

---

## 6.1 Why Transfer Learning Works

Deep neural networks learn hierarchical representations:

```
Layer 1:  edges, corners, colour gradients
Layer 2:  textures, simple patterns
Layer 3:  object parts (eyes, wheels, fur)
Layer 4+: semantic concepts (faces, cars, animals)
```

These lower-level features are largely **task-agnostic**. A backbone pretrained on ImageNet-1K (1.2M images, 1000 classes) encodes genuinely useful visual priors that benefit almost any vision task.

**When to use transfer learning:**
- Your dataset is small (< 10K samples) — always
- Your dataset is medium (10K–100K) — almost always
- Your dataset is large (> 1M) — still beneficial for faster convergence
- Domain is similar to pretraining domain — use more layers
- Domain is very different — use only early layers

---

## 6.2 Loading Pretrained Models

### From torchvision

```python
import torch
import torch.nn as nn
import torchvision.models as models

# ── Modern API (torchvision >= 0.13): use Weights enum ───────────────────────
from torchvision.models import ResNet50_Weights

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Print default preprocessing required by pretrained weights
transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
print(transform)   # Resize(232) → CenterCrop(224) → Normalize(...)

# Available pretrained models
models.resnet18   (weights="DEFAULT")
models.efficientnet_b0(weights="DEFAULT")
models.vit_b_16   (weights="DEFAULT")
models.convnext_tiny(weights="DEFAULT")
models.swin_t     (weights="DEFAULT")

# ── Legacy API (still works) ──────────────────────────────────────────────────
model_legacy = models.resnet50(pretrained=True)   # shows deprecation warning
```

### From timm (PyTorch Image Models)

```python
# pip install timm
import timm

# List available pretrained models
print(timm.list_models("convnext*", pretrained=True)[:5])

# Load any model with pretrained weights
model = timm.create_model(
    "convnext_base",
    pretrained=True,
    num_classes=0,       # 0 = remove the classification head
)

# Get the model's recommended preprocessing
data_config = timm.data.resolve_model_data_config(model)
transform   = timm.data.create_transform(**data_config, is_training=False)

# Or load with a custom number of classes
model_custom = timm.create_model(
    "efficientnet_b4",
    pretrained=True,
    num_classes=5,       # new head with 5 outputs
)
print(model_custom.get_classifier())   # Linear(1792, 5)
```

---

## 6.3 Strategy 1: Feature Extraction (Frozen Backbone)

Freeze all backbone parameters; only train a new classification head. Best for very small datasets.

```python
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def build_feature_extractor(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """Load ResNet-50, replace the classification head."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the head with a custom classifier
    in_features = model.fc.in_features   # 2048 for ResNet-50
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    # The new head parameters will have requires_grad=True by default

    return model


model = build_feature_extractor(num_classes=10, freeze_backbone=True)

# Verify: only head parameters are trainable
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
# Trainable: ~1.05M / 25.6M (4.1%)

# Only pass the trainable params to the optimizer
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
)
```

---

## 6.4 Strategy 2: Partial Fine-Tuning

Unfreeze the last N layers of the backbone. Good balance between speed and performance.

```python
def build_partial_finetune_model(num_classes: int, unfreeze_from_layer: str = "layer4"):
    """
    Freeze all layers up to (and not including) `unfreeze_from_layer`.
    """
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Step 1: freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: unfreeze from a specific layer onward
    layers = ["layer1", "layer2", "layer3", "layer4", "fc"]
    start_unfreezing = False
    for name, module in model.named_children():
        if name == unfreeze_from_layer:
            start_unfreezing = True
        if start_unfreezing:
            for param in module.parameters():
                param.requires_grad = True

    # Step 3: replace head
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
```

---

## 6.5 Strategy 3: Full Fine-Tuning with Discriminative LR

Train all layers but with **different learning rates** per layer group. Lower LR for early layers (already well-trained), higher for the head (randomly initialised).

```python
def build_discriminative_lr_optimizer(
    model: nn.Module,
    base_lr: float = 1e-4,
    lr_multiplier: float = 10.0,
    weight_decay: float = 1e-2,
) -> torch.optim.Optimizer:
    """
    Layer-wise learning rate: backbone gets base_lr, head gets base_lr * lr_multiplier.
    For ResNet-50: layer groups = [stem, layer1, layer2, layer3, layer4, fc]
    """
    layer_groups = [
        {"params": model.conv1.parameters(),   "lr": base_lr * 0.1},
        {"params": model.layer1.parameters(),  "lr": base_lr * 0.2},
        {"params": model.layer2.parameters(),  "lr": base_lr * 0.4},
        {"params": model.layer3.parameters(),  "lr": base_lr * 0.7},
        {"params": model.layer4.parameters(),  "lr": base_lr * 1.0},
        {"params": model.fc.parameters(),      "lr": base_lr * lr_multiplier},
    ]

    optimizer = torch.optim.AdamW(
        layer_groups,
        lr=base_lr,
        weight_decay=weight_decay,
    )
    return optimizer

# Usage
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, 5)
optimizer = build_discriminative_lr_optimizer(model)
```

---

## 6.6 Gradual Unfreezing

Unfreeze layers progressively during training. Popularised by fast.ai's ULMFiT approach.

```python
class GradualUnfreezeTrainer:
    """
    Gradually unfreeze ResNet layers from top to bottom.
    Phase 0: train only fc
    Phase 1: unfreeze layer4 + fc
    Phase 2: unfreeze layer3 + layer4 + fc
    ...
    """

    def __init__(self, model: nn.Module, base_lr: float = 1e-4):
        self.model   = model
        self.base_lr = base_lr
        self.phases  = ["fc", "layer4", "layer3", "layer2", "layer1"]
        self.current_phase = 0

        # Freeze everything initially
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.base_lr,
        )

    def advance_phase(self):
        """Unfreeze the next layer group."""
        if self.current_phase >= len(self.phases):
            return

        self.current_phase += 1
        layer_name = self.phases[self.current_phase - 1]
        layer = getattr(self.model, layer_name)
        for param in layer.parameters():
            param.requires_grad = True

        # Reinitialize optimizer with new param groups
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.base_lr / (2 ** (self.current_phase - 1)),  # decay lr for older layers
        )
        print(f"Unfroze {layer_name}. Trainable params: "
              f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
```

---

## 6.7 Fine-Tuning Vision Transformers (ViT)

```python
import timm
import torch
import torch.nn as nn

def finetune_vit(
    model_name: str = "vit_base_patch16_224",
    num_classes: int = 10,
    strategy: str = "full",   # 'head_only', 'last_n_blocks', 'full'
    n_blocks_to_unfreeze: int = 4,
) -> nn.Module:

    model = timm.create_model(model_name, pretrained=True)

    if strategy == "head_only":
        for param in model.parameters():
            param.requires_grad = False
        # unfreeze norm and head
        for param in model.norm.parameters():
            param.requires_grad = True

    elif strategy == "last_n_blocks":
        for param in model.parameters():
            param.requires_grad = False
        # unfreeze the last N transformer blocks
        total_blocks = len(model.blocks)
        for i in range(total_blocks - n_blocks_to_unfreeze, total_blocks):
            for param in model.blocks[i].parameters():
                param.requires_grad = True
        for param in model.norm.parameters():
            param.requires_grad = True

    # Replace classifier head
    model.head = nn.Linear(model.head.in_features, num_classes)

    return model


# Recommended: linear probing first, then full fine-tune
vit = finetune_vit("vit_base_patch16_224", num_classes=10, strategy="head_only")

# Stage 1: linear probing (10 epochs)
optim_probe = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, vit.parameters()),
    lr=1e-3,
)

# Stage 2: full fine-tune at lower LR
for param in vit.parameters():
    param.requires_grad = True
optim_full = torch.optim.AdamW(vit.parameters(), lr=2e-5, weight_decay=0.05)
```

---

## 6.8 Domain Adaptation: Non-Standard Inputs

```python
# ── 1-channel (grayscale) input with ImageNet pretrained model ───────────────
def adapt_to_grayscale(model: nn.Module) -> nn.Module:
    """
    Modify the first conv layer of a model to accept 1-channel input
    by averaging the pretrained 3-channel weights.
    """
    first_conv = model.conv1   # or model.features[0] for VGG
    new_conv = nn.Conv2d(
        1,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )
    # Average weights across the 3 input channels
    new_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)
    model.conv1 = new_conv
    return model

# ── Multi-spectral (N channels) input ─────────────────────────────────────────
def adapt_to_n_channels(model: nn.Module, n_channels: int) -> nn.Module:
    """Replicate 3-channel weights for N-channel input."""
    first_conv = model.conv1
    new_weight = first_conv.weight.data.repeat(1, (n_channels + 2) // 3, 1, 1)
    new_weight = new_weight[:, :n_channels, :, :]  # trim to exactly n_channels
    new_conv = nn.Conv2d(
        n_channels, first_conv.out_channels,
        kernel_size=first_conv.kernel_size, stride=first_conv.stride,
        padding=first_conv.padding, bias=False,
    )
    new_conv.weight.data = new_weight / (n_channels / 3.0)  # scale to preserve magnitude
    model.conv1 = new_conv
    return model
```

---

## 6.9 Complete Fine-Tuning Pipeline

```python
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def fine_tune_on_custom_dataset(
    data_dir: str,
    num_classes: int,
    epochs_head: int = 5,
    epochs_full: int = 20,
    batch_size: int = 32,
    device: torch.device = torch.device("cuda"),
) -> nn.Module:
    """
    Two-phase fine-tuning:
    1. Train only the new head (fast, avoids corrupting backbone)
    2. Fine-tune the whole model with low LR
    """
    # ── Model ───────────────────────────────────────────────────────────────
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes),
    )
    model = model.to(device)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_tf = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
    # Override with augmentations for training
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()

    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_tf)
    val_ds   = datasets.ImageFolder(f"{data_dir}/val",   transform=val_tf)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Phase 1: head only ───────────────────────────────────────────────────
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_head)
    _run_epochs(model, train_dl, val_dl, optimizer, criterion, scheduler, device, epochs_head, "Phase 1")

    # ── Phase 2: full fine-tune ──────────────────────────────────────────────
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 1e-4},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_full)
    _run_epochs(model, train_dl, val_dl, optimizer, criterion, scheduler, device, epochs_full, "Phase 2")

    return model


def _run_epochs(model, train_dl, val_dl, optimizer, criterion, scheduler, device, epochs, tag):
    for epoch in range(epochs):
        model.train()
        loss_sum = correct = total = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct  += (out.argmax(1) == y).sum().item()
            total    += len(y)
        scheduler.step()

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                val_correct += (preds == y).sum().item()
                val_total   += len(y)

        print(f"[{tag}] Epoch {epoch+1}/{epochs} | "
              f"train_acc={correct/total:.4f} | val_acc={val_correct/val_total:.4f}")
```

---

## Exercises

**Exercise 6.1** Fine-tune `EfficientNet-B2` on the Oxford Flowers 102 dataset (102 classes, 8K images). Report top-1 accuracy using each of the three strategies. Compare to training from scratch.

**Exercise 6.2** Implement `LayerWiseLRDecay`: a function that takes a model and assigns exponentially decreasing learning rates to each layer from top to bottom (LR × decay_factor per layer).

**Exercise 6.3** Adapt a `ResNet-18` to accept 6-channel satellite imagery (RGB + NIR + SWIR + thermal). Fine-tune on a custom satellite scene classification task.

---

## Module Summary

| Strategy | When to Use | Typical Result |
|----------|-------------|---------------|
| Feature extraction (frozen) | Very small dataset, similar domain | Fast, ~85% of full fine-tune |
| Partial fine-tune (last N layers) | Small-medium dataset | 95% of full fine-tune |
| Full fine-tune with disc. LR | Medium-large dataset | Best performance |
| Gradual unfreezing | Any size; avoids catastrophic forgetting | Stable training |
| Linear probe → full fine-tune | Transformer models (ViT) | Best for large models |

---

## Quiz

1. What is catastrophic forgetting and how does gradual unfreezing help?
2. Why is it recommended to train only the head first before fine-tuning the backbone?
3. What is the purpose of discriminative learning rates?
4. How do you adapt a pretrained ImageNet model to a 1-channel input?
5. Why does `timm.create_model(..., num_classes=0)` remove the head?
6. What is linear probing and when is it used?

---

*Next: [Module 07 — Recurrent Networks & Sequence Models](./07_recurrent_networks_and_sequences.md)*
