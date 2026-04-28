# Module 06: Transfer Learning & Fine-Tuning — Standing on Giants' Shoulders

> **Goal:** Understand *why* transfer learning works, when to use which strategy, and how to properly fine-tune pretrained models for your own task.

---

## Learning Objectives

By the end of this module, you will:
- **Understand** why features learned on large datasets transfer to new tasks
- **Implement** three fine-tuning strategies: feature extraction, partial, and full
- **Apply** discriminative learning rates (different LRs per layer group)
- **Use** torchvision and timm for pretrained models
- **Handle** non-standard inputs: different image sizes, grayscale, multi-channel
- **Avoid** catastrophic forgetting with gradual unfreezing

---

## Part 1: Why Transfer Learning Works

### 1.1 The Feature Hierarchy

Deep CNNs learn features in a hierarchical way:

```
Layer depth:    Early (1-3)      Mid (4-7)        Deep (8+)
Feature type:   Edges, Colors    Textures, Parts  Objects, Scenes
Examples:       Horizontal edge  Wheel, Eye        Car, Face
Transferability: HIGH            MEDIUM            LOW (task-specific)
```

**Key insight:** Early layers are universal detectors of low-level patterns (edges, colors, textures) that exist in ANY natural image. Later layers are more task-specific.

Training on ImageNet (1.2M images, 1000 classes) forces a model to learn excellent, reusable visual features.

### 1.2 The Spectrum of Strategies

```
STRATEGY                  WHEN TO USE                        FROZEN LAYERS
─────────────────────────────────────────────────────────────────────────────
Feature Extraction        Small dataset (<1K), similar task  ALL (freeze everything)
Head Fine-tuning          Small dataset (<1K), any task      All except new head
Partial Fine-tuning       Medium dataset, any task           Early layers frozen
Full Fine-tuning          Large dataset (10K+)               None frozen
Domain Adaptation         Very different domain              Careful per-layer
```

---

## Part 2: Feature Extraction

### 2.1 Using the Pretrained Backbone as-is

```python
import torch
import torch.nn as nn
import torchvision.models as models

# Load pretrained ResNet50 (trained on ImageNet)
backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# FREEZE all backbone parameters — they won't be updated
for param in backbone.parameters():
    param.requires_grad = False

# Replace the classifier head with a new one for our task
# ResNet50's fc layer: Linear(2048, 1000)
n_features = backbone.fc.in_features  # 2048
backbone.fc = nn.Linear(n_features, 5)  # Our task: 5 flower categories

# Only the new head has requires_grad=True
trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
total     = sum(p.numel() for p in backbone.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
# Trainable: 10,245 / 25,610,245 (0.04%)

# Pass only the new head's parameters to optimizer
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, backbone.parameters()),
    lr=1e-3
)
```

---

## Part 3: Partial and Full Fine-Tuning

### 3.1 Gradual Unfreezing — The Best Strategy

Unfreeze layers from top (task-specific) to bottom (general), training each stage before the next.

```python
class FineTuner:
    """
    Manages progressive fine-tuning of a pretrained model.
    
    Stage 1: Train only the head (frozen backbone) — fast, safe
    Stage 2: Unfreeze later layers (moderate LR)
    Stage 3: Unfreeze all layers (small LR, discriminative)
    """
    
    def __init__(self, model_name: str = 'resnet50', n_classes: int = 10):
        # Load pretrained model
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Replace head for our task
        n_feat = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_feat, n_classes)
        )
        
        # Group layers for controlled unfreezing
        # ResNet50 has: conv1, bn1, layer1, layer2, layer3, layer4, fc
        self.layer_groups = [
            [self.model.conv1, self.model.bn1],                  # Group 0: stem
            [self.model.layer1],                                   # Group 1: early
            [self.model.layer2],                                   # Group 2: mid-early
            [self.model.layer3],                                   # Group 3: mid-late
            [self.model.layer4],                                   # Group 4: late
            [self.model.fc],                                       # Group 5: head
        ]
    
    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_group(self, group_idx: int):
        """Unfreeze a specific layer group."""
        for module in self.layer_groups[group_idx]:
            for param in module.parameters():
                param.requires_grad = True
    
    def build_optimizer(self, base_lr: float = 1e-4):
        """
        Discriminative learning rates: later layers get HIGHER LR.
        Early layers learned good general features → need small LR
        Later layers are more task-specific → need larger LR
        """
        param_groups = []
        n_groups = len(self.layer_groups)
        
        for i, group_modules in enumerate(self.layer_groups):
            # Exponential scaling: later groups get larger LR
            # Group 0 (earliest): base_lr * 0.1
            # Group 5 (head):     base_lr * 1.0
            lr_scale = 0.1 ** (n_groups - 1 - i)  # 0.1^5, 0.1^4, ..., 0.1^0
            group_lr = base_lr * lr_scale
            
            params = []
            for module in group_modules:
                params.extend(p for p in module.parameters() if p.requires_grad)
            
            if params:
                param_groups.append({'params': params, 'lr': group_lr})
        
        return torch.optim.AdamW(param_groups, weight_decay=0.01)
    
    def stage1_head_only(self):
        """Stage 1: Train only the head"""
        self.freeze_all()
        self.unfreeze_group(5)  # Only head
        return self.build_optimizer(base_lr=1e-3)
    
    def stage2_top_layers(self):
        """Stage 2: Unfreeze last 2 groups"""
        self.freeze_all()
        self.unfreeze_group(4)  # layer4
        self.unfreeze_group(5)  # head
        return self.build_optimizer(base_lr=1e-4)
    
    def stage3_full(self):
        """Stage 3: Full fine-tuning with discriminative LRs"""
        for i in range(len(self.layer_groups)):
            self.unfreeze_group(i)
        return self.build_optimizer(base_lr=1e-5)


# Usage
fine_tuner = FineTuner(n_classes=10)

# Stage 1: 5 epochs — head only
optimizer = fine_tuner.stage1_head_only()
# ... train for 5 epochs ...

# Stage 2: 5 epochs — unfreeze top layers
optimizer = fine_tuner.stage2_top_layers()
# ... train for 5 epochs ...

# Stage 3: 10 epochs — full fine-tune
optimizer = fine_tuner.stage3_full()
# ... train for 10 epochs ...
```

---

## Part 4: Using timm — A Treasure Trove of Models

```python
import timm

# List available models
print(timm.list_models('efficientnet*'))

# Create model with pretrained weights
model = timm.create_model(
    'efficientnet_b4',
    pretrained=True,
    num_classes=10,       # Automatically replaces the head
    drop_rate=0.3,        # Dropout in the classifier
)

# Get model-specific preprocessing config
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=True)
print(data_config)
# {'input_size': (3, 380, 380), 'mean': (0.485, 0.456, 0.406), ...}

# Feature extraction (remove head, get features)
feature_extractor = timm.create_model(
    'efficientnet_b4',
    pretrained=True,
    num_classes=0,   # num_classes=0 removes the head
    global_pool='avg'
)
feature_extractor.eval()
with torch.no_grad():
    x = torch.randn(1, 3, 380, 380)
    feats = feature_extractor(x)
    print(f"Feature shape: {feats.shape}")  # (1, 1792) for EfficientNet-B4
```

---

## Part 5: Handling Non-Standard Inputs

### 5.1 Grayscale Images (1 channel)

```python
# Problem: pretrained models expect 3-channel RGB input
# Solution A: repeat grayscale to 3 channels (simple, works well)
# Solution B: modify first conv layer (fewer redundant parameters)

# Solution A: Expand grayscale to 3 channels
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(2048, n_classes)

class GrayscaleAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # x shape: (B, 1, H, W)
        x = x.repeat(1, 3, 1, 1)  # → (B, 3, H, W)
        return self.model(x)

# Solution B: Replace first conv layer
# Average the 3 RGB weight channels into 1
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
old_conv = model.conv1  # (64, 3, 7, 7)
new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# Initialize: average pretrained weights across RGB dimension
new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
model.conv1 = new_conv
```

---

## Part 6: Catastrophic Forgetting — The Hidden Danger

### 6.1 What Is Catastrophic Forgetting?

When you fine-tune a pretrained model, aggressive training on the new task can **overwrite** the useful pretrained representations. The model "forgets" what it learned on ImageNet.

**Signs:**
- Validation accuracy drops after initially rising
- Training on new task achieves high accuracy but test performance on new samples is poor
- Activations in early layers change dramatically from pretrained values

### 6.2 Prevention Strategies

```python
# Strategy 1: Small learning rate (most important)
# Use 10-100x smaller LR than training from scratch
# Training from scratch: lr=0.1
# Fine-tuning: lr=1e-4 to 1e-5

# Strategy 2: Elastic Weight Consolidation (EWC) — protect important weights
class EWCLoss(nn.Module):
    """
    EWC adds a penalty for weights that change from their pretrained values.
    Important weights (high Fisher information) are penalized more.
    """
    def __init__(self, model: nn.Module, dataloader, lamda: float = 1000):
        super().__init__()
        self.lamda = lamda
        # Save original (pretrained) weights
        self.orig_params = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
        }
        # Compute Fisher information (importance of each weight)
        self.fisher = self._compute_fisher(model, dataloader)
    
    def _compute_fisher(self, model, dataloader):
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        model.eval()
        for x, y in dataloader:
            model.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2 / len(dataloader)
        return fisher
    
    def forward(self, model: nn.Module) -> torch.Tensor:
        """Penalty term: Σ F_i * (θ_i - θ*_i)²"""
        penalty = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                penalty += (self.fisher[n] * (p - self.orig_params[n]) ** 2).sum()
        return self.lamda * penalty / 2
```

---

## Key Takeaways

| Strategy | Dataset Size | Training Time | Expected Accuracy |
|----------|-------------|---------------|-------------------|
| Feature extraction | < 500 | Very fast | Good baseline |
| Head fine-tuning | 500–2,000 | Fast | Good |
| Partial fine-tuning | 2,000–10,000 | Medium | Better |
| Full fine-tuning | 10,000+ | Slow | Best |

---

## Quiz

1. **Which layers of a CNN are most transferable to new tasks?**
   - Answer: Early layers (edges, textures) — they're universal; later layers are task-specific

2. **What is catastrophic forgetting?**
   - Answer: Overwriting useful pretrained representations when fine-tuning aggressively on a new task

3. **What are discriminative learning rates?**
   - Answer: Using different LRs for different layer groups — smaller for early layers, larger for later

4. **How do you freeze all parameters in a model?**
   - Answer: `for param in model.parameters(): param.requires_grad = False`

5. **Why is `num_classes=0` useful in timm?**
   - Answer: Removes the classification head, returning raw feature vectors for custom use

6. **What is the recommended LR for fine-tuning vs training from scratch?**
   - Answer: Fine-tuning: 10–100x smaller (e.g., 1e-4 vs 1e-2 from scratch)

7. **How do you adapt a pretrained RGB model for grayscale inputs?**
   - Answer: Either repeat the single channel 3 times, or average the pretrained 3-channel weights into 1

8. **What does gradual unfreezing mean?**
   - Answer: Unfreeze layers progressively from top to bottom, training after each stage

9. **What is the key advantage of timm over torchvision?**
   - Answer: More models (600+), consistent API, model-specific transforms, up-to-date SOTA

10. **When should you NOT use transfer learning?**
    - Answer: When your domain is very different from ImageNet (e.g., medical scans, satellite imagery) AND you have large amounts of domain-specific data
