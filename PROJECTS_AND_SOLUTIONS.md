# Capstone Projects & Solutions

This file contains **5 end-to-end capstone projects** covering the full ML lifecycle — from data to deployment. Each project is self-contained with a problem statement, dataset, architecture guide, evaluation criteria, and a complete reference solution.

---

## Project 1: Image Classification — Food Recognition App

**Level:** Intermediate | **Modules:** 05, 06, 09, 13, 14  
**Estimated time:** 5–7 days

### Problem Statement
Build a production-quality food image classifier that recognises 101 food categories (Food-101 dataset). The model must achieve >85% Top-1 accuracy and serve predictions via a REST API with < 50ms p95 latency.

### Dataset
```python
import torchvision.datasets as datasets

# Food-101: 101 classes, 75,750 train / 25,250 test images
train_ds = datasets.Food101(root="./data", split="train", download=True)
test_ds  = datasets.Food101(root="./data", split="test",  download=True)
```

### Requirements
- Transfer learning from EfficientNet-B4 (pretrained on ImageNet)
- Two-phase fine-tuning (head first, then full)
- Data augmentation: RandAugment, Mixup, CutMix
- AMP training (BF16)
- Export to TorchScript + ONNX
- FastAPI serving with preprocessing
- Accuracy ≥ 85%, p95 latency < 50ms on GPU

### Reference Solution

```python
# project1_food_classifier.py
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torch.cuda.amp import GradScaler, autocast
import torch.onnx
import json
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
class Config:
    data_dir        = "./data"
    model_dir       = "./food101_artifacts"
    n_classes       = 101
    batch_size      = 64
    n_epochs_head   = 5
    n_epochs_full   = 25
    lr_head         = 1e-3
    lr_backbone     = 1e-5
    lr_head_full    = 1e-4
    weight_decay    = 0.01
    label_smoothing = 0.1
    mixup_alpha     = 0.2
    device          = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
Path(cfg.model_dir).mkdir(exist_ok=True)

# ── Transforms ─────────────────────────────────────────────────────────────────
train_tf = T.Compose([
    T.RandomResizedCrop(380, scale=(0.4, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandAugment(num_ops=2, magnitude=9),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = T.Compose([
    T.Resize(440), T.CenterCrop(380),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.Food101(cfg.data_dir, split="train", transform=train_tf, download=True)
test_ds  = datasets.Food101(cfg.data_dir, split="test",  transform=val_tf,   download=True)
train_dl = DataLoader(train_ds, cfg.batch_size, shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
test_dl  = DataLoader(test_ds,  cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ── Model ───────────────────────────────────────────────────────────────────────
weights = EfficientNet_B4_Weights.IMAGENET1K_V1
model   = efficientnet_b4(weights=weights)
in_feat = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_feat, cfg.n_classes),
)
model = model.to(cfg.device)

# ── Phase 1: head only ──────────────────────────────────────────────────────────
for p in model.features.parameters():
    p.requires_grad = False

criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
optim1    = torch.optim.AdamW(model.classifier.parameters(), lr=cfg.lr_head)
sched1    = torch.optim.lr_scheduler.CosineAnnealingLR(optim1, T_max=cfg.n_epochs_head)
scaler    = GradScaler()


def train_epoch(model, loader, optim, sched, scaler, mixup=False):
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)
        optim.zero_grad(set_to_none=True)

        if mixup:
            lam = torch.distributions.Beta(cfg.mixup_alpha, cfg.mixup_alpha).sample().item()
            idx = torch.randperm(x.size(0), device=cfg.device)
            x_mix = lam * x + (1 - lam) * x[idx]
            with autocast(device_type="cuda"):
                out  = model(x_mix)
                loss = lam * criterion(out, y) + (1 - lam) * criterion(out, y[idx])
        else:
            with autocast(device_type="cuda"):
                out  = model(x)
                loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim); scaler.update()

        total_loss += loss.item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
        total      += len(y)
    sched.step()
    return total_loss / total, correct / total


@torch.inference_mode()
def eval_epoch(model, loader):
    model.eval()
    top1 = top5 = total = 0
    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)
        out  = model(x)
        top5_preds = out.topk(5, dim=-1).indices
        top1 += (out.argmax(1) == y).sum().item()
        top5 += (top5_preds == y.unsqueeze(1)).any(1).sum().item()
        total += len(y)
    return top1 / total, top5 / total


print("=== Phase 1: Training head ===")
for epoch in range(cfg.n_epochs_head):
    tr_loss, tr_acc = train_epoch(model, train_dl, optim1, sched1, scaler, mixup=False)
    val_top1, val_top5 = eval_epoch(model, test_dl)
    print(f"E{epoch+1:02d}: train_acc={tr_acc:.4f} | val_top1={val_top1:.4f} val_top5={val_top5:.4f}")

# ── Phase 2: full fine-tune ──────────────────────────────────────────────────
for p in model.parameters():
    p.requires_grad = True

optim2 = torch.optim.AdamW([
    {"params": model.features.parameters(), "lr": cfg.lr_backbone},
    {"params": model.classifier.parameters(), "lr": cfg.lr_head_full},
], weight_decay=cfg.weight_decay)
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, T_max=cfg.n_epochs_full)

print("\n=== Phase 2: Full fine-tuning ===")
best_acc = 0.0
for epoch in range(cfg.n_epochs_full):
    tr_loss, tr_acc = train_epoch(model, train_dl, optim2, sched2, scaler, mixup=True)
    val_top1, val_top5 = eval_epoch(model, test_dl)
    print(f"E{epoch+1:02d}: train_acc={tr_acc:.4f} | val_top1={val_top1:.4f} val_top5={val_top5:.4f}")
    if val_top1 > best_acc:
        best_acc = val_top1
        torch.save(model.state_dict(), f"{cfg.model_dir}/best_model.pt")

# ── Export ─────────────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(f"{cfg.model_dir}/best_model.pt"))
model.eval()

# TorchScript
dummy = torch.randn(1, 3, 380, 380).to(cfg.device)
with torch.no_grad():
    traced = torch.jit.trace(model, dummy)
traced.save(f"{cfg.model_dir}/model_traced.pt")

# ONNX
torch.onnx.export(
    model.cpu(), dummy.cpu(), f"{cfg.model_dir}/model.onnx",
    input_names=["image"], output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)

print(f"\nBest Top-1 Accuracy: {best_acc:.4f}")
print(f"Artifacts saved to: {cfg.model_dir}")
```

### Evaluation Criteria

| Criterion | Target |
|-----------|--------|
| Top-1 Accuracy | ≥ 85% |
| Top-5 Accuracy | ≥ 97% |
| p95 Latency (GPU, batch=1) | < 50ms |
| ONNX matches PyTorch | Max diff < 1e-4 |

---

## Project 2: NLP Sentiment Analysis Pipeline

**Level:** Intermediate | **Modules:** 07, 08, 12, 14  
**Estimated time:** 4–6 days

### Problem Statement
Build an end-to-end sentiment analysis system for product reviews. Support: (1) training a BiLSTM baseline, (2) fine-tuning DistilBERT, and (3) comparing both with full evaluation including calibration and interpretability.

### Dataset
```python
from datasets import load_dataset
dataset = load_dataset("amazon_polarity")
# 3.6M train, 400K test; binary sentiment
```

### Reference Solution

```python
# project2_sentiment.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import numpy as np

# ── Part A: Fine-tune DistilBERT ──────────────────────────────────────────────
dataset   = load_dataset("amazon_polarity")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_fn(batch):
    return tokenizer(batch["content"], truncation=True, padding="max_length", max_length=256)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["title", "content"])
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

acc_metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return acc_metric.compute(predictions=np.argmax(logits, -1), references=labels)

args = TrainingArguments(
    output_dir="./sentiment_distilbert",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    evaluation_strategy="steps",
    eval_steps=2000,
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    dataloader_num_workers=4,
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=tokenized["train"].select(range(100_000)),  # 100K for speed
    eval_dataset=tokenized["test"].select(range(10_000)),
    compute_metrics=compute_metrics,
)

trainer.train()

# ── Part B: Evaluate with calibration ────────────────────────────────────────
# Get probabilities on test set
preds_output = trainer.predict(tokenized["test"].select(range(10_000)))
logits = preds_output.predictions
labels = preds_output.label_ids
probs  = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()

# Temperature scaling
model.eval()
temperature = nn.Parameter(torch.ones(1))
optim_t = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

logits_t = torch.from_numpy(logits)
labels_t = torch.from_numpy(labels)

def closure():
    optim_t.zero_grad()
    loss = nn.CrossEntropyLoss()(logits_t / temperature.clamp(0.1), labels_t)
    loss.backward()
    return loss

optim_t.step(closure)
print(f"Calibrated temperature: {temperature.item():.4f}")

# Compare ECE before/after
from sklearn.calibration import calibration_curve
frac_pos_raw, mean_conf_raw = calibration_curve(labels, probs[:, 1], n_bins=10)
probs_cal = torch.softmax(logits_t / temperature.clamp(0.1), -1).detach().numpy()
frac_pos_cal, mean_conf_cal = calibration_curve(labels, probs_cal[:, 1], n_bins=10)

print(f"ECE before: {np.mean(np.abs(frac_pos_raw - mean_conf_raw)):.4f}")
print(f"ECE after:  {np.mean(np.abs(frac_pos_cal - mean_conf_cal)):.4f}")
```

### Evaluation Criteria

| Criterion | Target |
|-----------|--------|
| BiLSTM Test Accuracy | ≥ 88% |
| DistilBERT Test Accuracy | ≥ 93% |
| ECE after calibration | < 0.05 |
| Inference p50 latency (batch=32) | < 20ms |

---

## Project 3: Object Detection — Safety Helmet Detection

**Level:** Advanced | **Modules:** 05, 09, 10, 13, 14  
**Estimated time:** 7–10 days

### Problem Statement
Detect safety helmets (and their absence) in industrial workplace images. Use a pretrained YOLO-style detector fine-tuned on a custom helmet dataset. Deploy with edge-optimised TorchScript.

### Reference Solution Overview

```python
# project3_helmet_detection.py
# Uses torchvision's FasterRCNN or Ultralytics YOLOv8
from ultralytics import YOLO

# Fine-tune YOLOv8 on custom dataset
model = YOLO("yolov8n.pt")   # nano variant for edge
model.train(
    data="helmet_dataset.yaml",  # custom YAML with class names and paths
    epochs=100,
    imgsz=640,
    batch=16,
    device="cuda",
    optimizer="AdamW",
    lr0=1e-3,
    lrf=0.01,
    augment=True,
    cos_lr=True,
    label_smoothing=0.1,
    mixup=0.1,
    copy_paste=0.1,
    project="helmet_runs",
)

# Evaluate
results = model.val()
print(f"mAP@0.5: {results.box.map50:.4f}")
print(f"mAP@0.5:0.95: {results.box.map:.4f}")

# Export to TorchScript
model.export(format="torchscript")

# helmet_dataset.yaml:
# path: ./helmet_data
# train: images/train
# val: images/val
# nc: 2
# names: ['helmet', 'no_helmet']
```

### Evaluation Criteria

| Criterion | Target |
|-----------|--------|
| mAP@0.5 | ≥ 0.80 |
| mAP@0.5:0.95 | ≥ 0.55 |
| FPS on GPU (640px) | ≥ 30 |
| False Negative Rate (missed helmets) | < 10% |

---

## Project 4: Generative Language Model (miniGPT)

**Level:** Advanced | **Modules:** 07, 08, 09, 10, 11  
**Estimated time:** 7–10 days

### Problem Statement
Train a GPT-2-scale (124M parameter) language model on the OpenWebText dataset. Implement BPE tokenization, cosine LR with warmup, gradient accumulation, AMP, and checkpoint resumption. Generate coherent text samples and evaluate with perplexity.

### Reference Solution

```python
# project4_minigpt_train.py
import torch
import torch.nn as nn
from pathlib import Path
import tiktoken

# Use GPT-2 tokenizer from tiktoken
enc = tiktoken.get_encoding("gpt2")

# ── Config ─────────────────────────────────────────────────────────────────────
class GPTConfig:
    block_size  = 1024      # context length
    vocab_size  = 50257     # GPT-2 vocabulary
    n_layer     = 12        # transformer blocks
    n_head      = 12        # attention heads
    n_embd      = 768       # embedding dimension (d_model)
    dropout     = 0.1
    bias        = False     # no bias in attn/ffn (like GPT-2)

class TrainConfig:
    batch_size          = 12
    gradient_accumulate = 40      # effective batch = 480 * 1024 tokens
    max_iters           = 600_000
    lr                  = 6e-4
    min_lr              = 6e-5
    warmup_iters        = 2000
    weight_decay        = 0.1
    beta1, beta2        = 0.9, 0.95
    grad_clip           = 1.0
    device              = "cuda"
    dtype               = torch.bfloat16
    checkpoint_dir      = "gpt2_checkpoints"

# Import MiniGPT from Module 08
# (Adapted to match GPTConfig above)
from module08 import MiniGPT   # your implementation

model = MiniGPT(
    vocab_size=GPTConfig.vocab_size,
    d_model=GPTConfig.n_embd,
    n_heads=GPTConfig.n_head,
    n_layers=GPTConfig.n_layer,
    d_ff=4 * GPTConfig.n_embd,
    max_len=GPTConfig.block_size,
    dropout=GPTConfig.dropout,
).to(TrainConfig.device)

# Weight decay: apply to 2D tensors, not biases/norms
decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW([
    {"params": decay_params,   "weight_decay": TrainConfig.weight_decay},
    {"params": nodecay_params, "weight_decay": 0.0},
], lr=TrainConfig.lr, betas=(TrainConfig.beta1, TrainConfig.beta2), fused=True)

# Compile for maximum throughput
model = torch.compile(model)

scaler = torch.cuda.amp.GradScaler(enabled=(TrainConfig.dtype == torch.float16))

def get_lr(it):
    if it < TrainConfig.warmup_iters:
        return TrainConfig.lr * it / TrainConfig.warmup_iters
    if it > TrainConfig.max_iters:
        return TrainConfig.min_lr
    import math
    progress = (it - TrainConfig.warmup_iters) / (TrainConfig.max_iters - TrainConfig.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return TrainConfig.min_lr + coeff * (TrainConfig.lr - TrainConfig.min_lr)

# Training loop (pseudo-code — real implementation needs DataLoader for OpenWebText)
best_val_loss = float("inf")
for step in range(TrainConfig.max_iters):
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # Gradient accumulation
    for micro_step in range(TrainConfig.gradient_accumulate):
        x, y = get_batch("train")   # your data loading function
        x, y = x.to(TrainConfig.device), y.to(TrainConfig.device)

        with torch.cuda.amp.autocast(dtype=TrainConfig.dtype):
            logits = model(x)
            loss   = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            ) / TrainConfig.gradient_accumulate

        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), TrainConfig.grad_clip)
    scaler.step(optimizer); scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if step % 500 == 0:
        val_loss = estimate_loss(model, "val")
        print(f"Step {step}: val_loss={val_loss:.4f} | perplexity={torch.exp(torch.tensor(val_loss)):.2f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                       f"{TrainConfig.checkpoint_dir}/best.pt")
```

### Evaluation Criteria

| Criterion | Target |
|-----------|--------|
| Validation Perplexity | < 20 |
| Training throughput | > 100K tokens/s on A100 |
| Checkpoint resumes correctly | Yes |
| Generated text is coherent | Human evaluation |

---

## Project 5: End-to-End MLOps — CIFAR-10 Classification System

**Level:** Advanced | **Modules:** 04, 09, 10, 12, 13, 14, 15  
**Estimated time:** 8–12 days

### Problem Statement
Build a complete ML system for CIFAR-10 with automated retraining, model versioning, A/B testing, serving, and monitoring. Implement the full lifecycle: data → train → evaluate → package → serve → monitor.

### System Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Project 5: Full MLOps Stack                               │
│                                                                              │
│  1. Training Pipeline          4. Serving                                    │
│     ├── ConvNeXt-Tiny              ├── FastAPI inference server               │
│     ├── AMP + compile             ├── Dynamic batching                       │
│     ├── EMA weights               ├── Docker container                       │
│     └── Checkpoint + resume       └── Prometheus metrics                     │
│                                                                              │
│  2. Model Optimization         5. Monitoring                                 │
│     ├── INT8 PTQ                   ├── Accuracy tracking                     │
│     ├── Structured pruning         ├── Latency p50/p95/p99                   │
│     └── TorchScript export         ├── Data drift detection                  │
│                                    └── Automatic alerts                      │
│  3. Evaluation                                                                │
│     ├── Top-1/Top-5 accuracy   6. A/B Testing                                │
│     ├── Calibration (ECE)          ├── 10% traffic to new model              │
│     ├── Grad-CAM explanations      ├── Statistical significance test         │
│     └── Fairness audit             └── Auto-promote on improvement           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Reference Solution: Training + Evaluation

```python
# project5_cifar10_system.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import timm
from copy import deepcopy
import json
from pathlib import Path
from datetime import datetime

class MLSystem:
    def __init__(self, experiment_name: str):
        self.exp_name  = experiment_name
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir  = Path(f"experiments/{experiment_name}")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def build_model(self) -> nn.Module:
        model = timm.create_model("convnext_tiny", pretrained=True, num_classes=10)
        return model.to(self.device)

    def build_loaders(self):
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        test_tf = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_ds = torchvision.datasets.CIFAR10("./data", True,  download=True, transform=train_tf)
        test_ds  = torchvision.datasets.CIFAR10("./data", False, download=True, transform=test_tf)
        return (
            DataLoader(train_ds, 128, shuffle=True,  num_workers=4, pin_memory=True, drop_last=True),
            DataLoader(test_ds,  256, shuffle=False, num_workers=4, pin_memory=True),
        )

    def train(self, n_epochs: int = 100):
        model      = self.build_model()
        ema_model  = deepcopy(model).eval()
        train_dl, test_dl = self.build_loaders()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scaler    = GradScaler()

        best_acc = 0.0
        history  = []

        for epoch in range(n_epochs):
            model.train()
            for x, y in train_dl:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type="cuda"):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()

                # EMA update
                with torch.no_grad():
                    for ep, mp in zip(ema_model.parameters(), model.parameters()):
                        ep.data.mul_(0.9999).add_(mp.data, alpha=0.0001)

            scheduler.step()

            # Evaluate EMA model
            ema_model.eval()
            correct = total = 0
            with torch.inference_mode():
                for x, y in test_dl:
                    x, y = x.to(self.device), y.to(self.device)
                    correct += (ema_model(x).argmax(1) == y).sum().item()
                    total   += len(y)
            acc = correct / total

            history.append({"epoch": epoch + 1, "acc": acc})
            print(f"Epoch {epoch+1:3d}: val_acc (EMA) = {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                torch.save(ema_model.state_dict(), self.save_dir / "best_ema.pt")

        # Save history
        with open(self.save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"\nBest EMA accuracy: {best_acc:.4f}")
        return best_acc

    def package(self):
        """Export to TorchScript + ONNX with metadata."""
        model = self.build_model()
        model.load_state_dict(torch.load(self.save_dir / "best_ema.pt"))
        model.eval()

        dummy = torch.randn(1, 3, 32, 32).to(self.device)

        # TorchScript
        with torch.no_grad():
            traced = torch.jit.trace(model, dummy)
        traced.save(str(self.save_dir / "model.pt"))

        # ONNX
        torch.onnx.export(
            model.cpu(), dummy.cpu(), str(self.save_dir / "model.onnx"),
            input_names=["image"], output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )

        # Metadata
        meta = {
            "experiment": self.exp_name,
            "created_at": datetime.now().isoformat(),
            "torch_version": torch.__version__,
            "architecture": "convnext_tiny",
            "n_classes": 10,
            "input_shape": [1, 3, 32, 32],
        }
        with open(self.save_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Model packaged: {self.save_dir}")


if __name__ == "__main__":
    system = MLSystem(f"cifar10_{datetime.now().strftime('%Y%m%d_%H%M')}")
    acc    = system.train(n_epochs=100)
    system.package()
    print(f"Final Top-1: {acc:.4f}")
    # Expected: >93% with ConvNeXt-Tiny on CIFAR-10
```

### Final Evaluation Criteria

| Criterion | Target |
|-----------|--------|
| Test Accuracy (CIFAR-10) | ≥ 93% |
| Latency p95 (batch=1, GPU) | < 10ms |
| INT8 accuracy drop | < 0.5% |
| API responds correctly | Yes |
| Docker image builds | Yes |
| Monitoring metrics exported | Yes |

---

## Scoring Rubric

For each project:

| Dimension | Points |
|-----------|--------|
| Correctness (model trains, evaluation matches targets) | 40 |
| Code quality (clean, modular, well-documented) | 20 |
| Best practices (AMP, EMA, proper eval, no data leakage) | 20 |
| Production readiness (export, serving, Docker) | 20 |
| **Total** | **100** |

---

*Completing all 5 projects = practical mastery of production PyTorch engineering.*
