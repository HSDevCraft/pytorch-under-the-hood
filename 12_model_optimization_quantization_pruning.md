# Module 12: Model Optimization — Quantization & Pruning

## Learning Objectives
By the end of this module you will be able to:
- Explain the mathematics of quantization and apply PTQ and QAT in PyTorch
- Use `torch.ao.quantization` for INT8 quantization of CNNs
- Apply `bitsandbytes` for LLM 4-bit and 8-bit quantization (NF4, GPTQ)
- Implement magnitude-based, structured, and unstructured pruning
- Apply knowledge distillation to train compact student models
- Measure and validate the accuracy-efficiency trade-off
- Deploy quantized models and measure real-world latency improvements

---

## 12.1 Why Model Optimization?

| Technique | Size Reduction | Latency Reduction | Accuracy Cost |
|-----------|---------------|------------------|--------------|
| FP16 quantization | 2× | 1.5–2× | ~0% |
| INT8 quantization | 4× | 2–4× | 0.5–1% |
| INT4 quantization | 8× | 3–6× | 1–3% |
| Magnitude pruning 50% | 2× (compressed) | 1.1–1.5× | 0.5–2% |
| Structured pruning 50% | 2× | 1.5–2× | 1–4% |
| Knowledge distillation | 10–100× | 10–100× | 2–8% |

---

## 12.2 Quantization Fundamentals

Quantization maps floating-point values to integers:

```
x_q = round(x / scale) + zero_point          ← quantize
x   = (x_q - zero_point) × scale             ← dequantize
```

**Affine (asymmetric) INT8:**
- scale = (x_max - x_min) / (2^8 - 1)
- zero_point = round(-x_min / scale) ∈ [0, 255]

**Symmetric INT8:**
- scale = max(|x_max|, |x_min|) / 127
- zero_point = 0

```python
import torch
import torch.nn as nn

def quantize_tensor(x: torch.Tensor, n_bits: int = 8, symmetric: bool = False) -> tuple:
    """Manual quantization implementation for illustration."""
    if symmetric:
        abs_max = x.abs().max()
        scale = abs_max / (2 ** (n_bits - 1) - 1)
        zero_point = 0
        q_min, q_max = -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1
    else:
        q_min = 0
        q_max = 2 ** n_bits - 1
        scale      = (x.max() - x.min()) / (q_max - q_min)
        zero_point = round(q_min - x.min() / scale)

    x_q = torch.clamp(torch.round(x / scale) + zero_point, q_min, q_max).to(torch.int8)
    return x_q, scale, zero_point

def dequantize_tensor(x_q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    return (x_q.float() - zero_point) * scale

# Verify round-trip error
x     = torch.randn(1000) * 5
x_q, scale, zp = quantize_tensor(x, n_bits=8)
x_rec = dequantize_tensor(x_q, scale, zp)
print(f"Max quantization error: {(x - x_rec).abs().max():.4f}")
print(f"SNR: {10 * torch.log10((x**2).mean() / ((x-x_rec)**2).mean()):.1f} dB")
```

---

## 12.3 Post-Training Quantization (PTQ)

Quantize a pretrained FP32 model without retraining. Requires a small calibration dataset to determine scale/zero_point.

```python
import torch
import torch.nn as nn
import torch.ao.quantization as quant

# ── Step 1: Prepare the model ──────────────────────────────────────────────
class QuantizableResNet(nn.Module):
    """Add QuantStub/DeQuantStub at model boundaries."""

    def __init__(self):
        super().__init__()
        self.quant   = quant.QuantStub()      # marks the start of quantization
        self.dequant = quant.DeQuantStub()    # marks the end
        self.model   = resnet18()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        return self.dequant(x)

    def fuse_model(self):
        """Fuse Conv+BN+ReLU for better quantization quality."""
        torch.ao.quantization.fuse_modules(self.model, [
            ["conv1", "bn1", "relu"],
            # ... add all Conv+BN+ReLU patterns
        ], inplace=True)


# ── Step 2: Configure quantization ────────────────────────────────────────
model_fp32 = QuantizableResNet()
model_fp32.eval()
model_fp32.fuse_model()

# Use x86 config for CPU inference
model_fp32.qconfig = quant.get_default_qconfig("x86")

# ── Step 3: Insert observers ──────────────────────────────────────────────
quant.prepare(model_fp32, inplace=True)

# ── Step 4: Calibrate with representative data ────────────────────────────
calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)
model_fp32.eval()
with torch.no_grad():
    for x, _ in calib_loader:
        model_fp32(x)   # observers collect statistics

# ── Step 5: Convert to INT8 ───────────────────────────────────────────────
model_int8 = quant.convert(model_fp32)

# ── Verify accuracy and latency ──────────────────────────────────────────
print(f"FP32 size: {get_model_size(model_fp32):.1f} MB")
print(f"INT8 size: {get_model_size(model_int8):.1f} MB")

def get_model_size(model: nn.Module) -> float:
    """Model size in MB."""
    torch.save(model.state_dict(), "/tmp/tmp.pt")
    import os
    size = os.path.getsize("/tmp/tmp.pt") / 1e6
    os.remove("/tmp/tmp.pt")
    return size
```

---

## 12.4 Quantization-Aware Training (QAT)

QAT simulates quantization during forward passes so the model learns to be robust to quantization errors. Usually gives 1–2% accuracy back vs PTQ.

```python
import torch.ao.quantization as quant

def train_qat(model: nn.Module, train_loader, val_loader, n_epochs: int = 5):
    model.eval()
    model.fuse_model()
    model.qconfig = quant.get_default_qat_qconfig("x86")

    # Insert fake-quantization nodes (simulate INT8 during FP32 training)
    quant.prepare_qat(model, inplace=True)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        # Disable observer updates after epoch 2 to lock scale/zero_point
        if epoch == 2:
            model.apply(quant.disable_observer)
        # Disable BatchNorm tracking after epoch 4
        if epoch == 4:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        val_acc = evaluate(model, val_loader)
        print(f"QAT Epoch {epoch}: val_acc={val_acc:.4f}")

    # Convert to real INT8 for deployment
    model.eval()
    model_int8 = quant.convert(model)
    return model_int8
```

---

## 12.5 LLM Quantization with bitsandbytes

For large language models, standard PTQ is insufficient. The following techniques dominate:

```python
# pip install bitsandbytes transformers accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# ── 8-bit quantization (LLM.int8()) ───────────────────────────────────────────
bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,   # outlier threshold; above → kept in FP16
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config_8bit,
    device_map="auto",
)

# ── 4-bit quantization (NF4 + double quantization) ───────────────────────────
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",        # NormalFloat4 — best for normally distributed weights
    bnb_4bit_use_double_quant=True,   # quantize the scale factors too
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config_4bit,
    device_map="auto",
)

# Memory: Llama-2-7B in BF16 ≈ 14 GB; in 4-bit NF4 ≈ 3.5 GB

# ── QLoRA: fine-tune a 4-bit quantized model ─────────────────────────────────
# pip install peft
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_4bit = prepare_model_for_kbit_training(model_4bit)

lora_config = LoraConfig(
    r=16,            # rank of low-rank matrices
    lora_alpha=32,   # scaling factor
    target_modules=["q_proj", "v_proj"],  # which layers to apply LoRA
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model_qlora = get_peft_model(model_4bit, lora_config)
model_qlora.print_trainable_parameters()
# trainable params: ~4M (0.06% of 7B) — only LoRA adapters trained
```

---

## 12.6 Knowledge Distillation

Train a small **student** model to mimic a large **teacher** model's soft predictions (logit distributions), not just the hard labels.

```
Loss = (1 - α) · CE(student_logits, hard_labels)
     + α · KL(soft_student, soft_teacher) · T²
```

where T (temperature) softens the distributions, revealing the teacher's dark knowledge (relative confidence between classes).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """
    Standard knowledge distillation loss (Hinton et al., 2015).
    Combines hard-label CE with soft-label KL divergence.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha
        self.ce    = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        # Soft targets from teacher (at temperature T)
        soft_teacher = F.softmax(teacher_logits / self.T, dim=-1)
        soft_student = F.log_softmax(student_logits / self.T, dim=-1)

        # KL divergence: D_KL(soft_teacher || soft_student)
        # Scaled by T² to compensate for the T scaling in gradients
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.T ** 2)

        # Hard label cross-entropy
        ce_loss = self.ce(student_logits, targets)

        total = (1 - self.alpha) * ce_loss + self.alpha * kl_loss
        return {"total": total, "ce": ce_loss, "kl": kl_loss}


def train_distillation(
    teacher: nn.Module,
    student: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 4.0,
    alpha: float = 0.7,
    n_epochs: int = 50,
) -> nn.Module:
    criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    teacher.eval()   # freeze teacher

    for epoch in range(n_epochs):
        student.train()
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                teacher_logits = teacher(x)    # teacher doesn't update

            student_logits = student(x)
            losses = criterion(student_logits, teacher_logits, y)

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            total_loss += losses["total"].item()

        print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")

    return student


# Example: ResNet-50 (teacher) → ResNet-18 (student)
teacher = resnet50(pretrained=True).eval().to(device)
student = resnet18().to(device)
optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
student = train_distillation(teacher, student, train_dl, optimizer, device)
```

---

## 12.7 Pruning

Pruning removes weights with small magnitude to create sparse (or smaller) models.

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# ── Unstructured pruning: remove individual weights ───────────────────────────
model = resnet18()

# Prune 40% of weights in a specific layer
prune.l1_unstructured(model.layer4[0].conv1, name="weight", amount=0.4)

# Verify sparsity
total = model.layer4[0].conv1.weight.numel()
zeros = (model.layer4[0].conv1.weight == 0).sum().item()
print(f"Sparsity: {100 * zeros / total:.1f}%")

# Remove the pruning reparameterization (make it permanent)
prune.remove(model.layer4[0].conv1, "weight")

# ── Global pruning: prune globally across all layers ─────────────────────────
parameters_to_prune = [
    (module, "weight")
    for module in model.modules()
    if isinstance(module, (nn.Conv2d, nn.Linear))
]

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,   # remove 30% of all weights globally
)

total = sum(p.numel() for n, p in model.named_parameters() if "weight" in n)
zeros = sum((p == 0).sum().item() for n, p in model.named_parameters() if "weight" in n)
print(f"Global sparsity: {100 * zeros / total:.1f}%")

# ── Structured pruning: remove entire filters ─────────────────────────────────
# Removing a filter reduces the output channels → actually smaller model
def structured_prune_conv(
    module: nn.Conv2d, amount: float = 0.3
) -> nn.Conv2d:
    """
    Remove the `amount` fraction of output filters with smallest L2 norm.
    Returns a new, smaller Conv2d layer.
    """
    weights = module.weight.data   # (C_out, C_in, kH, kW)
    norms   = weights.view(weights.size(0), -1).norm(dim=1)   # (C_out,)
    n_keep  = max(1, int((1 - amount) * weights.size(0)))
    _, keep_idx = norms.topk(n_keep, largest=True)
    keep_idx    = keep_idx.sort().values

    new_conv = nn.Conv2d(
        module.in_channels, n_keep, module.kernel_size,
        stride=module.stride, padding=module.padding,
        groups=module.groups, bias=module.bias is not None,
    )
    new_conv.weight.data = weights[keep_idx]
    if module.bias is not None:
        new_conv.bias.data = module.bias.data[keep_idx]

    return new_conv, keep_idx
```

---

## 12.8 LoRA: Low-Rank Adaptation

LoRA adds trainable low-rank matrices to frozen pretrained weights, enabling efficient fine-tuning:

```
W' = W₀ + ΔW = W₀ + B·A
where A ∈ ℝ^(r×d_in), B ∈ ℝ^(d_out×r), rank r << min(d_in, d_out)
```

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    Freezes W₀ and trains only A, B (much fewer params).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        pretrained_weight: torch.Tensor = None,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank         = rank
        self.scaling      = lora_alpha / rank

        # Frozen pretrained weights
        self.weight = nn.Parameter(
            pretrained_weight if pretrained_weight is not None
            else torch.randn(out_features, in_features),
            requires_grad=False,
        )

        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Init: A ~ N(0, 1), B = 0 (so initial ΔW = 0)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = x @ self.weight.T
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * lora_out

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights into a single Linear for inference."""
        merged_weight = self.weight + self.scaling * (self.lora_B @ self.lora_A)
        linear = nn.Linear(self.in_features, self.out_features, bias=False)
        linear.weight.data = merged_weight
        return linear


def add_lora_to_model(model: nn.Module, rank: int = 8, target_modules: list = None) -> nn.Module:
    """Replace Linear layers in `target_modules` with LoRALinear."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    for name, module in model.named_modules():
        if any(t in name for t in target_modules) and isinstance(module, nn.Linear):
            parent_name, child_name = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parent_name]
            lora = LoRALinear(
                module.in_features, module.out_features, rank=rank,
                pretrained_weight=module.weight.data.clone(),
            )
            setattr(parent, child_name, lora)

    return model
```

---

## 12.9 Accuracy-Efficiency Trade-off Evaluation

```python
import torch
import time

def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_repeats: int = 100,
) -> dict:
    model.eval()
    model = model.to(device)

    # Accuracy
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(-1)
            correct += (preds == y).sum().item()
            total += len(y)

    # Latency (single sample)
    x_single = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(10): model(x_single)   # warmup
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        model(x_single)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = 1000 * (time.perf_counter() - t0) / n_repeats

    return {
        "accuracy":    correct / total,
        "latency_ms":  latency_ms,
        "size_mb":     get_model_size(model),
        "n_params":    sum(p.numel() for p in model.parameters()),
    }
```

---

## Exercises

**Exercise 12.1** Apply INT8 PTQ to a ResNet-18 trained on CIFAR-10. Measure the accuracy drop, latency improvement (CPU), and model size reduction. Calibrate with 100 training samples.

**Exercise 12.2** Implement magnitude-based iterative pruning: prune 10% of weights every 5 epochs, then fine-tune for 5 epochs. Compare accuracy vs one-shot 50% pruning.

**Exercise 12.3** Use QLoRA to fine-tune `google/flan-t5-base` on a custom QA dataset (SQuAD subset). Report: number of trainable parameters, VRAM usage, exact match score.

---

## Module Summary

| Technique | Params Reduced | Latency | Accuracy | Retraining |
|-----------|---------------|---------|----------|-----------|
| PTQ INT8 | 4× | 2–3× | −0.5–1% | No |
| QAT INT8 | 4× | 2–3× | ~0% | Yes |
| NF4 4-bit | 8× | 3–4× | −1–2% | No |
| Magnitude pruning | Variable | ~1× (sparse) | −1–3% | Fine-tune |
| Structured pruning | 2–5× | 1.5–3× | −2–5% | Fine-tune |
| Knowledge distillation | 10–100× | 10–100× | −2–8% | Full train |
| LoRA | 0.01–1% params | Same | ~0% | Partial |

---

## Quiz

1. What is the difference between symmetric and asymmetric quantization?
2. Why does QAT outperform PTQ at the same bit-width?
3. What is the temperature T in knowledge distillation and what does a high T do?
4. Why is unstructured pruning hard to speed up on real hardware?
5. What is the NF4 quantization type and why is it preferred for LLM weights?
6. Why does LoRA initialize matrix B to zero?
7. What is double quantization in bitsandbytes?

---

*Next: [Module 13 — Deployment: TorchScript & ONNX](./13_deployment_torchscript_onnx.md)*
