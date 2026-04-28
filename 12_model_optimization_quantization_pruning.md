# Module 12: Model Optimization — Quantization, Pruning & LoRA

> **Goal:** Learn how to shrink, compress, and adapt large models for deployment and resource-constrained environments — without sacrificing much accuracy.

---

## Learning Objectives

By the end of this module, you will:
- **Understand** quantization mathematically and implement PTQ and QAT
- **Apply** bitsandbytes 4-bit/8-bit quantization for LLMs
- **Use** knowledge distillation to train smaller, faster student models
- **Implement** structured and unstructured pruning
- **Apply** LoRA for efficient LLM fine-tuning with minimal parameters
- **Evaluate** the accuracy-efficiency trade-off for each technique

---

## Part 1: Quantization — Fewer Bits, Same (Almost) Accuracy

### 1.1 What Is Quantization?

Float32 weights use 32 bits per value. **Quantization** maps these to lower-precision integers (INT8, INT4) using a scale factor.

```
FP32 weight: [-1.5,  0.3, -0.7,  2.1]   (4 × 32 bits = 128 bits)
INT8  weight: [-76,   15,  -35,  107]   (4 × 8 bits = 32 bits)

Formula (symmetric):
  scale = max(|x|) / 127.0   = 2.1 / 127 ≈ 0.01654
  x_int = round(x / scale)
  x_deq = x_int * scale      (quantize then dequantize = approximation)

Memory: 4× reduction (FP32 → INT8)
Speed:  2-4× faster on hardware with INT8 tensor cores (NVIDIA)
```

```python
import torch
import torch.nn as nn

def quantize_tensor(x: torch.Tensor, n_bits: int = 8) -> tuple:
    """
    Symmetric quantization to n_bits integers.
    
    Returns: (quantized_int, scale_factor)
    To reconstruct: x_approx = quantized_int * scale
    """
    n_levels = 2 ** (n_bits - 1) - 1  # Max positive value: 127 for INT8
    
    # Scale: maps [-max_val, max_val] → [-n_levels, n_levels]
    max_val = x.abs().max().item()
    scale = max_val / n_levels if max_val > 0 else 1.0
    
    # Quantize: float → int (with rounding)
    x_int = torch.round(x / scale).clamp(-n_levels, n_levels).to(torch.int8)
    
    return x_int, scale

def dequantize_tensor(x_int: torch.Tensor, scale: float) -> torch.Tensor:
    """Reconstruct approximate float from quantized int"""
    return x_int.float() * scale


# Demonstrate quantization error
x = torch.tensor([-1.5, 0.3, -0.7, 2.1, 0.001, -0.002])
x_int, scale = quantize_tensor(x, n_bits=8)
x_reconstructed = dequantize_tensor(x_int, scale)

print(f"Original:      {x}")
print(f"Quantized:     {x_int}")
print(f"Reconstructed: {x_reconstructed}")
print(f"Max error:     {(x - x_reconstructed).abs().max():.6f}")
# Error is small for INT8, larger for INT4
```

### 1.2 Post-Training Quantization (PTQ)

PTQ quantizes a **already-trained** model without any re-training. Fast but less accurate.

```python
import torch.quantization as quant

def apply_ptq(model: nn.Module, calibration_loader) -> nn.Module:
    """
    Post-Training Quantization steps:
    1. Insert fake quantization observers
    2. Run calibration data to determine scale/zero_point ranges
    3. Convert to quantized model
    """
    model.eval()
    
    # Step 1: Set quantization configuration
    # qconfig specifies which observer to use for scale/zero_point calibration
    model.qconfig = quant.get_default_qconfig('fbgemm')  # x86 CPU
    # For mobile: quant.get_default_qconfig('qnnpack')
    
    # Step 2: Fuse Conv+BN+ReLU into single optimized operation
    # This must happen BEFORE quantization
    model_fused = quant.fuse_modules(model, [['conv', 'bn', 'relu']])
    
    # Step 3: Prepare for calibration (inserts observers)
    model_prepared = quant.prepare(model_fused)
    
    # Step 4: Calibration — run representative data to record activation ranges
    # CRITICAL: use representative data, not just 1-2 samples!
    print("Calibrating quantization ranges...")
    with torch.no_grad():
        for i, (x, _) in enumerate(calibration_loader):
            model_prepared(x)
            if i >= 100:  # 100 batches is usually sufficient
                break
    
    # Step 5: Convert observers → actual quantization operations
    model_quantized = quant.convert(model_prepared)
    
    return model_quantized


# Evaluate size and speed reduction
def compare_model_size(fp32_model, int8_model, input_tensor):
    import os
    import time
    
    # Save and compare file sizes
    torch.save(fp32_model.state_dict(), '/tmp/fp32.pt')
    torch.save(int8_model.state_dict(), '/tmp/int8.pt')
    
    fp32_size = os.path.getsize('/tmp/fp32.pt') / 1e6
    int8_size = os.path.getsize('/tmp/int8.pt') / 1e6
    
    print(f"FP32 model: {fp32_size:.1f} MB")
    print(f"INT8 model: {int8_size:.1f} MB")
    print(f"Size reduction: {fp32_size/int8_size:.1f}×")
    
    # Speed comparison (CPU inference)
    fp32_model.eval()
    int8_model.eval()
    
    with torch.no_grad():
        t0 = time.time()
        for _ in range(100):
            _ = fp32_model(input_tensor)
        fp32_time = (time.time() - t0) / 100
        
        t0 = time.time()
        for _ in range(100):
            _ = int8_model(input_tensor)
        int8_time = (time.time() - t0) / 100
    
    print(f"FP32 latency: {fp32_time*1000:.2f} ms")
    print(f"INT8 latency: {int8_time*1000:.2f} ms")
    print(f"Speed: {fp32_time/int8_time:.2f}×")
```

### 1.3 Quantization-Aware Training (QAT)

QAT simulates quantization **during training**, letting the model adapt. More accurate than PTQ.

```python
def apply_qat(model: nn.Module, train_loader, n_epochs=5) -> nn.Module:
    """
    Quantization-Aware Training:
    - Inserts fake quantization nodes (round to n_bit integers, then back to float)
    - Model learns to minimize error introduced by quantization
    - After training, convert to actual quantized model
    
    Key insight: gradients flow through fake quantize nodes using
    Straight-Through Estimator (STE): treat round() as identity during backward
    """
    model.train()
    model.qconfig = quant.get_default_qat_qconfig('fbgemm')
    
    model_fused = quant.fuse_modules(model, [['conv', 'bn', 'relu']])
    model_qat = quant.prepare_qat(model_fused)  # Insert fake quant nodes
    
    optimizer = torch.optim.AdamW(model_qat.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(n_epochs):
        model_qat.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model_qat(x), y)
            loss.backward()  # STE: grad flows through round()
            optimizer.step()
        print(f"QAT Epoch {epoch+1}/{n_epochs}")
    
    model_qat.eval()
    model_quantized = quant.convert(model_qat)
    return model_quantized

# QAT vs PTQ accuracy gap:
# PTQ: -1% to -3% accuracy on ImageNet (ResNet)
# QAT: -0.1% to -0.5% accuracy on ImageNet (much better!)
```

---

## Part 2: 4-bit Quantization for LLMs (bitsandbytes)

### 2.1 Why 4-bit Matters for LLMs

```
LLaMA-7B weights in different formats:
  FP32:  28GB  — won't fit on consumer GPU
  FP16:  14GB  — fits on 16GB GPU, but slow inference
  INT8:   7GB  — fits on 8GB GPU
  INT4:   3.5GB — fits on 4GB GPU!  ← game changer for consumer hardware

Using 4-bit NF4 (NormalFloat4):
  - 4-bit quantization that minimizes error for normally-distributed weights
  - LLM weights ≈ N(0,1) distribution → NF4 is optimal
  - Requires dequantization before computation → compute still in FP16
```

```python
# pip install bitsandbytes transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def load_4bit_llm(model_name: str = 'meta-llama/Llama-2-7b-hf'):
    """
    Load an LLM in 4-bit quantization using bitsandbytes.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                 # Use 4-bit quantization
        bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in BF16 after deq.
        bnb_4bit_use_double_quant=True,    # Double quantization:
                                            # Quantize the scale factors too!
                                            # Saves extra 0.4 bits/param
        bnb_4bit_quant_type='nf4',         # NormalFloat4:
                                            # Optimal 4-bit for normal distributions
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='auto',  # Automatically place layers on available GPUs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Memory after loading
    mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"Model memory: {mem:.2f} GB")
    
    return model, tokenizer
```

---

## Part 3: Knowledge Distillation — Teaching Small Models

### 3.1 The Concept

```
Teacher (large, accurate): ResNet50, BERT-Large
Student (small, fast): ResNet18, DistilBERT

Standard training: student learns from ground-truth labels only
Distillation:      student learns from teacher's SOFT outputs too

Why soft outputs? Teacher's confidence distribution encodes:
  - "This looks mostly like a cat, a bit like a dog, nothing like a car"
  - Much more information than just "cat" (one-hot label)
  
This "dark knowledge" helps the student generalize better.
```

```python
class DistillationLoss(nn.Module):
    """
    Combined loss: hard label cross-entropy + soft KL divergence with teacher.
    
    L = α * L_soft + (1-α) * L_hard
    
    L_soft: KL divergence between teacher and student distributions at temperature T
    L_hard: standard cross-entropy with true labels
    
    Temperature T:
    - T=1: standard distributions
    - T>1: softer distributions (more information about inter-class similarities)
    - T=4-8 is typical; must multiply by T² to maintain gradient magnitude
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha  # Weight for soft loss (0.7 = 70% distillation)
    
    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        student_logits: (batch, n_classes) — student raw outputs
        teacher_logits: (batch, n_classes) — teacher raw outputs (no grad)
        labels:         (batch,) — ground truth class indices
        """
        # ── Soft loss: KL(student || teacher) ────────────────────────────
        # Divide logits by T before softmax → softer probability distributions
        student_soft = nn.functional.log_softmax(student_logits / self.T, dim=-1)
        teacher_soft = nn.functional.softmax(teacher_logits / self.T, dim=-1)
        
        # KL divergence: how far is student from teacher?
        soft_loss = nn.functional.kl_div(
            student_soft, teacher_soft,
            reduction='batchmean'
        )
        # Multiply by T² to normalize gradient magnitude (Hinton et al. 2015)
        soft_loss = soft_loss * (self.T ** 2)
        
        # ── Hard loss: cross-entropy with true labels ─────────────────────
        hard_loss = nn.functional.cross_entropy(student_logits, labels)
        
        # ── Combined loss ─────────────────────────────────────────────────
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


def train_with_distillation(teacher, student, train_loader, device, n_epochs=10):
    """Distillation training loop"""
    teacher.eval()  # Teacher is FROZEN — never trains
    for param in teacher.parameters():
        param.requires_grad = False
    teacher = teacher.to(device)
    
    student.train()
    student = student.to(device)
    
    distill_criterion = DistillationLoss(temperature=4.0, alpha=0.7)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Get teacher predictions (no grad — teacher doesn't train)
            with torch.no_grad():
                teacher_logits = teacher(x)
            
            # Get student predictions
            student_logits = student(x)
            
            # Combined distillation loss
            loss = distill_criterion(student_logits, teacher_logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}")
```

---

## Part 4: Pruning — Removing Unnecessary Weights

### 4.1 Unstructured Pruning

```python
import torch.nn.utils.prune as prune

# Unstructured pruning: zero out individual weights based on magnitude
# Small magnitude weights → likely unimportant
# But: sparse matrices don't speed up on GPUs without specialized kernels!
# Best for: reducing model size (zeros compress well), specialized sparse hardware

model = nn.Linear(100, 50)

# Prune 30% of weights with smallest L1 norm
prune.l1_unstructured(
    module=model,
    name='weight',    # Which parameter to prune
    amount=0.30,      # Fraction to prune
)

# Check sparsity
weight_mask = model.weight_mask  # Binary mask (0=pruned, 1=active)
sparsity = 1.0 - weight_mask.float().mean().item()
print(f"Weight sparsity: {sparsity:.1%}")  # 30%

# Make pruning permanent (remove mask, actually zero out weights)
prune.remove(model, 'weight')
```

### 4.2 Structured Pruning — Hardware Friendly

```python
def structured_channel_pruning(conv_layer: nn.Conv2d, prune_ratio: float = 0.3):
    """
    Structured pruning: remove entire output channels (filters).
    
    Unlike unstructured pruning, this creates a SMALLER dense model
    that runs fast on any hardware — no sparse kernel needed!
    
    Strategy: prune channels with smallest L2 norm (least important)
    """
    weight = conv_layer.weight.data  # (out_ch, in_ch, kH, kW)
    
    # Compute importance score per output channel: L2 norm
    channel_norms = weight.view(weight.shape[0], -1).norm(dim=1)  # (out_ch,)
    
    # Find how many channels to keep
    n_keep = int(weight.shape[0] * (1 - prune_ratio))
    
    # Select indices of top-N most important channels
    _, keep_indices = channel_norms.topk(n_keep)
    keep_indices = keep_indices.sort().values
    
    # Create pruned layer
    new_out_ch = len(keep_indices)
    new_conv = nn.Conv2d(
        conv_layer.in_channels, new_out_ch,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=conv_layer.bias is not None
    )
    
    # Copy kept channel weights
    new_conv.weight.data = weight[keep_indices]
    if conv_layer.bias is not None:
        new_conv.bias.data = conv_layer.bias.data[keep_indices]
    
    print(f"Pruned: {weight.shape[0]} → {new_out_ch} output channels "
          f"({prune_ratio:.0%} removed)")
    return new_conv, keep_indices
```

---

## Part 5: LoRA — Low-Rank Adaptation

### 5.1 The Problem with Full Fine-Tuning

Fine-tuning a 7B LLM:
- **Updates all 7 billion parameters** — requires 100GB+ GPU memory
- **Overwrites pretrained knowledge** — risk of catastrophic forgetting
- **Slow** — 7B backward pass takes significant time

**LoRA's insight:** The change in weights during fine-tuning has **low intrinsic rank**. We don't need to update all W — just a low-rank approximation ΔW.

### 5.2 LoRA Mathematics

```
Instead of: W_new = W_pretrained + ΔW  (both are d×d, millions of params)

LoRA:       W_new = W_pretrained + B @ A
            where A ∈ ℝ^(r×d), B ∈ ℝ^(d×r), r << d

Parameters: 2 × r × d   vs   d × d  for full ΔW
Example: d=4096, r=8:  2×8×4096 = 65,536 vs 4096×4096 = 16,777,216
Reduction: 256× fewer parameters!
```

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation of a linear layer.
    
    Only trains A and B (low-rank matrices).
    Pretrained W is frozen throughout.
    
    Initialization:
    - A: random normal (introduces some signal)
    - B: zeros (so ΔW = B@A = 0 at start → model = pretrained at step 0)
    """
    
    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        
        self.rank = rank
        
        # Scaling factor: alpha/rank
        # alpha controls the magnitude of the LoRA update
        # Setting alpha=2*rank is a safe default
        self.scaling = alpha / rank
        
        # Low-rank matrices (the only trainable parameters!)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))  # Zero init!
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        LoRA output: scaling * (x @ A.T @ B.T)
        This is added to the pretrained layer's output in LoRALinear
        """
        return self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds LoRA adaptation.
    
    W_frozen (pretrained) → fixed, no gradient
    LoRA branch (A, B) → trainable, low-rank update
    
    Output = W_frozen(x) + LoRA(x)
           = W_frozen(x) + scaling * x @ A.T @ B.T
    """
    
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        
        # Copy and freeze the pretrained linear layer
        self.linear = linear
        self.linear.weight.requires_grad = False  # FREEZE
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False  # FREEZE
        
        # Add LoRA branch
        self.lora = LoRALayer(
            linear.in_features, linear.out_features,
            rank=rank, alpha=alpha
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pretrained output (frozen) + LoRA update (trainable)
        return self.linear(x) + self.lora(x)
    
    def merge_weights(self):
        """
        Merge LoRA weights into W for efficient inference.
        After merging: no overhead vs original linear layer!
        
        W_merged = W_frozen + scaling * B @ A
        """
        delta_W = self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
        self.linear.weight.data += delta_W
        # Now the layer behaves identically but without the LoRA branch
        self.lora = None  # Remove LoRA to save memory


def add_lora_to_model(model: nn.Module, rank: int = 8, alpha: float = 16.0,
                       target_modules: list = None) -> nn.Module:
    """
    Add LoRA to specific linear layers in the model.
    
    target_modules: layer name patterns to apply LoRA to
    Common choices: ['q_proj', 'v_proj'] for transformers
                    ['query', 'value'] for BERT
    """
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj']  # Query and Value by default
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(target in name for target in target_modules):
                # Replace with LoRA-wrapped version
                parent_name, child_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, child_name, lora_layer)
                print(f"Added LoRA to: {name}")
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return model


# Using HuggingFace PEFT library (production LoRA)
# pip install peft
try:
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,              # Rank
        lora_alpha=16,    # Alpha (scaling = alpha/r = 2)
        target_modules=['c_attn'],  # Apply to attention projections
        lora_dropout=0.05,
        bias='none',      # Don't train biases
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    # "trainable params: 294,912 || all params: 124,736,512 || trainable%: 0.24%"
    # Only 0.24% of params are trainable — but fine-tuning works!
    
except ImportError:
    print("Install peft: pip install peft")
```

---

## Key Takeaways

| Technique | Params Reduced | Accuracy Loss | Hardware Friendly |
|-----------|---------------|---------------|-------------------|
| **INT8 PTQ** | 4× memory | 0.5–2% | Yes (CPU/GPU INT8) |
| **INT8 QAT** | 4× memory | 0.1–0.5% | Yes |
| **INT4 (bitsandbytes)** | 8× memory | 1–3% | Partial (CPU dequant) |
| **Structured Pruning** | 2–10× compute | 1–5% | Yes (smaller dense model) |
| **Unstructured Pruning** | 4× memory | 0.5–2% | No (needs sparse kernels) |
| **LoRA** | 256× fewer trainable | <0.5% vs full FT | Yes (merged at inference) |

---

## Quiz

1. **What is the difference between PTQ and QAT?**
   - Answer: PTQ quantizes after training (fast, less accurate); QAT simulates quantization during training (slower, more accurate)

2. **Why does QAT use the Straight-Through Estimator?**
   - Answer: The rounding function has zero gradient everywhere; STE treats it as identity during backward so gradients still flow

3. **What is NF4 quantization?**
   - Answer: NormalFloat4 — a 4-bit data type with quantization levels spaced according to normal distribution quantiles, optimal for normally-distributed LLM weights

4. **Why do soft labels in distillation carry more information than hard labels?**
   - Answer: They encode inter-class similarities (teacher's uncertainty); "this looks like 70% cat, 20% dog" vs just "cat"

5. **What is the temperature T's role in distillation?**
   - Answer: Higher T produces softer distributions that reveal more inter-class information; multiply by T² to maintain gradient magnitude

6. **Why is structured pruning more hardware-friendly than unstructured?**
   - Answer: Structured pruning creates smaller dense matrices; unstructured creates sparse irregular patterns that standard hardware can't accelerate

7. **How does LoRA's weight initialization guarantee the model starts identically to pretrained?**
   - Answer: B is initialized to zeros, so ΔW = B@A = 0 at step 0

8. **What is double quantization in bitsandbytes?**
   - Answer: Quantizing the scale factors (quantization constants) themselves, saving an extra ~0.4 bits per parameter

9. **What is the scaling factor in LoRA and why is it important?**
   - Answer: scaling = alpha/rank; it controls the magnitude of the LoRA update, decoupling rank from learning rate sensitivity

10. **How do you merge LoRA weights for efficient inference?**
    - Answer: W_merged = W_frozen + (alpha/rank) × B @ A; after merging, the model runs at full speed with no LoRA overhead
