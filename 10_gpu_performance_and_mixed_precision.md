# Module 10: GPU Performance & Mixed Precision

## Learning Objectives
By the end of this module you will be able to:
- Profile GPU utilisation and identify performance bottlenecks
- Understand the CUDA memory hierarchy and optimise data transfers
- Apply operator fusion, `torch.compile`, and kernel-level optimisations
- Use `torch.profiler` and NVIDIA Nsight Systems for deep profiling
- Implement efficient DataLoader pipelines to eliminate CPU bottlenecks
- Tune batch sizes, tensor layouts, and memory formats for peak throughput
- Apply BF16 and FP8 training on modern hardware (A100, H100)

---

## 10.1 GPU Architecture Essentials

Understanding GPU hardware prevents performance mistakes:

```
┌─────────────────────────────────────────────────────────┐
│                     GPU (e.g. A100)                      │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │        Streaming Multiprocessors (SMs) × 108    │    │
│  │  ┌──────────────┐  ┌──────────────────────────┐ │    │
│  │  │ CUDA Cores   │  │ Tensor Cores (FP16/BF16) │ │    │
│  │  │ (FP32/INT)   │  │ (matrix multiply units)  │ │    │
│  │  └──────────────┘  └──────────────────────────┘ │    │
│  │         L1 Cache / Shared Memory (192 KB/SM)     │    │
│  └─────────────────────────────────────────────────┘    │
│                  L2 Cache (40 MB)                        │
│                  HBM2e (80 GB, 2 TB/s)                   │
└─────────────────────────────────────────────────────────┘
         ↕ PCIe 4.0 / NVLink (host ↔ GPU)
┌─────────────────────────────────────────────────────────┐
│                   CPU + System RAM                        │
└─────────────────────────────────────────────────────────┘
```

**Key metrics to care about:**
- **Compute throughput** (TFLOPS): how fast the GPU computes
- **Memory bandwidth** (TB/s): how fast data moves from HBM to SMs
- **SM occupancy**: fraction of SMs actively computing (target > 70%)
- **Memory-bound vs compute-bound**: most DL ops are memory-bound

---

## 10.2 Basic Profiling

```python
import torch
import torch.nn as nn
import time

# ── Simple timing with CUDA events (most accurate) ───────────────────────────
def cuda_time(fn, n_warmup=5, n_repeat=20):
    """Measure GPU wall time in milliseconds."""
    for _ in range(n_warmup):
        fn()

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_repeat):
        fn()
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_repeat   # ms per call


device = torch.device("cuda")
model  = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
    num_layers=6,
).to(device).half()   # FP16

x = torch.randn(32, 128, 512, device=device, dtype=torch.float16)
ms = cuda_time(lambda: model(x))
print(f"Forward pass: {ms:.2f} ms")

# ── PyTorch Profiler ─────────────────────────────────────────────────────────
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("model_forward"):
        out = model(x)
    with record_function("loss_backward"):
        out.sum().backward()

# Print top 20 ops by CUDA time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export for Chrome trace viewer (chrome://tracing)
prof.export_chrome_trace("trace.json")
```

---

## 10.3 Memory Management

```python
import torch

# ── Monitor memory ───────────────────────────────────────────────────────────
def print_gpu_memory(msg: str = ""):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved()  / 1e9
    print(f"{msg}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

# ── Memory snapshot for debugging OOM ─────────────────────────────────────────
torch.cuda.memory._record_memory_history(max_entries=100_000)
# ... run your code ...
torch.cuda.memory._dump_snapshot("memory_snapshot.pkl")
# Visualise with: python -m torch.cuda._memory_viz snapshot memory_snapshot.pkl

# ── Avoiding memory leaks ─────────────────────────────────────────────────────

# BAD: storing tensors with grad_fn in a list
losses = []
for batch in loader:
    loss = model(batch).sum()
    losses.append(loss)   # keeps entire computation graph alive!

# GOOD: detach or use .item()
losses = []
for batch in loader:
    loss = model(batch).sum()
    losses.append(loss.item())   # Python float, no graph stored

# ── del + empty_cache ────────────────────────────────────────────────────────
large_tensor = torch.randn(1000, 1000, 1000, device="cuda")
del large_tensor
torch.cuda.empty_cache()    # releases cached but unused memory back to OS
# NOTE: does not free allocated memory, only the reserved pool

# ── Context manager for memory-safe operations ─────────────────────────────
from contextlib import contextmanager

@contextmanager
def gpu_memory_report(label: str = ""):
    torch.cuda.reset_peak_memory_stats()
    yield
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[{label}] Peak GPU memory: {peak:.3f} GB")

with gpu_memory_report("ResNet-50 forward"):
    out = model(torch.randn(64, 3, 224, 224, device="cuda"))
```

---

## 10.4 DataLoader Performance Tuning

The DataLoader is often the bottleneck — the GPU starves while waiting for data.

```python
import torch
from torch.utils.data import DataLoader
import time

# ── Profile your DataLoader ───────────────────────────────────────────────────
def benchmark_loader(loader: DataLoader, n_batches: int = 100) -> float:
    t0 = time.perf_counter()
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
    elapsed = time.perf_counter() - t0
    samples_per_sec = n_batches * loader.batch_size / elapsed
    print(f"DataLoader: {samples_per_sec:.0f} samples/s")
    return samples_per_sec

# ── Optimal DataLoader configuration ─────────────────────────────────────────
optimal_loader = DataLoader(
    dataset,
    batch_size=256,
    num_workers=8,            # rule of thumb: 2–4 × num_GPUs; tune empirically
    pin_memory=True,          # page-locked RAM → faster host→device transfer
    persistent_workers=True,  # keep workers alive between epochs (avoid fork overhead)
    prefetch_factor=4,        # batches prefetched per worker
    drop_last=True,
)

# ── NVIDIA DALI (extremely fast DataLoader for vision) ─────────────────────
# pip install nvidia-dali-cuda120
# DALI runs the entire preprocessing pipeline on GPU:
# decode JPEG → resize → normalize — all in C++/CUDA with zero Python overhead
```

---

## 10.5 Tensor Memory Layout

```python
import torch

# ── channels_last (NHWC) vs channels_first (NCHW) ─────────────────────────
# NHWC is often 2–4× faster on GPU for CNNs because:
# - Better memory access patterns for convolution
# - cuDNN kernels optimised for this layout

model = torch.hub.load("pytorch/vision", "resnet50", pretrained=True).cuda()

# Convert model to channels_last
model = model.to(memory_format=torch.channels_last)

# Input must also be channels_last
x = torch.randn(32, 3, 224, 224, device="cuda")
x = x.to(memory_format=torch.channels_last)

# Now convolutions run in NHWC internally — faster on Ampere+
out = model(x)

# ── Verify memory format ──────────────────────────────────────────────────────
print(x.is_contiguous(memory_format=torch.channels_last))  # True

# ── Contiguous vs strided ─────────────────────────────────────────────────────
t = torch.randn(8, 64, 224, 224, device="cuda")
# Transpose creates non-contiguous tensor
t_T = t.permute(0, 2, 3, 1)
print(t_T.is_contiguous())  # False — bad for most ops

# Make contiguous to avoid implicit copies inside ops
t_contig = t_T.contiguous()
print(t_contig.is_contiguous())  # True
```

---

## 10.6 torch.compile (PyTorch 2.0+)

`torch.compile` uses TorchDynamo + TorchInductor to JIT-compile your model into optimised kernels (fused ops, better memory access, Triton kernels).

```python
import torch
import torch.nn as nn

model = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
    num_layers=6,
).cuda()

# ── Basic compilation ─────────────────────────────────────────────────────────
compiled_model = torch.compile(model)

# ── Compilation modes ─────────────────────────────────────────────────────────
# "default": good balance of speed and compile time (most common)
compiled_model = torch.compile(model, mode="default")

# "reduce-overhead": minimise Python overhead; better for small models
compiled_model = torch.compile(model, mode="reduce-overhead")

# "max-autotune": exhaustive search for best kernels (slow compile, fastest runtime)
compiled_model = torch.compile(model, mode="max-autotune")

# ── Works seamlessly with AMP and DDP ─────────────────────────────────────────
from torch.cuda.amp import autocast

with autocast(device_type="cuda"):
    out = compiled_model(x)   # ← compiled and AMP

# ── Benchmark compile vs eager ────────────────────────────────────────────────
x = torch.randn(32, 128, 512, device="cuda")

eager_ms   = cuda_time(lambda: model(x))
compile_ms = cuda_time(lambda: compiled_model(x))
print(f"Eager: {eager_ms:.1f}ms | Compiled: {compile_ms:.1f}ms | Speedup: {eager_ms/compile_ms:.2f}×")
# Typical speedup: 1.2–2.0× on transformer; 1.5–3× on CNNs
```

---

## 10.7 Operator Fusion

Fusing multiple operations into one kernel reduces memory roundtrips:

```python
import torch
import torch.nn.functional as F

# UNFUSED: multiple CUDA kernel launches, 3 reads + 3 writes to HBM
def unfused_attention(q, k, v):
    scores = q @ k.T
    scores = scores / q.size(-1) ** 0.5
    weights = torch.softmax(scores, dim=-1)
    return weights @ v

# FUSED: one kernel, computed entirely in SRAM
# PyTorch 2.0+ F.scaled_dot_product_attention uses FlashAttention automatically
def fused_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)

# FUSED: Linear + activation
import torch.nn as nn
# torch.compile fuses these automatically:
class FusedMLP(nn.Module):
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)     # compile() fuses the linear + gelu kernel
        return self.fc2(x)
```

---

## 10.8 Mixed Precision: FP16 vs BF16 vs FP8

| Format | Exponent | Mantissa | Range | Precision | GPU Support |
|--------|---------|---------|-------|-----------|-------------|
| FP32 | 8 | 23 | ±3.4e38 | ~7 digits | All |
| FP16 | 5 | 10 | ±65504 | ~3 digits | Pascal+ |
| BF16 | 8 | 7 | ±3.4e38 | ~2 digits | Ampere+ |
| FP8 (E4M3) | 4 | 3 | ±448 | ~1 digit | Hopper (H100) |

```python
import torch
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda")
model = nn.Linear(1024, 1024).to(device)
x = torch.randn(256, 1024, device=device)

# ── FP16: needs GradScaler to prevent gradient underflow ──────────────────────
scaler = GradScaler()
with autocast(device_type="cuda", dtype=torch.float16):
    out = model(x)
loss = out.sum()
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# ── BF16: safer, no scaler needed on A100/H100 ───────────────────────────────
with autocast(device_type="cuda", dtype=torch.bfloat16):
    out = model(x)
loss = out.sum()
loss.backward()   # works directly — no underflow risk with BF16
optimizer.step()

# ── FP8 (Transformer Engine, H100 only) ──────────────────────────────────────
# pip install transformer-engine
import transformer_engine.pytorch as te
# Replace linear layers with FP8-aware equivalents:
fp8_linear = te.Linear(1024, 1024)
with te.fp8_autocast():
    out = fp8_linear(x)   # 2× throughput vs BF16 on H100
```

---

## 10.9 Performance Tuning Checklist

```python
# ── One-time setup at start of training ──────────────────────────────────────
torch.backends.cudnn.benchmark = True   # benchmarks convolution algorithms; best for fixed-size inputs
# WARNING: can slow down runs with variable input sizes

torch.backends.cuda.matmul.allow_tf32 = True    # use TF32 for matmul (Ampere+)
torch.backends.cudnn.allow_tf32 = True          # use TF32 for convolutions

# ── Memory-efficient attention ────────────────────────────────────────────────
torch.backends.cuda.enable_flash_sdp(True)       # Flash Attention
torch.backends.cuda.enable_mem_efficient_sdp(True)  # xFormers mem-efficient

# ── Set seeds deterministically ──────────────────────────────────────────────
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

### GPU Throughput Benchmarking Script

```python
import torch
import torch.nn as nn
import time
from dataclasses import dataclass

@dataclass
class BenchResult:
    throughput_samples_s: float
    peak_memory_gb: float
    avg_step_ms: float

def benchmark_training_step(
    model: nn.Module,
    batch: tuple,
    n_warmup: int = 5,
    n_steps: int = 50,
    use_amp: bool = True,
    use_compile: bool = True,
    device: torch.device = torch.device("cuda"),
) -> BenchResult:
    model = model.to(device)
    if use_compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters())
    scaler    = torch.cuda.amp.GradScaler() if use_amp else None
    criterion = nn.CrossEntropyLoss()

    x, y = [b.to(device) for b in batch]

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out  = model(x)
            loss = criterion(out, y)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()

    # Warmup
    for _ in range(n_warmup): step()
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n_steps): step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return BenchResult(
        throughput_samples_s=n_steps * x.shape[0] / elapsed,
        peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
        avg_step_ms=1000 * elapsed / n_steps,
    )
```

---

## Exercises

**Exercise 10.1** Profile a ResNet-50 training step using `torch.profiler`. Identify the top-3 operations by CUDA time and top-3 by memory usage. Export the Chrome trace.

**Exercise 10.2** Benchmark 4 configurations on your GPU: (FP32 eager) vs (FP16+scaler eager) vs (BF16 eager) vs (BF16+compile). Report throughput, peak memory, and final accuracy after 10 epochs on CIFAR-10.

**Exercise 10.3** Implement a custom `fused_linear_gelu` using a custom autograd function or `torch.compile`. Benchmark against `nn.Linear → F.gelu` with different batch sizes.

---

## Module Summary

| Technique | Impact | Complexity |
|-----------|--------|-----------|
| AMP (BF16) | 2× throughput, 2× memory | Low |
| `torch.compile` | 1.5–3× speedup | Low (one line) |
| `channels_last` | 1.5–2× for CNNs | Low |
| `cudnn.benchmark=True` | 10–30% for fixed shapes | Low |
| Fused attention | 3–5× for long sequences | Low |
| DataLoader tuning | Eliminate CPU bottleneck | Medium |
| Gradient checkpointing | 3–4× more model capacity | Medium |
| Flash Attention | O(T) memory vs O(T²) | Low |

---

## Quiz

1. What is the difference between `memory_allocated()` and `memory_reserved()`?
2. Why is BF16 safer than FP16 for training and which GPUs support it natively?
3. What does `cudnn.benchmark = True` do and when is it counterproductive?
4. What is operator fusion and why does it improve performance?
5. Why is `set_to_none=True` in `optimizer.zero_grad()` faster than the default?
6. What is the memory bandwidth bottleneck and how does channels_last address it?
7. What compilation mode would you choose for maximum inference throughput?

---

*Next: [Module 11 — Distributed Training & Scaling](./11_distributed_training_and_scaling.md)*
