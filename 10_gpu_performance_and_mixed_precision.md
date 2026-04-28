# Module 10: GPU Performance & Mixed Precision — Squeezing Every FLOP

> **Goal:** Understand how GPU hardware works, how PyTorch uses it, and how to systematically profile and optimize training and inference for maximum throughput.

---

## Learning Objectives

By the end of this module, you will:
- **Understand** GPU architecture: CUDA cores, Tensor Cores, memory hierarchy
- **Profile** PyTorch code to find real bottlenecks
- **Optimize** DataLoader, memory layout, and operator fusion
- **Use** `torch.compile` for automatic kernel optimization
- **Master** FP16, BF16, and FP8 mixed precision formats
- **Implement** all techniques to achieve 3-5× training speedup

---

## Part 1: GPU Architecture Fundamentals

### 1.1 Why GPUs Are Fast for Deep Learning

A CPU has 8–64 cores optimized for **serial, low-latency** tasks.
A GPU has thousands of cores optimized for **parallel, high-throughput** tasks.

```
CPU (Intel i9):                GPU (A100):
- 24 cores                     - 6,912 CUDA cores
- 5 GHz clock                  - 1.41 GHz clock
- Optimized for latency        - Optimized for throughput
- Complex branch prediction    - Simple, repetitive operations
- Large cache per core         - Small cache, massive bandwidth

Matrix multiply on CPU:  ~77 TFLOPS (FP32)
Matrix multiply on A100: ~312 TFLOPS (FP16 Tensor Cores)
```

### 1.2 Memory Hierarchy

```
Register file    : ~4MB,  clock speed (~0 latency)
L1 cache/Shared  : ~100KB per SM, ~30 cycles
L2 cache         : ~40MB, ~200 cycles
HBM (GPU VRAM)   : 40-80GB, ~400 cycles, 2TB/s bandwidth
PCIe/NVLink      : System RAM ↔ GPU, ~64 GB/s (PCIe4)
```

**The key insight:** Most deep learning ops are **memory-bandwidth limited**, not compute limited. Moving data between memory levels is the bottleneck, not the arithmetic.

```python
import torch
import time

# Demonstrate memory bandwidth bottleneck
def benchmark_elementwise_vs_matmul(size=4096):
    """
    Element-wise ops (memory-bound): primarily limited by bandwidth
    Matrix multiply (compute-bound): primarily limited by TFLOPS
    """
    A = torch.randn(size, size, device='cuda')
    B = torch.randn(size, size, device='cuda')
    
    # Warmup
    for _ in range(5):
        _ = A * B  # element-wise

    # Benchmark element-wise (memory-bound)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        C = A * B
    torch.cuda.synchronize()
    ew_time = (time.time() - t0) / 100
    
    # Benchmark matmul (compute-bound)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        C = A @ B
    torch.cuda.synchronize()
    mm_time = (time.time() - t0) / 100
    
    ew_gflops  = 2 * size**2 / ew_time / 1e9
    mm_gflops  = 2 * size**3 / mm_time / 1e9
    
    print(f"Element-wise:   {ew_time*1000:.2f}ms, {ew_gflops:.0f} GFLOPS")
    print(f"Matrix multiply:{mm_time*1000:.2f}ms, {mm_gflops:.0f} GFLOPS")
    print(f"matmul achieves {mm_gflops/ew_gflops:.1f}x more GFLOPS (compute-bound)")

benchmark_elementwise_vs_matmul()
```

---

## Part 2: Profiling — Finding the Real Bottleneck

### 2.1 PyTorch Profiler

**Never optimize without profiling first.** Intuition is often wrong about where time is spent.

```python
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

model = nn.Sequential(
    nn.Linear(1024, 4096),
    nn.ReLU(),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Linear(4096, 1000),
).cuda()

x = torch.randn(64, 1024, device='cuda')

# Profile CPU and CUDA activity
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,      # Record tensor shapes
    profile_memory=True,     # Track memory allocation
    with_stack=True,         # Stack traces for attribution
) as prof:
    
    with record_function("model_inference"):  # Named region
        for _ in range(20):
            y = model(x)
    
    torch.cuda.synchronize()  # Wait for all GPU work to complete

# Print top-20 most expensive operations by CUDA time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export to Chrome trace viewer
prof.export_chrome_trace("trace.json")
# Open in chrome://tracing for interactive visualization
```

### 2.2 Simple Timing with CUDA Events

```python
def cuda_timer(func, *args, warmup=3, reps=10, **kwargs):
    """
    Precise GPU timing using CUDA events.
    
    Why not use time.time()? 
    CUDA operations are ASYNC — time.time() measures CPU time, 
    not GPU execution time. CUDA events are synchronized to GPU.
    """
    # Warmup (JIT compile, cache warmup)
    for _ in range(warmup):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    
    # Timing
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(reps):
        func(*args, **kwargs)
    end.record()
    
    torch.cuda.synchronize()  # Wait for all GPU work
    
    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / reps
    return avg_ms

# Example: compare different implementations
x = torch.randn(1024, 1024, device='cuda')
layer = nn.Linear(1024, 1024, device='cuda')

# Time forward pass
t_forward = cuda_timer(layer, x)
print(f"Linear forward: {t_forward:.3f} ms")
```

---

## Part 3: Optimizing Data Loading

### 3.1 DataLoader Bottlenecks

```python
import torchvision
from torch.utils.data import DataLoader
import time

def benchmark_dataloader(num_workers, pin_memory, prefetch_factor):
    """Find optimal DataLoader configuration"""
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=torchvision.transforms.ToTensor()
    )
    
    loader = DataLoader(
        dataset,
        batch_size=256,
        num_workers=num_workers,   # Parallel CPU workers
        pin_memory=pin_memory,     # Page-lock memory for fast GPU transfer
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    start = time.time()
    for i, (x, y) in enumerate(loader):
        if torch.cuda.is_available():
            x = x.cuda(non_blocking=True)  # non_blocking for async transfer
        if i >= 50:
            break
    elapsed = time.time() - start
    
    throughput = 50 * 256 / elapsed
    print(f"workers={num_workers}, pin_memory={pin_memory}: "
          f"{throughput:.0f} samples/sec")

# Test different configurations
for workers in [0, 2, 4, 8]:
    benchmark_dataloader(workers, pin_memory=True, prefetch_factor=2)
```

### 3.2 Channels-Last Memory Format

By default, PyTorch stores 4D tensors in **NCHW** format. NVIDIA GPUs prefer **NHWC** (channels-last) for convolutions — it matches how cuDNN internally stores data, eliminating memory transpositions.

```python
# Default NCHW: (batch, channels, height, width)
# Memory layout: all red pixels, then all green, then all blue
x_nchw = torch.randn(32, 3, 224, 224, device='cuda')

# NHWC: (batch, height, width, channels) — "channels last"
# Memory layout: R,G,B,R,G,B,R,G,B (spatially adjacent)
x_nhwc = x_nchw.to(memory_format=torch.channels_last)

# Convert the entire model to channels_last
model = torchvision.models.resnet50().cuda()
model = model.to(memory_format=torch.channels_last)

# Apply channels_last to input before forward pass
x_input = torch.randn(32, 3, 224, 224, device='cuda')
x_input = x_input.to(memory_format=torch.channels_last)
output = model(x_input)

# Speedup on A100: ~5-15% on ResNet50
```

---

## Part 4: torch.compile — Automatic Optimization

### 4.1 What torch.compile Does

`torch.compile` is PyTorch 2.0's JIT compiler. It:
1. **Traces** the model's computation graph
2. **Fuses** adjacent operators into single kernels (eliminates HBM round-trips)
3. **Selects** the fastest CUDA kernel for each operation
4. **Generates** optimized code using Triton

```python
import torch

model = nn.Sequential(
    nn.Linear(1024, 4096),
    nn.GELU(),
    nn.Linear(4096, 1000),
).cuda()

# Compile the model — first call is slow (compilation), subsequent calls are fast
compiled_model = torch.compile(
    model,
    mode='default',  # Options: 'default', 'reduce-overhead', 'max-autotune'
    # default:        balanced compilation time + runtime speed
    # reduce-overhead: faster compilation, less runtime optimization
    # max-autotune:   exhaustive search for fastest kernels (slow compile)
    fullgraph=False,  # True: entire model as one graph (faster, but less flexible)
    dynamic=False,    # True: handle variable shapes (slower but more flexible)
)

x = torch.randn(64, 1024, device='cuda')

# Warmup pass (triggers compilation)
with torch.no_grad():
    y = compiled_model(x)

# Subsequent passes use compiled code
t_eager  = cuda_timer(model, x)
t_compiled = cuda_timer(compiled_model, x)
print(f"Eager:    {t_eager:.3f} ms")
print(f"Compiled: {t_compiled:.3f} ms")
print(f"Speedup:  {t_eager/t_compiled:.2f}x")
# Typical speedup: 1.5–4× depending on model and hardware
```

### 4.2 Operator Fusion — The Key Benefit

```python
# Without fusion: each operation reads/writes to GPU memory separately
# With fusion: chained operations computed in one GPU kernel
#
# Example: Linear + Bias + GELU
#   Without fusion:
#     read W from HBM, read x from HBM
#     compute x@W → write to HBM
#     read from HBM → add bias → write to HBM
#     read from HBM → apply GELU → write to HBM
#   With fusion (torch.compile):
#     read W and x once, compute x@W + bias + GELU, write once

# Flash Attention: state-of-the-art attention fusion
# Instead of: QK^T (HBM write) → softmax (HBM write) → @V
# Flash Attention computes everything in SRAM tiles (no HBM writes!)
# Memory: O(n²) → O(n)  |  Speed: 2-4× faster

try:
    # PyTorch 2.0+ has Flash Attention built in via scaled_dot_product_attention
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,      # Flash Attention
        enable_math=False,      # Standard attention
        enable_mem_efficient=False
    ):
        q = torch.randn(2, 8, 512, 64, device='cuda')
        k = torch.randn(2, 8, 512, 64, device='cuda')
        v = torch.randn(2, 8, 512, 64, device='cuda')
        
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        print(f"Flash Attention output: {output.shape}")
except Exception as e:
    print(f"Flash Attention not available: {e}")
```

---

## Part 5: Mixed Precision Formats Deep Dive

### 5.1 Understanding Floating Point Formats

```
Format | Sign | Exponent | Mantissa | Range          | Precision
FP32   |  1   |    8     |    23    | ±3.4×10³⁸      | ~7 decimal digits
FP16   |  1   |    5     |    10    | ±65,504        | ~3 decimal digits
BF16   |  1   |    8     |     7    | ±3.4×10³⁸      | ~2 decimal digits
FP8 E4 |  1   |    4     |     3    | ±448           | ~1 decimal digit
FP8 E5 |  1   |    5     |     2    | ±57,344        | ~0.5 decimal digits
```

Key observations:
- **FP16** has small exponent range → overflow risk (weight gradients can exceed 65,504!)
- **BF16** has same range as FP32 → safe for most operations, no GradScaler needed
- **FP8** (Hopper GPU, H100) → 4× memory reduction, extreme throughput

```python
# Demonstrating overflow/underflow risks
import torch

# FP16 overflow example
x_fp32 = torch.tensor(70000.0)  # Large value
x_fp16 = x_fp32.half()          # Convert to FP16
print(f"FP32: {x_fp32}")  # 70000.0
print(f"FP16: {x_fp16}")  # inf (overflow! max FP16 = 65504)

# BF16 handles the same value safely
x_bf16 = x_fp32.bfloat16()
print(f"BF16: {x_bf16}")  # 70016 (within range, slight precision loss)

# FP16 underflow example
x_small = torch.tensor(1e-8)
print(f"FP16 small: {x_small.half()}")    # 0.0 (underflow!)
print(f"BF16 small: {x_small.bfloat16()}")  # 1e-8 (preserved)
```

### 5.2 Choosing the Right Precision

```python
# Decision guide:
# 
# FP32:  Stable, slow. Use for: loss, optimizer state, final model weights
# BF16:  Safe, fast. Use for: forward pass, backward pass on Ampere+ GPUs
# FP16:  Tricky, fast. Use for: inference, older GPUs (V100, T4)
# FP8:   Extreme, H100 only. Use for: LLM training at scale
#
# Rule of thumb for modern training:
# - Ampere+ (A100, A10, 3090): use BF16 with autocast
# - Volta/Turing (V100, T4): use FP16 with GradScaler
# - Production inference: FP16 or INT8 (see Module 12)

def select_dtype_and_scaler(gpu_type: str):
    if gpu_type in ['A100', 'A10', 'H100', 'RTX3090', 'RTX4090']:
        dtype = torch.bfloat16
        scaler = None  # No scaler needed!
    else:
        dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler()
    return dtype, scaler
```

---

## Part 6: GPU Memory Management

### 6.1 Avoiding OOM Errors

```python
# Check memory usage
def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved() / 1e9
        print(f"Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")

# Clear unused memory
torch.cuda.empty_cache()  # Releases reserved but unallocated memory

# Common OOM causes and fixes:
# 1. Batch size too large → reduce batch_size + use gradient accumulation
# 2. Not using no_grad() during validation → wrap val loop in torch.no_grad()
# 3. Accumulating loss for logging → use loss.item() not loss
# 4. Model too large → use gradient checkpointing (Module 09)
# 5. Optimizer state too large → use AdaFactor or 8-bit Adam (bitsandbytes)
```

### 6.2 Memory-Efficient Optimizer

```python
# 8-bit Adam (bitsandbytes): stores optimizer states in INT8 → 4× less memory
# Works almost identically to AdamW in practice

# pip install bitsandbytes
try:
    import bitsandbytes as bnb
    optimizer_8bit = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
    print("Using 8-bit AdamW — saves ~75% optimizer memory!")
except ImportError:
    print("bitsandbytes not installed, using standard AdamW")
    optimizer_8bit = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

---

## Part 7: Putting It All Together — Optimized Training Template

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

def create_optimized_trainer(model, train_loader, val_loader, config):
    """Production-grade training setup with all performance optimizations."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Move model to device + channels_last (for CNNs)
    model = model.to(device)
    if config.get('channels_last', False):
        model = model.to(memory_format=torch.channels_last)
    
    # 2. Compile model (PyTorch 2.0+)
    if config.get('compile', True) and hasattr(torch, 'compile'):
        model = torch.compile(model, mode='default')
    
    # 3. Enable cuDNN benchmarking (finds fastest conv algorithms)
    # Only useful when input shapes are FIXED across batches
    torch.backends.cudnn.benchmark = config.get('cudnn_benchmark', True)
    
    # 4. Enable TF32 on Ampere (slightly lower precision, much faster)
    # TF32 uses 10-bit mantissa for matmuls (FP32 has 23-bit)
    # Negligible accuracy impact, significant speed gain
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 5. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0.01),
        fused=True  # Fused AdamW: faster on GPU, requires CUDA
    )
    
    # 6. AMP
    is_bf16_supported = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    amp_dtype = torch.bfloat16 if is_bf16_supported else torch.float16
    scaler = None if is_bf16_supported else GradScaler()
    
    return model, optimizer, amp_dtype, scaler

# Full optimized training step
def optimized_step(model, x, y, optimizer, criterion, scaler, amp_dtype,
                   accumulate_steps, step, device):
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    
    if amp_dtype == torch.bfloat16:
        # BF16 path (no scaler)
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits, y) / accumulate_steps
        loss.backward()
    else:
        # FP16 path (with scaler)
        with autocast(device_type='cuda', dtype=torch.float16):
            logits = model(x)
            loss = criterion(logits, y) / accumulate_steps
        scaler.scale(loss).backward()
    
    if (step + 1) % accumulate_steps == 0:
        if scaler:
            scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    return loss.item() * accumulate_steps
```

---

## Key Takeaways

| Optimization | Typical Speedup | Effort |
|-------------|-----------------|--------|
| **BF16/FP16 AMP** | 2–4× | Low |
| **torch.compile** | 1.5–4× | Very low |
| **Channels-last** | 5–15% | Low |
| **DataLoader (workers)** | 1.5–2× | Low |
| **TF32** | 1.2–1.5× | Very low |
| **Flash Attention** | 2–4× (attention) | Low |
| **Gradient checkpointing** | Memory only | Low |

---

## Quiz

1. **What is the difference between memory_allocated() and memory_reserved()?**
   - Answer: Allocated = memory used by live tensors; Reserved = total memory requested from CUDA (includes cached blocks)

2. **Why does channels_last improve performance?**
   - Answer: Matches cuDNN's internal NHWC format for convolutions, eliminating memory transpositions

3. **What does torch.backends.cudnn.benchmark = True do?**
   - Answer: Benchmarks multiple conv algorithms on first run and selects the fastest (only effective for fixed input shapes)

4. **What does `non_blocking=True` do in tensor.to(device)?**
   - Answer: Allows asynchronous CPU→GPU transfer so CPU can continue while GPU loads data

5. **What is TF32 and what trade-off does it make?**
   - Answer: NVIDIA format using 10-bit mantissa for matmuls; slight precision loss for ~1.5× speedup

6. **What is Flash Attention and what problem does it solve?**
   - Answer: Attention computed in SRAM tiles without materializing the full attention matrix; reduces memory from O(n²) to O(n)

7. **Why is `torch.cuda.synchronize()` needed for timing?**
   - Answer: CUDA operations are async; synchronize() waits for all GPU work to complete before measuring time

8. **What is `fused=True` in AdamW?**
   - Answer: Combines all optimizer operations into a single GPU kernel; faster than sequential operations

9. **What does `torch.compile(model, mode='max-autotune')` do?**
   - Answer: Exhaustively benchmarks all candidate kernels to find the absolute fastest for each operation (slow compile)

10. **When should you NOT enable cudnn.benchmark?**
    - Answer: When input shapes vary across batches — benchmark mode re-profiles on each new shape, adding overhead
