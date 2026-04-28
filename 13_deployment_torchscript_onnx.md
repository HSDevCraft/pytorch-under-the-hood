# Module 13: Deployment — TorchScript, ONNX & TensorRT

> **Goal:** Take a trained PyTorch model and make it production-ready — optimized, portable, and fast across different runtimes and hardware.

---

## Learning Objectives

By the end of this module, you will:
- **Understand** the deployment pipeline from research to production
- **Export** models with TorchScript (tracing vs scripting) and know which to use
- **Export** to ONNX and run with ONNX Runtime for cross-platform deployment
- **Optimize** with TensorRT for maximum GPU inference speed
- **Benchmark** inference latency, throughput, and memory correctly
- **Version** and manage model artifacts in production

---

## Part 1: The Deployment Pipeline

### 1.1 Why Not Just Use Raw PyTorch in Production?

Raw PyTorch models require:
- The Python interpreter (slow startup, GIL, not embedded-friendly)
- PyTorch installed as a dependency (large, version-sensitive)
- Gradient tracking overhead (unnecessary for inference)

Production deployments need:
- **Low latency** (< 10ms for real-time applications)
- **High throughput** (thousands of requests/second)
- **Cross-platform** (C++, Java, mobile, embedded systems)
- **No Python dependency** (for C++ serving, edge deployment)

```
Training Pipeline:
  Data → Model (PyTorch, Python) → Checkpoint (.pt file)

Deployment Pipeline:
  Checkpoint → Export → Optimize → Serve
               ↓
  ┌────────────────────────────────────────────────────────┐
  │  TorchScript  → C++ LibTorch, TorchServe              │
  │  ONNX         → ONNX Runtime (CPU/GPU/Edge)            │
  │  TensorRT     → NVIDIA GPU, maximum speed              │
  │  ExecuTorch   → Mobile (iOS, Android), Edge            │
  └────────────────────────────────────────────────────────┘
```

---

## Part 2: TorchScript

### 2.1 TorchScript Tracing

Tracing **records** the operations executed during a single forward pass with a dummy input.

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleModel().eval()

# ── Tracing ────────────────────────────────────────────────────────────────────
# Pro:  Simple, works with most models
# Con:  Only records ONE execution path — misses conditional branches!
# Use when: model has no data-dependent control flow

dummy_input = torch.randn(1, 784)  # Must match your actual input shape

with torch.no_grad():
    traced_model = torch.jit.trace(model, dummy_input)

# Verify outputs match
with torch.no_grad():
    x = torch.randn(1, 784)
    original_out = model(x)
    traced_out = traced_model(x)
    max_diff = (original_out - traced_out).abs().max()
    print(f"Max output difference: {max_diff:.8f}")  # Should be ~0

# Save the traced model
traced_model.save("model_traced.pt")

# Load and run (no Python/PyTorch model definition needed!)
loaded = torch.jit.load("model_traced.pt")
loaded.eval()
with torch.no_grad():
    out = loaded(x)
print(f"Loaded model output: {out.shape}")
```

### 2.2 TorchScript Scripting

Scripting **compiles** the Python source code directly — handles all control flow correctly.

```python
class ModelWithBranching(nn.Module):
    """
    This model has data-dependent control flow:
    the path through the network depends on the INPUT value.
    
    Tracing would only capture one branch — WRONG!
    Scripting reads the source code — captures ALL branches — CORRECT!
    """
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.fc_large = nn.Linear(10, 5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Data-dependent branch: tracing bakes in ONE path!
        if x.mean() > 0:
            return torch.relu(self.fc(x))
        else:
            return torch.sigmoid(self.fc_large(x))

model = ModelWithBranching().eval()

# ── Scripting ──────────────────────────────────────────────────────────────────
# Pro: Handles all control flow correctly (if/for/while)
# Con: Python code must be TorchScript-compatible (subset of Python)
#      Cannot use arbitrary Python objects, only TorchScript types
# Use when: model has conditional logic, loops, or recursive calls

try:
    scripted_model = torch.jit.script(model)
    print("Scripting succeeded!")
    scripted_model.save("model_scripted.pt")
except Exception as e:
    print(f"Scripting failed: {e}")

# ── When does scripting fail? ─────────────────────────────────────────────────
# 1. Using non-TorchScript types (e.g., pandas DataFrames, PIL Images)
# 2. Using Python builtins not supported in TorchScript
# 3. Dynamic dispatch on Python objects
# Solution: refactor the offending code or use @torch.jit.script on specific functions
```

### 2.3 Optimizing TorchScript for Inference

```python
# Additional optimizations after scripting/tracing:
model_scripted = torch.jit.script(SimpleModel().eval())

# 1. Freeze: inline constants and remove Python dispatch overhead
model_frozen = torch.jit.freeze(model_scripted)

# 2. Optimize for inference: fuse BatchNorm into Conv, constant folding
model_opt = torch.jit.optimize_for_inference(model_frozen)

# 3. Run with torch.no_grad() for speed (no gradient tracking)
x = torch.randn(1, 784)
with torch.no_grad():
    out = model_opt(x)

# Benchmark the speedup
import time
n_reps = 1000

with torch.no_grad():
    t0 = time.time()
    for _ in range(n_reps):
        _ = model(x)
    eager_ms = (time.time() - t0) * 1000 / n_reps
    
    t0 = time.time()
    for _ in range(n_reps):
        _ = model_opt(x)
    opt_ms = (time.time() - t0) * 1000 / n_reps

print(f"Eager:    {eager_ms:.3f} ms")
print(f"Optimized:{opt_ms:.3f} ms")
print(f"Speedup:  {eager_ms/opt_ms:.2f}×")
```

---

## Part 3: ONNX Export

### 3.1 What Is ONNX?

**Open Neural Network Exchange (ONNX)** is an open format for ML models:
- Supported by: PyTorch, TensorFlow, scikit-learn, and 30+ frameworks
- Runs on: ONNX Runtime (CPU, GPU, mobile), TensorRT, OpenVINO, CoreML
- The universal "lingua franca" of model deployment

```python
import torch
import torch.onnx

model = SimpleModel().eval()
dummy_input = torch.randn(1, 784)

# ── Export to ONNX ─────────────────────────────────────────────────────────────
torch.onnx.export(
    model,                         # Model to export
    dummy_input,                   # Example input (defines shapes)
    "model.onnx",                  # Output file path
    
    input_names=["input"],         # Names for ONNX graph inputs
    output_names=["logits"],       # Names for ONNX graph outputs
    
    # Dynamic axes: specify which dimensions can vary
    # Without this, model is fixed to the dummy_input shapes
    dynamic_axes={
        "input":  {0: "batch_size"},   # Batch dimension can vary
        "logits": {0: "batch_size"},
    },
    
    opset_version=17,              # ONNX opset version
                                    # Higher = more ops supported
                                    # Must be supported by your runtime
    
    do_constant_folding=True,      # Pre-compute constants → faster inference
    
    export_params=True,            # Include trained weights in the file
    verbose=False,                  # Print computation graph (debug mode)
)

print("ONNX export successful!")

# ── Verify the ONNX model ─────────────────────────────────────────────────────
import onnx

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)  # Validates graph structure
print(f"ONNX model valid!")
print(f"Opset: {onnx_model.opset_import[0].version}")

# Inspect the model
for node in onnx_model.graph.node[:3]:
    print(f"Op: {node.op_type}, Inputs: {list(node.input)}")
```

### 3.2 ONNX Runtime Inference

```python
import onnxruntime as ort
import numpy as np

# ── Create ONNX Runtime session ────────────────────────────────────────────────
# ORT automatically selects the best execution provider
session = ort.InferenceSession(
    "model.onnx",
    providers=[
        'CUDAExecutionProvider',    # GPU (preferred if available)
        'CPUExecutionProvider',     # CPU fallback
    ]
)

# Inspect input/output info
for inp in session.get_inputs():
    print(f"Input: {inp.name}, shape={inp.shape}, dtype={inp.type}")
for out in session.get_outputs():
    print(f"Output: {out.name}, shape={out.shape}, dtype={out.type}")

# ── Run inference ─────────────────────────────────────────────────────────────
# ONNX Runtime uses NumPy arrays, not PyTorch tensors!
x_np = np.random.randn(32, 784).astype(np.float32)

ort_outputs = session.run(
    output_names=["logits"],      # Which outputs to compute
    input_feed={"input": x_np},  # Dictionary: name → numpy array
)

output = ort_outputs[0]  # (32, 10) numpy array
print(f"ORT output shape: {output.shape}")
print(f"ORT output dtype: {output.dtype}")

# ── Verify ORT matches PyTorch ────────────────────────────────────────────────
x_torch = torch.from_numpy(x_np)
with torch.no_grad():
    pytorch_output = model(x_torch).numpy()

max_diff = np.abs(pytorch_output - output).max()
print(f"Max difference PyTorch vs ORT: {max_diff:.8f}")
# Should be < 1e-5 for well-behaved models
```

---

## Part 4: TensorRT — Maximum GPU Inference Speed

### 4.1 What TensorRT Does

TensorRT is NVIDIA's inference optimization engine:
1. **Analyzes** the network graph
2. **Fuses** operators (Conv+BN+ReLU → single kernel)
3. **Selects** optimal CUDA kernel for each operation on your specific GPU
4. **Quantizes** to INT8 or FP16 automatically
5. **Generates** a highly optimized execution plan

**Typical speedups:** 2–8× over raw ONNX Runtime on NVIDIA GPUs.

```python
# Method 1: torch_tensorrt (simplest)
# pip install torch-tensorrt
try:
    import torch_tensorrt
    
    model = SimpleModel().eval().cuda()
    
    # Convert to TensorRT — provides optimal plan for THIS specific GPU
    trt_model = torch_tensorrt.compile(
        model,
        inputs=[
            torch_tensorrt.Input(
                min_shape=(1, 784),    # Minimum batch size
                opt_shape=(32, 784),   # Optimal (most common) batch size
                max_shape=(128, 784),  # Maximum batch size
                dtype=torch.float32
            )
        ],
        enabled_precisions={torch.float16},  # Allow FP16 for speed
    )
    
    # Run TRT inference
    x = torch.randn(32, 784, device='cuda')
    with torch.no_grad():
        out = trt_model(x)
    print(f"TensorRT output: {out.shape}")
    
    # Save TRT engine for later
    torch.jit.save(trt_model, "model_trt.pt")
    
except ImportError:
    print("torch_tensorrt not installed")
```

---

## Part 5: Benchmarking Inference Correctly

### 5.1 Latency vs Throughput

```python
import torch
import time

def benchmark_inference(model, input_shape, batch_sizes=[1, 8, 32, 64],
                         n_warmup=20, n_reps=200, device='cuda'):
    """
    Comprehensive inference benchmark.
    
    Key concepts:
    - Latency: time for ONE batch (critical for real-time, interactive)
    - Throughput: samples/second (critical for batch processing, serving)
    - Memory: peak GPU memory during inference
    
    Common mistakes:
    1. Not warming up (first few runs include JIT compilation)
    2. Not using torch.no_grad() (adds gradient tracking overhead)
    3. Forgetting torch.cuda.synchronize() (GPU is async!)
    4. Not setting model.eval() (Dropout/BN behave differently)
    """
    model = model.to(device).eval()
    results = {}
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, *input_shape, device=device)
        
        # Step 1: Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(x)
        if device == 'cuda':
            torch.cuda.synchronize()  # Wait for warmup to finish
        
        # Step 2: Track peak memory
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        # Step 3: Benchmark with CUDA events (most accurate for GPU)
        if device == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with torch.no_grad():
                for _ in range(n_reps):
                    _ = model(x)
            end_event.record()
            torch.cuda.synchronize()
            
            total_ms = start_event.elapsed_time(end_event)
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_reps):
                    _ = model(x)
            total_ms = (time.perf_counter() - t0) * 1000
        
        latency_ms = total_ms / n_reps
        throughput = batch_size * n_reps / (total_ms / 1000)
        
        if device == 'cuda':
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
        else:
            peak_mem_mb = 0
        
        results[batch_size] = {
            'latency_ms': latency_ms,
            'throughput': throughput,
            'peak_mem_mb': peak_mem_mb,
        }
        
        print(f"Batch {batch_size:4d}: "
              f"latency={latency_ms:.2f}ms, "
              f"throughput={throughput:.0f} samples/s, "
              f"memory={peak_mem_mb:.0f}MB")
    
    return results

# Run benchmark
model = SimpleModel()
results = benchmark_inference(model, input_shape=(784,), device='cpu')
```

---

## Part 6: Model Versioning and Artifact Management

### 6.1 Production Artifact Structure

```python
import json
import hashlib
from pathlib import Path
from datetime import datetime

def save_model_artifact(model: torch.nn.Module, save_dir: str,
                         metadata: dict) -> str:
    """
    Save a complete model artifact with reproducibility metadata.
    
    A model artifact = model weights + metadata + config + checksums
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    weights_path = save_dir / "model.pt"
    torch.save(model.state_dict(), weights_path)
    
    # Compute checksum for integrity verification
    with open(weights_path, 'rb') as f:
        checksum = hashlib.sha256(f.read()).hexdigest()
    
    # Save metadata (everything needed to reproduce this model)
    full_metadata = {
        "timestamp": datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "checksum_sha256": checksum,
        "model_params": sum(p.numel() for p in model.parameters()),
        **metadata  # User-provided: accuracy, training config, etc.
    }
    
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(full_metadata, f, indent=2)
    
    print(f"Saved artifact to: {save_dir}")
    print(f"SHA256: {checksum[:16]}...")
    return checksum


# Example artifact metadata
save_model_artifact(
    model=SimpleModel(),
    save_dir="artifacts/resnet50_cifar10_v1.2",
    metadata={
        "model_name": "SimpleModel",
        "dataset": "CIFAR-10",
        "version": "1.2.0",
        "val_accuracy": 0.921,
        "training_epochs": 100,
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
    }
)
```

---

## Key Takeaways

| Format | Platform | Speed | Portability | Use Case |
|--------|----------|-------|-------------|----------|
| **TorchScript** | PyTorch-compatible | Good | C++, Python | PyTorch serving |
| **ONNX** | Any ONNX Runtime | Good | Any platform | Cross-platform |
| **TensorRT** | NVIDIA GPU only | Best | NVIDIA only | Max GPU speed |
| **ExecuTorch** | Mobile/Edge | Great | iOS, Android | Edge devices |

---

## Quiz

1. **What is the key difference between tracing and scripting?**
   - Answer: Tracing records one execution path; scripting compiles source code and handles all branches

2. **When would tracing fail silently?**
   - Answer: When the model has data-dependent if/else — traced model always takes the branch from the dummy input

3. **What does `dynamic_axes` do in torch.onnx.export?**
   - Answer: Specifies which dimensions can vary (e.g., batch size) so the ONNX model accepts variable input shapes

4. **What is `do_constant_folding=True`?**
   - Answer: Pre-computes operations on constants at export time, reducing inference computation

5. **Why must you call `torch.cuda.synchronize()` when timing GPU operations?**
   - Answer: GPU ops are asynchronous; without synchronize, you measure CPU scheduling time, not GPU execution time

6. **What is latency vs throughput?**
   - Answer: Latency = time for one batch (real-time requirement); throughput = samples/second (batch processing)

7. **What is TensorRT's main mechanism for speedup?**
   - Answer: Analyzes the computation graph, fuses operators, selects optimal CUDA kernels for specific GPU, optionally quantizes

8. **Why is warmup needed before benchmarking?**
   - Answer: First runs include JIT compilation, GPU cache loading; warmup ensures steady-state performance is measured

9. **What is ONNX opset version?**
   - Answer: The version of ONNX operator specification; higher opsets support more operations but require newer runtimes

10. **Why save a model checksum in production?**
    - Answer: Verify model file integrity (detect corruption/tampering) and ensure the correct version is deployed
