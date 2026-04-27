# Module 13: Deployment — TorchScript & ONNX

## Learning Objectives
By the end of this module you will be able to:
- Convert PyTorch models to TorchScript via tracing and scripting
- Export models to ONNX and run inference with ONNX Runtime
- Apply TensorRT optimisation for maximum GPU inference performance
- Validate numerical equivalence between export formats and the original model
- Handle dynamic shapes, custom ops, and common export pitfalls
- Package and version production model artifacts
- Choose the right export format for your deployment target

---

## 13.1 Deployment Format Decision Tree

```
Your model → where does it run?
│
├─ Python server (same process)
│   └─ Raw PyTorch + torch.inference_mode()  [simplest]
│
├─ C++ application / embedded
│   └─ TorchScript (.pt)  [best PyTorch → C++ path]
│
├─ Cross-framework (TF, TFLite, etc.) / ONNX Runtime
│   └─ ONNX (.onnx)  [most portable]
│
├─ NVIDIA GPU (maximum throughput)
│   └─ TensorRT (.engine)  [fastest GPU]
│
├─ Mobile / browser
│   ├─ TorchScript + PyTorch Mobile  [iOS/Android]
│   └─ ONNX → ONNX Runtime Mobile
│
└─ CPU-only, maximum compatibility
    └─ ONNX → ONNX Runtime CPU  [good cross-platform]
```

---

## 13.2 TorchScript

TorchScript compiles your model into a statically typed, portable intermediate representation that can run without Python.

**Two methods:**
1. **Tracing** — run the model once; record the operations; best for models without data-dependent control flow
2. **Scripting** — parse the Python source code with a restricted subset of Python; handles if/loops

### Tracing

```python
import torch
import torch.nn as nn

model = resnet50(weights="DEFAULT").eval()

# Trace with a representative example input
example_input = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    traced = torch.jit.trace(model, example_input)

# Save
traced.save("resnet50_traced.pt")

# Load (no Python class needed)
loaded = torch.jit.load("resnet50_traced.pt")
out = loaded(torch.randn(1, 3, 224, 224))

# Verify equivalence
torch.testing.assert_close(
    model(example_input),
    traced(example_input),
    rtol=1e-4,
    atol=1e-4,
)
print("Traced model output matches original!")

# Inspect the generated IR
print(traced.graph)
print(traced.code)   # Python-like representation of the traced graph
```

### Scripting

```python
# Scripting handles dynamic control flow

class DynamicModel(nn.Module):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.fc = nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Data-dependent branch: tracing would miss one path!
        if x.mean() > self.threshold:
            return self.fc(x)
        else:
            return self.fc(-x)

model = DynamicModel().eval()

# Script compiles both branches
scripted = torch.jit.script(model)
scripted.save("dynamic_model_scripted.pt")

# Scripted models can be optimised
torch.jit.optimize_for_inference(scripted)  # fuses ops, removes dead code

# ── Scripting individual functions ────────────────────────────────────────────
@torch.jit.script
def top_k_softmax(x: torch.Tensor, k: int) -> torch.Tensor:
    probs = torch.softmax(x, dim=-1)
    top_k_vals, top_k_idx = probs.topk(k, dim=-1)
    return top_k_idx

# ── Hybrid: script a module that uses traced sub-modules ─────────────────────
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Trace the backbone (no control flow)
        backbone = resnet50().eval()
        self.backbone = torch.jit.trace(backbone, torch.randn(1, 3, 224, 224))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        # Script handles the conditional
        if features.max() > 1.0:
            return features.clamp(max=1.0)
        return features

scripted_hybrid = torch.jit.script(HybridModel())
```

### Common TorchScript Pitfalls

```python
# ── Type annotations are mandatory ────────────────────────────────────────────
class GoodModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ← required
        return x * 2

# ── Dictionary and list types must be declared ────────────────────────────────
from typing import Dict, List

@torch.jit.script
def process_multi_output(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {"logits": x, "probs": x.softmax(-1)}

# ── No *args/**kwargs in scripted modules ─────────────────────────────────────
# ── No non-primitive Python objects (e.g. dataclasses) ───────────────────────
# ── Use torch.jit.is_scripting() to branch ────────────────────────────────────
class BranchableModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting():
            return x  # simplified path for TorchScript
        else:
            return self._full_forward(x)
```

---

## 13.3 ONNX Export

ONNX (Open Neural Network Exchange) is a cross-framework model format supported by TensorFlow, TFLite, CoreML, ONNX Runtime, TensorRT, and more.

```python
import torch
import onnx
import onnxruntime as ort
import numpy as np

model = resnet50(weights="DEFAULT").eval()
x     = torch.randn(1, 3, 224, 224)

# ── Export to ONNX ────────────────────────────────────────────────────────────
with torch.no_grad():
    torch.onnx.export(
        model,
        x,
        "resnet50.onnx",
        opset_version=17,              # use latest stable opset
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={                 # enable batch-size flexibility
            "image":  {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        export_params=True,
        do_constant_folding=True,      # fold constant expressions
        verbose=False,
    )

# ── Validate ONNX graph ───────────────────────────────────────────────────────
onnx_model = onnx.load("resnet50.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

# Optional: pretty-print the graph
print(onnx.helper.printable_graph(onnx_model.graph))

# ── Run with ONNX Runtime ─────────────────────────────────────────────────────
# Providers: CUDAExecutionProvider, TensorrtExecutionProvider, CPUExecutionProvider
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 4

ort_session = ort.InferenceSession(
    "resnet50.onnx",
    sess_options,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

# Run inference
np_input = x.numpy()
ort_out   = ort_session.run(["logits"], {"image": np_input})[0]  # numpy array

# ── Numerical validation ──────────────────────────────────────────────────────
with torch.no_grad():
    pt_out = model(x).numpy()

np.testing.assert_allclose(pt_out, ort_out, rtol=1e-3, atol=1e-5)
print(f"Max absolute difference: {np.abs(pt_out - ort_out).max():.6f}")
print("ONNX Runtime output matches PyTorch!")
```

---

## 13.4 Dynamic Shapes and Complex Exports

```python
import torch
import torch.nn as nn

# ── Transformer with dynamic sequence length ──────────────────────────────────
class TextClassifier(nn.Module):
    def __init__(self, vocab_size=30522, d_model=256, n_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.enc   = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True),
            num_layers=4,
        )
        self.head  = nn.Linear(d_model, n_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self.enc(x)
        return self.head(x[:, 0])   # [CLS] token

model = TextClassifier().eval()
dummy_input = torch.randint(0, 30522, (1, 64))

torch.onnx.export(
    model,
    dummy_input,
    "text_classifier.onnx",
    opset_version=17,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq_len"},
        "logits":    {0: "batch"},
    },
)

# Verify dynamic shapes work
sess = ort.InferenceSession("text_classifier.onnx")
for seq_len in [32, 64, 128, 256]:
    inp = np.random.randint(0, 30522, (2, seq_len)).astype(np.int64)
    out = sess.run(None, {"input_ids": inp})
    print(f"seq_len={seq_len}: output shape={out[0].shape}")

# ── Custom op: register for ONNX export ──────────────────────────────────────
class CustomNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x / x.norm(dim=-1, keepdim=True)

    @staticmethod
    def symbolic(g, x):
        # Map to a standard ONNX op
        norm = g.op("ReduceL2", x, axes_i=[-1], keepdims_i=1)
        return g.op("Div", x, norm)
```

---

## 13.5 TensorRT Optimisation

TensorRT is NVIDIA's inference optimisation library — it fuses ops, uses INT8/FP16, selects the best kernel for each layer, and is typically 3–10× faster than ONNX Runtime on GPU.

```python
# Option 1: via torch_tensorrt (PyTorch-native)
# pip install torch-tensorrt
import torch_tensorrt

model = resnet50(weights="DEFAULT").eval().cuda()
x     = torch.randn(32, 3, 224, 224, device="cuda")

trt_model = torch_tensorrt.compile(
    model,
    inputs=[
        torch_tensorrt.Input(
            min_shape=[1, 3, 224, 224],
            opt_shape=[32, 3, 224, 224],
            max_shape=[64, 3, 224, 224],
            dtype=torch.half,
        )
    ],
    enabled_precisions={torch.half},   # use FP16
    workspace_size=1 << 30,            # 1 GB workspace for optimization
)

# Save TensorRT-compiled model
torch.jit.save(trt_model, "resnet50_trt.ts")

# ── Benchmark comparison ──────────────────────────────────────────────────────
def benchmark(fn, n_warmup=20, n_runs=100):
    for _ in range(n_warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs): fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / n_runs  # ms

x_fp16 = x.half()
print(f"PyTorch FP32: {benchmark(lambda: model(x)):.2f} ms")
print(f"PyTorch FP16: {benchmark(lambda: model(x_fp16)):.2f} ms")
print(f"TensorRT FP16: {benchmark(lambda: trt_model(x_fp16)):.2f} ms")
# Typical: TRT is 2–5× faster than PT FP16

# Option 2: via polygraphy / trtexec CLI
# trtexec --onnx=resnet50.onnx --fp16 --saveEngine=resnet50_trt.engine \
#         --minShapes=image:1x3x224x224 --optShapes=image:32x3x224x224 --maxShapes=image:64x3x224x224
```

---

## 13.6 Model Artifact Versioning

```python
import torch
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

@dataclass
class ModelMetadata:
    model_name: str
    version: str
    architecture: str
    training_data: str
    input_shape: list
    output_shape: list
    accuracy: float
    latency_ms: float
    created_at: str
    torch_version: str
    export_format: str
    notes: str = ""

def package_model(
    model: torch.nn.Module,
    metadata: ModelMetadata,
    save_dir: str = "model_artifacts",
    export_formats: list = ["torchscript", "onnx"],
) -> dict:
    """
    Package a model with metadata into versioned artifacts.
    Returns dict of {format: path}.
    """
    save_path = Path(save_dir) / metadata.version
    save_path.mkdir(parents=True, exist_ok=True)

    artifacts = {}
    model.eval()
    dummy = torch.randn(*metadata.input_shape)

    # Save TorchScript
    if "torchscript" in export_formats:
        with torch.no_grad():
            traced = torch.jit.trace(model, dummy)
        ts_path = save_path / "model.pt"
        traced.save(str(ts_path))
        artifacts["torchscript"] = str(ts_path)

    # Save ONNX
    if "onnx" in export_formats:
        onnx_path = save_path / "model.onnx"
        with torch.no_grad():
            torch.onnx.export(
                model, dummy, str(onnx_path),
                input_names=["input"], output_names=["output"],
                opset_version=17,
            )
        artifacts["onnx"] = str(onnx_path)

    # Save metadata
    meta_path = save_path / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(asdict(metadata), f, indent=2)

    print(f"Model artifacts saved to: {save_path}")
    print(f"  {', '.join(artifacts.keys())}")
    return artifacts
```

---

## 13.7 Inference Warm-Up and Benchmarking

```python
import torch
import time
import statistics

class InferenceBenchmark:
    """Comprehensive inference benchmarking."""

    def __init__(self, model, device, n_warmup: int = 20, n_runs: int = 100):
        self.model   = model
        self.device  = device
        self.n_warmup = n_warmup
        self.n_runs   = n_runs

    def run(self, input_tensor: torch.Tensor) -> dict:
        self.model.eval()
        x = input_tensor.to(self.device)

        # Warmup (fill GPU pipeline, compile any JIT lazily)
        with torch.no_grad():
            for _ in range(self.n_warmup):
                _ = self.model(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Timed runs
        latencies = []
        with torch.no_grad():
            for _ in range(self.n_runs):
                if self.device.type == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    end   = torch.cuda.Event(enable_timing=True)
                    start.record()
                    _ = self.model(x)
                    end.record()
                    torch.cuda.synchronize()
                    latencies.append(start.elapsed_time(end))
                else:
                    t0 = time.perf_counter()
                    _ = self.model(x)
                    latencies.append(1000 * (time.perf_counter() - t0))

        return {
            "p50_ms":  statistics.median(latencies),
            "p95_ms":  sorted(latencies)[int(0.95 * len(latencies))],
            "p99_ms":  sorted(latencies)[int(0.99 * len(latencies))],
            "mean_ms": statistics.mean(latencies),
            "std_ms":  statistics.stdev(latencies),
            "throughput_sps": 1000 * x.shape[0] / statistics.median(latencies),
        }
```

---

## Exercises

**Exercise 13.1** Export `MiniGPT` (Module 08) to ONNX with dynamic batch and sequence dimensions. Validate that ONNX Runtime output matches PyTorch for batch sizes 1, 4, 16 and sequence lengths 32, 64, 128.

**Exercise 13.2** Script the `BiLSTMClassifier` from Module 07 using `torch.jit.script`. Fix any type annotation issues. Benchmark the scripted version vs eager on CPU (100 batches, batch_size=32).

**Exercise 13.3** Write a complete `export_pipeline(model, model_name, version)` function that: exports TorchScript + ONNX, validates both, benchmarks all three (eager/TS/ONNX), and writes a JSON report.

---

## Module Summary

| Format | Portability | Speed | Dynamic Shapes | Use Case |
|--------|------------|-------|---------------|---------|
| Eager PyTorch | Python only | Baseline | Full | Development |
| TorchScript | Python + C++ | +10–20% | With scripting | C++ apps |
| ONNX | All frameworks | +20–50% | Yes | Cross-platform |
| TensorRT | NVIDIA only | 3–10× | Yes | Max GPU perf |

---

## Quiz

1. What is the key difference between `torch.jit.trace` and `torch.jit.script`?
2. Why does tracing fail for models with data-dependent control flow?
3. What does `dynamic_axes` do in `torch.onnx.export`?
4. What is `do_constant_folding=True` and what does it optimise?
5. What is the ONNX opset version and why does it matter?
6. Why do you need to warm up the model before benchmarking inference?
7. What is TensorRT engine calibration and when is it needed?

---

*Next: [Module 14 — Serving & Production Systems](./14_serving_and_production.md)*
