# Module 14: Serving & Production Systems

## Learning Objectives
By the end of this module you will be able to:
- Build a production-ready REST API for model inference using FastAPI
- Deploy models with TorchServe for multi-model, batched serving
- Implement request batching, caching, and async inference
- Write health checks, readiness probes, and graceful shutdown
- Containerise PyTorch inference services with Docker
- Set up Kubernetes deployments with GPU node selectors and autoscaling
- Monitor model performance, latency, and drift in production

---

## 14.1 Production Serving Architecture

```
                    ┌─────────────────────┐
         Client     │   Load Balancer      │
         Requests → │   (nginx / k8s SVC)  │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
       ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
       │ API Pod 0   │  │ API Pod 1   │  │ API Pod 2   │
       │ FastAPI /   │  │ FastAPI /   │  │ FastAPI /   │
       │ TorchServe  │  │ TorchServe  │  │ TorchServe  │
       │ GPU:0       │  │ GPU:1       │  │ GPU:2       │
       └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
              │                │                │
              └────────────────┴────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │  Model Registry /    │
                    │  Object Storage     │
                    │  (S3/GCS/MLflow)    │
                    └─────────────────────┘
```

---

## 14.2 Production FastAPI Inference Server

```python
# inference_server.py
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional
import threading

import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn


# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    inputs: List[List[float]] = Field(..., description="Batch of input feature vectors")
    return_probabilities: bool = False

class PredictionResult(BaseModel):
    prediction_id: str
    predictions: List[int]
    probabilities: Optional[List[List[float]]] = None
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    uptime_s: float


# ── Global model holder ────────────────────────────────────────────────────────
class ModelHolder:
    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.device: Optional[torch.device] = None
        self._lock = asyncio.Lock()
        self.n_requests = 0
        self.total_latency_ms = 0.0
        self.start_time = time.time()

    def load(self, model_path: str, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load TorchScript model
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        # Warmup — fill GPU pipeline
        dummy = torch.zeros(1, 10, device=self.device)
        with torch.inference_mode():
            for _ in range(5):
                self.model(dummy)

        print(f"Model loaded on {self.device}")


holder = ModelHolder()


# ── Lifespan: startup & shutdown ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    holder.load("model.pt", device="auto")
    yield
    # Shutdown
    print("Shutting down inference server...")
    if holder.device and holder.device.type == "cuda":
        torch.cuda.empty_cache()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PyTorch Inference API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if holder.model is not None else "loading",
        model_loaded=holder.model is not None,
        device=str(holder.device),
        uptime_s=time.time() - holder.start_time,
    )


@app.post("/predict", response_model=PredictionResult)
async def predict(req: PredictRequest):
    if holder.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    t0 = time.perf_counter()
    prediction_id = str(uuid.uuid4())

    async with holder._lock:
        try:
            x = torch.tensor(req.inputs, dtype=torch.float32, device=holder.device)

            with torch.inference_mode():
                logits = holder.model(x)
                probs  = torch.softmax(logits, dim=-1)
                preds  = probs.argmax(dim=-1)

        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Inference failed: {e}")

    latency_ms = 1000 * (time.perf_counter() - t0)
    holder.n_requests += 1
    holder.total_latency_ms += latency_ms

    return PredictionResult(
        prediction_id=prediction_id,
        predictions=preds.tolist(),
        probabilities=probs.tolist() if req.return_probabilities else None,
        latency_ms=round(latency_ms, 3),
    )


@app.get("/metrics")
async def metrics():
    avg_latency = (
        holder.total_latency_ms / holder.n_requests if holder.n_requests > 0 else 0.0
    )
    mem_info = {}
    if holder.device and holder.device.type == "cuda":
        mem_info = {
            "gpu_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
            "gpu_reserved_gb":  round(torch.cuda.memory_reserved() / 1e9, 3),
        }
    return {
        "n_requests": holder.n_requests,
        "avg_latency_ms": round(avg_latency, 3),
        **mem_info,
    }


if __name__ == "__main__":
    uvicorn.run("inference_server:app", host="0.0.0.0", port=8000, workers=1)
```

---

## 14.3 Dynamic Batching

Instead of processing each request independently, accumulate requests and process them as a batch — dramatically improves GPU utilisation.

```python
import asyncio
import time
import torch
from typing import List
from dataclasses import dataclass, field

@dataclass
class InferenceRequest:
    tensor: torch.Tensor
    result_future: asyncio.Future = field(default_factory=asyncio.Future)

class DynamicBatcher:
    """
    Collects individual requests, batches them, runs inference,
    then distributes results back to each request's future.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        max_batch_size: int = 32,
        max_wait_ms: float = 10.0,
    ):
        self.model          = model
        self.device         = device
        self.max_batch_size = max_batch_size
        self.max_wait_ms    = max_wait_ms
        self.queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def start(self):
        self._running = True
        asyncio.create_task(self._batch_worker())

    async def stop(self):
        self._running = False

    async def infer(self, x: torch.Tensor) -> torch.Tensor:
        """Submit a single tensor; wait for result."""
        future = asyncio.get_event_loop().create_future()
        await self.queue.put(InferenceRequest(tensor=x, result_future=future))
        return await future

    async def _batch_worker(self):
        while self._running:
            # Collect requests up to max_batch_size or max_wait_ms
            requests: List[InferenceRequest] = []
            deadline = time.monotonic() + self.max_wait_ms / 1000

            # Wait for at least one request
            try:
                req = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                requests.append(req)
            except asyncio.TimeoutError:
                continue

            # Drain more requests until batch is full or deadline passed
            while len(requests) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    requests.append(req)
                except asyncio.TimeoutError:
                    break

            # Run batched inference
            batch = torch.stack([r.tensor for r in requests]).to(self.device)
            try:
                with torch.inference_mode():
                    outputs = self.model(batch)
                for i, req in enumerate(requests):
                    req.result_future.set_result(outputs[i])
            except Exception as e:
                for req in requests:
                    req.result_future.set_exception(e)
```

---

## 14.4 TorchServe

TorchServe is PyTorch's official production model server. It handles multi-model serving, batching, versioning, and management APIs.

```python
# ── 1. Write a custom handler ──────────────────────────────────────────────────
# my_handler.py
import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import map_class_to_label

class ClassificationHandler(BaseHandler):
    """
    Custom TorchServe handler for image classification.
    Inherits BaseHandler which manages model loading + context.
    """

    def initialize(self, context):
        """Load model and preprocessing config."""
        super().initialize(context)
        self.model.eval()

    def preprocess(self, data):
        """Convert raw request bytes to tensor."""
        from PIL import Image
        import torchvision.transforms as T
        import io

        transform = T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensors = []
        for row in data:
            img_bytes = row.get("data") or row.get("body")
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            tensors.append(transform(img))
        return torch.stack(tensors)

    def inference(self, x: torch.Tensor):
        with torch.no_grad():
            return self.model(x.to(self.device))

    def postprocess(self, logits: torch.Tensor):
        probs = F.softmax(logits, dim=-1)
        top5_probs, top5_idx = probs.topk(5, dim=-1)
        results = []
        for probs_i, idx_i in zip(top5_probs, top5_idx):
            results.append({
                str(i.item()): float(p.item())
                for p, i in zip(probs_i, idx_i)
            })
        return results
```

```bash
# ── 2. Package model archive ────────────────────────────────────────────────
torch-model-archiver \
    --model-name resnet50 \
    --version 1.0 \
    --serialized-file resnet50_traced.pt \
    --handler my_handler.py \
    --extra-files imagenet_classes.json \
    --output-path model_store/

# ── 3. Start TorchServe ────────────────────────────────────────────────────
torchserve \
    --start \
    --model-store model_store \
    --models resnet50=resnet50.mar \
    --ts-config config.properties \
    --log-config log4j.properties

# config.properties
# inference_address=http://0.0.0.0:8080
# management_address=http://0.0.0.0:8081
# metrics_address=http://0.0.0.0:8082
# number_of_netty_threads=4
# job_queue_size=1000
# batch_size=32
# max_batch_delay=100  # ms
# default_workers_per_model=2

# ── 4. Use the API ────────────────────────────────────────────────────────
# curl -X POST http://localhost:8080/predictions/resnet50 -T image.jpg
# curl http://localhost:8081/models/resnet50  # management API
# curl http://localhost:8082/metrics           # Prometheus metrics
```

---

## 14.5 Dockerising the Inference Server

```dockerfile
# Dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and model
COPY inference_server.py .
COPY model.pt .

# Non-root user
RUN useradd -m -u 1000 mluser
USER mluser

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "inference_server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop"]
```

```bash
# Build and push
docker build -t my-inference-server:v1.0 .
docker push my-inference-server:v1.0

# Run with GPU
docker run --gpus all -p 8000:8000 my-inference-server:v1.0

# Test
curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"inputs": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]}'
```

---

## 14.6 Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pytorch-inference
  labels:
    app: pytorch-inference
    version: v1.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pytorch-inference
  template:
    metadata:
      labels:
        app: pytorch-inference
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/path: "/metrics"
        prometheus.io/port: "8000"
    spec:
      containers:
      - name: inference
        image: my-inference-server:v1.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          failureThreshold: 3
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
      nodeSelector:
        accelerator: nvidia-tesla-a100
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pytorch-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pytorch-inference
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## 14.7 Production Monitoring & Alerting

```python
# monitoring.py — Prometheus metrics for the inference server
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests",
    labelnames=["status", "model_version"],
)
LATENCY = Histogram(
    "inference_latency_seconds",
    "Request latency in seconds",
    labelnames=["model_version"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
BATCH_SIZE = Histogram(
    "inference_batch_size",
    "Batch size per request",
    buckets=[1, 2, 4, 8, 16, 32, 64],
)
GPU_MEMORY = Gauge("gpu_memory_allocated_bytes", "GPU memory allocated", labelnames=["device"])


def record_inference(batch_size: int, latency_s: float, success: bool, version: str = "v1"):
    status = "success" if success else "error"
    REQUEST_COUNT.labels(status=status, model_version=version).inc()
    LATENCY.labels(model_version=version).observe(latency_s)
    BATCH_SIZE.observe(batch_size)
    if torch.cuda.is_available():
        GPU_MEMORY.labels(device="cuda:0").set(torch.cuda.memory_allocated())


# Data drift detection
class DriftDetector:
    """
    Monitors statistical properties of incoming data.
    Alerts when distribution drifts from training baseline.
    """

    def __init__(self, baseline_stats: dict, threshold: float = 0.1):
        self.baseline = baseline_stats  # {"mean": [...], "std": [...]}
        self.threshold = threshold
        self.buffer = []
        self.buffer_size = 1000

    def observe(self, x: torch.Tensor):
        self.buffer.append(x.detach().cpu())
        if len(self.buffer) >= self.buffer_size:
            self._check_drift()
            self.buffer.clear()

    def _check_drift(self):
        data = torch.stack(self.buffer)
        curr_mean = data.mean(0).numpy()
        baseline_mean = self.baseline["mean"]

        rel_drift = abs(curr_mean - baseline_mean) / (abs(baseline_mean) + 1e-8)
        if rel_drift.max() > self.threshold:
            print(f"ALERT: Data drift detected! Max relative drift: {rel_drift.max():.3f}")
            # In production: send to PagerDuty, Slack, etc.
```

---

## 14.8 A/B Testing Models

```python
import random
from typing import Callable
import torch

class ABTestRouter:
    """
    Routes a fraction of traffic to a canary (new) model.
    Records outcomes for statistical comparison.
    """

    def __init__(
        self,
        control_model: torch.nn.Module,  # current production model
        canary_model: torch.nn.Module,   # new model being tested
        canary_fraction: float = 0.1,    # 10% traffic to canary
    ):
        self.control = control_model
        self.canary  = canary_model
        self.canary_fraction = canary_fraction
        self.control_results = []
        self.canary_results  = []

    def predict(self, x: torch.Tensor, ground_truth=None) -> tuple:
        use_canary = random.random() < self.canary_fraction
        model_name = "canary" if use_canary else "control"
        model      = self.canary if use_canary else self.control

        with torch.inference_mode():
            output = model(x)

        if ground_truth is not None:
            result = (output.argmax(-1) == ground_truth).float().mean().item()
            if use_canary:
                self.canary_results.append(result)
            else:
                self.control_results.append(result)

        return output, model_name

    def report(self) -> dict:
        if not self.control_results or not self.canary_results:
            return {"status": "insufficient_data"}

        ctrl_acc   = sum(self.control_results) / len(self.control_results)
        canary_acc = sum(self.canary_results)  / len(self.canary_results)
        return {
            "control_accuracy":      ctrl_acc,
            "canary_accuracy":       canary_acc,
            "delta":                 canary_acc - ctrl_acc,
            "control_samples":       len(self.control_results),
            "canary_samples":        len(self.canary_results),
            "recommendation":        "promote" if canary_acc > ctrl_acc + 0.005 else "hold",
        }
```

---

## Exercises

**Exercise 14.1** Build a complete image classification serving API using FastAPI + TorchScript (ResNet-50, ImageNet). Add: input validation (image size, channels), preprocessing, top-5 class names in response, and Prometheus metrics.

**Exercise 14.2** Implement request-level caching using an LRU cache keyed on input tensor hash. Measure hit rate and latency improvement for repeated requests.

**Exercise 14.3** Write a Kubernetes deployment manifest for the inference server with: GPU node selector, HPA based on GPU memory utilisation (custom metrics via DCGM), and a PodDisruptionBudget ensuring at least 1 pod is always running.

---

## Module Summary

| Component | Technology | Purpose |
|-----------|----------|--------|
| REST API | FastAPI + uvicorn | Single-model HTTP inference |
| Multi-model server | TorchServe | Production multi-model, batching, versioning |
| Dynamic batching | asyncio Queue | GPU utilisation for async servers |
| Container | Docker (nvidia runtime) | Reproducible, portable deployment |
| Orchestration | Kubernetes + HPA | Scaling, HA, rolling updates |
| Monitoring | Prometheus + Grafana | Latency, throughput, drift |
| A/B testing | Custom router | Safe canary rollouts |

---

## Quiz

1. Why must you warm up the model before serving real traffic?
2. What is dynamic batching and why does it improve GPU utilisation?
3. What is the difference between readiness and liveness probes in Kubernetes?
4. How does A/B testing differ from canary deployment?
5. What does `asyncio.Lock()` protect in the inference server?
6. Why use `workers=1` in uvicorn for GPU inference servers?
7. What is data drift and how would you detect it in production?

---

*Next: [Module 15 — Evaluation & Model Interpretability](./15_evaluation_and_interpretability.md)*
