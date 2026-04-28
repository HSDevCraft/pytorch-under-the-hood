# Module 14: Serving & Production — From Model to Live API

> **Goal:** Build production-grade ML serving systems — from a simple FastAPI server to auto-scaling Kubernetes deployments with real-time monitoring.

---

## Learning Objectives

By the end of this module, you will:
- **Build** a production FastAPI inference server with proper error handling
- **Implement** dynamic batching to maximize GPU utilization
- **Configure** TorchServe for scalable model serving
- **Containerize** with Docker and deploy on Kubernetes
- **Monitor** with Prometheus metrics (latency, throughput, errors)
- **Run** A/B tests between model versions

---

## Part 1: The Production Serving Architecture

### 1.1 Why Production Serving Is Hard

A research notebook is not a serving system. Production requires:
- **Concurrency:** Handle 1000 simultaneous requests
- **Reliability:** 99.99% uptime, graceful error handling
- **Performance:** < 50ms p95 latency under load
- **Observability:** Know what's failing and why
- **Scalability:** Auto-scale with traffic
- **Safety:** Input validation, rate limiting, authentication

```
Client Request Flow:
Browser/App → Load Balancer → Inference Server → Model → Response
                                     ↓
                               Monitoring (Prometheus)
                               Logging (structlog)
                               Tracing (OpenTelemetry)
```

---

## Part 2: FastAPI Inference Server

### 2.1 Production-Grade Server Implementation

```python
# inference_server.py
import torch
import torch.nn as nn
import asyncio
import time
import logging
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, validator
import numpy as np

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Model wrapper ─────────────────────────────────────────────────────────────
class ModelInference:
    """
    Thread-safe model wrapper with device management and error handling.
    
    Design decisions:
    1. Singleton pattern — load model once, reuse for all requests
    2. eval() mode always set — Dropout off, BN uses running stats
    3. torch.no_grad() — no gradient tracking = faster + less memory
    4. asyncio.Lock() — prevent concurrent GPU access (race conditions)
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = self._select_device(device)
        self.model = self._load_model(model_path)
        self._lock = asyncio.Lock()  # Prevent concurrent GPU access
        logger.info(f"Model loaded on {self.device}")
    
    def _select_device(self, device: str) -> torch.device:
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self, path: str) -> nn.Module:
        # Load TorchScript model (no model class definition needed!)
        model = torch.jit.load(path, map_location=self.device)
        model.eval()
        return model
    
    async def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Async prediction with exclusive GPU access.
        
        async + Lock prevents multiple coroutines from accessing GPU simultaneously.
        Without Lock: GPU context switching overhead, potential memory errors.
        """
        async with self._lock:
            # Move input to model's device
            x = x.to(self.device)
            
            with torch.no_grad():
                logits = self.model(x)
            
            return logits.cpu()  # Return to CPU (serializable)


# ── Request/Response schemas ──────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """
    Pydantic models provide automatic:
    - Type coercion (list → np.array → tensor)
    - Input validation (shapes, ranges)
    - OpenAPI documentation
    """
    instances: List[List[float]]  # List of feature vectors
    
    @validator('instances')
    def validate_shape(cls, v):
        if not v:
            raise ValueError("instances cannot be empty")
        n_features = len(v[0])
        if not all(len(row) == n_features for row in v):
            raise ValueError("All instances must have the same number of features")
        if n_features != 784:
            raise ValueError(f"Expected 784 features, got {n_features}")
        return v

class PredictResponse(BaseModel):
    predictions: List[int]          # Predicted class indices
    probabilities: List[List[float]] # Class probabilities
    latency_ms: float                # Server-side inference time
    model_version: str               # For tracking which model served

# ── Application lifecycle ─────────────────────────────────────────────────────
# Lifespan: modern FastAPI pattern for startup/shutdown (replaces on_event)
model_inference: ModelInference = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, release at shutdown"""
    global model_inference
    
    logger.info("Loading model...")
    model_inference = ModelInference(
        model_path="model_traced.pt",
        device='auto'
    )
    logger.info("Model ready — server starting")
    
    yield  # Application runs here
    
    # Cleanup on shutdown
    logger.info("Shutting down — releasing model")
    del model_inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="ML Inference API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Middleware: request logging ───────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with method, path, and response time"""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} "
        f"latency={duration_ms:.1f}ms"
    )
    return response

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """
    Kubernetes liveness + readiness probe.
    Returns 200 if model is loaded and ready.
    """
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    device = str(model_inference.device)
    return {
        "status": "healthy",
        "device": device,
        "model_loaded": True,
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Main inference endpoint.
    
    Error handling:
    - 422: Pydantic validation errors (bad input format)
    - 500: Model inference errors (internal)
    - 503: Model not loaded (startup incomplete)
    """
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    inference_start = time.perf_counter()
    
    try:
        # Convert list → numpy → tensor
        x = torch.tensor(request.instances, dtype=torch.float32)
        
        # Run inference
        logits = await model_inference.predict(x)
        
        # Post-process
        probabilities = torch.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        
        latency_ms = (time.perf_counter() - inference_start) * 1000
        
        return PredictResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities.tolist(),
            latency_ms=latency_ms,
            model_version="1.0.0",
        )
    
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# Run: uvicorn inference_server:app --host 0.0.0.0 --port 8000 --workers 1
# Note: workers=1 for GPU servers! Multiple workers = multiple model copies in GPU memory
```

---

## Part 3: Dynamic Batching

### 3.1 Why Batch Requests Together

```
Without batching:
  Request 1 → GPU → Response   (GPU 5% utilized)
  Request 2 → GPU → Response   (GPU 5% utilized)
  ...
  Throughput: 20 req/sec

With dynamic batching:
  Request 1 ─┐
  Request 2  ├─ wait 10ms → batch(1,2,3,4) → GPU → Responses
  Request 3  │                               (GPU 80% utilized)
  Request 4 ─┘
  Throughput: 200 req/sec (10× improvement!)
```

```python
import asyncio
from typing import Tuple
import torch

class DynamicBatcher:
    """
    Collects individual inference requests and batches them for GPU efficiency.
    
    Key parameters:
    - max_batch_size: hard cap on batch size (memory constraint)
    - max_wait_ms: maximum wait time before processing partial batch
      - Small: low latency, low throughput
      - Large: higher latency, high throughput
      - Tune based on SLA requirements
    """
    
    def __init__(self, model: nn.Module, max_batch_size: int = 32,
                 max_wait_ms: float = 10.0):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000.0
        
        # Queue of (tensor, Future) pairs
        # Each request adds itself to the queue and waits for a result
        self._queue: asyncio.Queue[Tuple[torch.Tensor, asyncio.Future]] = asyncio.Queue()
        self._worker_task: asyncio.Task = None
    
    async def start(self):
        """Start the background batching worker"""
        self._worker_task = asyncio.create_task(self._worker())
    
    async def stop(self):
        """Clean up worker"""
        if self._worker_task:
            self._worker_task.cancel()
            await asyncio.gather(self._worker_task, return_exceptions=True)
    
    async def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Submit a single sample for batched inference.
        Returns when the batch is processed.
        """
        future = asyncio.get_event_loop().create_future()
        await self._queue.put((x, future))
        return await future  # Wait for batch to complete
    
    async def _worker(self):
        """
        Background worker: collects requests and runs batched inference.
        
        Strategy:
        1. Wait for first request (block until something arrives)
        2. Collect more requests for up to max_wait_ms
        3. Run batch inference
        4. Set results on individual futures
        """
        while True:
            # Step 1: Wait for at least one request
            first_tensor, first_future = await self._queue.get()
            
            batch_tensors = [first_tensor]
            batch_futures = [first_future]
            
            # Step 2: Collect more requests up to limits
            deadline = asyncio.get_event_loop().time() + self.max_wait_s
            
            while (len(batch_tensors) < self.max_batch_size and
                   asyncio.get_event_loop().time() < deadline):
                try:
                    # Non-blocking check for more requests (tiny timeout)
                    tensor, future = await asyncio.wait_for(
                        self._queue.get(), timeout=0.001
                    )
                    batch_tensors.append(tensor)
                    batch_futures.append(future)
                except asyncio.TimeoutError:
                    break  # No more requests ready; process what we have
            
            # Step 3: Run batch inference
            batch = torch.stack(batch_tensors, dim=0)
            
            try:
                with torch.no_grad():
                    outputs = self.model(batch)  # Shape: (batch_size, n_classes)
                
                # Step 4: Return individual results
                for i, future in enumerate(batch_futures):
                    if not future.cancelled():
                        future.set_result(outputs[i])
            
            except Exception as e:
                # On failure, propagate exception to all waiting requests
                for future in batch_futures:
                    if not future.cancelled():
                        future.set_exception(e)
```

---

## Part 4: Docker Containerization

### 4.1 Production Dockerfile

```dockerfile
# Dockerfile
# Multi-stage build: keep production image slim

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

# Install dependencies in virtual env (isolate from system Python)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Production ───────────────────────────────────────────────────────
FROM python:3.11-slim

# Security: run as non-root user
RUN useradd --create-home --uid 1001 appuser

WORKDIR /app

# Copy virtual env from builder (avoids re-installing)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY inference_server.py .
COPY model_traced.pt .

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Health check: Docker marks container unhealthy if /health fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

```bash
# Build and run
docker build -t ml-inference:v1.0 .
docker run -p 8000:8000 --gpus all ml-inference:v1.0

# Test the running server
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[0.0]*784]}'
```

---

## Part 5: Monitoring with Prometheus

### 5.1 Instrumenting Your Server

```python
# metrics.py — Prometheus metrics for ML serving
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)
from fastapi import Response

# ── Define metrics ────────────────────────────────────────────────────────────
# Counter: monotonically increasing (requests, errors)
REQUEST_COUNT = Counter(
    'ml_request_total',
    'Total inference requests',
    labelnames=['endpoint', 'status'],  # Dimensions for filtering
)

# Histogram: distribution of values (latency percentiles)
REQUEST_LATENCY = Histogram(
    'ml_request_latency_seconds',
    'Inference request latency',
    labelnames=['endpoint'],
    # Bucket boundaries for percentile computation
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

# Gauge: current value (can go up or down)
BATCH_SIZE = Gauge('ml_current_batch_size', 'Current batch size being processed')
GPU_MEMORY = Gauge('ml_gpu_memory_bytes', 'Current GPU memory usage')

# ── Instrument the predict endpoint ──────────────────────────────────────────
import functools
import time

def track_metrics(endpoint_name: str):
    """Decorator to automatically track request metrics"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            REQUEST_COUNT.labels(endpoint=endpoint_name, status='started').inc()
            start = time.perf_counter()
            
            try:
                result = await func(*args, **kwargs)
                REQUEST_COUNT.labels(endpoint=endpoint_name, status='success').inc()
                return result
            except Exception as e:
                REQUEST_COUNT.labels(endpoint=endpoint_name, status='error').inc()
                raise
            finally:
                latency = time.perf_counter() - start
                REQUEST_LATENCY.labels(endpoint=endpoint_name).observe(latency)
                
                # Update GPU memory gauge
                if torch.cuda.is_available():
                    GPU_MEMORY.set(torch.cuda.memory_allocated())
        
        return wrapper
    return decorator

# Apply to endpoint:
@app.post("/predict")
@track_metrics("predict")
async def predict(request: PredictRequest):
    ...  # Same as before

# Metrics endpoint for Prometheus scraping
@app.get("/metrics")
async def metrics():
    """Prometheus scrapes this endpoint every 15 seconds"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

---

## Part 6: A/B Testing Model Versions

```python
import random
from enum import Enum

class ModelVersion(str, Enum):
    V1 = "v1"  # Current production model (stable)
    V2 = "v2"  # New candidate model (under test)

class ABTestingRouter:
    """
    Routes requests between model versions based on traffic split.
    
    Example: 90% → v1 (stable), 10% → v2 (candidate)
    
    For statistical significance with 10% traffic split:
    - Need ~1000 requests to see 1% accuracy difference
    - Use proper statistical test (z-test for proportions)
    """
    
    def __init__(self, model_v1: ModelInference, model_v2: ModelInference,
                 v2_traffic_pct: float = 0.10):
        self.models = {
            ModelVersion.V1: model_v1,
            ModelVersion.V2: model_v2,
        }
        self.v2_traffic_pct = v2_traffic_pct
        self.results = {v: {'correct': 0, 'total': 0} for v in ModelVersion}
    
    def route_request(self) -> ModelVersion:
        """Random routing based on traffic split"""
        if random.random() < self.v2_traffic_pct:
            return ModelVersion.V2
        return ModelVersion.V1
    
    async def predict(self, x: torch.Tensor,
                      true_label: int = None) -> dict:
        version = self.route_request()
        model = self.models[version]
        
        logits = await model.predict(x)
        pred = logits.argmax(dim=-1).item()
        
        # Track accuracy if ground truth available (shadow mode)
        if true_label is not None:
            self.results[version]['total'] += 1
            if pred == true_label:
                self.results[version]['correct'] += 1
        
        return {
            'prediction': pred,
            'model_version': version.value,
        }
    
    def get_stats(self) -> dict:
        """Compare model versions statistically"""
        stats = {}
        for version, counts in self.results.items():
            if counts['total'] > 0:
                accuracy = counts['correct'] / counts['total']
                stats[version.value] = {
                    'accuracy': accuracy,
                    'n_requests': counts['total'],
                }
        return stats
    
    def should_promote_v2(self, min_requests: int = 1000,
                           min_improvement: float = 0.005) -> bool:
        """
        Check if v2 should replace v1 in production.
        Conservative threshold: >0.5% improvement with >1000 samples.
        """
        stats = self.get_stats()
        if 'v1' not in stats or 'v2' not in stats:
            return False
        if stats['v2']['n_requests'] < min_requests:
            return False
        improvement = stats['v2']['accuracy'] - stats['v1']['accuracy']
        return improvement >= min_improvement
```

---

## Key Takeaways

| Component | Purpose | Key Decision |
|-----------|---------|-------------|
| **FastAPI + asyncio** | Handle concurrent requests | workers=1 for GPU servers |
| **Dynamic Batching** | Maximize GPU utilization | Tune max_wait_ms vs max_batch_size |
| **Docker** | Reproducible deployment | Non-root user, health checks |
| **Prometheus** | Observability | Track latency p50/p95/p99 |
| **A/B Testing** | Safe model updates | 10% traffic → collect stats → decide |

---

## Quiz

1. **Why use `workers=1` for GPU inference servers?**
   - Answer: Multiple uvicorn workers create separate processes each loading the model into GPU memory, causing OOM

2. **What is dynamic batching and why does it improve throughput?**
   - Answer: Groups multiple individual requests into one GPU batch; GPUs are efficient with larger batches; amortizes kernel launch overhead

3. **What does `asyncio.Lock()` protect in the inference server?**
   - Answer: Prevents concurrent GPU access from multiple async handlers; prevents race conditions and memory errors

4. **What is the role of the `/health` endpoint?**
   - Answer: Kubernetes liveness/readiness probe; returns 200 only when model is loaded and ready

5. **What is a Prometheus Counter vs Gauge vs Histogram?**
   - Answer: Counter: only increases (requests, errors); Gauge: current value (memory, connections); Histogram: distribution for percentiles

6. **Why should production Docker containers run as non-root?**
   - Answer: Security principle of least privilege; limits damage if container is compromised

7. **What is multi-stage Docker build?**
   - Answer: Uses a builder image for compilation, copies only artifacts to slim runtime image; reduces production image size

8. **What is the p95 latency metric?**
   - Answer: 95th percentile — 95% of requests are faster than this; better than average for catching tail latency

9. **How does A/B testing differ from a canary deployment?**
   - Answer: A/B: fixed traffic split for statistical comparison; Canary: gradually increase traffic % while monitoring for regressions

10. **Why use pydantic models for request/response?**
    - Answer: Automatic validation, type coercion, clear error messages, and auto-generated OpenAPI documentation
