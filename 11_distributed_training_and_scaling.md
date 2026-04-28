# Module 11: Distributed Training & Scaling — Training at Scale

> **Goal:** Understand how to scale training from one GPU to hundreds — the strategies, trade-offs, and real implementation patterns used for training large models.

---

## Learning Objectives

By the end of this module, you will:
- **Understand** the taxonomy of parallelism: data, model, tensor, pipeline
- **Implement** DDP (DistributedDataParallel) for multi-GPU data parallelism
- **Use** FSDP (Fully Sharded Data Parallel) for memory-efficient large model training
- **Launch** distributed jobs with `torchrun`
- **Debug** common distributed training issues
- **Apply** the linear scaling rule and gradient synchronization strategies

---

## Part 1: Why Distributed Training?

### 1.1 The Scaling Problem

```
Single A100 (80GB VRAM):
- GPT-3 (175B params): 175B × 4 bytes/param = 700GB weights alone!
  → Won't fit on a single GPU
- LLaMA-7B (7B params): 7B × 4 = 28GB weights + optimizer state = ~100GB
  → Won't fit on a single GPU in training mode

Even if the model fits, training on more GPUs = faster iteration!
```

### 1.2 Parallelism Taxonomy

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Distributed Training                          │
│                                                                      │
│  Data Parallelism          Model Parallelism                         │
│  ┌───────────────────┐     ┌────────────────────────────────────┐   │
│  │ Same model on     │     │ Different parts of model on        │   │
│  │ all GPUs          │     │ different GPUs                     │   │
│  │                   │     │                                    │   │
│  │ DDP (gradients)   │     │ Tensor Parallelism │ Pipeline Par. │   │
│  │ FSDP (sharded)    │     │ (split layers)    │ (split stages)│   │
│  └───────────────────┘     └────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: DistributedDataParallel (DDP)

### 2.1 How DDP Works

DDP is the **most common** distributed training strategy:
1. Every GPU has a **full copy** of the model
2. Each GPU processes a **different batch** (data parallelism)
3. After backward, gradients are **all-reduced** across all GPUs
4. All GPUs update identically → models stay in sync

```
GPU 0: batch_0 → forward → loss_0 → backward → grad_0 ─┐
GPU 1: batch_1 → forward → loss_1 → backward → grad_1 ─┤─ All-Reduce ─> avg_grad
GPU 2: batch_2 → forward → loss_2 → backward → grad_2 ─┤               (Σgrad/N)
GPU 3: batch_3 → forward → loss_3 → backward → grad_3 ─┘
                                                          │
All GPUs: param = param - lr * avg_grad ←────────────────┘
```

### 2.2 Complete DDP Training Script

```python
# train_ddp.py — Launch with: torchrun --nproc_per_node=4 train_ddp.py
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision

def setup_distributed():
    """
    Initialize the distributed process group.
    
    torchrun sets these environment variables automatically:
    - RANK: global rank of this process (0 to world_size-1)
    - LOCAL_RANK: rank on this machine (0 to n_gpus_per_node-1)
    - WORLD_SIZE: total number of processes
    - MASTER_ADDR, MASTER_PORT: rendezvous point
    """
    dist.init_process_group(
        backend='nccl',  # NCCL: NVIDIA's optimized collective ops
                          # Use 'gloo' for CPU-only distributed training
        init_method='env://',  # Use environment variables set by torchrun
    )
    
    # Each process controls one GPU
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    return local_rank

def cleanup_distributed():
    """Must call at end of script to cleanly shut down process group"""
    dist.destroy_process_group()

def is_main_process():
    """Only rank 0 should print/log/save to avoid duplicates"""
    return dist.get_rank() == 0

def main():
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')
    
    if is_main_process():
        print(f"Training with {world_size} GPUs")
    
    # ── Dataset ─────────────────────────────────────────────────────────────
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = torchvision.datasets.CIFAR10('./data', train=True,
                                            download=True, transform=transform)
    
    # DistributedSampler: ensures each GPU sees a DIFFERENT subset
    # Each epoch, every GPU processes 1/world_size of the data
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,  # Total processes
        rank=rank,                 # This process's rank
        shuffle=True,              # Shuffle within each rank's subset
        drop_last=True,            # Drop incomplete last batch
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=256,            # Per-GPU batch size
                                    # Effective batch = 256 * world_size
        sampler=sampler,           # USE sampler, NOT shuffle=True
        num_workers=4,
        pin_memory=True,
    )
    
    # ── Model ────────────────────────────────────────────────────────────────
    model = torchvision.models.resnet50(pretrained=False)
    model = model.to(device)
    
    # Wrap model with DDP
    model = DDP(
        model,
        device_ids=[local_rank],   # GPUs this process uses
        output_device=local_rank,  # Where to gather output
        
        # find_unused_parameters=True: required for models with conditional computation
        # Warning: adds overhead! Only use if needed
        find_unused_parameters=False,
        
        # gradient_as_bucket_view=True: slight memory optimization
        gradient_as_bucket_view=True,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # ── Training Loop ─────────────────────────────────────────────────────
    for epoch in range(10):
        # CRITICAL: set epoch in sampler for correct shuffling
        # Without this, all GPUs get the same data order every epoch!
        sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0.0
        
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()  # DDP all-reduces gradients automatically!
            
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Aggregate loss across all GPUs for logging
        avg_loss_tensor = torch.tensor(total_loss, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        
        if is_main_process():
            print(f"Epoch {epoch}: avg_loss = {avg_loss_tensor.item():.4f}")
            
            # Only save from rank 0 to avoid duplicate writes
            # Save model.module.state_dict() (not model.state_dict())
            # model.module is the underlying model before DDP wrapping
            torch.save(model.module.state_dict(), f'checkpoint_epoch{epoch}.pt')
    
    cleanup_distributed()

if __name__ == '__main__':
    main()
```

### 2.3 Launching DDP Jobs

```bash
# Launch on single machine with 4 GPUs:
torchrun --standalone --nproc_per_node=4 train_ddp.py

# Launch on 2 machines, 4 GPUs each (8 GPUs total):
# On machine 1 (master):
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
         --master_addr=<machine1_ip> --master_port=29500 train_ddp.py

# On machine 2:
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \
         --master_addr=<machine1_ip> --master_port=29500 train_ddp.py
```

### 2.4 Gradient Accumulation with DDP

```python
# With gradient accumulation, avoid syncing gradients on micro-steps
# (only sync on the "real" optimizer step)

ACCUMULATE = 4  # Sync every 4 steps

for step, (x, y) in enumerate(train_loader):
    # model.no_sync(): temporarily disables gradient all-reduce
    # This avoids expensive communication on every micro-step
    if (step + 1) % ACCUMULATE != 0:
        with model.no_sync():  # No all-reduce during accumulation
            loss = criterion(model(x), y) / ACCUMULATE
            loss.backward()
    else:
        # Last accumulation step: perform all-reduce
        loss = criterion(model(x), y) / ACCUMULATE
        loss.backward()  # All-reduce happens here
        
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

---

## Part 3: FSDP — Fully Sharded Data Parallel

### 3.1 DDP vs FSDP Memory Usage

```
Model: 10B parameters

DDP (each GPU stores full model):
- 4 GPUs × (40GB weights + 40GB gradients + 120GB optimizer) = 4 × 200GB
- Total GPU memory: 800GB
- Problem: Each GPU needs 200GB → only works on H100/A100 with 80GB

FSDP (shards everything across GPUs):
- Weights: 40GB / 4 GPUs = 10GB per GPU
- Gradients: 40GB / 4 GPUs = 10GB per GPU
- Optimizer: 120GB / 4 GPUs = 30GB per GPU
- Total per GPU: 50GB
- FSDP enables training 10B model on 4×80GB GPUs!

During forward/backward: gather full layer momentarily, compute, discard
```

### 3.2 FSDP Implementation

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
import functools

def setup_fsdp_training(model, local_rank):
    """Configure FSDP for a transformer model"""
    
    # Mixed precision: forward in BF16, reduce in FP32
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,      # Store params in BF16
        reduce_dtype=torch.float32,       # Reduce gradients in FP32 (for stability)
        buffer_dtype=torch.bfloat16,      # Buffers (running mean/var) in BF16
    )
    
    # FSDP sharding strategies:
    # FULL_SHARD:     Shard params + gradients + optimizer state (maximum savings)
    # SHARD_GRAD_OP:  Shard gradients + optimizer state (less savings, faster)
    # NO_SHARD:       Same as DDP (no sharding)
    # HYBRID_SHARD:   Full shard within node, replicate across nodes
    
    # Auto-wrap policy: wrap each transformer block independently
    # This ensures FSDP gathers/scatters at the right granularity
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={GPT2Block},  # Wrap each transformer block
    )
    
    # Wrap model with FSDP
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        
        # BackwardPrefetch: pre-fetches next layer's params during backward
        # Overlaps communication with computation → reduces idle time
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        
        device_id=local_rank,
    )
    
    return fsdp_model
```

---

## Part 4: The Linear Scaling Rule

### 4.1 Scaling Batch Size with Learning Rate

When you increase the batch size by N, the gradient is averaged over N times more samples. To maintain the same effective gradient update, scale the learning rate proportionally.

```
Rule: LR_new = LR_base × (batch_size_new / batch_size_base)
Example: base LR=0.01 with batch=256, scale to batch=1024:
         new LR = 0.01 × (1024/256) = 0.04

Works well up to:   batch ≈ 8,192 (for ImageNet classification)
Breaks down above: batch > ~8K (gradient noise is beneficial for generalization)
When batch > threshold: use LR warmup to stabilize
```

```python
def scale_lr_for_batch(base_lr: float, base_batch: int, actual_batch: int,
                        n_gpus: int, accumulate_steps: int) -> float:
    """
    Compute the scaled learning rate for distributed training.
    
    Effective batch = per_gpu_batch × n_gpus × accumulate_steps
    """
    effective_batch = actual_batch * n_gpus * accumulate_steps
    scale = effective_batch / base_batch
    return base_lr * scale

# Example: Training GPT-2
base_lr    = 6e-4    # OpenAI's original single-GPU LR
base_batch = 512     # Original batch size

# Training on 8 GPUs with batch=64, accumulate=8
# Effective batch = 64 × 8 × 8 = 4096
new_lr = scale_lr_for_batch(base_lr, base_batch, 64, n_gpus=8, accumulate_steps=8)
print(f"Scaled LR: {new_lr:.2e}")  # 6e-4 × (4096/512) = 4.8e-3
```

---

## Part 5: Collective Communication Operations

### 5.1 Understanding All-Reduce, Broadcast, and All-Gather

```python
# All-reduce: every process contributes a tensor, all receive the combined result
# DDP uses this to average gradients

if dist.is_initialized():
    tensor = torch.tensor([dist.get_rank() + 1.0], device='cuda')
    print(f"Before all_reduce, rank {dist.get_rank()}: {tensor}")
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"After all_reduce (sum): {tensor}")  # Sum of all ranks
    
    # Average: divide by world_size
    tensor /= dist.get_world_size()
    print(f"After averaging: {tensor}")  # Average across all ranks

# Broadcast: one process sends its tensor to all others
# Use case: ensure all GPUs start with the same model
    tensor = torch.zeros(3) if dist.get_rank() != 0 else torch.tensor([1.0, 2.0, 3.0])
    dist.broadcast(tensor, src=0)  # Rank 0 sends to all
    print(f"After broadcast: {tensor}")  # All ranks have [1, 2, 3]

# All-gather: collect different tensors from all ranks into one
    local = torch.tensor([dist.get_rank()], device='cuda', dtype=torch.float32)
    gathered = [torch.zeros(1, device='cuda') for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    # Now: gathered = [[0], [1], [2], [3]] on all ranks
```

---

## Part 6: HuggingFace Accelerate — The Easy Way

```python
# Accelerate: abstraction over DDP, FSDP, DeepSpeed
# Same training code works on 1 GPU, 8 GPUs, or 8 machines!

from accelerate import Accelerator

accelerator = Accelerator(
    gradient_accumulation_steps=4,
    mixed_precision='bf16',  # 'fp16', 'bf16', or 'no'
    log_with='tensorboard',
)

# Prepare everything — accelerate handles device placement and wrapping
model, optimizer, train_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, scheduler
)

# Training loop — almost identical to single GPU!
for epoch in range(n_epochs):
    model.train()
    for x, y in train_loader:
        # x, y are already on the correct device
        with accelerator.accumulate(model):  # Handles gradient accumulation
            logits = model(x)
            loss = criterion(logits, y)
            accelerator.backward(loss)  # Handles scaled backward
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
    # Only save from main process
    if accelerator.is_main_process:
        accelerator.save_state('checkpoint/')

# Launch: accelerate launch --num_processes=4 train.py
# No code changes needed — same script for 1 or N GPUs!
```

---

## Key Takeaways

| Strategy | When to Use | Memory | Speed |
|----------|------------|--------|-------|
| **DDP** | Model fits on 1 GPU | N × full model | Linear scaling |
| **FSDP** | Model too large for 1 GPU | ~1/N of DDP | Linear scaling |
| **Tensor Parallel** | Ultra-large models (100B+) | ~1/N per layer | Communication overhead |
| **Pipeline Parallel** | Ultra-large models | Stage memory | Pipeline bubbles |

---

## Quiz

1. **What does DDP's all-reduce operation do?**
   - Answer: Averages gradients across all GPUs so every process has the same gradient for the optimizer step

2. **Why must you call `sampler.set_epoch(epoch)` in DDP?**
   - Answer: Without it, all GPUs shuffle the same way every epoch, violating data parallelism

3. **What is the key difference between DDP and FSDP?**
   - Answer: DDP keeps a full model copy on each GPU; FSDP shards parameters, gradients, and optimizer state across GPUs

4. **What does `model.no_sync()` do in DDP?**
   - Answer: Defers gradient all-reduce to the next sync point, used during gradient accumulation micro-steps

5. **What is the linear scaling rule?**
   - Answer: When batch size increases by N, learning rate should also increase by N

6. **What is `BackwardPrefetch.BACKWARD_PRE` in FSDP?**
   - Answer: Pre-fetches next layer's parameters during backward pass, overlapping communication with compute

7. **How do you save a DDP-wrapped model's state_dict correctly?**
   - Answer: `model.module.state_dict()` — access `.module` to get the unwrapped underlying model

8. **What is the NCCL backend and when is it used?**
   - Answer: NVIDIA's optimized collective communication library; used for GPU-GPU communication in NCCL

9. **What does Accelerate's `accelerator.prepare()` do?**
   - Answer: Wraps model, optimizer, and dataloader for the target hardware (DDP, FSDP, DeepSpeed, etc.)

10. **Why does FSDP gather parameters during forward and discard after?**
    - Answer: Parameters are sharded across GPUs; gather = reconstruct full layer temporarily for computation, then discard to reclaim memory
