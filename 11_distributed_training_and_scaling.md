# Module 11: Distributed Training & Scaling

## Learning Objectives
By the end of this module you will be able to:
- Understand data parallelism, model parallelism, and pipeline parallelism
- Launch multi-GPU training with `DistributedDataParallel` (DDP)
- Apply Fully Sharded Data Parallel (FSDP) for training billion-parameter models
- Configure gradient communication strategies and understand their trade-offs
- Use `torchrun` and `accelerate` for multi-node, multi-GPU training
- Implement model parallelism (tensor parallelism and pipeline parallelism)
- Debug distributed training: deadlocks, rank mismatches, and gradient errors

---

## 11.1 Parallelism Strategies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Parallelism Taxonomy                                   │
│                                                                              │
│  DATA PARALLELISM         MODEL PARALLELISM        PIPELINE PARALLELISM     │
│  ─────────────────        ─────────────────        ────────────────────     │
│  Full model on each GPU   Model split across GPUs  Layers split in stages   │
│  Different data each GPU  Same batch on all GPUs   Micro-batches overlap    │
│                                                                              │
│  DDP, FSDP               Tensor Parallelism        GPipe, PipeDream         │
│                           (Megatron-style)                                   │
│                                                                              │
│  Best for: most tasks     Best for: models too     Best for: very deep      │
│                           large for 1 GPU          models (GPT-3 scale)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 11.2 DistributedDataParallel (DDP)

DDP is the standard approach: each GPU holds a complete model replica and processes a different data shard. After each backward pass, gradients are **all-reduced** (averaged across GPUs), then all replicas apply the same update.

```python
# ── ddp_train.py: run with torchrun ──────────────────────────────────────────
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup_ddp():
    """Initialise the process group. Called once per process."""
    dist.init_process_group(backend="nccl")  # NCCL: fastest for GPU-GPU comms
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def train(rank: int, world_size: int):
    # ── Setup ────────────────────────────────────────────────────────────────
    local_rank = setup_ddp()
    device     = torch.device(f"cuda:{local_rank}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = ResNet50().to(device)
    # Wrap with DDP: gradients auto-synchronized during backward
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    # Access underlying model: model.module

    # ── Data: each GPU gets a different shard ─────────────────────────────────
    dataset = ImageFolderDataset("data/train", transform=train_transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=dist.get_rank(),
        shuffle=True,
        drop_last=True,
    )
    loader = DataLoader(
        dataset, batch_size=64, sampler=sampler,
        num_workers=4, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 * world_size)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.cuda.amp.GradScaler()

    for epoch in range(50):
        # CRITICAL: set epoch so sampler reshuffles differently each epoch
        sampler.set_epoch(epoch)

        model.train()
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                loss = criterion(model(x), y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        # Only rank 0 saves checkpoints and prints
        if dist.get_rank() == 0:
            torch.save(model.module.state_dict(), f"ckpt_epoch{epoch}.pt")
            print(f"Epoch {epoch}: loss={loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    train(rank=0, world_size=1)
```

### Launch Command

```bash
# Single node, 4 GPUs
torchrun --standalone --nproc_per_node=4 ddp_train.py

# Multi-node: 2 nodes × 4 GPUs = 8 GPUs total
# On node 0 (master):
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    ddp_train.py

# On node 1:
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    ddp_train.py
```

---

## 11.3 DDP Communication Strategies

```python
# ── Gradient buckets: control when all-reduce fires ──────────────────────────
model = DDP(
    model,
    device_ids=[local_rank],
    bucket_cap_mb=25,        # bucket size before all-reduce (default: 25MB)
    find_unused_parameters=False,  # faster; set True only if some params unused
    gradient_as_bucket_view=True,  # saves memory by aliasing grad buffers
)

# ── No-sync context: defer gradient sync for gradient accumulation ─────────
accumulate_steps = 4
for step, (x, y) in enumerate(loader):
    # Only sync on the last micro-step
    sync_context = model.no_sync() if (step + 1) % accumulate_steps != 0 else torch.no_grad.__class__()

    if (step + 1) % accumulate_steps != 0:
        with model.no_sync():
            loss = criterion(model(x), y) / accumulate_steps
            loss.backward()
    else:
        loss = criterion(model(x), y) / accumulate_steps
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# ── Collective operations ─────────────────────────────────────────────────────
# all_reduce: sum/average a tensor across all ranks
local_loss = torch.tensor([loss.item()], device=device)
dist.all_reduce(local_loss, op=dist.ReduceOp.AVG)
global_loss = local_loss.item()

# all_gather: collect tensors from all ranks
local_tensor = torch.randn(4, device=device)
gathered = [torch.zeros(4, device=device) for _ in range(dist.get_world_size())]
dist.all_gather(gathered, local_tensor)

# broadcast: send from rank 0 to all ranks
data = torch.zeros(100, device=device)
if dist.get_rank() == 0:
    data = torch.randn(100, device=device)
dist.broadcast(data, src=0)
```

---

## 11.4 Fully Sharded Data Parallel (FSDP)

DDP keeps a full model copy on each GPU. For models with billions of parameters, this is infeasible. **FSDP** shards model parameters, gradients, and optimizer states across GPUs — each GPU holds only 1/world_size of the total model.

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import functools

# ── FSDP mixed precision policy ───────────────────────────────────────────────
bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,     # store params in BF16
    reduce_dtype=torch.bfloat16,    # gradient reduction in BF16
    buffer_dtype=torch.bfloat16,    # buffers (BN stats etc.)
)

# ── Auto-wrap policy: wrap each transformer block independently ───────────────
# This is critical: wrapping too large a unit defeats the sharding
gpt2_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={GPT2Block},   # wrap each GPT-2 block independently
)

# ── Build FSDP model ──────────────────────────────────────────────────────────
def build_fsdp_model(model: nn.Module, local_rank: int) -> FSDP:
    return FSDP(
        model,
        auto_wrap_policy=gpt2_wrap_policy,
        mixed_precision=bf16_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,   # maximum sharding
        # ShardingStrategy.SHARD_GRAD_OP: only shard during backward (less comm)
        # ShardingStrategy.NO_SHARD: same as DDP (for debugging)
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # prefetch next shard
        cpu_offload=CPUOffload(offload_params=False),     # True for CPU offloading
        device_id=local_rank,
    )

model = build_fsdp_model(GPT2LMHeadModel(GPT2Config()), local_rank)

# ── Saving with FSDP ──────────────────────────────────────────────────────────
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

# Gather all shards to rank 0 for saving
with FSDP.state_dict_type(
    model,
    StateDictType.FULL_STATE_DICT,
    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
):
    state_dict = model.state_dict()

if dist.get_rank() == 0:
    torch.save(state_dict, "fsdp_model.pt")
```

---

## 11.5 Tensor Parallelism

Splits individual weight matrices across GPUs (Megatron-LM style). Used for the largest models (GPT-3, LLaMA-2 70B).

```python
# Simplified illustration of column vs row parallelism
# Column parallelism: Y = X @ W, split W by columns
# Row parallelism: Y = X @ W, split X and W by rows, then all-reduce

class ColumnParallelLinear(nn.Module):
    """
    Splits output features across GPUs.
    Each GPU computes a slice of the output: Y_i = X @ W_i
    No all-reduce needed — the next layer is row-parallel.
    """

    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int):
        super().__init__()
        assert out_features % world_size == 0
        self.local_out = out_features // world_size
        self.weight = nn.Parameter(torch.randn(in_features, self.local_out) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight   # (B, T, local_out) — no communication


class RowParallelLinear(nn.Module):
    """
    Splits input features across GPUs.
    Each GPU computes Y_i = X_i @ W_i, then all-reduce sums across GPUs.
    """

    def __init__(self, in_features: int, out_features: int, world_size: int):
        super().__init__()
        assert in_features % world_size == 0
        self.local_in = in_features // world_size
        self.weight   = nn.Parameter(torch.randn(self.local_in, out_features) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_out = x @ self.weight            # (B, T, out_features)
        dist.all_reduce(local_out, op=dist.ReduceOp.SUM)
        return local_out
```

---

## 11.6 HuggingFace Accelerate

`accelerate` abstracts away the boilerplate of DDP/FSDP/TPU/CPU training with a unified API.

```python
# pip install accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed

def train_with_accelerate():
    set_seed(42)
    accelerator = Accelerator(
        mixed_precision="bf16",      # "fp16", "bf16", or "no"
        gradient_accumulation_steps=4,
        # fsdp_plugin=... for FSDP
        # deepspeed_plugin=... for DeepSpeed
    )

    model     = MyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_dl  = DataLoader(train_ds, batch_size=32, shuffle=True)

    # Accelerate handles: device placement, DDP wrapping, mixed precision
    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)

    for epoch in range(10):
        model.train()
        for x, y in train_dl:
            # No .to(device) needed — accelerate handles it
            with accelerator.accumulate(model):   # handles gradient accumulation
                loss = criterion(model(x), y)
                accelerator.backward(loss)        # handles scaler
                optimizer.step()
                optimizer.zero_grad()

        if accelerator.is_main_process:
            print(f"Epoch {epoch}: loss = {loss.item():.4f}")

    # Save: unwrap and get plain model
    unwrapped = accelerator.unwrap_model(model)
    torch.save(unwrapped.state_dict(), "model.pt")

# Launch:
# accelerate config         # interactive setup
# accelerate launch train.py
```

---

## 11.7 Batch Size Scaling Rules

When scaling to multiple GPUs, the effective batch size increases:

```
effective_batch = per_gpu_batch × n_gpus × accumulate_steps
```

**Linear scaling rule** (Goyal et al., 2017): multiply LR proportionally to batch size:
```
lr_new = lr_base × (effective_batch / base_batch)
```

**Square-root scaling** (sometimes better for Adam):
```
lr_new = lr_base × √(effective_batch / base_batch)
```

```python
def scale_lr(base_lr: float, base_batch: int, effective_batch: int, rule: str = "linear") -> float:
    ratio = effective_batch / base_batch
    if rule == "linear":
        return base_lr * ratio
    elif rule == "sqrt":
        return base_lr * ratio ** 0.5
    return base_lr

# Example: base: lr=1e-3 for batch=256; scale to 8 GPUs × batch=256 = 2048
lr = scale_lr(1e-3, base_batch=256, effective_batch=2048, rule="linear")
print(f"Scaled LR: {lr}")  # 8e-3

# Always use warmup when scaling LR!
```

---

## 11.8 Debugging Distributed Training

```python
import torch.distributed as dist
import os

def ddp_sanity_checks():
    """Run at the start of training to catch common DDP bugs."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    # 1. Verify all ranks see the same model weights
    model = nn.Linear(4, 4).to(device)
    # DDP syncs weights on init, but let's verify
    weight_sum = model.weight.sum()
    dist.all_reduce(weight_sum, op=dist.ReduceOp.SUM)
    mean_weight = weight_sum / world_size
    if rank == 0:
        assert torch.allclose(model.weight.sum(), mean_weight, atol=1e-4), \
            "Rank 0 weights differ from mean — init sync failed!"

    # 2. Verify data is different per rank
    local_idx = torch.tensor([rank * 1000], device=device, dtype=torch.float)
    all_idx = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(all_idx, local_idx)
    all_idx = torch.cat(all_idx)
    assert len(all_idx.unique()) == world_size, "Ranks are receiving the same data!"

    # 3. Verify gradient sync works
    x = torch.randn(4, 4, device=device)
    model_ddp = DDP(model, device_ids=[device.index])
    loss = model_ddp(x).sum()
    loss.backward()

    grad = model_ddp.module.weight.grad.clone()
    dist.all_reduce(grad, op=dist.ReduceOp.AVG)
    assert torch.allclose(model_ddp.module.weight.grad, grad, atol=1e-4), \
        "Gradients not properly synced!"

    if rank == 0:
        print(f"All {world_size} ranks passed DDP sanity checks!")
```

---

## Exercises

**Exercise 11.1** Convert the `Trainer` class from Module 04 to support DDP. Add: `DistributedSampler`, rank-0-only checkpointing, and proper cleanup. Launch on 2 GPUs.

**Exercise 11.2** Train `MiniGPT` from Module 08 on a 4-GPU machine using FSDP with BF16. Compare memory footprint per GPU vs DDP. Verify that gradients are equivalent.

**Exercise 11.3** Use `accelerate` to run the same CIFAR-10 ResNet training on: (1) 1 GPU, (2) 4 GPUs, (3) 4 GPUs + gradient accumulation ×4. Report: time to accuracy, peak GPU memory.

---

## Module Summary

| Strategy | Memory per GPU | Communication | Best For |
|----------|---------------|---------------|---------|
| DDP | Full model | Gradient all-reduce | < 80B params |
| FSDP | 1/N of model | Param gather + grad all-reduce | Multi-billion params |
| Tensor Parallelism | 1/N of each layer | All-reduce per layer | Extremely large |
| Pipeline Parallelism | 1/stages of model | Activations between stages | Deepest models |

---

## Quiz

1. What is the difference between DDP and FSDP in terms of what is sharded?
2. Why must you call `sampler.set_epoch(epoch)` in every epoch?
3. What does `model.no_sync()` do and when is it needed?
4. What is the linear scaling rule and when does it break down?
5. Why is `find_unused_parameters=True` expensive and when is it required?
6. What is `BackwardPrefetch` in FSDP and how does it hide communication latency?
7. What command launches DDP on 4 GPUs with `torchrun`?

---

*Next: [Module 12 — Model Optimization: Quantization & Pruning](./12_model_optimization_quantization_pruning.md)*
