# Quiz Bank — Complete Question Set

All quiz questions from across the course, organised by module and difficulty. Use for self-assessment, spaced repetition, or interview preparation.

**Difficulty key:** ★ = Beginner | ★★ = Intermediate | ★★★ = Advanced

---

## Module 00: Prerequisites & Setup

| # | Question | Difficulty | Answer |
|---|----------|-----------|--------|
| 1 | What does `np.random.randn(3, 4)` produce? What is its shape? | ★ | 3×4 array drawn from N(0,1); shape (3, 4) |
| 2 | If **A** ∈ ℝ^(4×3) and **B** ∈ ℝ^(3×5), what is the shape of **AB**? | ★ | (4, 5) |
| 3 | Write the chain rule symbolically for y = f(g(x)). | ★ | dy/dx = (df/dg)(dg/dx) |
| 4 | How do you verify PyTorch can see your GPU? | ★ | `torch.cuda.is_available()` |
| 5 | What is the memory advantage of a generator over a list comprehension? | ★ | Generator yields one item at a time (O(1) memory); list stores all items (O(n)) |
| 6 | What does `axis=0` mean in `np.mean(data, axis=0)` for shape (100, 10)? | ★ | Computes mean across the 100 rows; result shape is (10,) |

---

## Module 01: Tensors & Fundamentals

| # | Question | Difficulty | Answer |
|---|----------|-----------|--------|
| 1 | What is the shape of `torch.randn(2, 3).unsqueeze(1)`? | ★ | (2, 1, 3) |
| 2 | Why does `x.T.view(-1)` sometimes raise an error? | ★★ | `.T` creates a non-contiguous tensor; `.view()` requires contiguous storage. Use `.contiguous().view(-1)` or `.reshape(-1)`. |
| 3 | What is the difference between `torch.cat` and `torch.stack`? | ★ | `cat` concatenates along an existing dimension; `stack` creates a new dimension |
| 4 | Given shapes (3, 1, 5) and (1, 4, 5), what is the broadcast result shape? | ★★ | (3, 4, 5) |
| 5 | What does `keepdim=True` do in `x.sum(dim=1, keepdim=True)`? | ★ | Preserves the summed dimension (size 1) instead of removing it; enables downstream broadcasting |
| 6 | Why is `torch.from_numpy(arr)` potentially dangerous? | ★★ | It shares memory with the NumPy array; modifying one changes the other |
| 7 | What is a stride in the context of tensor memory layout? | ★★ | Number of memory elements to skip to advance one step along each dimension |
| 8 | When would you use `einsum` over `@`? | ★★ | For operations involving more than 2 tensors, contracting over non-standard axes, outer products, or traces — anything requiring explicit index notation |
| 9 | What is the output of `torch.arange(12).reshape(3, 4)[1, 2]`? | ★ | 6 (row 1, column 2 of the reshaped tensor) |
| 10 | What does `x.numel()` return for `x = torch.randn(4, 3, 2)`? | ★ | 24 (total number of elements) |

---

## Module 02: Autograd & Computation Graphs

| # | Question | Difficulty | Answer |
|---|----------|-----------|--------|
| 1 | What does `retain_graph=True` do and when do you need it? | ★★ | Keeps the computation graph after backward; needed when you call backward multiple times on the same graph |
| 2 | Why must you call `optimizer.zero_grad()` before each backward pass? | ★ | PyTorch accumulates gradients into `.grad`; not zeroing causes incorrect gradient updates |
| 3 | What is the difference between `detach()` and `torch.no_grad()`? | ★★ | `detach()` cuts a specific tensor from the graph (shares data); `no_grad()` disables all gradient tracking for a block |
| 4 | What is a VJP and why does `backward()` compute it? | ★★★ | Vector-Jacobian Product: v^T J. Since loss is scalar (v=1), backward computes the gradient efficiently without materialising the full Jacobian |
| 5 | What happens if you call `.backward()` on a non-scalar without a gradient argument? | ★★ | RuntimeError: grad can be implicitly created only for scalar outputs |
| 6 | How does `gradcheck` verify a custom autograd function? | ★★ | Compares analytical gradients (from backward) against numerical finite differences |
| 7 | When would you use `retain_grad()` on a non-leaf tensor? | ★★ | Debugging: to inspect gradients at intermediate computations that PyTorch doesn't retain by default |
| 8 | What is the leaf tensor property `.is_leaf`? | ★ | True for tensors created by user (not results of operations); `.grad` is only populated for leaf tensors |
| 9 | What does `set_to_none=True` in `optimizer.zero_grad()` do vs the default? | ★★ | Sets `.grad` to None instead of filling with zeros; saves memory and can be slightly faster |
| 10 | What is the default gradient of a newly created tensor with `requires_grad=True`? | ★ | None (not yet computed) |

---

## Module 03: Neural Networks with nn.Module

| # | Question | Difficulty | Answer |
|---|----------|-----------|--------|
| 1 | What is the difference between `nn.ReLU()` and `F.relu()`? | ★ | `nn.ReLU()` is a Module (has state, can be placed in Sequential); `F.relu()` is a stateless function. Functionally identical for inference. |
| 2 | Why should `bias=False` be used before BatchNorm? | ★★ | BatchNorm has its own learnable shift parameter β that makes the conv bias redundant |
| 3 | What happens if you store sub-modules in a plain Python `list` instead of `nn.ModuleList`? | ★★ | Their parameters are NOT tracked by `model.parameters()`, so they won't be trained or saved in `state_dict()` |
| 4 | What is the dead ReLU problem and how is it mitigated? | ★★ | Neurons whose inputs are always negative output 0, receiving zero gradient, and never recover. Fix: Leaky ReLU, ELU, careful init, lower LR |
| 5 | Why should you call `model.eval()` before inference? | ★ | Disables Dropout (pass-through) and makes BatchNorm use running statistics instead of batch statistics |
| 6 | What does Kaiming initialisation set as the variance of weights, and why? | ★★ | Var(W) = 2/fan_in; preserves the variance of activations through ReLU layers (which zeros half the inputs) |
| 7 | What is the skip connection's role in a residual block? | ★★ | Provides a gradient highway during backpropagation; allows very deep networks to train without vanishing gradients |
| 8 | What is `nn.Parameter` and how does it differ from a regular tensor? | ★ | A tensor that is automatically registered as a model parameter and included in `model.parameters()` |
| 9 | What does `model.state_dict()` return? | ★ | An OrderedDict mapping parameter names to their current tensor values |
| 10 | What activation is recommended for transformer hidden layers? | ★★ | GELU (used in BERT, GPT-2, etc.); smooth approximation that outperforms ReLU on NLP tasks |

---

## Module 04: Training Pipeline Fundamentals

| # | Question | Difficulty | Answer |
|---|----------|-----------|--------|
| 1 | What does `pin_memory=True` do and when is it beneficial? | ★★ | Allocates DataLoader batches in page-locked (pinned) RAM, enabling faster async host-to-GPU transfers. Beneficial when training on GPU. |
| 2 | Why should you use `BCEWithLogitsLoss` instead of `BCELoss + sigmoid`? | ★★ | Numerically more stable — uses log-sum-exp trick internally; avoids floating-point overflow in sigmoid for large logit magnitudes |
| 3 | What is the difference between Adam and AdamW? | ★★ | AdamW uses decoupled weight decay (applied directly to weights, not through gradient update), which is mathematically correct. Adam's L2 regularisation is incorrectly scaled by the adaptive LR. |
| 4 | When would you call `scheduler.step()` per batch vs per epoch? | ★★ | Per batch: `OneCycleLR`, `CyclicLR`. Per epoch: most others (`CosineAnnealingLR`, `StepLR`). Check the scheduler's documentation. |
| 5 | Why does `drop_last=True` in DataLoader help with BatchNorm? | ★★ | BatchNorm requires ≥ 2 samples to compute variance; a last incomplete batch of size 1 would cause an error or degenerate stats |
| 6 | What does it mean if loss is NaN at step 0? | ★★ | Typically: LR too high, bad weight initialisation, NaN in the dataset, or exploding logits. Check data first. |
| 7 | Why is the "overfit one batch" test a valuable sanity check? | ★★ | If a model can't overfit a single batch, there's a bug in the model, loss function, or data pipeline — the model has sufficient capacity to memorise. |
| 8 | What is label smoothing and what does it prevent? | ★★ | Replaces hard 0/1 targets with (1-ε) and ε/K; prevents the model from being overconfident, improves calibration and generalisation |
| 9 | What is a `WeightedRandomSampler` and when do you use it? | ★★ | A sampler that over-represents minority classes during batching; used to handle severe class imbalance |
| 10 | What is the purpose of `persistent_workers=True` in DataLoader? | ★★ | Keeps worker processes alive between epochs; avoids the overhead of forking workers at each epoch start |

---

## Module 05: Convolutional Neural Networks

| # | Question | Difficulty | Answer |
|---|----------|-----------|--------|
| 1 | What is the output size of a conv with H=28, k=5, p=0, s=1? | ★ | (28 + 2×0 - 5)/1 + 1 = 24 |
| 2 | How many parameters does a 3×3 depthwise-separable conv have vs standard conv (C_in=C_out=64)? | ★★ | Standard: 64×64×3×3=36,864. Depthwise-sep: 64×1×3×3 + 64×64×1×1 = 576 + 4096 = 4,672 (8× reduction) |
| 3 | Why does dilation expand the receptive field without adding parameters? | ★★ | Dilation inserts gaps between kernel elements; the kernel size stays the same but it samples over a larger area |
| 4 | What problem do skip connections solve, and how? | ★★ | Vanishing gradients in deep networks. The identity shortcut provides a gradient path that bypasses non-linearities. |
| 5 | Why is `nn.AdaptiveAvgPool2d((1,1))` preferred over flattening the full feature map? | ★★ | Produces fixed-size output regardless of input spatial size; enables the model to accept variable-resolution inputs |
| 6 | What is label smoothing and why does it help? | ★★ | Softens one-hot targets; prevents overconfidence and acts as implicit regularisation |
| 7 | Why is `bias=False` common when followed by BatchNorm? | ★ | BatchNorm's β parameter serves the same role as bias; having both is redundant |
| 8 | What is the Squeeze-and-Excitation block? | ★★ | Channel attention: GAP → FC → ReLU → FC → Sigmoid → multiply channels. Learns to reweight feature channels. |
| 9 | What is Global Average Pooling and what advantage does it have over flattening? | ★★ | Averages each feature map to a single value; greatly reduces parameters and avoids spatial overfitting |
| 10 | What is the effective receptive field of stacking three 3×3 convolutions vs one 7×7? | ★★ | Both have RF = 7, but three 3×3 convs have 3×(9×C²) = 27C² params vs 49C² for 7×7 — fewer params and more non-linearity |

---

## Module 06: Transfer Learning & Fine-Tuning

| # | Question | Difficulty | Answer |
|---|----------|-----------|--------|
| 1 | What is catastrophic forgetting and how does gradual unfreezing help? | ★★ | A pre-trained network "forgets" old knowledge when fine-tuned aggressively. Gradual unfreezing slowly activates earlier layers at low LR, preserving learned representations. |
| 2 | Why is it recommended to train only the head first before fine-tuning the backbone? | ★★ | The randomly-initialised head has large gradients early on; those would corrupt pre-trained backbone weights if we fine-tune everything from the start. |
| 3 | What is the purpose of discriminative learning rates? | ★★ | Early layers already encode general features — they need smaller updates. Later layers need to adapt more — larger LR. Prevents over-updating useful low-level features. |
| 4 | How do you adapt a pretrained ImageNet model to a 1-channel input? | ★★ | Replace the first conv layer and initialise its weights by averaging the pretrained 3-channel weights along the channel dimension. |
| 5 | Why does `timm.create_model(..., num_classes=0)` remove the head? | ★ | Returns the model as a feature extractor; you can then add your own classification head for a custom task |
| 6 | What is linear probing and when is it used? | ★★ | Training only the final linear layer (head) while keeping all backbone layers frozen. Used as a baseline for transfer learning evaluation (especially for self-supervised models). |
| 7 | What is the key difference between feature extraction and partial fine-tuning? | ★ | Feature extraction: entire backbone frozen. Partial fine-tuning: only some layers (usually later) are unfrozen and trained. |
| 8 | Why is fine-tuning generally preferred over training from scratch for small datasets? | ★★ | Pre-trained models have already learned general features (edges, textures, shapes). Training from scratch with little data leads to severe overfitting. |

---

## Module 07: Recurrent Networks & Sequences

| # | Question | Difficulty | Answer |
|---|----------|-----------|--------|
| 1 | Why does the vanilla RNN suffer from vanishing gradients but the LSTM cell state does not? | ★★ | RNN gradients involve repeated multiplication by W_hh; LSTM cell state is updated additively via forget/input gates, enabling gradients to flow with near-unity magnitude through the forget gate |
| 2 | What is the shape of `h_n` for a bidirectional 2-layer LSTM with batch=8, hidden=128? | ★★ | (4, 8, 128) — 2 layers × 2 directions = 4 |
| 3 | What is teacher forcing and what is its drawback? | ★★ | Using ground-truth previous tokens as decoder input during training. Drawback: exposure bias — at inference, the model sees its own (potentially wrong) predictions, causing distributional shift. |
| 4 | Why must you detach hidden states between BPTT chunks? | ★★ | To prevent backpropagating through the entire sequence history; without detaching, the computation graph grows unboundedly and OOM errors occur |
| 5 | What does `pack_padded_sequence` do internally to avoid computing on padding? | ★★ | Packs variable-length sequences into a contiguous block, skipping padding positions entirely during the RNN computation |
| 6 | What is the difference between `output` and `h_n` from `nn.LSTM`? | ★ | `output`: hidden states at every timestep (B, T, H); `h_n`: only the final hidden state per layer/direction (layers, B, H) |
| 7 | What is the GRU's update gate z_t and what does z_t ≈ 1 mean? | ★★ | The fraction of the new candidate hidden state to use. z_t ≈ 1 means the GRU ignores the previous hidden state and fully adopts the new candidate (forget and update). |
| 8 | What is truncated BPTT and why is it used? | ★★ | Backpropagating only through K timesteps instead of the full sequence; reduces memory and compute for very long sequences |

---

## Module 08: Attention & Transformers

| # | Question | Difficulty | Answer |
|---|----------|-----------|--------|
| 1 | Why divide by √d_k in scaled dot-product attention? | ★★ | Without scaling, dot products grow with d_k, pushing softmax into saturation (near-zero gradients). Dividing by √d_k keeps the variance of QK^T at ~1. |
| 2 | What is the difference between encoder and decoder attention patterns? | ★★ | Encoder: bidirectional (can attend to all positions). Decoder: causal/autoregressive (can only attend to past positions). Cross-attention: decoder attends to encoder output. |
| 3 | Why does sinusoidal PE generalise to longer sequences than learned PE? | ★★ | Sinusoidal has a mathematical formula covering all positions; learned PE only has embeddings for positions seen in training (up to max_len). |
| 4 | What is weight tying in language models? | ★★ | Sharing the token embedding matrix with the LM head (output projection). Reduces parameters and often improves perplexity. |
| 5 | Why does pre-LN (normalise before attention) train more stably than post-LN? | ★★★ | Pre-LN keeps the gradient magnitude bounded throughout the network; post-LN can have very large gradient norms in early training, requiring warmup. |
| 6 | What memory complexity does standard attention have and how does Flash Attention improve it? | ★★★ | Standard: O(T²) in memory (stores the full T×T attention matrix). Flash Attention: O(T) by computing attention in tiles in SRAM without materialising the full matrix. |
| 7 | What is the "context window" of a transformer and what limits it? | ★★ | The maximum sequence length the model can process. Limited by: (1) O(T²) attention memory, (2) positional embedding max_len, (3) training context. |
| 8 | What is multi-head attention's advantage over single-head attention? | ★★ | Different heads learn to attend to different representation subspaces simultaneously; richer relational structure captured with the same compute |
| 9 | What is a key-query-value (KQV) decomposition? | ★★ | Separate linear projections produce queries (what I'm looking for), keys (what I offer), values (my content). Allows flexible many-to-many matching. |
| 10 | What is RoPE (Rotary Position Embedding)? | ★★★ | Encodes position by rotating Q and K vectors in complex space; relative position emerges naturally from the dot product; enables length generalisation (used in LLaMA, Mistral) |

---

## Module 09: Advanced Training Techniques

| # | Question | Difficulty | Answer |
|---|----------|-----------|--------|
| 1 | Why must you divide the loss by `accumulate_steps` when doing gradient accumulation? | ★★ | So the accumulated gradient is equivalent to a single forward pass on the full batch. Without scaling, the gradient magnitude is N× larger than expected. |
| 2 | What does `GradScaler` do and why is it needed for FP16 but not BF16? | ★★ | Multiplies the loss by a large scale factor before backward to prevent FP16 gradient underflow. BF16 has the same exponent range as FP32 so gradients don't underflow. |
| 3 | When would you use gradient checkpointing and what is the memory-compute tradeoff? | ★★ | When training large models that OOM. Trade: recomputes activations during backward (~33% extra compute) to avoid storing them (~O(√N) vs O(N) memory). |
| 4 | Why does EMA give better test accuracy than using the raw model weights? | ★★ | EMA averages out noise from stochastic gradient updates, producing smoother weight trajectories that often land in flatter, better-generalising minima. |
| 5 | What is the difference between DropPath and Dropout? | ★★ | Dropout zeros individual neurons. DropPath zeros the entire residual branch contribution (per sample), effectively dropping whole blocks stochastically. |
| 6 | Why use warmup before cosine annealing in transformer training? | ★★ | Transformers with Adam are sensitive to early LR; large early updates can push weights to bad regions. Warmup lets Adam accumulate reliable gradient statistics before taking large steps. |
| 7 | What is label smoothing and what regularisation effect does it have? | ★★ | Prevents the model from assigning full probability mass to the correct class; penalises overconfident predictions and improves calibration. |
| 8 | What is Mixup and what invariance does it encourage? | ★★ | Linearly interpolates pairs of training samples and their labels. Encourages the model to behave linearly between training examples, improving generalisation. |

---

## Module 10: GPU Performance & Mixed Precision

| # | Question | Difficulty | Answer |
|---|----------|-----------|--------|
| 1 | What is the difference between `memory_allocated()` and `memory_reserved()`? | ★★ | `memory_allocated`: memory currently used by live tensors. `memory_reserved`: total memory PyTorch has requested from CUDA (cached blocks not yet returned to OS). |
| 2 | Why is BF16 safer than FP16 for training and which GPUs support it natively? | ★★ | BF16 has 8 exponent bits (same as FP32); no risk of overflow. FP16 only has 5 exponent bits. BF16: Ampere+, A100, H100, TPUs. |
| 3 | What does `cudnn.benchmark = True` do and when is it counterproductive? | ★★ | Benchmarks convolution algorithms at the first call and caches the fastest. Counterproductive when input shapes vary (benchmarks run repeatedly, adding overhead). |
| 4 | What is operator fusion and why does it improve performance? | ★★ | Merging multiple ops into a single CUDA kernel; eliminates intermediate HBM writes/reads (bandwidth-bound bottleneck). E.g., Linear+GELU fused → 2–3× faster. |
| 5 | Why is `set_to_none=True` in `optimizer.zero_grad()` faster than the default? | ★★ | Sets `.grad` to None (no memory write) instead of filling with zeros (requires memset). Slightly less memory used too. |
| 6 | What is the memory bandwidth bottleneck and how does channels_last address it? | ★★★ | Convolutions in NCHW layout have non-contiguous channel access; NHWC (channels_last) matches the access pattern of cuDNN kernels, reducing HBM reads. |
| 7 | What compilation mode would you choose for maximum inference throughput? | ★★ | `torch.compile(model, mode="max-autotune")` — exhaustively searches for the best kernel for each op. |
| 8 | What is TF32 and how does it affect numerical precision? | ★★★ | TF32 is an NVIDIA internal format on Ampere+: rounds FP32 to 10-bit mantissa for matmuls; ~8× faster than FP32 matmul with negligible accuracy loss. Enabled by `torch.backends.cuda.matmul.allow_tf32 = True`. |

---

## Module 11: Distributed Training & Scaling

| # | Question | Difficulty | Answer |
|---|----------|-----------|--------|
| 1 | What is the difference between DDP and FSDP in terms of what is sharded? | ★★ | DDP: each GPU holds the full model; only gradients are all-reduced. FSDP: parameters, gradients, AND optimizer states are all sharded across GPUs. |
| 2 | Why must you call `sampler.set_epoch(epoch)` in every epoch? | ★★ | `DistributedSampler` uses a deterministic shuffle based on epoch; without setting the epoch, all processes shuffle identically and get the same data. |
| 3 | What does `model.no_sync()` do and when is it needed? | ★★ | Defers gradient all-reduce to the next sync point. Needed for gradient accumulation with DDP to avoid syncing on every micro-step. |
| 4 | What is the linear scaling rule and when does it break down? | ★★ | LR_new = LR_base × (effective_batch / base_batch). Breaks down at very large batch sizes (typically > ~8K images) where the noise from smaller batches that helped generalisation is lost. |
| 5 | Why is `find_unused_parameters=True` expensive and when is it required? | ★★ | It adds a traversal of the compute graph after backward to identify params not used; required for models with conditional computation paths (e.g., MoE, some multi-task models). |
| 6 | What is `BackwardPrefetch` in FSDP and how does it hide communication latency? | ★★★ | While the backward pass computes gradients for layer i, FSDP pre-fetches (gathers) the parameters for layer i-1 that will be needed next — overlapping compute with communication. |
| 7 | What command launches DDP on 4 GPUs with `torchrun`? | ★ | `torchrun --standalone --nproc_per_node=4 train.py` |
| 8 | What is all-reduce and what operation does DDP use for gradient synchronisation? | ★★ | All-reduce: each process contributes a tensor; all receive the combined result (sum or average). DDP uses `SUM` then divides by world_size (= AVG all-reduce). |

---

## Module 12: Model Optimization — Quantization & Pruning

| # | Question | Difficulty | Answer |
|---|----------|-----------|--------|
| 1 | What is the difference between symmetric and asymmetric quantization? | ★★ | Symmetric: zero_point=0, scale = max(|x|)/(2^{n-1}-1). Asymmetric: zero_point ≠ 0, covers the full range [x_min, x_max]. Asymmetric is more accurate but slightly more complex. |
| 2 | Why does QAT outperform PTQ at the same bit-width? | ★★ | QAT lets the model adapt its weights to quantization error during training (via fake-quantization); PTQ post-processes a model that was trained without awareness of quantization. |
| 3 | What is the temperature T in knowledge distillation and what does a high T do? | ★★ | T softens the teacher's softmax output (divides logits). High T → softer distribution → reveals more inter-class similarity ("dark knowledge"). |
| 4 | Why is unstructured pruning hard to speed up on real hardware? | ★★★ | Modern hardware (GPUs, CPUs) is optimised for dense tensor operations; sparse irregular patterns in unstructured pruning don't map well to SIMD/tensor cores without specialised sparse kernels. |
| 5 | What is the NF4 quantization type and why is it preferred for LLM weights? | ★★★ | NormalFloat-4: a 4-bit data type quantised to have equal-sized quantile bins for a normal distribution. LLM weights are approximately normally distributed, so NF4 minimises quantization error. |
| 6 | Why does LoRA initialize matrix B to zero? | ★★ | So that ΔW = B·A = 0 at the start of training; the model starts identical to the pretrained model and learns adaptations from scratch. |
| 7 | What is double quantization in bitsandbytes? | ★★★ | Quantising the quantization constants (scale factors) themselves — 32-bit scales → 8-bit. Saves an additional ~0.37 bits per parameter on top of 4-bit quantization. |
| 8 | What is structured pruning vs unstructured pruning? | ★★ | Structured: removes entire filters, channels, or layers (yields a smaller dense model). Unstructured: zeroes individual weights (sparse model, harder to accelerate without sparse kernels). |

---

## Module 13: Deployment — TorchScript & ONNX

| # | Question | Difficulty | Answer |
|---|----------|-----------|--------|
| 1 | What is the key difference between `torch.jit.trace` and `torch.jit.script`? | ★★ | Tracing records operations executed on a specific input — misses branches not taken. Scripting compiles the Python source — handles all control flow paths. |
| 2 | Why does tracing fail for models with data-dependent control flow? | ★★ | The tracer only records one execution path; an `if x.mean() > 0` branch is baked in as the branch taken during tracing, not as a conditional. |
| 3 | What does `dynamic_axes` do in `torch.onnx.export`? | ★ | Declares which dimensions are variable (e.g., batch size, sequence length); the ONNX model accepts tensors of different sizes along those axes. |
| 4 | What does `do_constant_folding=True` optimise? | ★★ | Pre-computes operations that depend only on constants (e.g., `weight * 2` where 2 is a constant), reducing runtime computation. |
| 5 | What is the ONNX opset version and why does it matter? | ★★ | Version of the ONNX operator specification used. Higher opsets support more ops and have better coverage; must be supported by your inference runtime. |
| 6 | Why do you need to warm up the model before benchmarking inference? | ★★ | GPU kernels may be JIT-compiled on first call; GPU pipelines need several batches to fill; cached memory allocations stabilise. Warmup ensures steady-state performance. |
| 7 | What is TensorRT engine calibration and when is it needed? | ★★★ | For INT8 TensorRT inference, calibration runs representative data through the model to determine per-layer scale factors (activation ranges). Needed because activations require dynamic range information unlike static weights. |
| 8 | What does `torch.jit.optimize_for_inference(model)` do? | ★★ | Applies post-processing optimisations to a frozen scripted model: folds batch normalisation into preceding convolution layers, removes unused code, etc. |

---

## Module 14: Serving & Production

| # | Question | Difficulty | Answer |
|---|----------|-----------|--------|
| 1 | Why must you warm up the model before serving real traffic? | ★★ | GPU kernels compile lazily; first calls are slow. Warmup ensures P99 latency from the first real request is representative. |
| 2 | What is dynamic batching and why does it improve GPU utilisation? | ★★ | Accumulates multiple single-sample requests into a batch before running GPU inference. Batched execution amortises kernel launch overhead and better utilises tensor cores. |
| 3 | What is the difference between readiness and liveness probes in Kubernetes? | ★★ | Readiness: is the pod ready to receive traffic? (fails = removed from load balancer). Liveness: is the pod alive? (fails = pod killed and restarted). |
| 4 | How does A/B testing differ from canary deployment? | ★★ | A/B testing: route a fixed percentage of traffic to each model and compare metrics statistically. Canary: gradually increase traffic to the new model (0% → 10% → 100%), monitoring for regressions at each stage. |
| 5 | What does `asyncio.Lock()` protect in the inference server? | ★★ | Ensures only one request uses the GPU at a time; prevents concurrent GPU kernel launches from different async handlers (which can cause race conditions or OOM). |
| 6 | Why use `workers=1` in uvicorn for GPU inference servers? | ★★ | Multiple uvicorn workers create separate Python processes, each trying to load the model onto GPU — causing OOM and doubling memory usage. Use 1 worker + async concurrency instead. |
| 7 | What is data drift and how would you detect it in production? | ★★ | When the statistical distribution of incoming production data shifts away from training data. Detect via: statistical tests (KS test, PSI), monitoring feature means/variances, tracking prediction distribution shifts. |
| 8 | What is TorchServe's management API used for? | ★ | Registering/unregistering models, changing number of workers, updating model versions, querying model status — all at runtime without restarting the server. |

---

## Module 15: Evaluation & Interpretability

| # | Question | Difficulty | Answer |
|---|----------|-----------|--------|
| 1 | What is ECE (Expected Calibration Error) and what does a value of 0.05 mean? | ★★ | Expected gap between confidence and accuracy across confidence bins. ECE=0.05 means on average, when the model says 80% confidence, the actual accuracy is 75–85%. |
| 2 | How does Grad-CAM differ from Integrated Gradients in its approach to attribution? | ★★ | Grad-CAM uses gradients flowing into the last conv layer + activation maps (spatial); IG integrates gradients along a path from baseline to input (pixel-level, all input features). |
| 3 | What is temperature scaling and does it change model accuracy? | ★★ | Post-hoc calibration: learn a single temperature T on a validation set; divide logits by T before softmax. Does NOT change predictions (argmax same) → accuracy unchanged; only calibration improves. |
| 4 | What is Attention Rollout and how does it improve over raw attention weights? | ★★★ | Recursively multiplies attention matrices layer-by-layer (adding residual identity), accounting for the fact that information flows through multiple paths. Raw weights only show one layer's view. |
| 5 | What is demographic parity and when should it be prioritised over accuracy? | ★★★ | Equalising the positive prediction rate across demographic groups. Prioritised in high-stakes decisions (lending, hiring, criminal justice) where disparate impact has legal/ethical consequences. |
| 6 | Why should you evaluate on a test set held out from calibration? | ★★ | Temperature scaling is fitted on the validation set; evaluating calibration on the same set overfits. The test set gives an unbiased estimate of real-world calibration. |
| 7 | What does a BLEU score of 40 mean for machine translation? | ★★ | ~40 corresponds to a high-quality translation (human quality ranges 50–60). BLEU is computed as the geometric mean of 1–4 gram precision × brevity penalty. |
| 8 | What is the purpose of SHAP values? | ★★ | Shapley Additive exPlanations: attributes each feature's contribution to a prediction using cooperative game theory; additive, locally accurate, and consistent. |

---

## Cross-Module: System Design Questions (Senior/Staff Level)

| # | Question | Difficulty |
|---|----------|-----------|
| 1 | Design an image search engine that returns the 10 most visually similar images from a corpus of 1B images. What is your ML pipeline, indexing strategy, and serving architecture? | ★★★ |
| 2 | You need to serve a 70B LLM with < 100ms first-token latency for 1,000 concurrent users. What deployment strategy do you use? | ★★★ |
| 3 | Your CIFAR-10 model achieves 95% on the test set but only 70% on production traffic. Diagnose and fix. | ★★★ |
| 4 | How would you train a model on 100TB of image data distributed across 10 data centres in 3 countries? | ★★★ |
| 5 | Design an online learning system that updates a fraud detection model every 10 minutes without downtime. | ★★★ |

---

## Quick Reference: Key Formulas

| Formula | Module |
|---------|--------|
| L2 norm: ‖x‖₂ = √(Σxᵢ²) | 00 |
| Chain rule: dy/dx = (df/dg)(dg/dx) | 00, 02 |
| Conv output: H_out = (H_in + 2p - k) / s + 1 | 05 |
| Scaled attention: softmax(QKᵀ/√d_k) V | 08 |
| Adam update: θ ← θ - η·m̂/(√v̂ + ε) | 04 |
| Cross-entropy: CE = -Σ yᵢ log ŷᵢ | 04 |
| KL divergence: KL(P‖Q) = Σ P log(P/Q) | 12 |
| ECE: Σ (|B_m|/n)|acc(B_m) - conf(B_m)| | 15 |
| Quantize: x_q = round(x/scale) + zero_point | 12 |
| LoRA: W' = W₀ + BA | 12 |
| BLEU: BP × exp(Σ wₙ log pₙ) | 15 |
| Perplexity: exp(-(1/N) Σ log p(xᵢ)) | 07, 08 |

---

*Total: 120+ questions across 16 modules. Use with spaced repetition for best retention.*
