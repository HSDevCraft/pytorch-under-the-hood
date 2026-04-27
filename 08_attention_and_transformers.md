# Module 08: Attention Mechanisms & Transformers

## Learning Objectives
By the end of this module you will be able to:
- Derive scaled dot-product attention from first principles
- Implement multi-head attention and understand why multiple heads help
- Build a complete Transformer encoder and decoder from scratch
- Understand positional encoding: sinusoidal and learned
- Implement a GPT-style causal language model
- Fine-tune BERT / GPT-2 using HuggingFace Transformers
- Apply Flash Attention and understand memory-efficient attention

---

## 8.1 The Attention Intuition

In a Seq2Seq model, the encoder compresses the whole source sequence into a single vector — a bottleneck. **Attention** lets the decoder look at *all* encoder states and focus on the relevant ones dynamically.

**Bahdanau attention (2015):**
```
eᵢⱼ = v^T · tanh(W_a · s_{i-1} + U_a · hⱼ)   ← alignment score
αᵢⱼ = softmax(eᵢⱼ)                              ← attention weights
cᵢ   = Σⱼ αᵢⱼ · hⱼ                              ← context vector
```

**Scaled Dot-Product Attention (Vaswani 2017):**
```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

Where:
- **Q (query):** what I'm looking for
- **K (key):** what each position offers
- **V (value):** what each position actually contains
- `d_k`: dimension of keys — divides to prevent softmax saturation

---

## 8.2 Scaled Dot-Product Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
) -> tuple:
    """
    Args:
        Q: (..., seq_q, d_k)
        K: (..., seq_k, d_k)
        V: (..., seq_k, d_v)
        mask: (..., seq_q, seq_k) — True where position should be masked
    Returns:
        output: (..., seq_q, d_v)
        weights: (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)   # (..., seq_q, seq_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))    # mask=True → -∞ → 0 after softmax

    weights = F.softmax(scores, dim=-1)                     # (..., seq_q, seq_k)
    weights = F.dropout(weights, p=dropout, training=True)

    output = weights @ V                                    # (..., seq_q, d_v)
    return output, weights


# ── Causal (autoregressive) mask ──────────────────────────────────────────────
def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Returns upper-triangular mask: position i cannot attend to j > i.
    Shape: (1, 1, seq_len, seq_len), True where masked.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, T, T)

# Test
Q = torch.randn(2, 8, 10, 64)   # (batch, heads, seq, d_k)
K = torch.randn(2, 8, 10, 64)
V = torch.randn(2, 8, 10, 64)
mask = causal_mask(10, Q.device)
out, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
print(out.shape)     # (2, 8, 10, 64)
print(weights.shape) # (2, 8, 10, 10)
# Verify causality: lower triangle only
print((weights[0, 0].triu(1).abs() < 1e-6).all())  # True
```

---

## 8.3 Multi-Head Attention

Running attention multiple times in parallel with different linear projections lets the model attend to information from different subspaces simultaneously:

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · W_O
where headᵢ = Attention(Q·W_Qᵢ, K·W_Kᵢ, V·W_Vᵢ)
```

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with optional causal masking.
    d_model must be divisible by n_heads.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        # Fused projection (4 * d_model instead of 4 separate projections)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.dropout = dropout
        self._init_weights()

    def _init_weights(self):
        for layer in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(layer.weight)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) → (B, h, T, d_k)"""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, h, T, d_k) → (B, T, d_model)"""
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        query: (B, T_q, d_model)
        key, value: (B, T_k, d_model)
        mask: (B, 1, T_q, T_k) or (B, n_heads, T_q, T_k)
        """
        Q = self._split_heads(self.W_Q(query))   # (B, h, T_q, d_k)
        K = self._split_heads(self.W_K(key))     # (B, h, T_k, d_k)
        V = self._split_heads(self.W_V(value))   # (B, h, T_k, d_k)

        attn_out, _ = scaled_dot_product_attention(
            Q, K, V, mask=mask,
            dropout=self.dropout if self.training else 0.0,
        )

        merged = self._merge_heads(attn_out)     # (B, T_q, d_model)
        return self.W_O(merged)


# Test
mha = MultiHeadAttention(d_model=512, n_heads=8)
x = torch.randn(2, 20, 512)
out = mha(x, x, x)    # self-attention
print(out.shape)       # (2, 20, 512)
```

---

## 8.4 Positional Encoding

Transformers have no built-in notion of sequence order. Positional encodings inject order information:

**Sinusoidal (fixed):**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Properties: different frequency per dimension, smooth relative position signal, generalises to unseen lengths.

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute PE table once at init
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # Register as buffer (not a parameter — not trained, but saved in state_dict)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        return self.dropout(x + self.pe[:, :x.size(1)])


class LearnedPositionalEncoding(nn.Module):
    """Learned embeddings for each position (used in BERT, GPT)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.pe      = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        positions = torch.arange(T, device=x.device)
        return self.dropout(x + self.pe(positions))
```

---

## 8.5 Transformer Encoder

```python
class FeedForward(nn.Module):
    """
    Position-wise FFN: Linear → GELU → Dropout → Linear
    Expands by factor r (typically 4) then contracts.
    """

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """
    Pre-LN Transformer encoder layer (more stable training):
    x → LN → MHA → + residual → LN → FFN → + residual
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff        = FeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN: normalise BEFORE attention (more stable)
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc   = LearnedPositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.scale = d_model ** 0.5
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: (B, T) token ids
        src_key_padding_mask: (B, T) True where padding
        Returns: (B, T, d_model)
        """
        # Build attention mask from padding mask
        if src_key_padding_mask is not None:
            attn_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        else:
            attn_mask = None

        emb = self.pos_enc(self.embedding(x) * self.scale)   # (B, T, d_model)
        for layer in self.layers:
            emb = layer(emb, attn_mask)
        return self.norm(emb)
```

---

## 8.6 GPT-Style Causal Decoder

```python
class CausalSelfAttention(nn.Module):
    """Self-attention with causal mask — each position only attends to past."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.attn  = MultiHeadAttention(d_model, n_heads, dropout)
        # Precompute causal mask
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        mask = self.causal_mask[:, :, :T, :T]
        return self.attn(x, x, x, mask=mask)


class GPTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, max_len: int):
        super().__init__()
        self.norm1  = nn.LayerNorm(d_model)
        self.attn   = CausalSelfAttention(d_model, n_heads, dropout, max_len)
        self.norm2  = nn.LayerNorm(d_model)
        self.ff     = FeedForward(d_model, d_ff, dropout)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class MiniGPT(nn.Module):
    """
    GPT-2 style causal language model.
    Architecture matches GPT-2 small (117M params) at:
    d_model=768, n_heads=12, n_layers=12, d_ff=3072, max_len=1024
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len,   d_model)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff, dropout, max_len) for _ in range(n_layers)
        ])
        self.norm  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying: share token embedding and LM head weights
        self.head.weight = self.token_emb.weight

        self._init_weights()
        # Scale residual projections (GPT-2 trick)
        for name, p in self.named_parameters():
            if name.endswith("W_O.weight") or name.endswith("net.3.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * n_layers))

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) → logits: (B, T, vocab_size)"""
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def loss(self, idx: torch.Tensor) -> torch.Tensor:
        """Language modelling loss: predict next token."""
        logits = self(idx[:, :-1])                    # (B, T-1, V)
        target = idx[:, 1:]                           # (B, T-1)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int,
        temperature: float = 1.0, top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive generation with top-k sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(idx)[:, -1, :] / temperature   # (B, V)
            # Top-k filtering
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1:]] = float("-inf")
            probs     = F.softmax(logits, dim=-1)
            next_tok  = torch.multinomial(probs, 1)
            idx       = torch.cat([idx, next_tok], dim=1)
        return idx


# Quick parameter count
model = MiniGPT()
n = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n/1e6:.1f}M")   # ~7M for the default config
```

---

## 8.7 Using HuggingFace Transformers

```python
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForCausalLM, BertConfig, GPT2Config,
    Trainer, TrainingArguments,
)
import torch

# ── BERT for sequence classification ─────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model     = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

texts  = ["I love PyTorch!", "This is mediocre."]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
# inputs: {input_ids, attention_mask, token_type_ids}

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits         # (2, 2)
probs  = torch.softmax(logits, dim=-1)

# ── GPT-2 text generation ─────────────────────────────────────────────────────
gpt2_tok   = AutoTokenizer.from_pretrained("gpt2")
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "The future of AI is"
inputs = gpt2_tok(prompt, return_tensors="pt")

generated = gpt2_model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1,
    pad_token_id=gpt2_tok.eos_token_id,
)
print(gpt2_tok.decode(generated[0], skip_special_tokens=True))

# ── Fine-tuning with HuggingFace Trainer ──────────────────────────────────────
from datasets import load_dataset

dataset   = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./imdb-distilbert",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
)

from evaluate import load as load_metric
import numpy as np

accuracy = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    compute_metrics=compute_metrics,
)
trainer.train()
```

---

## 8.8 Flash Attention (Memory-Efficient)

Standard attention is O(T²) in memory. Flash Attention (Dao et al., 2022) is O(T) by tiling the computation. In PyTorch 2.0+:

```python
import torch
import torch.nn.functional as F

# PyTorch 2.0+ has native Flash Attention via scaled_dot_product_attention
# It automatically selects the optimal backend (Flash Attention 2, xFormers, or math)
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
    output = F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=None,      # or boolean mask
        dropout_p=0.1,
        is_causal=True,      # uses causal masking efficiently
    )
# No explicit mask needed for causal; is_causal=True handles it

# For custom MHA using F.scaled_dot_product_attention:
class EfficientMHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, is_causal=False):
        B, T, C = x.shape
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_O(out)
```

---

## 8.9 Best Practices for Transformers

| Practice | Why |
|----------|-----|
| Pre-LN (LayerNorm before attention) | More stable training; no warmup needed |
| Weight tying (embedding ↔ LM head) | Fewer parameters; better generalisation |
| Gradient clipping (max_norm=1.0) | Attention gradients can spike |
| Warmup + cosine LR schedule | Transformers are sensitive to initial LR |
| Dropout on attention weights AND residuals | Regularisation in data-scarce settings |
| Flash Attention (`F.scaled_dot_product_attention`) | 2–4× faster, 10× less memory for long sequences |
| `torch.compile(model)` (PyTorch 2.0+) | 20–50% faster on GPU |
| FP16 or BF16 training | 2× throughput; BF16 preferred (better range) |

---

## Exercises

**Exercise 8.1** Implement Rotary Position Embeddings (RoPE) as used in LLaMA. Apply a rotation matrix to Q and K before the dot product. Verify relative position invariance.

**Exercise 8.2** Build a Transformer for machine translation (English→French) on the Multi30K dataset. Use the Seq2Seq architecture from module 07 as a baseline and compare BLEU scores.

**Exercise 8.3** Fine-tune `roberta-base` on the SST-2 sentiment dataset. Implement a custom training loop (no HuggingFace Trainer). Report the accuracy curve and final test accuracy.

---

## Module Summary

| Component | Function | Key Equation |
|-----------|---------|-------------|
| Scaled dot-product attention | Match queries to keys, aggregate values | softmax(QKᵀ/√d_k)V |
| Multi-head attention | Attend to multiple subspaces | Concat(headsᵢ) W_O |
| Positional encoding | Inject sequence order | PE(pos, 2i) = sin(pos/10000^{2i/d}) |
| Encoder layer | Bidirectional contextualisation | SA → Add+Norm → FFN → Add+Norm |
| GPT block | Causal token prediction | Causal-SA → Add+Norm → FFN → Add+Norm |
| Flash Attention | Memory-efficient O(T) attention | Tiled computation, HBM-aware |

---

## Quiz

1. Why divide by √d_k in scaled dot-product attention?
2. What is the difference between encoder and decoder attention patterns?
3. Why does sinusoidal PE generalise to longer sequences than learned PE?
4. What is weight tying in language models?
5. Why does pre-LN (normalise before attention) train more stably than post-LN?
6. What memory complexity does standard attention have and how does Flash Attention improve it?
7. What is the "context window" of a transformer and what limits it?

---

*Next: [Module 09 — Advanced Training Techniques](./09_advanced_training_techniques.md)*
