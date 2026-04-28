# Module 08: Attention & Transformers — The Architecture That Changed Everything

> **Goal:** Understand attention from first principles — the intuition, the mathematics, and how GPT/BERT are built from these primitives.

---

## Learning Objectives

By the end of this module, you will:
- **Understand** why attention was invented and what problem it solves
- **Derive** scaled dot-product attention from scratch
- **Implement** multi-head attention and understand why multiple heads help
- **Build** a full Transformer encoder and GPT-style decoder
- **Use** positional encodings (sinusoidal and learned)
- **Fine-tune** pretrained LLMs from HuggingFace

---

## Part 1: The Problem Attention Solves

### 1.1 The Bottleneck in Seq2Seq

In encoder-decoder RNNs, the encoder compresses an entire input sequence into a **single fixed-size vector** (the final hidden state). This is the bottleneck:

- "The bank by the river was steep" → [final hidden vector] → translation
- All 8 words must be squeezed into one vector
- Long sequences: early words are "forgotten" before encoding finishes

**Attention says:** "Instead of compressing to one vector, let the decoder *look back* at all encoder states and choose which ones are relevant at each decoding step."

### 1.2 The Attention Intuition

Think of attention like a **soft database lookup**:
- You have a `Query` (what you're looking for)
- The database has `Keys` (what each entry is about)
- Each entry has a `Value` (the actual content)
- Attention: match Query to Keys → weighted sum of Values

```
Query: "I want to translate the word 'bank'"
Keys:  ["The", "bank", "by", "the", "river", "was", "steep"]
Scores: [0.05, 0.70, 0.05, 0.05, 0.10, 0.03, 0.02]  ← soft match
Values: [embedding1, embedding2, ...]

Output = 0.05*emb1 + 0.70*emb2 + 0.05*emb3 + ...
       ≈ mostly embedding of "bank" — the relevant word!
```

---

## Part 2: Scaled Dot-Product Attention

### 2.1 The Mathematical Formula

```
Attention(Q, K, V) = softmax(Q @ K.T / √d_k) @ V
```

Breaking this down:
1. `Q @ K.T` — compute similarity between query and all keys
2. `/ √d_k` — scale by key dimension (prevents softmax saturation)
3. `softmax(...)` — convert similarities to probabilities (sum to 1)
4. `@ V` — weighted average of values

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    Q: torch.Tensor,  # (batch, n_heads, seq_q, d_k)
    K: torch.Tensor,  # (batch, n_heads, seq_k, d_k)
    V: torch.Tensor,  # (batch, n_heads, seq_k, d_v)
    mask: torch.Tensor = None  # Optional: prevents attending to certain positions
) -> tuple:
    """
    Compute attention and return weighted values + attention weights.
    
    d_k: dimension of each key/query vector
    The √d_k scaling is CRITICAL:
    
    Why scale? If d_k is large (e.g., 512), Q@K.T values can be very large.
    Large values → softmax becomes very "peaked" (near one-hot) → gradients vanish.
    Dividing by √d_k keeps the variance of the dot product ≈ 1.
    
    Mathematical justification:
    If Q,K ~ N(0,1), then Q@K.T ~ N(0, d_k)
    Dividing by √d_k → variance = 1 (unit variance)
    """
    d_k = Q.shape[-1]
    
    # Step 1: Compute raw attention scores
    # Q @ K.T: (batch, heads, seq_q, d_k) @ (batch, heads, d_k, seq_k)
    #        = (batch, heads, seq_q, seq_k)
    scores = Q @ K.transpose(-2, -1)  # Raw dot products
    scores = scores / math.sqrt(d_k)   # Scale — stabilizes softmax
    
    # Step 2: Apply mask (if provided)
    # Used for:
    # - Causal masking in decoder (can't look at future tokens)
    # - Padding mask (don't attend to padding tokens)
    if mask is not None:
        # Replace masked positions with very large negative value
        # After softmax, exp(-inf) = 0 → these positions get 0 attention
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Step 3: Softmax → attention weights (probabilities)
    # dim=-1: softmax over the key dimension (each query attends to all keys)
    attention_weights = F.softmax(scores, dim=-1)  # (batch, heads, seq_q, seq_k)
    
    # Step 4: Weighted sum of values
    output = attention_weights @ V  # (batch, heads, seq_q, d_v)
    
    return output, attention_weights


# Concrete example: self-attention on a 5-word sentence
batch, seq_len, d_model = 2, 5, 512
d_k = d_v = 64  # Typically d_model // n_heads

Q = torch.randn(batch, 1, seq_len, d_k)  # 1 head for simplicity
K = torch.randn(batch, 1, seq_len, d_k)
V = torch.randn(batch, 1, seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Attention output: {output.shape}")  # (2, 1, 5, 64)
print(f"Attention weights: {weights.shape}") # (2, 1, 5, 5) — every word → every word
print(f"Weights sum to 1: {weights.sum(dim=-1)}")  # All 1.0
```

### 2.2 Causal Masking for Autoregressive Generation

```python
def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Creates a lower-triangular mask that prevents each position
    from attending to future positions.
    
    Position 0 can only see position 0
    Position 1 can see positions 0,1
    Position 2 can see positions 0,1,2
    ...
    
    Mask (1=attend, 0=block):
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq) for broadcasting

mask = make_causal_mask(seq_len=5, device=torch.device('cpu'))
print(f"Causal mask:\n{mask.squeeze()}")
```

---

## Part 3: Multi-Head Attention

### 3.1 Why Multiple Heads?

One attention head can only learn **one type of relationship** at a time. Multiple heads learn different relationships simultaneously:
- Head 1: "which word is the subject?"
- Head 2: "what does this word's syntax depend on?"
- Head 3: "which word is semantically similar?"
- Head 4: "what is the verb associated with this noun?"

Each head projects Q, K, V into a **different subspace** and learns a different pattern.

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention: run h attention heads in parallel, 
    then concatenate and project.
    
    Architecture:
    Q, K, V
      ↓ Linear projections (split into h heads)
    [Head₁ Attention] [Head₂ Attention] ... [Headₕ Attention]
      ↓ Concatenate
    Linear projection → output
    
    The key: each head projects into d_k = d_model/h dimensions
    Total cost ≈ same as single-head with d_model dimensions
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        # Separate projections for Q, K, V (all learned)
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # Query projection
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # Key projection
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # Value projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # Output projection
        
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (n_heads, d_k).
        (batch, seq, d_model) → (batch, n_heads, seq, d_k)
        """
        batch, seq, d_model = x.shape
        x = x.reshape(batch, seq, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, n_heads, seq, d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        query: (batch, seq_q, d_model)
        key:   (batch, seq_k, d_model)
        value: (batch, seq_k, d_model)
        
        For self-attention: query = key = value = x
        For cross-attention: query from decoder, key/value from encoder
        """
        batch = query.shape[0]
        
        # Project Q, K, V
        Q = self.split_heads(self.W_q(query))  # (batch, n_heads, seq_q, d_k)
        K = self.split_heads(self.W_k(key))    # (batch, n_heads, seq_k, d_k)
        V = self.split_heads(self.W_v(value))  # (batch, n_heads, seq_k, d_k)
        
        # Compute attention for all heads simultaneously
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_output: (batch, n_heads, seq_q, d_k)
        
        # Concatenate heads and project back
        attn_output = attn_output.transpose(1, 2)  # (batch, seq_q, n_heads, d_k)
        attn_output = attn_output.reshape(batch, -1, self.d_model)  # (batch, seq_q, d_model)
        
        output = self.W_o(attn_output)  # (batch, seq_q, d_model)
        return output

# Test
mha = MultiHeadAttention(d_model=512, n_heads=8, dropout=0.1)
x = torch.randn(2, 10, 512)  # (batch=2, seq=10, d_model=512)
output = mha(x, x, x)         # Self-attention
print(f"MHA output: {output.shape}")  # (2, 10, 512)
```

---

## Part 4: Positional Encoding

### 4.1 Why Positional Encoding Is Needed

Attention has no notion of **order** — `attention(ABCDE) = attention(BACED)`. We need to inject position information.

```python
class SinusoidalPositionalEncoding(nn.Module):
    """
    Positional encoding using sine/cosine functions.
    
    For position pos and dimension i:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Key properties:
    1. Deterministic (no learned parameters)
    2. Unique encoding for each position
    3. Generalizes to sequences longer than training length
    4. Relative positions can be expressed as linear combinations
       (PE(pos+k) is a linear function of PE(pos))
    
    Intuition: Like a binary counter, but smooth and continuous.
    Different frequencies for different dimensions:
    - Low-dim: fast oscillations (captures fine-grained position)
    - High-dim: slow oscillations (captures coarse position)
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Precompute all positional encodings
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        
        # Compute the division term for each dimension
        # 1/10000^(2i/d) = exp(-2i * log(10000) / d)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) — batch broadcasting
        
        # Register as buffer: part of state_dict but not a parameter (not trained)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        Adds positional encoding to the token embeddings.
        """
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len, :]  # Add positional info
        return self.dropout(x)
```

---

## Part 5: The Full Transformer Block

```python
class TransformerEncoderBlock(nn.Module):
    """
    One Transformer encoder layer.
    
    Architecture (Pre-LN — more stable than original Post-LN):
    x → LayerNorm → MultiHeadAttention → Add (residual) → x'
    x' → LayerNorm → FeedForward → Add (residual) → output
    
    Feed-Forward Network (FFN): Two linear layers with GELU.
    - Projects to 4*d_model then back: captures non-linear interactions
    - Each position processed independently (position-wise FFN)
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network: d_model → d_ff → d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),               # GELU: standard in modern transformers
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        
        # Layer normalization: normalize across features (not batch!)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Pre-LN: normalize BEFORE each sub-layer (more stable gradients)
        x: (batch, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_out = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attn_out)  # Residual connection
        
        # Feed-forward with residual connection
        ff_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ff_out)  # Residual connection
        
        return x


class TransformerEncoder(nn.Module):
    """Stack of N encoder blocks — the full BERT-style encoder"""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 d_ff: int, n_layers: int, max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)  # Final normalization
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """x: (batch, seq_len) — token IDs"""
        x = self.embedding(x)      # (batch, seq_len, d_model)
        x = self.pos_encoding(x)   # Add positional encoding
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)  # (batch, seq_len, d_model)


class GPTDecoderBlock(nn.Module):
    """
    GPT-style decoder block (decoder-only, no cross-attention).
    
    Key difference from encoder: uses CAUSAL masking
    → each position can only attend to previous positions
    → enables autoregressive language modelling
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        # Causal mask: position i can only attend to positions ≤ i
        causal_mask = make_causal_mask(seq_len, x.device)
        
        attn_out = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                   mask=causal_mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class MiniGPT(nn.Module):
    """
    GPT-style language model.
    
    Training: predict next token (causal LM)
    Generation: autoregressive sampling
    """
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # GPT uses LEARNED positional embeddings (not sinusoidal)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            GPTDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # LM head: project to vocabulary
        # Weight tying: share weights between embedding and LM head
        # Reduces parameters, often improves perplexity
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying!
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len) — token IDs
        Returns: (batch, seq_len, vocab_size) — logits for next token prediction
        """
        batch, seq_len = x.shape
        
        # Token embeddings + positional embeddings
        tok_emb = self.token_embedding(x)  # (batch, seq, d_model)
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embedding(positions)  # (seq, d_model) → broadcast
        
        x = self.dropout(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return self.lm_head(x)  # (batch, seq_len, vocab_size)
    
    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """
        Autoregressive text generation.
        temperature: > 1 = more random, < 1 = more focused
        top_k: only sample from top-k most probable tokens
        """
        for _ in range(max_new_tokens):
            # Trim context to max length
            x_cond = x[:, -1024:]
            
            # Get logits for last position
            logits = self(x_cond)[:, -1, :]  # (batch, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k sampling: zero out all but top-k logits
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, top_k)
                logits[logits < top_k_vals[:, [-1]]] = -float('inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append and continue
            x = torch.cat([x, next_token], dim=1)
        
        return x

# Build GPT-2 Small equivalent
model = MiniGPT(
    vocab_size=50257,  # GPT-2 BPE vocabulary
    d_model=768,       # Embedding dimension
    n_heads=12,        # 768/12 = 64 d_k per head
    n_layers=12,       # 12 transformer blocks
    d_ff=3072,         # 4 * d_model (standard)
    max_len=1024,      # Context length
    dropout=0.1
)

total_params = sum(p.numel() for p in model.parameters())
print(f"GPT-2 Small equivalent: {total_params/1e6:.1f}M parameters")  # ~117M

# Test forward pass
x = torch.randint(0, 50257, (2, 10))  # (batch=2, seq_len=10)
logits = model(x)
print(f"Logits shape: {logits.shape}")  # (2, 10, 50257)
```

---

## Part 6: Using HuggingFace Transformers

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

# Load pretrained BERT for classification
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize text
texts = ["PyTorch is amazing!", "I hate bugs in my code."]
encoding = tokenizer(
    texts,
    truncation=True,         # Truncate to max_length
    padding=True,            # Pad to longest in batch
    max_length=128,
    return_tensors="pt"      # Return PyTorch tensors
)
print(f"Input IDs: {encoding['input_ids'].shape}")  # (2, 128)

# Forward pass
with torch.no_grad():
    outputs = model(**encoding)
    logits = outputs.logits
    print(f"Logits: {logits.shape}")  # (2, 2)
    probs = torch.softmax(logits, dim=-1)
    print(f"Probabilities: {probs}")
```

---

## Key Takeaways

| Concept | Why It Matters |
|---------|----------------|
| **Attention mechanism** | Directly connects any two positions regardless of distance |
| **Scaled dot-product** | √d_k scaling prevents softmax saturation |
| **Multi-head attention** | Different heads learn different relationship types |
| **Causal masking** | Enables autoregressive generation (GPT-style) |
| **Positional encoding** | Injects order information into order-invariant attention |
| **Weight tying** | Embedding and LM head share weights → fewer params, better PPL |
| **Pre-LN** | More stable gradients than original Post-LN formulation |

---

## Quiz

1. **Why divide attention scores by √d_k?**
   - Answer: Prevents dot products from becoming too large, which would cause softmax to saturate and gradients to vanish

2. **What is the difference between encoder and decoder self-attention?**
   - Answer: Encoder uses bidirectional attention; decoder uses causal (lower-triangular) masking to prevent attending to future tokens

3. **What is weight tying in language models?**
   - Answer: Sharing embedding matrix weights with the LM head, reducing parameters and improving perplexity

4. **Why does multi-head attention use h smaller heads instead of one big head?**
   - Answer: Each head learns to attend to different relationship types; jointly they capture richer information

5. **What is the FFN's role in a transformer block?**
   - Answer: Position-wise non-linear transformation that allows each position to combine features independently

6. **Why do modern transformers use Pre-LN instead of Post-LN?**
   - Answer: Pre-LN keeps gradient magnitudes more stable during training, removing the need for learning rate warmup

7. **What does the causal mask contain?**
   - Answer: Lower-triangular matrix (1s below/on diagonal, 0s above) preventing attention to future positions

8. **What is temperature in text generation?**
   - Answer: Divides logits before softmax; <1 = sharper/focused distribution, >1 = flatter/random

9. **What does `register_buffer` do vs `nn.Parameter`?**
   - Answer: Buffer is part of state_dict but not trained; Parameter is trained by optimizer

10. **What is the computational complexity of self-attention?**
    - Answer: O(n²·d) where n is sequence length — quadratic in sequence length
