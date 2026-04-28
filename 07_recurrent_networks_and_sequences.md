# Module 07: Recurrent Networks & Sequences — Learning from Order

> **Goal:** Understand recurrent neural networks from first principles — why they exist, how LSTMs solve the vanishing gradient problem, and when to use them vs transformers.

---

## Learning Objectives

By the end of this module, you will:
- **Understand** why sequential data needs special architectures
- **Implement** a vanilla RNN from scratch to see how state flows
- **Grasp** the LSTM cell gates mathematically and intuitively
- **Handle** variable-length sequences with padding and packing
- **Build** a BiLSTM text classifier and a Seq2Seq model
- **Know** when RNNs are appropriate vs when to use Transformers

---

## Part 1: The Sequence Problem

### 1.1 Why Feedforward Networks Fail on Sequences

A feedforward network takes a fixed-size input and produces a fixed-size output. For sequences, we need to:
1. Process inputs of **variable length** (sentences can be 3 or 300 words)
2. Maintain **order sensitivity** ("dog bites man" ≠ "man bites dog")
3. Capture **long-range dependencies** ("The cat, which was sitting by the window, **was** hungry")

### 1.2 The Core RNN Idea: Hidden State

An RNN processes sequences one step at a time, passing a **hidden state** from step to step. The hidden state is the network's "memory."

```
Input:   x₁  →  x₂  →  x₃  →  ...  → xₜ
          ↓       ↓       ↓             ↓
h₀  →  [RNN] → [RNN] → [RNN] → ... → [RNN] → hₜ → output
         h₁      h₂      h₃            hₜ
```

At each step: `hₜ = tanh(W_hh · h_{t-1} + W_xh · xₜ + b)`

---

## Part 2: Vanilla RNN from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaRNNCell(nn.Module):
    """
    Single RNN step: takes one input and previous hidden state,
    produces new hidden state.
    
    Math: h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)
    
    W_xh: input → hidden (how much does input affect memory?)
    W_hh: hidden → hidden (how much does memory affect itself?)
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # Combined weight matrix: [W_xh | W_hh] applied to [x_t; h_{t-1}]
        # This is mathematically equivalent but computed in one operation
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        x_t:   (batch, input_size)  — current input token
        h_prev: (batch, hidden_size) — previous hidden state
        
        Returns h_t: (batch, hidden_size)
        """
        # Concatenate input and hidden state
        combined = torch.cat([x_t, h_prev], dim=-1)  # (batch, input+hidden)
        # Apply linear + tanh activation
        h_t = torch.tanh(self.linear(combined))
        return h_t


class VanillaRNN(nn.Module):
    """Full RNN that processes an entire sequence"""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = VanillaRNNCell(input_size, hidden_size)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        x: (batch, seq_len, input_size)
        Returns: (all_hidden, last_hidden)
            all_hidden: (batch, seq_len, hidden_size) — all time steps
            last_hidden: (batch, hidden_size) — final hidden state
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state to zeros
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        all_hidden = []
        for t in range(seq_len):
            x_t = x[:, t, :]         # (batch, input_size) — t-th input
            h = self.cell(x_t, h)    # (batch, hidden_size) — new hidden state
            all_hidden.append(h)
        
        # Stack all hidden states along time dimension
        all_hidden = torch.stack(all_hidden, dim=1)  # (batch, seq_len, hidden)
        
        return all_hidden, h  # Return all states AND final state

# Test
rnn = VanillaRNN(input_size=10, hidden_size=32)
x = torch.randn(8, 20, 10)   # batch=8, seq_len=20, features=10
all_h, last_h = rnn(x)
print(f"All hidden: {all_h.shape}")   # (8, 20, 32)
print(f"Last hidden: {last_h.shape}") # (8, 32)
```

### 2.1 The Vanishing Gradient Problem in RNNs

```python
# Why vanilla RNNs struggle with long sequences
# 
# During backpropagation through time (BPTT):
# gradient at step t = ∏ₜ (W_hh * tanh'(z_t))
#
# If tanh'(z) ≈ 0.5 and |W_hh| < 2 (common):
# gradient after 100 steps ≈ 0.5^100 ≈ 10^-30 → VANISHED!
#
# If |W_hh| > 2:
# gradient after 100 steps → ∞ → EXPLODED!

def demonstrate_vanishing_gradient():
    """Show how gradients vanish over time in vanilla RNN"""
    input_size, hidden_size, seq_len = 10, 32, 50
    rnn = VanillaRNN(input_size, hidden_size)
    x = torch.randn(1, seq_len, input_size)
    
    all_h, last_h = rnn(x)
    loss = last_h.sum()
    loss.backward()
    
    # Gradient of FIRST input should capture long-range dependency
    # In practice, it's nearly zero
    first_input_grad = x.grad
    if first_input_grad is not None:
        grad_norm = first_input_grad.norm()
        print(f"Gradient norm at step 0 (seq_len={seq_len}): {grad_norm:.6f}")
        # Expected: very small for long sequences
```

---

## Part 3: LSTM — Long Short-Term Memory

### 3.1 The LSTM Solution: Separate Cell State

The LSTM introduces a **cell state** (Cₜ) — a "conveyor belt" that runs through the sequence with only minor, controlled modifications. Information can flow largely unchanged across many timesteps.

```
         h_{t-1}        x_t
            │             │
            └──────┬───────┘
                   │
          ┌────────┴────────┐
          │    Gate Logic    │
          │  ┌─────────────┐│
          │  │ Forget Gate ││  ← "What from memory to erase?"
          │  │  fₜ = σ(...)││
          │  └──────┬──────┘│
     ┌────┤         │       ├────┐
     │    │  ┌──────┤       │   │
C_{t-1}─(×)─┤  ┌─────────┐│   (×)─→ C_t ─→ (tanh) ─→ hₜ
             │  │ Input G.││             ↑
             │  │  iₜ=σ(..)│         (×) Output Gate
             │  │  g̃ₜ=tanh││         oₜ = σ(...)
             │  │ (iₜ * g̃ₜ)│
             │  └─────────┘│
             └─────────────┘
```

### 3.2 LSTM Equations Explained

```python
class LSTMCell(nn.Module):
    """
    LSTM cell implementing the full gating mechanism.
    
    Gates (all use sigmoid — output in [0,1]):
    - Forget gate fₜ:  "What fraction of old memory to keep?" (0=forget, 1=keep)
    - Input gate iₜ:   "How much of new info to write?"
    - Output gate oₜ:  "What to read out from memory to hidden state?"
    
    Candidate:
    - g̃ₜ (tanh): "What new information to potentially write?"
    
    Cell state update:
    Cₜ = fₜ ⊙ C_{t-1} + iₜ ⊙ g̃ₜ
         ─────────────   ──────────
         keep fraction   add fraction of new info
    
    Hidden state:
    hₜ = oₜ ⊙ tanh(Cₜ)   ← output gate controls what to expose
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # One combined linear layer for efficiency
        # Projects [xₜ, h_{t-1}] → [fₜ, iₜ, g̃ₜ, oₜ] (4 × hidden)
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)
    
    def forward(self, x_t, h_prev, c_prev):
        """
        x_t:    (batch, input_size)
        h_prev: (batch, hidden_size)
        c_prev: (batch, hidden_size)
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x_t, h_prev], dim=1)  # (batch, input+hidden)
        
        # Compute all 4 gates at once (more efficient)
        gates = self.gates(combined)  # (batch, 4*hidden)
        
        # Split into individual gates
        f, i, g, o = gates.chunk(4, dim=1)  # Each: (batch, hidden)
        
        # Apply activations
        f = torch.sigmoid(f)   # Forget gate: [0,1]
        i = torch.sigmoid(i)   # Input gate:  [0,1]
        g = torch.tanh(g)      # Candidate:   [-1,1]
        o = torch.sigmoid(o)   # Output gate: [0,1]
        
        # Update cell state: keep some of old, add some of new
        c_t = f * c_prev + i * g  # Element-wise multiplication
        
        # Compute hidden state: gated output of cell state
        h_t = o * torch.tanh(c_t)
        
        return h_t, c_t


class LSTM(nn.Module):
    """Multi-step LSTM that processes full sequences"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        # Use PyTorch's built-in LSTM (optimized with cuDNN)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,       # input: (batch, seq, features) instead of (seq, batch, features)
            dropout=dropout if num_layers > 1 else 0,  # Dropout between layers
            bidirectional=bidirectional
        )
        # If bidirectional, hidden size is doubled (forward + backward)
        self.out_size = hidden_size * 2 if bidirectional else hidden_size
    
    def forward(self, x: torch.Tensor, lengths=None):
        """
        x:       (batch, seq_len, input_size)
        lengths: optional tensor of actual sequence lengths (for packing)
        Returns: (output, (h_n, c_n))
            output: (batch, seq_len, out_size) — hidden states at each step
            h_n:    (num_layers, batch, out_size) — final hidden state
            c_n:    (num_layers, batch, out_size) — final cell state
        """
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)

# Test LSTM
lstm = LSTM(input_size=50, hidden_size=128, num_layers=2,
            dropout=0.3, bidirectional=True)
x = torch.randn(16, 30, 50)  # batch=16, seq_len=30, embed_dim=50
output, (h_n, c_n) = lstm(x)
print(f"Output: {output.shape}")   # (16, 30, 256) — 128*2 for bidirectional
print(f"h_n: {h_n.shape}")        # (4, 16, 128) — 2 layers * 2 directions
```

### 3.3 GRU — Simplified LSTM

```python
# GRU (Gated Recurrent Unit) — Simpler than LSTM, often comparable performance
# 
# Key difference: No separate cell state! Only hidden state.
# Uses only 2 gates instead of 3:
# - Reset gate rₜ: "How much past to consider for candidate?"
# - Update gate zₜ: "How much new vs old hidden state to use?"
#
# h̃ₜ = tanh(Wxh*xₜ + rₜ ⊙ (Whh*h_{t-1}))   ← candidate
# hₜ = (1-zₜ) ⊙ h_{t-1} + zₜ ⊙ h̃ₜ        ← final update
#
# When zₜ → 0: hₜ → h_{t-1} (keep old state)
# When zₜ → 1: hₜ → h̃ₜ (full update)

# Built-in nn.GRU is optimized and easy to use
gru = nn.GRU(input_size=50, hidden_size=128, num_layers=2,
             batch_first=True, dropout=0.3, bidirectional=True)
x = torch.randn(16, 30, 50)
output, h_n = gru(x)   # GRU returns (output, h_n) — no c_n!
print(f"GRU output: {output.shape}")  # (16, 30, 256)
print(f"GRU h_n: {h_n.shape}")       # (4, 16, 128)

# GRU vs LSTM:
# GRU: fewer parameters (2 gates vs 3), faster to train
# LSTM: more expressive, slightly better on long dependencies
# In practice: try GRU first, use LSTM if more capacity needed
```

---

## Part 4: Handling Variable-Length Sequences

### 4.1 Padding and Packing

```python
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# Problem: sequences in a batch have different lengths
# Solution: pad shorter sequences with zeros, pack for efficiency

sequences = [
    torch.randn(5, 10),  # seq_len=5
    torch.randn(8, 10),  # seq_len=8
    torch.randn(3, 10),  # seq_len=3
]

# Pad to same length (max_length=8)
padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
print(f"Padded shape: {padded.shape}")  # (3, 8, 10) — 3 seqs, max_len=8, features=10

# Record original lengths
lengths = torch.tensor([5, 8, 3])

# Pack: removes padding, tells LSTM where sequences end
# IMPORTANT: sequences must be sorted by length (descending) before packing!
sorted_lengths, sort_idx = lengths.sort(descending=True)
padded_sorted = padded[sort_idx]

packed = pack_padded_sequence(padded_sorted, sorted_lengths.cpu(), batch_first=True)

# Run through LSTM (avoids computing on padding → faster + correct!)
lstm = nn.LSTM(input_size=10, hidden_size=32, batch_first=True)
output_packed, (h_n, c_n) = lstm(packed)

# Unpack back to padded format
output_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=True)
print(f"Output shape: {output_padded.shape}")  # (3, 8, 32) — padded back
```

---

## Part 5: Text Classification with BiLSTM

```python
class TextClassifier(nn.Module):
    """
    Bidirectional LSTM for sentiment/topic classification.
    
    Architecture:
    Input tokens → Embedding → BiLSTM → Pooling → Classifier
    
    Why bidirectional? Captures context from BOTH directions:
    "The movie was NOT good" — "NOT" affects "good" which comes after
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 n_classes: int, n_layers: int = 2, dropout: float = 0.3,
                 pad_idx: int = 0):
        super().__init__()
        
        # Embedding: maps token IDs to dense vectors
        # padding_idx: zero-out embeddings for padding tokens
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embed_dim, hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 2*hidden because bidirectional
        # Use mean pooling over time (captures all positions)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x:       (batch, seq_len) — token IDs
        lengths: (batch,) — actual lengths (before padding)
        """
        # Embed: (batch, seq_len) → (batch, seq_len, embed_dim)
        embedded = self.dropout(self.embedding(x))
        
        # Pack for efficient computation
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM
        packed_output, (h_n, c_n) = self.lstm(packed)
        
        # Unpack
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # output: (batch, seq_len, hidden*2)
        
        # Pooling strategies:
        # Option 1: Last hidden state of both directions
        # h_n: (n_layers*2, batch, hidden) → take last layer
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden*2)
        
        # Option 2: Mean pooling (often better)
        # Mask out padding positions before averaging
        mask = (x != 0).unsqueeze(-1).float()  # (batch, seq, 1)
        h_mean = (output * mask).sum(dim=1) / mask.sum(dim=1)  # (batch, hidden*2)
        
        # Use mean pooling
        return self.classifier(h_mean)

# Test
model = TextClassifier(vocab_size=10000, embed_dim=128, hidden_size=256,
                       n_classes=2, n_layers=2, dropout=0.3)
x = torch.randint(0, 10000, (32, 50))  # batch=32, seq_len=50
lengths = torch.randint(20, 50, (32,))  # varying lengths
out = model(x, lengths)
print(f"Output: {out.shape}")  # (32, 2) — binary classification
```

---

## Part 6: When RNN vs Transformer

| Aspect | RNN/LSTM | Transformer |
|--------|----------|-------------|
| **Parallelism** | Sequential (slow training) | Fully parallel (fast training) |
| **Long-range deps** | Struggles (vanishing gradient) | Excellent (direct attention) |
| **Memory** | O(seq_len) | O(seq_len²) — expensive for long |
| **Best for** | Short sequences, streaming | NLP, long context |
| **Datasets** | Works with less data | Needs more data |
| **Use today** | Time-series, edge inference | NLP, vision, most DL tasks |

---

## Quiz

1. **What problem does the LSTM cell state solve vs vanilla RNN?**
   - Answer: The vanishing gradient problem — cell state allows gradients to flow across many timesteps nearly unchanged

2. **What does the forget gate compute?**
   - Answer: A value in [0,1] that controls how much of the previous cell state to keep

3. **Why use bidirectional LSTMs?**
   - Answer: Captures context from both past and future positions simultaneously

4. **What does `pack_padded_sequence` do?**
   - Answer: Removes padding positions so the LSTM doesn't compute on padded tokens

5. **What is `batch_first=True` in nn.LSTM?**
   - Answer: Changes input shape from (seq, batch, features) to (batch, seq, features)

6. **What is the key difference between LSTM and GRU?**
   - Answer: GRU has no separate cell state and uses 2 gates (reset, update) instead of 3

7. **What is BPTT?**
   - Answer: Backpropagation Through Time — unrolling the RNN across time steps and computing gradients

8. **Why does mean pooling often outperform using the last hidden state?**
   - Answer: Mean pooling captures information from all positions, not just the last; more robust

9. **What is the vanishing gradient problem in RNNs?**
   - Answer: Gradients become exponentially small when backpropagating through many time steps

10. **When should you choose RNN over Transformer?**
    - Answer: Online/streaming inference, very long sequences with memory constraints, or when training data is limited
