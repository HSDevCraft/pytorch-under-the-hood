# Module 07: Recurrent Networks & Sequence Models

## Learning Objectives
By the end of this module you will be able to:
- Explain the vanishing gradient problem in vanilla RNNs and how LSTMs solve it
- Implement RNN, LSTM, and GRU from scratch and using nn.RNN/LSTM/GRU
- Build sequence-to-sequence models with encoder-decoder architectures
- Handle variable-length sequences with padding and packing
- Apply teacher forcing and scheduled sampling for sequence generation
- Build character-level and word-level language models
- Implement bidirectional and stacked (multi-layer) RNNs correctly

---

## 7.1 The Recurrence Equation

A **Recurrent Neural Network** processes a sequence x₁, x₂, ..., xₜ one step at a time, maintaining a hidden state hₜ that carries information from previous steps:

```
hₜ = tanh(W_hh · h_{t-1} + W_xh · xₜ + b_h)
yₜ = W_hy · hₜ + b_y
```

**Backpropagation Through Time (BPTT):**

Gradients flow backward through all T timesteps:
```
∂L/∂W_hh = Σ_{t=1}^{T} ∂Lₜ/∂W_hh
```

The chain rule produces factors of `W_hh^T` in the gradient — if eigenvalues of `W_hh` < 1: **vanishing gradients**; if > 1: **exploding gradients**.

---

## 7.2 Vanilla RNN in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Manual RNN cell ──────────────────────────────────────────────────────────
class RNNCell(nn.Module):
    """Single RNN step: hₜ = tanh(x_t @ W_xh.T + h_{t-1} @ W_hh.T + b)"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.W_xh = nn.Linear(input_size,  hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.W_xh(x_t) + self.W_hh(h_prev))


class ManualRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = RNNCell(input_size, hidden_size)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        x: (batch, seq_len, input_size)
        returns: (outputs, final_hidden)
        """
        batch, T, _ = x.shape
        h = torch.zeros(batch, self.hidden_size, device=x.device)
        outputs = []
        for t in range(T):
            h = self.cell(x[:, t, :], h)
            outputs.append(h)
        outputs = torch.stack(outputs, dim=1)   # (batch, T, hidden)
        return self.fc(outputs), h


# ── PyTorch built-in RNN ─────────────────────────────────────────────────────
rnn = nn.RNN(
    input_size=10,
    hidden_size=32,
    num_layers=2,        # stacked RNN
    batch_first=True,    # (batch, seq, feature) — preferred
    dropout=0.3,         # applied between layers (not after last layer)
    bidirectional=False,
    nonlinearity="tanh", # 'tanh' or 'relu'
)

x = torch.randn(8, 20, 10)  # (batch=8, seq=20, features=10)
output, h_n = rnn(x)
# output: (8, 20, 32) — all hidden states
# h_n:   (num_layers, 8, 32) — final hidden state per layer
```

---

## 7.3 LSTM: Long Short-Term Memory

The LSTM solves the vanishing gradient problem by introducing a **cell state** cₜ that flows through time with minimal perturbation (via additive updates rather than multiplicative).

**LSTM equations:**
```
fₜ = σ(W_f · [h_{t-1}, xₜ] + b_f)    ← forget gate
iₜ = σ(W_i · [h_{t-1}, xₜ] + b_i)    ← input gate
c̃ₜ = tanh(W_c · [h_{t-1}, xₜ] + b_c)  ← candidate cell
cₜ = fₜ ⊙ c_{t-1} + iₜ ⊙ c̃ₜ         ← cell state update
oₜ = σ(W_o · [h_{t-1}, xₜ] + b_o)    ← output gate
hₜ = oₜ ⊙ tanh(cₜ)                   ← hidden state
```

Key insight: the cell state gradient has a "highway" through `fₜ` (forget gate), which can be ≈1, allowing gradients to flow hundreds of steps back.

```python
# ── Manual LSTM cell ─────────────────────────────────────────────────────────
class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # Combined gate matrix: [W_f, W_i, W_c, W_o] stacked
        # More efficient than 4 separate linears
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor
    ) -> tuple:
        combined = torch.cat([x_t, h_prev], dim=-1)   # (batch, input+hidden)
        raw      = self.gates(combined)                # (batch, 4*hidden)

        # Split into 4 gate activations
        f, i, c_tilde, o = raw.chunk(4, dim=-1)

        f       = torch.sigmoid(f)
        i       = torch.sigmoid(i)
        c_tilde = torch.tanh(c_tilde)
        o       = torch.sigmoid(o)

        c_t = f * c_prev + i * c_tilde    # cell update (additive!)
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


# ── PyTorch built-in LSTM ─────────────────────────────────────────────────────
lstm = nn.LSTM(
    input_size=64,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    dropout=0.3,
    bidirectional=True,
)

x = torch.randn(32, 50, 64)   # (batch=32, seq=50, features=64)
output, (h_n, c_n) = lstm(x)
# output: (32, 50, 512)   — 256 * 2 (bidirectional)
# h_n:   (4,  32, 256)    — 2 layers * 2 directions
# c_n:   (4,  32, 256)

# For classification using the last hidden state:
# Concatenate forward and backward final states
final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (32, 512)
```

---

## 7.4 GRU: Gated Recurrent Unit

Simpler than LSTM (no cell state), nearly as effective, faster to train:

```
zₜ = σ(W_z · [h_{t-1}, xₜ])          ← update gate
rₜ = σ(W_r · [h_{t-1}, xₜ])          ← reset gate
h̃ₜ = tanh(W · [rₜ ⊙ h_{t-1}, xₜ])    ← candidate hidden
hₜ = (1 - zₜ) ⊙ h_{t-1} + zₜ ⊙ h̃ₜ   ← hidden update
```

```python
gru = nn.GRU(input_size=64, hidden_size=256, num_layers=2,
             batch_first=True, dropout=0.3, bidirectional=False)

x = torch.randn(32, 50, 64)
output, h_n = gru(x)
# output: (32, 50, 256)
# h_n:   (2, 32, 256)
```

### RNN vs GRU vs LSTM: When to Use Which

| Model | Parameters | Speed | Memory | Best For |
|-------|-----------|-------|--------|---------|
| RNN | Fewest | Fastest | Least | Short sequences (< 20 tokens) |
| GRU | Medium | Fast | Less | Medium sequences; resource-limited |
| LSTM | Most | Slower | More | Long sequences; standard NLP |

---

## 7.5 Variable-Length Sequences: Padding & Packing

Real sequences have variable lengths. PyTorch's `pack_padded_sequence` avoids computing on padding tokens — critical for training efficiency and correct gradients.

```python
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# ── Padding a batch of sequences ──────────────────────────────────────────────
sequences = [
    torch.randn(5, 64),   # seq_len=5
    torch.randn(3, 64),   # seq_len=3
    torch.randn(7, 64),   # seq_len=7
]
# Sort by length descending (required for pack_padded_sequence)
sequences.sort(key=lambda x: x.shape[0], reverse=True)
lengths = torch.tensor([s.shape[0] for s in sequences])

padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
# shape: (batch=3, max_len=7, features=64)

# ── Pack → LSTM → Unpack ──────────────────────────────────────────────────────
packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=True)

lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
packed_output, (h_n, c_n) = lstm(packed)

output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
# output: (batch, max_len, hidden) — padding zeros restored

# ── Alternative: use padding_mask in attention (for transformers) ─────────────
# Many modern implementations just use attention masks instead of packing
```

---

## 7.6 Language Model: Character-Level

```python
import torch
import torch.nn as nn
from typing import Tuple

class CharLM(nn.Module):
    """
    Character-level language model.
    Input: (batch, seq_len) integer character IDs
    Output: (batch, seq_len, vocab_size) logits
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, n_layers: int, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm      = nn.LSTM(
            embed_dim, hidden_size, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0,
        )
        self.drop   = nn.Dropout(dropout)
        self.fc     = nn.Linear(hidden_size, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self, x: torch.Tensor, hidden: Tuple = None
    ) -> Tuple[torch.Tensor, Tuple]:
        emb    = self.drop(self.embedding(x))       # (B, T, E)
        out, hidden = self.lstm(emb, hidden)        # (B, T, H)
        logits = self.fc(self.drop(out))            # (B, T, V)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple:
        h = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        c = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        return h, c

    @torch.no_grad()
    def generate(
        self, start_token: int, max_len: int = 200, temperature: float = 1.0, device: str = "cpu"
    ) -> list:
        """Autoregressive generation with temperature sampling."""
        self.eval()
        x = torch.tensor([[start_token]], device=device)
        hidden = self.init_hidden(1, device)
        generated = [start_token]

        for _ in range(max_len):
            logits, hidden = self(x, hidden)       # (1, 1, V)
            logits = logits[:, -1, :] / temperature
            probs  = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            x = torch.tensor([[next_token]], device=device)

        return generated


# Training with truncated BPTT
def train_lm_epoch(model, loader, optimizer, criterion, device, bptt_len=35):
    model.train()
    total_loss = 0
    hidden = model.init_hidden(loader.dataset.batch_size, device)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # Detach hidden to prevent backprop into previous chunks
        hidden = tuple(h.detach() for h in hidden)

        optimizer.zero_grad()
        logits, hidden = model(x, hidden)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)
```

---

## 7.7 Sequence-to-Sequence with Encoder-Decoder

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, n_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm      = nn.LSTM(embed_dim, hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> Tuple:
        """src: (batch, src_len) → context: (h_n, c_n)"""
        emb = self.dropout(self.embedding(src))
        _, (h_n, c_n) = self.lstm(emb)
        return h_n, c_n


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, n_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm      = nn.LSTM(embed_dim, hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.fc        = nn.Linear(hidden_size, vocab_size)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, tgt_token: torch.Tensor, hidden: Tuple) -> Tuple:
        """tgt_token: (batch,) → logits: (batch, vocab), hidden"""
        emb    = self.dropout(self.embedding(tgt_token.unsqueeze(1)))  # (B, 1, E)
        out, hidden = self.lstm(emb, hidden)                           # (B, 1, H)
        logits = self.fc(out.squeeze(1))                               # (B, V)
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, pad_idx: int = 0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        src: (batch, src_len)
        tgt: (batch, tgt_len)
        Returns: logits (batch, tgt_len - 1, vocab_size)
        """
        batch_size, tgt_len = tgt.shape
        vocab_size = self.decoder.fc.out_features

        # Encode source
        hidden = self.encoder(src)

        # First decoder input = <BOS> token
        dec_input  = tgt[:, 0]
        all_logits = []

        for t in range(1, tgt_len):
            logits, hidden = self.decoder(dec_input, hidden)
            all_logits.append(logits)

            # Teacher forcing: use ground truth vs model prediction
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            dec_input = tgt[:, t] if use_teacher else logits.argmax(-1)

        return torch.stack(all_logits, dim=1)   # (B, tgt_len-1, V)

    @torch.no_grad()
    def generate(
        self, src: torch.Tensor, bos_idx: int, eos_idx: int,
        max_len: int = 100, device: str = "cpu"
    ) -> list:
        self.eval()
        hidden = self.encoder(src)
        dec_input  = torch.full((src.size(0),), bos_idx, dtype=torch.long, device=device)
        all_preds  = []

        for _ in range(max_len):
            logits, hidden = self.decoder(dec_input, hidden)
            preds = logits.argmax(-1)
            all_preds.append(preds)
            dec_input = preds
            if (preds == eos_idx).all():
                break

        return torch.stack(all_preds, dim=1)   # (batch, out_len)
```

---

## 7.8 Text Classification with Bidirectional LSTM

```python
class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for text classification.
    Takes word indices as input, outputs class logits.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_size: int = 256,
        n_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, n_layers,
            batch_first=True, dropout=dropout, bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        # 4 features: [max_pool, mean_pool, forward_final, backward_final]
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        emb = self.dropout(self.embedding(x))              # (B, T, E)

        if lengths is not None:
            packed  = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, (h_n, _) = self.lstm(packed)
            out, _ = pad_packed_sequence(out, batch_first=True)  # (B, T, 2H)
        else:
            out, (h_n, _) = self.lstm(emb)

        # Pooling strategies
        max_pool  = out.max(dim=1).values                  # (B, 2H)
        mean_pool = out.mean(dim=1)                        # (B, 2H)

        # Final states from both directions (last layer)
        fwd_final = h_n[-2]                                # (B, H)
        bwd_final = h_n[-1]                                # (B, H)

        combined  = torch.cat([max_pool, mean_pool, fwd_final, bwd_final], dim=-1)  # (B, 4H)
        return self.classifier(self.dropout(combined))
```

---

## 7.9 Best Practices for RNNs

| Practice | Why |
|----------|-----|
| Use LSTM/GRU over vanilla RNN | Vanishing gradients are fatal for sequences > 20 steps |
| `batch_first=True` | Matches convention of (B, T, F); easier to work with |
| Detach hidden state between chunks (TBPTT) | Prevents accumulating the entire sequence graph in memory |
| Sort sequences by length (or use `enforce_sorted=False`) | Required for `pack_padded_sequence` |
| Use bidirectional for classification; unidirectional for generation | Bidir needs full sequence; generation is causal |
| Gradient clipping (`max_norm=0.25` for char-LM) | BPTT through many steps amplifies gradients |
| Initialize LSTM biases: forget gate bias = 1.0 | Encourage the network to remember at the start |
| Orthogonal init for recurrent weights | Helps preserve gradients in early training |

```python
# Forget gate bias initialisation
def init_forget_bias(lstm: nn.LSTM):
    for name, param in lstm.named_parameters():
        if "bias" in name:
            n = param.size(0)
            # LSTM bias layout: [b_ii, b_if, b_ig, b_io]
            # forget gate is the second quarter
            param.data[n//4 : n//2].fill_(1.0)
```

---

## Exercises

**Exercise 7.1** Implement a GRU from scratch (GRUCell) and verify it matches `nn.GRUCell` on the same weights. Use `gradcheck` to verify backward correctness.

**Exercise 7.2** Build a character-level language model trained on Shakespeare's works. Report bits-per-character (BPC) and generate 500-character samples at temperatures 0.5, 1.0, and 1.5.

**Exercise 7.3** Implement attention-enhanced Seq2Seq: add a Bahdanau attention layer (dot-product between decoder hidden state and all encoder outputs), producing a context vector at each decode step.

---

## Module Summary

| Model | Key Innovation | Equation |
|-------|---------------|---------|
| RNN | Hidden state | hₜ = tanh(W_xh·xₜ + W_hh·h_{t-1}) |
| LSTM | Cell state + 3 gates | cₜ = fₜ⊙c_{t-1} + iₜ⊙c̃ₜ; hₜ = oₜ⊙tanh(cₜ) |
| GRU | 2 gates, no cell | hₜ = (1-zₜ)⊙h_{t-1} + zₜ⊙h̃ₜ |

---

## Quiz

1. Why does the vanilla RNN suffer from vanishing gradients but the LSTM cell state does not?
2. What is the shape of `h_n` for a bidirectional 2-layer LSTM with batch=8, hidden=128?
3. What is teacher forcing and what is its drawback?
4. Why must you detach hidden states between BPTT chunks?
5. What does `pack_padded_sequence` do internally to avoid computing on padding?
6. What is the difference between `output` and `h_n` from `nn.LSTM`?

---

*Next: [Module 08 — Attention Mechanisms & Transformers](./08_attention_and_transformers.md)*
