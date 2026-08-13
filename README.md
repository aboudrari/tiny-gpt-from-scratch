# Tiny GPT From Scratch

> A character-level GPT implemented end-to-end in **pure NumPy** — no PyTorch, no Keras, no autograd. Every forward pass, every gradient, every optimizer update written by hand.

---

## What this is

This project builds a working Transformer language model from the ground up, one function at a time. Starting from raw text tokenization and basic array operations, it grows into a full GPT with multi-head self-attention, LayerNorm, a position-wise FFN, residual connections, and an Adam optimizer — all in ~166 functions across ~1700 lines of NumPy.

The goal was not to use a framework. The goal was to understand exactly what happens inside one.

---

## Architecture

```
Input tokens (B, T)
      │
      ▼
Token Embedding  (vocab_size → d_model)
      +
Positional Embedding  (block_size → d_model)
      │
      ▼
┌─────────────────────────────┐
│   Transformer Block × N     │
│                             │
│  LayerNorm                  │
│  Multi-Head Self-Attention  │  ← causal mask, scaled dot-product
│  Residual connection        │
│                             │
│  LayerNorm                  │
│  Feed-Forward Network       │  ← Linear → ReLU → Linear
│  Residual connection        │
└─────────────────────────────┘
      │
      ▼
Final LayerNorm
      │
      ▼
LM Head  (d_model → vocab_size)
      │
      ▼
Logits (B, T, vocab_size)
```

**All gradients are derived and implemented manually** — including backprop through multi-head attention, LayerNorm, and the embedding scatter-add.

---

## What was implemented

| Part | Topics covered |
|------|---------------|
| Tokenization | `build_vocab`, character encoding/decoding, corpus preparation |
| NumPy fundamentals | Array creation, indexing, slicing, broadcasting, matmul |
| Softmax | Naive, overflow-safe 1D and 2D row-wise variants |
| Data pipeline | Batch sampling, train/val split, sliding window construction |
| Bigram model | Count-based with Laplace smoothing, NLL evaluation, autoregressive sampling |
| Neural bigram | One-hot encoding, weight lookup equivalence, SGD training loop |
| Backpropagation | Linear, bias, ReLU, softmax+cross-entropy gradients from first principles |
| LayerNorm | Forward (mean, variance, normalize, affine) and full backward pass |
| Embeddings | Token embedding forward/backward, learned positional embeddings |
| Attention | QKV projections, scaled dot-product, causal masking, output projection — forward and backward |
| Multi-head attention | Head splitting/merging, parallel head computation, output projection |
| Transformer block | Pre-LN sublayer pattern, FFN, residual connections — forward and backward |
| Full model | End-to-end forward, end-to-end backward, cache management |
| Adam optimizer | Moment initialization, bias correction, recursive tree update |
| Generation | Temperature scaling, top-k filtering, autoregressive sampling loop |

---

## How to run

```bash
python scaffold.py
```

Expected output:
```
vocab_size=28, vocab[:10]=['\n', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
train=1836 val=204
batch X shape=(4, 8) Y shape=(4, 8)
val_loss ~ 3.33
generated: 'hello ...'
```

---

## Key implementation details

**Numerically stable softmax** — subtracts the row max before exponentiating to prevent overflow, critical for long sequences where logits grow large.

**Causal masking** — the attention score matrix is masked with `-inf` at future positions before softmax, so each token can only attend to itself and what came before.

**Scatter-add gradients** — both the embedding backward and the neural bigram backward use `np.add.at` to accumulate gradients into the right rows without a Python loop.

**LayerNorm backward** — derived from scratch: `dx = (1/std) * (dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat))`.

**Recursive Adam update** — the optimizer walks the nested parameter tree (dicts and lists of arrays) in lockstep with the gradient tree, applying bias-corrected moment updates at every leaf.

---

## Dependencies

```
numpy
```

That's it.

---

## Built on

[Deep-ML](https://www.deep-ml.com) · 166 steps · Pure NumPy
