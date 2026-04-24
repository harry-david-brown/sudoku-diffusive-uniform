# Sudoku Uniform Diffusion

Part of a project comparing autoregressive and diffusion-based approaches to constraint satisfaction problems. Sudoku is the benchmark.

---

## What This Is

A discrete uniform diffusion model trained to solve Sudoku puzzles. Instead of masking unknown cells, they are replaced with random digits during training — the model must learn to detect which tokens are wrong and correct them.

---

## Architecture

| Component | Value |
|---|---|
| Sequence length | 81 tokens (one per cell, row-major) |
| Input vocabulary | 10 tokens (digits 0–9) |
| Output vocabulary | 10 tokens (digits 0–9) |
| Embedding dim | 128 |
| Attention heads | 4 |
| Transformer layers | 4 |
| Feedforward dim | 512 |
| Attention | Bidirectional — no causal mask |
| Total parameters | 806,026 |

---

## Training

```
Dataset:      Kaggle 1M Sudoku dataset (bryanpark/sudoku)
              500,000 puzzles used for training
Noise:        Per-puzzle random corruption of unknown cells
              Corrupted cells replaced with random digit 1–9
              Given cells never corrupted
Epochs:       20
Batch:        64
Optimizer:    Adam, lr=1e-3
Loss:         CrossEntropyLoss on corrupted positions only
Device:       Apple MPS (M-series Mac)
Time:         ~9 hours
```

---

## Results

**Loss curve — 500k/20ep:**

```
Epoch 1:  1.1356
Epoch 5:  0.8585
Epoch 10: 0.8190
Epoch 20: 0.7986
```

Compare to masked diffusion 500k/20ep final loss: 0.0213. Uniform diffusion converges far more slowly.

**Easy puzzles (n=1000 sample):**

```
One-shot:
  Cell accuracy:         44.96%
  Puzzle accuracy:       0.00%
  Puzzles w/ violations: 1000/1000
  Avg violations:        25.14

Iterative decoding:
  k=1  — Accuracy: 0.60% — Avg violations: 15.64
  k=5  — Accuracy: 0.20% — Avg violations: 16.28
  k=10 — Accuracy: 0.10% — Avg violations: 17.11
  k=20 — Accuracy: 0.00% — Avg violations: 19.70
  k=81 — Accuracy: 0.00% — Avg violations: 25.17
```

**Scaling behavior:**

```
100k/10ep:  Cell 43.76%, Puzzle 0.00%
100k/20ep:  Cell 44.75%, Puzzle 0.00%
500k/20ep:  Cell 44.96%, Puzzle 0.00%
```

---

## What the Model Learned

Despite zero puzzle accuracy, the model is not outputting random noise. Confidence analysis on unknown cells:

```
Mean max confidence: 0.5842  (random baseline: 0.1111)
Min max confidence:  0.2826
Max max confidence:  0.9684
```

The model has learned to rule out unlikely digits — near-zero probabilities appear for most positions. It has learned *what is unlikely* better than *what is correct*. This is the first half of the error-detection task; the second half (identifying the correct replacement) requires more compute to emerge.

---

## Why It Fails at This Scale

With masked diffusion, a MASK token unambiguously signals "predict something here." With uniform noise, a corrupted cell is indistinguishable from a given cell by appearance alone — the model must infer from global context whether any token is correct or wrong. This requires simultaneously learning two distinct skills:

1. **Error detection** — is this token wrong?
2. **Error correction** — if wrong, what should it be?

The current single-output-head architecture conflates these tasks, producing a diffuse gradient signal. Von Rütte et al. conjecture that explicitly separating these into two heads, one for the holding distribution (when to change) and one for the jump chain (what to change to), would produce cleaner gradients and faster convergence.

