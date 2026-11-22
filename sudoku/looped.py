"""Looped transformer for Sudoku: a port of Cheng Lou's Sotaku (github.com/chenglou/sotaku, v2).

One 4-layer transformer whose weights are reused on every iteration. Each iteration adds the previous
iteration's digit probabilities back into the hidden state, runs the four layers, and predicts every blank
cell again. Training supervises all 16 iterations; at test time the same network can simply be run for more.
This is not a diffusion model: nothing is masked or noised, and no cell is ever committed.

Module names match the reference so its published weights load directly (see load_solver in checkpoint.py).
Reference: stabilize/exp_testbed_20k.py (SudokuTransformer, RoPETransformerLayer).
"""
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL, N_HEADS, D_FF, N_LAYERS = 128, 4, 512, 4
ROPE_BASE = 10.0


def _rope_tables():
    """2D rotary position tables: half of each attention head's dimensions rotate with the row index,
    half with the column index. The model is given the grid layout and nothing about boxes."""
    head_dim = D_MODEL // N_HEADS          # 32
    rope_half = head_dim // 2              # 16 dims per axis
    pairs = rope_half // 2                 # 8 frequency pairs per axis
    freqs = 1.0 / (ROPE_BASE ** (torch.arange(pairs).float() * 2 / rope_half))
    rows = torch.tensor([i // 9 for i in range(81)]).float()
    cols = torch.tensor([i % 9 for i in range(81)]).float()
    angles = torch.cat([rows[:, None] * freqs[None], cols[:, None] * freqs[None]], dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    B, H, L, D = x.shape
    pairs = x.reshape(B, H, L, D // 2, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    c, s = cos[None, None], sin[None, None]
    return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).reshape(B, H, L, D)


class RoPETransformerLayer(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=0.1):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.q_proj, self.k_proj = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.v_proj, self.out_proj = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.linear1, self.linear2 = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)
        self.dropout, self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)
        self.attn_dropout_p = dropout

    def forward(self, x, cos, sin):
        h = self.norm1(x)
        B, L, D = h.shape
        split = lambda t: t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.q_proj(h)), split(self.k_proj(h)), split(self.v_proj(h))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout_p if self.training else 0.0)
        x = x + self.dropout1(self.out_proj(a.transpose(1, 2).contiguous().view(B, L, D)))
        return x + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(self.norm2(x))))))


class LoopedTransformer(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.initial_encoder = nn.Linear(10, D_MODEL)      # one-hot cell (0 = blank, 1..9) -> hidden
        self.pred_proj = nn.Linear(9, D_MODEL)             # previous digit probabilities -> hidden
        self.layers = nn.ModuleList([RoPETransformerLayer(dropout=dropout) for _ in range(N_LAYERS)])
        self.output_head = nn.Linear(D_MODEL, 9)
        cos, sin = _rope_tables()
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

    def start(self, puzzles):
        """puzzles: [B,81] ints 0..9 -> initial (hidden state, digit probabilities)"""
        h = self.initial_encoder(F.one_hot(puzzles, 10).float())
        return h, torch.zeros(puzzles.shape[0], 81, 9, device=puzzles.device)

    def step(self, h, preds):
        """One iteration: feed the last predictions back in, run the four layers, predict again."""
        h = h + self.pred_proj(preds)
        for layer in self.layers:
            h = layer(h, self.rope_cos, self.rope_sin)
        logits = self.output_head(h)
        return h, logits, F.softmax(logits, dim=-1)

    def forward(self, puzzles, iters=16, state=None, seeds=None):
        """-> (list of logits, one per iteration; final state).
        seeds: optional list of ints, one per iteration, fixing that iteration's dropout."""
        h, preds = self.start(puzzles) if state is None else state
        all_logits = []
        for i in range(iters):
            if seeds is not None:
                torch.manual_seed(seeds[i])
            h, logits, preds = self.step(h, preds)
            all_logits.append(logits)
        return all_logits, (h, preds)


def autocast(amp):
    """amp: None for ordinary 32-bit arithmetic, or (device type, dtype) for mixed precision,
    e.g. ('cuda', torch.float16) on an NVIDIA card."""
    return torch.autocast(amp[0], dtype=amp[1]) if amp else contextlib.nullcontext()


def looped_loss(all_logits, puzzles, solutions):
    """Cross-entropy on the blank cells, averaged over every iteration (deep supervision)."""
    blank = (puzzles == 0).float()
    target = (solutions - 1).reshape(-1)
    loss = 0
    for logits in all_logits:
        ce = F.cross_entropy(logits.float().reshape(-1, 9), target, reduction='none').view_as(blank)
        loss = loss + (ce * blank).sum() / blank.sum()
    return loss / len(all_logits)


@torch.no_grad()
def solve_looped(model, puzzles, iters=128):
    """Run `iters` iterations and read off the answer. Given cells are kept as given."""
    h, preds = model.start(puzzles)
    for _ in range(iters):
        h, logits, preds = model.step(h, preds)
    return torch.where(puzzles != 0, puzzles, logits.argmax(-1) + 1)


def backward_low_memory(model, puzzles, solutions, iters=16, state=None, generator=None, amp=None, scale=1.0):
    """Compute the same loss and gradients as looped_loss(...).backward(), holding one iteration in memory
    at a time instead of all of them.

    Ordinary backpropagation through 16 iterations keeps every layer's activations for all 16: about
    130 MB per puzzle, so a batch of 256 needs over 30 GB. Here the iterations are first run without
    gradients, keeping only each one's small input state. Then they are replayed one at a time from last to
    first; each replay gets the gradient flowing back from the iteration after it and passes its own
    back. Dropout is seeded per iteration so the replay matches the first run.
    Memory is about a sixteenth; time is roughly a third more. Fills p.grad; returns (loss, last logits).

    amp, scale: mixed precision. Gradients are multiplied by `scale` (pass GradScaler.get_scale()) so that
    16-bit gradients do not underflow; the scaler divides it back out when it steps the optimizer."""
    blank = (puzzles == 0).float(); n_blank = blank.sum(); target = (solutions - 1).reshape(-1)
    params = [p for p in model.parameters() if p.requires_grad]
    seeds = torch.randint(0, 2 ** 31 - 1, (iters,), generator=generator).tolist()
    with torch.no_grad():
        h, preds = model.start(puzzles) if state is None else state
        inputs = []
        for i in range(iters):
            inputs.append((h, preds))
            torch.manual_seed(seeds[i])
            with autocast(amp):
                h, logits, preds = model.step(h, preds)
        last_logits = logits.float()
    grads = [torch.zeros_like(p) for p in params]
    g_h = g_p = None; total = 0.0

    def accumulate(found):
        for acc, g in zip(grads, found):
            if g is not None:
                acc += g.to(acc.dtype)

    for i in reversed(range(iters)):
        h_in = inputs[i][0].detach().requires_grad_(True); p_in = inputs[i][1].detach().requires_grad_(True)
        torch.manual_seed(seeds[i])
        with autocast(amp):
            h_out, logits, p_out = model.step(h_in, p_in)
        ce = F.cross_entropy(logits.float().reshape(-1, 9), target, reduction='none').view_as(blank)
        objective = (ce * blank).sum() / n_blank / iters
        total += objective.item()
        objective = objective * scale
        if g_h is not None:                      # gradient arriving from the iteration after this one
            objective = objective + (h_out * g_h).sum() + (p_out * g_p).sum()
        found = torch.autograd.grad(objective, [h_in, p_in] + params, allow_unused=True)
        g_h, g_p = found[0], found[1]
        accumulate(found[2:])
    if state is None:                            # the initial encoder produced the first hidden state
        h0, _ = model.start(puzzles)
        accumulate(torch.autograd.grad((h0 * g_h).sum(), params, allow_unused=True))
    for p, g in zip(params, grads):
        p.grad = g
    return total, last_logits
