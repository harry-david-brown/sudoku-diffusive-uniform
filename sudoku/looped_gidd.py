"""Looped GIDD: the GIDD hybrid (mask + uniform noise) diffusion model with a looped denoiser.

The diffusion side is sudoku/gidd.py, unchanged: the same noise schedule (a share p_u of random digits
mixed into the masks), the same ELBO loss, the same samplers including self-correction. What changes is
the network that predicts the clean grid. In gidd.py it is four transformer layers run once. Here it is
four layers run `iters` times with shared weights, each pass feeding its own digit probabilities back in
(the looping mechanism of sudoku/looped.py), and the loss is applied after every pass.

Inputs to the network, per cell: the noisy token (a digit or MASK), whether the cell is a given, and the
noise level t. A non-given digit may be wrong (that is what uniform noise does), so the network has to be
told which digits to trust; the plain looped model has no such input because it never sees a wrong digit.

Layer code is imported from sudoku/looped.py. Parameter names differ from the looped model's on purpose,
so that a checkpoint of one is never loaded as the other.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .looped import RoPETransformerLayer, _rope_tables, D_MODEL, N_LAYERS, autocast
from .gidd import (MASK, N_DIGITS, N_STATES, T_EPS, HybridSchedule, tokens_to_state, gidd_loss, sample_t,
                   make_noisy, sample_confidence, sample_ancestral, self_correct, _start)


class LoopedGIDD(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.cell_encoder = nn.Linear(N_STATES + 1, D_MODEL)   # one-hot noisy state (9 digits + MASK) + given flag
        self.time_mlp = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.SiLU(), nn.Linear(D_MODEL, D_MODEL))
        self.feedback = nn.Linear(N_DIGITS, D_MODEL)           # previous pass's digit probabilities -> hidden
        self.layers = nn.ModuleList([RoPETransformerLayer(dropout=dropout) for _ in range(N_LAYERS)])
        self.output = nn.Linear(D_MODEL, N_DIGITS)
        cos, sin = _rope_tables()
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

    def time_embedding(self, t):
        half = D_MODEL // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * 1000 * freqs[None]
        return torch.cat([args.cos(), args.sin()], dim=-1)

    def start(self, z_tok, t, given):
        x = torch.cat([F.one_hot(tokens_to_state(z_tok), N_STATES).float(), given.float().unsqueeze(-1)], dim=-1)
        h = self.cell_encoder(x) + self.time_mlp(self.time_embedding(t))[:, None, :]
        return h, torch.zeros(z_tok.shape[0], 81, N_DIGITS, device=z_tok.device)

    def step(self, h, preds):
        h = h + self.feedback(preds)
        for layer in self.layers:
            h = layer(h, self.rope_cos, self.rope_sin)
        logits = self.output(h)
        return h, logits, F.softmax(logits, dim=-1)

    def forward(self, z_tok, t, given, iters=16, state=None):
        """-> (list of logits [B,81,9], one per pass; final state)"""
        h, preds = self.start(z_tok, t, given) if state is None else state
        out = []
        for _ in range(iters):
            h, logits, preds = self.step(h, preds)
            out.append(logits)
        return out, (h, preds)


def looped_gidd_loss(all_logits, x_state, z_state, t, unknown, sched, weighting='dynamic'):
    """The GIDD loss of sudoku/gidd.py applied after every pass and averaged (deep supervision).
    -> (training loss, ELBO of the last pass)"""
    loss = 0
    for logits in all_logits:
        l, elbo, _, _ = gidd_loss(logits.float(), x_state, z_state, t, unknown, sched, weighting)
        loss = loss + l
    return loss / len(all_logits), elbo


class Denoiser:
    """Makes a looped model look like gidd.py's one-pass model, model(z, t, given) -> logits, by running
    `iters` passes and returning the last. Lets every sampler in gidd.py be used unchanged."""

    def __init__(self, model, iters=16):
        self.model, self.iters = model, iters

    def __call__(self, z_tok, t, given):
        return self.model(z_tok, t, given, self.iters)[0][-1].float()


@torch.no_grad()
def solve(model, puzzles, sched, sampler='confidence', iters=16, k=4, steps=32):
    """sampler:
      'loop'        one denoising call from the fully masked grid, `iters` passes, read everything off.
                    The looped model's way of solving; no cell is committed and no noise level is stepped.
      'confidence'  commit the k most confident masked cells per call (cannot revise)
      'ancestral'   the paper's sampler, `steps` denoising calls; with p_u > 0 a filled cell can change
      'ancestral+correct'  ancestral, then the paper's self-correction loop
    Network passes: loop = iters; confidence = iters x ceil(blanks / k); ancestral = iters x (steps + 1)."""
    den = Denoiser(model, iters)
    if sampler == 'loop':
        z, given = _start(puzzles)
        t = torch.full((puzzles.shape[0],), 1 - T_EPS, device=puzzles.device)
        return torch.where(given, puzzles, den(z, t, given).argmax(-1) + 1)
    if sampler == 'confidence':
        return sample_confidence(den, puzzles, k)
    z = sample_ancestral(den, puzzles, sched, steps)
    if sampler == 'ancestral+correct':
        z = self_correct(den, z, puzzles != 0)
    return z
