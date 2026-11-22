"""Masked diffusion with a looped denoiser: the looped transformer (sudoku/looped.py) used as the network
inside a masked diffusion model.

Two things are added to the looped model, and nothing in it is changed:

  Training (mask_for_training): the model no longer always starts from the bare puzzle. Each puzzle gets a
    noise level t in (0, 1]; every blank cell stays masked with probability t and is otherwise filled in
    from the solution. The network sees that partly solved grid and must predict the cells still masked,
    on every one of its iterations. At t = 1 this is exactly the looped model's training; lower t are the
    intermediate states a solver passes through. This is the masked ("absorbing") diffusion objective with
    the givens held fixed. The loss is the looped model's own (looped_loss), which averages over masked
    cells across the batch, so heavily masked grids count for more than the diffusion bound's 1/t weighting
    would give them. The network gets no time input: for masked diffusion the grid itself shows how many
    cells are masked.

  Solving (solve_commit): run the loop for a few iterations, commit the k blank cells the model is most
    sure of, and start again from the grid with those cells filled in, until nothing is blank. Committed
    cells are never revisited. solve_looped (run the loop, read off everything at the end) remains
    available for the same weights, so the two ways of solving can be compared on one model.
"""
import torch
from .looped import LoopedTransformer  # noqa: F401  (same network; re-exported for convenience)


def mask_for_training(puzzles, solutions, generator=None, t_min=0.0):
    """-> grid [B,81]: givens kept, each blank cell masked (0) with probability t, else filled from the
    solution; t is drawn per puzzle, uniform on (t_min, 1]. At least one cell per puzzle stays masked."""
    B = puzzles.shape[0]
    t = 1.0 - torch.rand(B, 1, generator=generator) * (1.0 - t_min)              # (t_min, 1]
    u = torch.rand(B, 81, generator=generator)
    blank = puzzles.cpu() == 0
    keep = blank & (u < t)
    # guarantee one masked cell: the blank with the smallest u
    first = torch.where(blank, u, torch.full_like(u, 2.0)).argmin(1)
    keep[torch.arange(B), first] = True
    keep = keep.to(puzzles.device)
    return torch.where((puzzles == 0) & ~keep, solutions, puzzles)


@torch.no_grad()
def solve_commit(model, puzzles, iters=16, k=4, margin=False):
    """Iterative unmasking. Each round: run `iters` iterations on the current grid, then commit the k blank
    cells with the highest confidence (top probability, or top-two gap if margin=True) per puzzle.
    Total network passes = iters x ceil(most blanks / k)."""
    z = puzzles.clone()
    while bool((z == 0).any()):
        h, preds = model.start(z)
        for _ in range(iters):
            h, logits, preds = model.step(h, preds)
        top2 = preds.float().topk(2, dim=-1).values
        conf = (top2[..., 0] - top2[..., 1]) if margin else top2[..., 0]
        conf = torch.where(z == 0, conf, torch.full_like(conf, -1.0))
        pick = conf.topk(k, dim=1).indices                                       # [B,k]
        chosen = torch.zeros_like(z, dtype=torch.bool).scatter_(1, pick, True) & (z == 0)
        z = torch.where(chosen, logits.argmax(-1) + 1, z)
    return z
