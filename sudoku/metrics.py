"""Scoring."""
import torch.nn.functional as F


def violations(grid):
    """grid: [B,81] digits 1..9 -> how many of the 27 rows/columns/boxes are not a permutation of 1..9."""
    g = grid.reshape(-1, 9, 9)
    oh = F.one_hot(g.clamp(1, 9) - 1, 9)
    rows = (oh.sum(2) != 1).any(-1).sum(1)
    cols = (oh.sum(1) != 1).any(-1).sum(1)
    boxes = (oh.reshape(-1, 3, 3, 3, 3, 9).sum((2, 4)) != 1).any(-1).sum((1, 2))
    return rows + cols + boxes


def score(out, puzzles, solutions):
    """-> (puzzles solved, unknown cells correct, unknown cells, total violated units)"""
    unk = puzzles == 0
    return ((out == solutions).all(1).sum().item(), ((out == solutions) & unk).sum().item(),
            unk.sum().item(), violations(out).sum().item())
