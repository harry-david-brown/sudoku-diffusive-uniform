"""Load any checkpoint in this repo as a solver: puzzles [B,81] -> filled grids [B,81]."""
import torch
from .gidd import SudokuGIDD, HybridSchedule, sample_confidence
from .looped import LoopedTransformer, solve_looped


def load_solver(path, device='cpu', k=1, iters=128):
    """-> (solve function, one-line description).
    GIDD checkpoints are decoded with the confidence sampler, k cells per pass.
    Looped checkpoints (ours, or Sotaku's published weights) are run for `iters` iterations."""
    ck = torch.load(path, map_location=device, weights_only=True)
    state = ck['model'] if 'model' in ck else ck
    if any(key.startswith('initial_encoder') for key in state):
        model = LoopedTransformer().to(device)
        model.load_state_dict(state)
        model.eval()
        info = ck.get('info', 'looped transformer') if 'model' in ck else 'looped transformer (external weights)'
        return (lambda p: solve_looped(model, p, iters)), f"{path}: {info}; {iters} iterations"
    model = SudokuGIDD().to(device)
    model.load_state_dict(state)
    model.eval()
    return (lambda p: sample_confidence(model, p, k)), \
        f"{path}: GIDD p_u={ck['pu']}, trained on {ck['data']}, epoch {ck['epoch']}; confidence sampler k={k}"
