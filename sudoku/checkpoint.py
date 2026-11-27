"""Load any checkpoint in this repo as a solver: puzzles [B,81] -> filled grids [B,81]."""
import torch
from .gidd import SudokuGIDD, HybridSchedule, sample_confidence
from .looped import LoopedTransformer, solve_looped
from .looped_diffusion import solve_commit


def load_solver(path, device='cpu', k=1, iters=128, commit=0, sampler=None, stop=None):
    """-> (solve function, one-line description).
    GIDD checkpoints are decoded with the confidence sampler, k cells per pass.
    Looped checkpoints (ours, or Sotaku's published weights) are run for `iters` iterations; with commit > 0
    they are instead decoded by iterative unmasking, `commit` cells per round of `iters` iterations.
    Looped GIDD checkpoints take `sampler`: loop, confidence, ancestral or ancestral+correct."""
    ck = torch.load(path, map_location=device, weights_only=True)
    state = ck['model'] if 'model' in ck else ck
    if isinstance(ck, dict) and ck.get('kind') == 'looped_gidd':
        from .looped_gidd import LoopedGIDD, solve
        model = LoopedGIDD().to(device); model.load_state_dict(state); model.eval()
        sched = HybridSchedule(ck['pu']); sampler = sampler or ('confidence' if commit else 'loop')
        how = {'loop': f"{iters} passes, nothing committed" + (f", stop = {stop}" if stop else ""),
               'confidence': f"commit {commit or 4} cells per call of {iters} passes"}.get(sampler, f"{sampler}, {iters} passes per call")
        return (lambda p: solve(model, p, sched, sampler, iters=iters, k=commit or 4, stop=stop)), f"{path}: {ck['info']}; {how}"
    if any(key.startswith('initial_encoder') for key in state):
        model = LoopedTransformer().to(device)
        model.load_state_dict(state)
        model.eval()
        info = ck.get('info', 'looped transformer') if 'model' in ck else 'looped transformer (external weights)'
        if commit:
            return (lambda p: solve_commit(model, p, iters, commit)), f"{path}: {info}; commit {commit} cells per round of {iters} iterations"
        return (lambda p: solve_looped(model, p, iters)), f"{path}: {info}; {iters} iterations"
    model = SudokuGIDD().to(device)
    model.load_state_dict(state)
    model.eval()
    return (lambda p: sample_confidence(model, p, k)), \
        f"{path}: GIDD p_u={ck['pu']}, trained on {ck['data']}, epoch {ck['epoch']}; confidence sampler k={k}"
