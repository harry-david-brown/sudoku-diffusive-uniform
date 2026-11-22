"""Kibble-Zurek check: do constraint violations fall as a power law in the number of decoding rounds?

For a masked (GIDD, p_u = 0) checkpoint, decode the same puzzles committing k cells per round for several k,
and record the exact number of rounds and the violated rows/columns/boxes. If violations ~ rounds^(-alpha),
log(violations) against log(rounds) is a straight line with slope -alpha.

  python kz_fit.py checkpoints/gidd_pu0_easyhard.pth 200
  python kz_fit.py checkpoints/gidd_pu0_easyhard.pth 200 '> 50'     # one band only
"""
import sys, math
import numpy as np, torch
from sudoku.data import HARD_TEST_PATH, read_hard, to_tensor
from sudoku.gidd import SudokuGIDD, sample_confidence
from sudoku.metrics import violations

ck_path, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200
KS = [81, 40, 20, 12, 8, 5, 3, 2, 1]
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
ck = torch.load(ck_path, map_location=device, weights_only=True)
model = SudokuGIDD().to(device); model.load_state_dict(ck['model']); model.eval()
df = read_hard(HARD_TEST_PATH); r = df['rating']
bands = [('rating 0 (not Kaggle, not 17-clue)', (r == 0) & ~df.source.isin(['puzzles0_kaggle', 'puzzles2_17_clue'])),
         ('rating 1-5', (r >= 1) & (r <= 5)), ('rating 6-50', (r > 5) & (r <= 50)), ('rating > 50', r > 50)]
only = sys.argv[3] if len(sys.argv) > 3 else ''      # optional: substring of a band name
print(ck_path)
for name, mask in bands:
    if only not in name:
        continue
    d = df[mask].sample(n=n, random_state=0)
    P = to_tensor(d.question.tolist()).to(device); unknown = (P == 0).sum(1).float()
    xs, ys = [], []
    print(f"\n{name}  (n={n}, {unknown.mean():.1f} unknown cells)\n   k   rounds   violations   solved")
    for k in KS:
        out = sample_confidence(model, P, k)
        S = to_tensor(d.answer.tolist()).to(device)
        rounds = torch.ceil(unknown / k).mean().item(); v = violations(out).float().mean().item()
        xs.append(rounds); ys.append(v)
        print(f"  {k:2d}   {rounds:6.1f}   {v:10.2f}   {(out == S).all(1).float().mean().item() * 100:5.1f}%", flush=True)
    for label, sel in (('all points', slice(None)), ('without one-shot', slice(1, None))):
        x, y = np.log(np.array(xs[sel])), np.log(np.maximum(np.array(ys[sel]), 1e-3))
        slope, icpt = np.polyfit(x, y, 1); res = y - (slope * x + icpt)
        r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
        local = np.diff(y) / np.diff(x)
        print(f"  fit, {label}: slope {slope:.2f}, R^2 {r2:.3f}; local slopes {' '.join(f'{s:.2f}' for s in local)}")
