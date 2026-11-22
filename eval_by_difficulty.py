"""Score a checkpoint by difficulty band on the hard test file. This is the standard report.

  python eval_by_difficulty.py checkpoints/gidd_pu0_easyhard.pth 500
  python eval_by_difficulty.py checkpoints/looped.pth 500 --iters 1024

Bands split the test file by backtrack rating and source (see sudoku/data.py). For each band: puzzles
solved, unknown cells correct, and violated rows/columns/boxes per puzzle.
"""
import argparse, time
import torch
from sudoku.data import HARD_TEST_PATH, read_hard, to_tensor, difficulty_bands
from sudoku.metrics import score
from sudoku.checkpoint import load_solver

ap = argparse.ArgumentParser()
ap.add_argument('ckpt'); ap.add_argument('n', type=int, nargs='?', default=500)
ap.add_argument('--k', type=int, default=1, help='GIDD: cells committed per pass')
ap.add_argument('--iters', type=int, default=128, help='looped: iterations at test time')
ap.add_argument('--commit', type=int, default=0, help='looped: commit this many cells per round of --iters iterations (0 = loop only)')
ap.add_argument('--batch', type=int, default=250); ap.add_argument('--bands', default='ABCDEFGH')
ap.add_argument('--sampler', default=None, help='looped GIDD: loop | confidence | ancestral | ancestral+correct')
a = ap.parse_args()

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
solve, info = load_solver(a.ckpt, device, k=a.k, iters=a.iters, commit=a.commit, sampler=a.sampler)
print(info)
df = read_hard(HARD_TEST_PATH)
df['givens'] = df['question'].str.count(r'[1-9]')
for name, mask in difficulty_bands(df):
    if name[0] not in a.bands:
        continue
    d = df[mask].sample(n=min(a.n, int(mask.sum())), random_state=0)
    P, S = to_tensor(d.question.tolist()), to_tensor(d.answer.tolist())
    solved = cells = unk = viol = 0; t0 = time.time()
    for i in range(0, len(P), a.batch):
        p, s = P[i:i + a.batch].to(device), S[i:i + a.batch].to(device)
        r = score(solve(p), p, s)
        solved += r[0]; cells += r[1]; unk += r[2]; viol += r[3]
    print(f"{name:26s} n={len(d):3d} givens {d.givens.mean():4.1f}  solved {solved / len(d) * 100:5.1f}%  "
          f"cells {cells / unk * 100:5.1f}%  viol {viol / len(d):5.2f}  ({time.time() - t0:.0f}s)", flush=True)
