"""Measure difficulty with a looped model: how many iterations does each puzzle take?

  python iters_to_solve.py checkpoints/looped_desktop.pth 500
  python iters_to_solve.py checkpoints/looped_desktop.pth 500 --iters 256 --bands DEFGH --csv out.csv

For every puzzle the model is run for `--iters` iterations and its answer is read off after each one.
Recorded per puzzle: the first iteration at which the answer is fully correct (or never), whether it is
still correct at the end, and, if it ends wrong, how many cells are wrong and whether the answer had
stopped changing (settled on a wrong grid) or was still moving.
Reported per band, then by clue count and by rating within the hard bands. `--csv` writes one row per puzzle.
"""
import argparse
import torch, pandas as pd
from sudoku.data import HARD_TEST_PATH, read_hard, to_tensor, difficulty_bands
from sudoku.looped import LoopedTransformer

ap = argparse.ArgumentParser()
ap.add_argument('ckpt'); ap.add_argument('n', type=int, nargs='?', default=500)
ap.add_argument('--iters', type=int, default=128); ap.add_argument('--batch', type=int, default=250)
ap.add_argument('--bands', default='ABCDEFGH'); ap.add_argument('--csv')
a = ap.parse_args()

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
ck = torch.load(a.ckpt, map_location=device, weights_only=True)
model = LoopedTransformer().to(device); model.load_state_dict(ck['model'] if 'model' in ck else ck); model.eval()


@torch.no_grad()
def trace(p, s):
    """-> first correct iteration (0 = never), wrong cells at the end, iterations since the answer last changed."""
    h, preds = model.start(p)
    first = torch.zeros(len(p), dtype=torch.long, device=p.device)
    last_change = torch.zeros_like(first); prev = None
    for it in range(1, a.iters + 1):
        h, logits, preds = model.step(h, preds)
        ans = torch.where(p != 0, p, logits.argmax(-1) + 1)
        ok = (ans == s).all(1)
        first = torch.where((first == 0) & ok, torch.full_like(first, it), first)
        if prev is not None:
            last_change = torch.where((ans != prev).any(1), torch.full_like(first, it), last_change)
        prev = ans
    return first.cpu(), (ans != s).sum(1).cpu(), (a.iters - last_change).cpu()


df = read_hard(HARD_TEST_PATH); df['givens'] = df['question'].str.count(r'[1-9]')
rows = []
for name, mask in difficulty_bands(df):
    if name[0] not in a.bands:
        continue
    d = df[mask].sample(n=min(a.n, int(mask.sum())), random_state=0)
    P, S = to_tensor(d.question.tolist()), to_tensor(d.answer.tolist())
    out = [trace(P[i:i + a.batch].to(device), S[i:i + a.batch].to(device)) for i in range(0, len(P), a.batch)]
    first, wrong, still = (torch.cat(x).numpy() for x in zip(*out))
    rows.append(pd.DataFrame({'band': name, 'rating': d.rating.values, 'givens': d.givens.values,
                              'first_correct': first, 'wrong_at_end': wrong, 'unchanged_for': still}))
r = pd.concat(rows, ignore_index=True)
r['solved'] = r.wrong_at_end == 0
if a.csv:
    r.to_csv(a.csv, index=False)

print(f"{a.ckpt}, {a.iters} iterations, up to {a.n} puzzles per band")
print("\nPer band. 'iterations' = median [quarter, three-quarter] of the first correct iteration, solved puzzles only.")
print("'lost' = correct at some iteration but wrong at the end. 'settled' = share of failures whose answer had")
print("not changed for at least 16 iterations (stuck on a wrong grid, as opposed to still moving).")
for name, g in r.groupby('band', sort=False):
    s = g[g.solved]; f = g[~g.solved]; q = s.first_correct.quantile([.25, .5, .75]).values if len(s) else [0, 0, 0]
    lost = int(((g.first_correct > 0) & ~g.solved).sum())
    print(f"{name:26s} solved {g.solved.mean() * 100:5.1f}%  iterations {q[1]:4.0f} [{q[0]:.0f}, {q[2]:.0f}]  lost {lost:3d}  "
          + (f"failures: {f.wrong_at_end.median():.0f} cells wrong (median), settled {(f.unchanged_for >= 16).mean() * 100:.0f}%" if len(f) else ""))

def rank_corr(x, y):
    """Spearman rank correlation (Pearson correlation of the ranks)."""
    return x.astype(float).rank().corr(y.astype(float).rank())


hard = r[r.rating > 0]
if len(hard):
    print("\nBy backtrack rating (puzzles rated 1+):")
    hard = hard.assign(bin=pd.cut(hard.rating, [0, 2, 5, 10, 20, 35, 50, 75, 100, 200, 10 ** 6]))
    for b, g in hard.groupby('bin', observed=True):
        s = g[g.solved]
        print(f"  rating {str(b):16s} n={len(g):4d}  solved {g.solved.mean() * 100:5.1f}%  median iterations {s.first_correct.median() if len(s) else 0:4.0f}  givens {g.givens.mean():.1f}")
    print("\nBy clue count (puzzles rated 6+):")
    h6 = r[r.rating > 5]
    for b, g in h6.groupby('givens'):
        if len(g) >= 20:
            s = g[g.solved]
            print(f"  {b} clues  n={len(g):4d}  solved {g.solved.mean() * 100:5.1f}%  median iterations {s.first_correct.median() if len(s) else 0:4.0f}  median rating {g.rating.median():.0f}")
    s = h6[h6.solved]
    print(f"\nRank correlation with solved (rating 6+, n={len(h6)}): rating {rank_corr(h6.solved, h6.rating):+.2f}, clue count {rank_corr(h6.solved, h6.givens):+.2f}")
    if len(s) > 10:
        print(f"Rank correlation with iterations needed (solved only, n={len(s)}): rating {rank_corr(s.first_correct, s.rating):+.2f}, clue count {rank_corr(s.first_correct, s.givens):+.2f}")
