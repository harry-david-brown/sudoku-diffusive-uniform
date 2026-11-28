"""Does the loop take its own guesses back? Trace a looped model pass by pass.
Works on looped GIDD checkpoints and on plain looped checkpoints (looped_sudoku.py, Sotaku's weights).

  python loop_trace.py checkpoints/loopgidd_ce_pu02_desktop.pth 200
  python loop_trace.py checkpoints/loopgidd_ce_pu0_desktop.pth 200 --bands EFGH

The model is run from the fully masked grid for `--iters` passes with nothing committed (the 'loop'
decoder) and its answer is read off after every pass. Reported per band:
  right after pass k     share of unknown cells correct at passes 1, 2, 4, 8, 16, 32 and the last
  early errors fixed     of the cells wrong after pass `--early`, the share correct at the end
  early rights lost      of the cells right after pass `--early`, the share wrong at the end
  changed its mind       cells whose answer differs between pass `--early` and the end, as a share of all
  still moving           share of cells whose answer changed during the last quarter of the passes
Compare a model trained with wrong digits against its blanks-only twin: if wrong-digit training teaches
the loop to revise, 'early errors fixed' should be clearly higher with 'early rights lost' no worse.
"""
import argparse
import torch
from sudoku.data import HARD_TEST_PATH, read_hard, to_tensor, difficulty_bands
from sudoku.gidd import _start, T_EPS
from sudoku.looped_gidd import LoopedGIDD

ap = argparse.ArgumentParser()
ap.add_argument('ckpt'); ap.add_argument('n', type=int, nargs='?', default=200)
ap.add_argument('--iters', type=int, default=64); ap.add_argument('--early', type=int, default=4)
ap.add_argument('--bands', default='DEFGH'); ap.add_argument('--batch', type=int, default=250)
a = ap.parse_args()

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
ck = torch.load(a.ckpt, map_location=device, weights_only=True)
state = ck['model'] if 'model' in ck else ck
gidd = isinstance(ck, dict) and ck.get('kind') == 'looped_gidd'
if gidd:
    model = LoopedGIDD().to(device)
elif any(key.startswith('initial_encoder') for key in state):
    from sudoku.looped import LoopedTransformer
    model = LoopedTransformer().to(device)
else:
    raise SystemExit("loop_trace.py needs a looped checkpoint")
model.load_state_dict(state); model.eval()
info = ck.get('info', 'looped transformer (external weights)') if 'model' in ck else 'looped transformer (external weights)'
print(f"{a.ckpt}: {info}; {a.iters} passes, nothing committed; 'early' = pass {a.early}")
marks = [k for k in (1, 2, 4, 8, 16, 32) if k < a.iters] + [a.iters]
print(f"{'':26s} solved   right after pass " + " ".join(f"{k:>5d}" for k in marks)
      + "   early errors fixed   early rights lost   changed its mind   still moving")

df = read_hard(HARD_TEST_PATH)
with torch.no_grad():
    for name, mask in difficulty_bands(df):
        if name[0] not in a.bands:
            continue
        d = df[mask].sample(n=min(a.n, int(mask.sum())), random_state=0)
        P, S = to_tensor(d.question.tolist()), to_tensor(d.answer.tolist())
        right_at = {k: 0 for k in marks}; unk_n = solved = 0
        e_wrong = e_fixed = e_right = e_lost = changed = moving = 0
        for i in range(0, len(P), a.batch):
            p, s = P[i:i + a.batch].to(device), S[i:i + a.batch].to(device)
            z, given = _start(p); unk = ~given
            if gidd:
                h, preds = model.start(z, torch.full((len(p),), 1 - T_EPS, device=device), given)
            else:
                h, preds = model.start(p)
            early = late_ref = None; late_moved = torch.zeros_like(unk)
            for k in range(1, a.iters + 1):
                h, logits, preds = model.step(h, preds)
                ans = logits.float().argmax(-1) + 1
                if k in right_at:
                    right_at[k] += ((ans == s) & unk).sum().item()
                if k == a.early:
                    early = ans
                if k >= a.iters - a.iters // 4:
                    if late_ref is not None:
                        late_moved |= (ans != late_ref) & unk
                    late_ref = ans
            unk_n += unk.sum().item(); solved += ((ans == s) | given).all(1).sum().item()
            ew, er = (early != s) & unk, (early == s) & unk
            e_wrong += ew.sum().item(); e_fixed += (ew & (ans == s)).sum().item()
            e_right += er.sum().item(); e_lost += (er & (ans != s)).sum().item()
            changed += ((ans != early) & unk).sum().item(); moving += late_moved.sum().item()
        pc = lambda x, y: f"{x / max(1, y) * 100:5.1f}%"
        print(f"{name:26s} {pc(solved, len(P))}                    " + " ".join(f"{right_at[k] / unk_n * 100:5.1f}" for k in marks)
              + f"   {pc(e_fixed, e_wrong):>18s}   {pc(e_lost, e_right):>17s}   {pc(changed, unk_n):>16s}   {pc(moving, unk_n):>12s}", flush=True)
