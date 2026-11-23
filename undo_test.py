"""Measure undo: can a model trained with uniform noise take back a wrong digit?

  python undo_test.py checkpoints/loopgidd_pu02_desktop.pth 200
  python undo_test.py checkpoints/gidd_pu02_hard.pth 200 --bands EFGH

Works on looped GIDD checkpoints and on the single-pass GIDD checkpoints (gidd_sudoku.py).

Part 1, random noise by noise level. Test puzzles are noised exactly as in training (masks plus a share of
random digits) at fixed noise levels t. Unknown cells then fall into three kinds, scored separately:
  masked        -> 'filled right': the model predicts the solution digit
  wrong digit   -> 'flagged': the model predicts something other than the wrong digit it was shown
                   'fixed': the model predicts the solution digit
  right digit   -> 'kept': the model leaves it alone
Low t is a nearly finished grid with a few bad cells; high t is a nearly empty one.

Part 2, the model's own errors. Each puzzle is first solved by committing cells (confidence sampler), which
cannot revise. The cells it got wrong are its own errors. Then, on that finished grid:
  one look     the model is shown its own grid once: how many of its errors does it flag, how many does it
               fix, and how many correct cells does it break?
  control      the same number of random wrong digits placed in the true solution, same one look
  conflicts    rows, columns and boxes with a repeated digit, in the model's grid and in the control grid
  self-correct the paper's self-correction loop is run on the model's grid: errors repaired, correct cells
               broken, puzzles solved before and after
"""
import argparse
import torch
from sudoku.data import HARD_TEST_PATH, read_hard, to_tensor, difficulty_bands
from sudoku.metrics import violations
from sudoku.gidd import SudokuGIDD, HybridSchedule, make_noisy, sample_confidence, self_correct, MASK

ap = argparse.ArgumentParser()
ap.add_argument('ckpt'); ap.add_argument('n', type=int, nargs='?', default=200)
ap.add_argument('--iters', type=int, default=16, help='looped GIDD: passes per denoising call')
ap.add_argument('--k', type=int, default=4, help='cells committed per call when solving')
ap.add_argument('--pu', type=float, default=0.2, help='share of random digits in the Part 1 test noise')
ap.add_argument('--bands', default='DEFGH'); ap.add_argument('--batch', type=int, default=250)
ap.add_argument('--t0', type=float, default=0.01, help='noise level the model is told when shown a finished grid')
a = ap.parse_args()

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
ck = torch.load(a.ckpt, map_location=device, weights_only=True)
if ck.get('kind') == 'looped_gidd':
    from sudoku.looped_gidd import LoopedGIDD, Denoiser
    model = LoopedGIDD().to(device); model.load_state_dict(ck['model']); model.eval()
    den = Denoiser(model, a.iters); what = f"{ck['info']}; {a.iters} passes per call"
else:
    model = SudokuGIDD().to(device); model.load_state_dict(ck['model']); model.eval()
    den = model; what = f"single-pass GIDD p_u={ck['pu']}, trained on {ck['data']}"
print(f"{a.ckpt}: {what}")
torch.manual_seed(0)
sched = HybridSchedule(a.pu)
pct = lambda num, den_: f"{num / den_ * 100:5.1f}%" if den_ else "   n/a"

df = read_hard(HARD_TEST_PATH)
bands = []
for name, mask in difficulty_bands(df):
    if name[0] in a.bands:
        d = df[mask].sample(n=min(a.n, int(mask.sum())), random_state=0)
        bands.append((name, to_tensor(d.question.tolist()), to_tensor(d.answer.tolist())))


@torch.no_grad()
def batches(P, S):
    for i in range(0, len(P), a.batch):
        yield P[i:i + a.batch].to(device), S[i:i + a.batch].to(device)


# ── Part 1 ─────────────────────────────────────────────────────────────────────
print(f"\nPart 1: random noise (p_u = {a.pu}), bands {a.bands} pooled, {sum(len(b[1]) for b in bands)} puzzles")
print("  t     masked: filled right     wrong digits: flagged   fixed     right digits: kept     (cells: masked / wrong / right)")
with torch.no_grad():
    for tv in (0.1, 0.3, 0.5, 0.7, 0.9):
        c = dict(m=0, mr=0, w=0, wf=0, wx=0, r=0, rk=0)
        for _, P, S in bands:
            for p, s in batches(P, S):
                t = torch.full((len(p),), tv, device=device)
                z_tok, z_state, x_state, unknown = make_noisy(p, s, t, sched)
                pred = den(z_tok, t, ~unknown).argmax(-1)
                masked = unknown & (z_state == 9); wrong = unknown & (z_state != 9) & (z_state != x_state)
                right = unknown & (z_state == x_state)
                c['m'] += masked.sum().item(); c['mr'] += (masked & (pred == x_state)).sum().item()
                c['w'] += wrong.sum().item(); c['wf'] += (wrong & (pred != z_state)).sum().item()
                c['wx'] += (wrong & (pred == x_state)).sum().item()
                c['r'] += right.sum().item(); c['rk'] += (right & (pred == x_state)).sum().item()
        print(f"  {tv:.1f}   {pct(c['mr'], c['m']):>20s}   {pct(c['wf'], c['w']):>29s}  {pct(c['wx'], c['w'])}   {pct(c['rk'], c['r']):>18s}     "
              f"({c['m']} / {c['w']} / {c['r']})")

# ── Part 2 ─────────────────────────────────────────────────────────────────────
print(f"\nPart 2: the model's own errors (solved by committing {a.k} cells per call), per band")
print("'conflicts' = rows, columns and boxes (of 27) containing a repeated digit, per failed puzzle: how visible the errors are.")
print("                             solved   own errors  conflicts   one look at own grid             control: random errors        after self-correction")
print("                                      per failed  own grid    flagged  fixed  correct broken   conflicts  flagged  fixed     repaired  broken  solved")
with torch.no_grad():
    for name, P, S in bands:
        c = dict(n=0, solved=0, failed=0, err=0, flag=0, fix=0, ok=0, brk=0, rerr=0, rflag=0, rfix=0, rep=0, brk2=0, solved2=0, viol=0, rviol=0)
        for p, s in batches(P, S):
            given = p != 0; unknown = ~given
            g = sample_confidence(den, p, a.k)
            err = unknown & (g != s); okc = unknown & (g == s)
            t = torch.full((len(p),), a.t0, device=device)
            pred = den(g, t, given).argmax(-1) + 1
            c['n'] += len(p); c['solved'] += (~err.any(1)).sum().item(); c['failed'] += err.any(1).sum().item()
            c['err'] += err.sum().item(); c['flag'] += (err & (pred != g)).sum().item(); c['fix'] += (err & (pred == s)).sum().item()
            c['ok'] += okc.sum().item(); c['brk'] += (okc & (pred != s)).sum().item()
            failed = err.any(1); c['viol'] += violations(g)[failed].sum().item()
            # control: the same number of random wrong digits, placed in the true solution
            n_err = err.sum(1, keepdim=True)
            score = torch.rand(p.shape, device=device).masked_fill(given, 2.0)
            rank = score.argsort(1).argsort(1)
            pick = (rank < n_err) & unknown
            r = torch.where(pick, (s - 1 + torch.randint(1, 9, p.shape, device=device)) % 9 + 1, s)
            rpred = den(r, t, given).argmax(-1) + 1
            c['rviol'] += violations(r)[failed].sum().item()
            c['rerr'] += pick.sum().item(); c['rflag'] += (pick & (rpred != r)).sum().item(); c['rfix'] += (pick & (rpred == s)).sum().item()
            # the paper's self-correction loop on the model's own grid
            g2 = self_correct(den, g.clone(), given)
            c['rep'] += (err & (g2 == s)).sum().item(); c['brk2'] += (okc & (g2 != s)).sum().item()
            c['solved2'] += (g2 == s).all(1).sum().item()
        nf = max(1, c['failed'])
        print(f"{name:26s} {pct(c['solved'], c['n'])}   {c['err'] / nf:8.1f}  {c['viol'] / nf:8.1f}    "
              f"{pct(c['flag'], c['err'])} {pct(c['fix'], c['err'])}  {pct(c['brk'], c['ok'])}        "
              f"{c['rviol'] / nf:8.1f}   {pct(c['rflag'], c['rerr'])} {pct(c['rfix'], c['rerr'])}       "
              f"{pct(c['rep'], c['err'])}  {pct(c['brk2'], c['ok'])}  {pct(c['solved2'], c['n'])}", flush=True)
