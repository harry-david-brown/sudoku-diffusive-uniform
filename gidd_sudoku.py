"""Train and evaluate the GIDD (masked / hybrid diffusion) Sudoku model. Model code: sudoku/gidd.py.

  python gidd_sudoku.py test                                   # math checks against the paper, no data needed
  python gidd_sudoku.py train --pu 0.0 --out checkpoints/gidd_pu0_easy.pth
  python gidd_sudoku.py train --pu 0.0 --data hard --init checkpoints/gidd_pu0_easy.pth --lr 1e-4 --out checkpoints/gidd_pu0_easyhard.pth
  python gidd_sudoku.py eval  --ckpt checkpoints/gidd_pu0_easy.pth --sampler confidence --k 5

--pu is the share of random digits in the noise at t = 1/2; 0 is masked diffusion.
For results by difficulty band use eval_by_difficulty.py.
"""
import argparse, os, time
import torch
import torch.nn.functional as F
from sudoku.data import load_data, load_easy, load_hard, HARD_TEST_PATH
from sudoku.metrics import violations
from sudoku.gidd import (HybridSchedule, SudokuGIDD, gidd_loss, sample_t, make_noisy, sample_confidence,
                         sample_ancestral, self_correct, N_STATES, N_DIGITS)


def evaluate(model, sched, puzzles, solutions, sampler, k, steps, correct, device, batch=250):
    model.eval()
    solved = cells = unknowns = viol = 0
    start = time.time()
    for i in range(0, len(puzzles), batch):
        p, s = puzzles[i:i + batch].to(device), solutions[i:i + batch].to(device)
        if sampler in ('confidence', 'margin'):
            out = sample_confidence(model, p, k, margin=(sampler == 'margin'))
        else:
            out = sample_ancestral(model, p, sched, steps)
        if correct:
            out = self_correct(model, out, p != 0)
        unk = p == 0
        solved += (out == s).all(1).sum().item()
        cells += ((out == s) & unk).sum().item(); unknowns += unk.sum().item()
        viol += violations(out).sum().item()
    n = len(puzzles)
    tag = f"{sampler}" + (f" k={k}" if sampler in ('confidence', 'margin') else f" steps={steps}") + (" +self-correct" if correct else "")
    print(f"{tag:34s} puzzle acc {solved / n * 100:6.2f}%   cell acc (unknowns) {cells / unknowns * 100:6.2f}%   "
          f"avg violations {viol / n:5.2f}   ({time.time() - start:.0f}s)")


# ── Commands ───────────────────────────────────────────────────────────────────
def get_device():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')


def cmd_train(a):
    device = get_device(); print(f"Using device: {device}")
    sched = HybridSchedule(a.pu)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    puzzles, solutions = load_data(a.data, a.n, min_rating=a.min_rating)
    print(f"Training puzzles: {len(puzzles):,} ({a.data}{'' if a.data == 'easy' else f', rating > {a.min_rating}'})   p_u = {a.pu}   weighting = {a.weighting}")
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(puzzles, solutions),
                                         batch_size=a.batch, shuffle=True, drop_last=True)
    model = SudokuGIDD().to(device)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=device)['model'])
        print(f"Initialised from {a.init}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    for epoch in range(a.epochs):
        model.train(); tot = dict(loss=0.0, elbo=0.0, kl=0.0, isd=0.0); start = time.time()
        for bp, bs in loader:
            bp, bs = bp.to(device), bs.to(device)
            t = sample_t(bp.shape[0], device)
            z_tok, z_state, x_state, unknown = make_noisy(bp, bs, t, sched)
            loss, elbo, kl, isd = gidd_loss(model(z_tok, t, ~unknown), x_state, z_state, t, unknown, sched, a.weighting)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot['loss'] += loss.item(); tot['elbo'] += elbo.item(); tot['kl'] += kl.item(); tot['isd'] += isd.item()
        n = len(loader)
        print(f"Epoch {epoch + 1}/{a.epochs} — Loss: {tot['loss'] / n:.4f} — KL: {tot['kl'] / n:.4f} — "
              f"IS: {tot['isd'] / n:.4f} — ELBO: {tot['elbo'] / n:.3f} — {time.time() - start:.0f}s", flush=True)
        torch.save({'model': model.state_dict(), 'pu': a.pu, 'epoch': epoch + 1, 'data': a.data}, a.out)
    print(f"Saved {a.out}")


def cmd_eval(a):
    device = get_device()
    ck = torch.load(a.ckpt, map_location=device)
    model = SudokuGIDD().to(device); model.load_state_dict(ck['model'])
    sched = HybridSchedule(ck['pu'])
    torch.manual_seed(a.seed)
    if a.data == 'easy':                    # rows the model never trained on
        puzzles, solutions = load_easy(n=a.n, skip=a.skip)
    else:
        puzzles, solutions = load_hard(HARD_TEST_PATH, n=None)
        idx = torch.randperm(len(puzzles))[:a.n]
        puzzles, solutions = puzzles[idx], solutions[idx]
    print(f"{a.ckpt}: p_u = {ck['pu']}, epoch {ck['epoch']}, trained on {ck['data']}; evaluating {len(puzzles)} {a.data} puzzles")
    ks = [a.k] if a.k else [1, 5, 10, 20, 81]
    if a.sampler in ('confidence', 'all'):
        for k in ks:
            for correct in ([False, True] if a.correct else [False]):
                evaluate(model, sched, puzzles, solutions, 'confidence', k, None, correct, device)
    if a.sampler == 'margin':
        for k in ks:
            for correct in ([False, True] if a.correct else [False]):
                evaluate(model, sched, puzzles, solutions, 'margin', k, None, correct, device)
    if a.sampler in ('ancestral', 'all'):
        for correct in ([False, True] if a.correct else [False]):
            evaluate(model, sched, puzzles, solutions, 'ancestral', None, a.steps, correct, device)


def cmd_test(a):
    """Checks of the math against the paper. No data or checkpoint needed."""
    torch.manual_seed(0)
    ok = lambda name, cond: print(f"{'PASS' if cond else 'FAIL'}  {name}")
    for pu in (0.0, 0.1, 0.2):
        s = HybridSchedule(pu); t = torch.tensor([0.05, 0.3, 0.5, 0.9])
        x_state = torch.randint(0, 9, (4, 81)); x = F.one_hot(x_state, N_STATES).float()
        q = s.probs_at_t(x, t)
        ok(f"p_u={pu}: q_t sums to 1", torch.allclose(q.sum(-1), torch.ones(4, 81), atol=1e-6))
        if pu > 0:
            share = s.alpha_betapi(t)[1][2, :9].sum()
            ok(f"p_u={pu}: random-digit share at t=1/2 is p_u ({share:.4f})", abs(share - pu) < 1e-4)
        # closed-form ELBO weight vs Eq. 13 by finite differences: w = [d/dt(beta*pi) - (alpha'/alpha) beta*pi](z) / q_t(z|x)
        h = 1e-5; t64 = t.double()
        a0, b0 = s.alpha_betapi(t64); a1, b1 = s.alpha_betapi(t64 + h); a_1, b_1 = s.alpha_betapi(t64 - h)
        rate = (b1 - b_1) / (2 * h) - ((a1 - a_1) / (2 * h) / a0) * b0                 # [4,10]
        for name, zs in (('mask', torch.full((4, 81), 9)), ('clean', x_state), ('wrong digit', (x_state + 1) % 9)):
            if pu == 0 and name != 'mask':
                continue
            w_fd = rate[:, None, :].expand(-1, 81, -1).gather(-1, zs.unsqueeze(-1)).squeeze(-1) / \
                   s.probs_at_t(x.double(), t64).gather(-1, zs.unsqueeze(-1)).squeeze(-1)
            w_cf, _ = s.weights(t, zs, x_state, 'elbo')
            ok(f"p_u={pu}: ELBO weight on {name} matches Eq. 13", torch.allclose(w_fd.float(), w_cf, rtol=2e-3))
        # loss is zero at the truth, positive otherwise
        z_state = s.sample_zt(x_state, t); unk = torch.ones(4, 81, dtype=torch.bool)
        perfect = torch.full((4, 81, 9), -30.0).scatter(-1, x_state.unsqueeze(-1), 30.0)
        l0 = gidd_loss(perfect, x_state, z_state, t, unk, s)[1]; l1 = gidd_loss(torch.zeros(4, 81, 9), x_state, z_state, t, unk, s)[1]
        ok(f"p_u={pu}: ELBO is 0 at the true distribution ({l0:.2e}) and positive otherwise ({l1:.3f})", l0 < 1e-4 and l1 > 0.01)
    # Corollary 3.8: p_u = 0 reduces to the masked-diffusion ELBO, CE/t on masked cells
    s = HybridSchedule(0.0); t = torch.tensor([0.2, 0.7]); x_state = torch.randint(0, 9, (2, 81))
    z_state = s.sample_zt(x_state, t); logits = torch.randn(2, 81, 9); unk = torch.ones(2, 81, dtype=torch.bool)
    elbo = gidd_loss(logits, x_state, z_state, t, unk, s)[1]
    ce = F.cross_entropy(logits.reshape(-1, 9), x_state.reshape(-1), reduction='none').reshape(2, 81)
    mdm = ((z_state == 9).float() * ce / t[:, None]).mean()
    ok(f"p_u=0: GIDD ELBO equals masked-diffusion ELBO ({elbo:.4f} vs {mdm:.4f})", abs(elbo - mdm) / mdm < 1e-3)
    # samplers recover the solution when the network is replaced by an oracle
    sol = torch.tensor([[(3 * (r % 3) + r // 3 + c) % 9 + 1 for r in range(9) for c in range(9)]] * 8)
    puz = torch.where(torch.rand(8, 81) < 0.4, sol, torch.zeros_like(sol))
    oracle = lambda z, t, given: torch.full((8, 81, 9), -30.0).scatter(-1, (sol - 1).unsqueeze(-1), 30.0)
    ok("valid grid has 0 violations; one duplicated cell breaks its row, column and box", violations(sol).sum().item() == 0 and violations(torch.where(torch.arange(81) == 0, sol[:1, 1:2], sol[:1])).item() == 3)
    ok("confidence sampler recovers the solution with an oracle", (sample_confidence(oracle, puz, 5) == sol).all().item())
    for pu in (0.0, 0.2):
        ok(f"p_u={pu}: ancestral sampler recovers the solution with an oracle",
           (sample_ancestral(oracle, puz, HybridSchedule(pu), 32) == sol).all().item())
    wrong = torch.where((puz == 0) & (torch.rand(8, 81) < 0.15), sol % 9 + 1, sol)
    ok("self-correction repairs wrong digits with an oracle", (self_correct(oracle, wrong, puz != 0) == sol).all().item())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    tr = sub.add_parser('train')
    tr.add_argument('--pu', type=float, default=0.0); tr.add_argument('--data', default='easy', choices=['easy', 'hard'])
    tr.add_argument('--n', type=int, default=500000); tr.add_argument('--epochs', type=int, default=20)
    tr.add_argument('--batch', type=int, default=64); tr.add_argument('--lr', type=float, default=1e-3)
    tr.add_argument('--wd', type=float, default=0.02); tr.add_argument('--weighting', default='dynamic', choices=['dynamic', 'clip', 'elbo'])
    tr.add_argument('--min-rating', type=float, default=50, help='hard data only: keep puzzles rated above this; -1 keeps all'); tr.add_argument('--init', default=None); tr.add_argument('--out', required=True); tr.set_defaults(fn=cmd_train)
    ev = sub.add_parser('eval')
    ev.add_argument('--ckpt', required=True); ev.add_argument('--data', default='easy', choices=['easy', 'hard'])
    ev.add_argument('--n', type=int, default=1000); ev.add_argument('--skip', type=int, default=900000)
    ev.add_argument('--sampler', default='all', choices=['confidence', 'margin', 'ancestral', 'all'])
    ev.add_argument('--k', type=int, default=None); ev.add_argument('--steps', type=int, default=64)
    ev.add_argument('--correct', action='store_true'); ev.add_argument('--seed', type=int, default=0); ev.set_defaults(fn=cmd_eval)
    te = sub.add_parser('test'); te.set_defaults(fn=cmd_test)
    a = ap.parse_args(); a.fn(a)
