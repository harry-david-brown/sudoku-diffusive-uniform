"""Train looped GIDD: hybrid (mask + uniform noise) diffusion with a looped denoiser. Model: sudoku/looped_gidd.py.

  python looped_gidd_sudoku.py test
  python looped_gidd_sudoku.py train --pu 0.2 --steps 100000 --batch 128 --lr 1e-3 --amp --out checkpoints/loopgidd_pu02.pth
  python eval_by_difficulty.py checkpoints/loopgidd_pu02.pth 500 --sampler loop --iters 64
  python eval_by_difficulty.py checkpoints/loopgidd_pu02.pth 500 --sampler confidence --iters 16 --commit 4
  python eval_by_difficulty.py checkpoints/loopgidd_pu02.pth 500 --sampler ancestral+correct --iters 16

--pu is the share of random digits in the noise at t = 1/2; --pu 0 is masked diffusion.
Optimizer, schedule, curriculum and 16-bit option are the looped model's (looped_sudoku.py), so runs are
comparable with it at equal --steps and --batch. Training keeps all passes in memory (about 130 MB per
puzzle in 32-bit, a quarter of that with --amp), so size the batch accordingly.
"""
import argparse, math, os, time
import torch
from sudoku.data import HARD_TRAIN_PATH, load_hard
from sudoku.gidd import HybridSchedule, sample_t, make_noisy
from sudoku.looped import autocast
from sudoku.looped_gidd import LoopedGIDD, looped_gidd_loss, solve
from looped_sudoku import get_device, amp_setting, LossScaler, lr_at, phase_pools


def cmd_train(a):
    device = get_device(); print(f"Using device: {device}")
    torch.manual_seed(a.seed); gen = torch.Generator().manual_seed(a.seed)
    P, S, R = load_hard(a.train_file, n=a.pool, min_rating=-1, with_rating=True)
    print(f"Training pool: {len(P):,} puzzles, all ratings   curriculum = {a.curriculum}   "
          f"{a.steps} steps x batch {a.batch} x {a.iters} passes   p_u = {a.pu}, weighting = {a.weighting}")
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    pools = phase_pools(R, a.steps, a.curriculum)
    sched = HybridSchedule(a.pu)
    model = LoopedGIDD().to(device)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=device, weights_only=True)['model'])
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95))
    amp = amp_setting(a, device); scaler = LossScaler(amp is not None); params = list(model.parameters())
    phase = -1; run_loss = run_elbo = run_mask = run_fix = run_keep = 0.0; start = time.time(); seen = 0
    for step in range(a.steps):
        while phase < 0 or step >= pools[phase][0]:
            phase += 1; print(f"— step {step}: pool is now {pools[phase][2]}", flush=True)
        idx = pools[phase][1][torch.randint(0, len(pools[phase][1]), (a.batch,), generator=gen)]
        p, s = P[idx].to(device), S[idx].to(device)
        t = sample_t(a.batch, device)
        z_tok, z_state, x_state, unknown = make_noisy(p, s, t, sched)
        for g in opt.param_groups:
            g['lr'] = lr_at(step, a.steps, a.lr)
        model.train()
        with autocast(amp):
            all_logits, _ = model(z_tok, t, ~unknown, a.iters)
        loss, elbo = looped_gidd_loss(all_logits, x_state, z_state, t, unknown, sched, a.weighting)
        opt.zero_grad(); (loss * scaler.scale).backward(); scaler.step(opt, params)
        loss = loss.item()
        if not math.isfinite(loss):
            raise SystemExit(f"loss is {loss} at step {step + 1}: stopping. Try a lower --lr.")
        with torch.no_grad():
            # three kinds of unknown cell in the noisy grid: masked, holding a wrong digit, holding the right digit
            right = all_logits[-1].argmax(-1) == x_state
            masked = unknown & (z_state == 9); wrong = unknown & (z_state != 9) & (z_state != x_state)
            kept = unknown & (z_state == x_state)
            share = lambda sel: (right & sel).sum().item() / max(1, sel.sum().item())
            run_loss += loss; run_elbo += elbo.item(); seen += 1
            run_mask += share(masked); run_fix += share(wrong); run_keep += share(kept)
        if (step + 1) % a.log_every == 0 or step + 1 == a.steps:
            el = time.time() - start
            print(f"Step {step + 1}/{a.steps} — Loss: {run_loss / seen:.4f} — ELBO: {run_elbo / seen:.4f} — "
                  f"masked filled right: {run_mask / seen * 100:.1f}% — wrong digits fixed: {run_fix / seen * 100:.1f}% — "
                  f"right digits kept: {run_keep / seen * 100:.1f}% — {(step + 1) * a.batch / el:.0f} puzzles/s — "
                  f"{el / 60:.0f} min elapsed, ~{el / (step + 1) * (a.steps - step - 1) / 60:.0f} min left"
                  + (f" — skipped {scaler.skipped}" if scaler.skipped else ""), flush=True)
            run_loss = run_elbo = run_mask = run_fix = run_keep = 0.0; seen = 0
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            torch.save({'model': model.state_dict(), 'step': step + 1, 'kind': 'looped_gidd', 'pu': a.pu,
                        'info': f"looped GIDD p_u={a.pu}, {step + 1}/{a.steps} steps x batch {a.batch}, "
                                f"{a.iters} training passes, curriculum {a.curriculum}"}, a.out)
    print(f"Saved {a.out}")


def cmd_test(a):
    torch.manual_seed(0)
    ok = lambda name, cond: print(f"{'PASS' if cond else 'FAIL'}  {name}")
    sol = torch.tensor([[(3 * (r % 3) + r // 3 + c) % 9 + 1 for r in range(9) for c in range(9)]] * 4000)
    puz = torch.where(torch.rand(4000, 81) < 0.3, sol, torch.zeros_like(sol))
    sched = HybridSchedule(0.2); t = torch.full((4000,), 0.5)
    z_tok, z_state, x_state, unknown = make_noisy(puz, sol, t, sched)
    ok("givens are never noised", bool((z_tok[~unknown] == puz[~unknown]).all()))
    share = ((z_tok != 10) & unknown & (z_tok != sol)).sum().item() / unknown.sum().item()
    ok(f"at t = 1/2 with p_u = 0.2, a fifth of unknown cells are random digits, 8 in 9 of them wrong ({share:.3f}, expected 0.178)", abs(share - 0.2 * 8 / 9) < 0.01)
    m = LoopedGIDD(); print(f"      parameters: {sum(p.numel() for p in m.parameters()):,} (plain looped model: 796,937)")
    b = slice(0, 6); tt = torch.rand(6)
    zt, zs, xs, unk = make_noisy(puz[b], sol[b], tt, sched)
    m.eval(); lg5, _ = m(zt, tt, ~unk, 5); lg2, _ = m(zt, tt, ~unk, 2, state=m(zt, tt, ~unk, 3)[1])
    ok("5 passes equal 3 passes then 2 more from the saved state", torch.allclose(lg5[-1], lg2[-1], atol=1e-5))
    m.train(); lg, _ = m(zt, tt, ~unk, 4); loss, elbo = looped_gidd_loss(lg, xs, zs, tt, unk, sched)
    ok(f"loss is finite at initialisation ({loss.item():.2f})", math.isfinite(loss.item()))
    loss.backward(); ok("every parameter receives a gradient", all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
    m.eval()
    for sampler in ('loop', 'confidence', 'ancestral', 'ancestral+correct'):
        out = solve(m, puz[b], sched, sampler, iters=2, k=8, steps=4)
        ok(f"sampler '{sampler}' fills every cell with a digit and keeps the givens",
           bool(((out >= 1) & (out <= 9)).all()) and bool((out[puz[b] != 0] == puz[b][puz[b] != 0]).all()))
    m0 = LoopedGIDD(); s0 = HybridSchedule(0.0); z0 = make_noisy(puz[:2000], sol[:2000], torch.full((2000,), 0.5), s0)[0]
    ok("p_u = 0 produces no wrong digits (masked diffusion)", bool(((z0 == 10) | (z0 == sol[:2000])).all()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    tr = sub.add_parser('train')
    tr.add_argument('--pu', type=float, default=0.2); tr.add_argument('--weighting', default='dynamic', choices=['dynamic', 'clip', 'elbo'])
    tr.add_argument('--steps', type=int, default=8000); tr.add_argument('--batch', type=int, default=64)
    tr.add_argument('--iters', type=int, default=16); tr.add_argument('--lr', type=float, default=2e-3)
    tr.add_argument('--pool', type=int, default=1000000); tr.add_argument('--curriculum', default='reverse', choices=['reverse', 'none'])
    tr.add_argument('--log-every', type=int, default=100); tr.add_argument('--save-every', type=int, default=500)
    tr.add_argument('--train-file', default=HARD_TRAIN_PATH); tr.add_argument('--seed', type=int, default=0); tr.add_argument('--init', default=None)
    tr.add_argument('--amp', action='store_true'); tr.add_argument('--out', required=True); tr.set_defaults(fn=cmd_train)
    te = sub.add_parser('test'); te.set_defaults(fn=cmd_test)
    a = ap.parse_args(); a.fn(a)
