"""Train masked diffusion with a looped denoiser. Model and sampler: sudoku/looped_diffusion.py.

  python looped_diffusion_sudoku.py test
  python looped_diffusion_sudoku.py train --steps 100000 --batch 128 --lr 1e-3 --amp --full-memory --out checkpoints/loopdiff.pth
  python eval_by_difficulty.py checkpoints/loopdiff.pth 500 --iters 64               # decode by looping only
  python eval_by_difficulty.py checkpoints/loopdiff.pth 500 --iters 16 --commit 4    # decode by committing 4 cells per round

This is looped_sudoku.py's training loop with one change: before each step, every puzzle in the batch has a
random share of its blank cells filled in from the solution (mask_for_training), and the model is trained
to predict the cells still masked. Everything else (network, 16 supervised iterations, optimizer, schedule,
reverse curriculum, 16-bit option) is imported from the looped model unchanged, so a run here and a run of
looped_sudoku.py with the same flags differ only in that one line.
"""
import argparse, math, os, time
import torch
from sudoku.data import HARD_TRAIN_PATH, load_hard
from sudoku.looped import LoopedTransformer, looped_loss, backward_low_memory, autocast
from sudoku.looped_diffusion import mask_for_training, solve_commit
from looped_sudoku import get_device, amp_setting, LossScaler, lr_at, phase_pools


def cmd_train(a):
    device = get_device(); print(f"Using device: {device}")
    torch.manual_seed(a.seed); gen = torch.Generator().manual_seed(a.seed)
    P, S, R = load_hard(a.train_file, n=a.pool, min_rating=-1, with_rating=True)
    print(f"Training pool: {len(P):,} puzzles, all ratings   curriculum = {a.curriculum}   "
          f"{a.steps} steps x batch {a.batch} x {a.iters} iterations   masked diffusion, t in ({a.t_min}, 1]")
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    pools = phase_pools(R, a.steps, a.curriculum)
    model = LoopedTransformer().to(device)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=device, weights_only=True)['model'])
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95))
    amp = amp_setting(a, device); scaler = LossScaler(amp is not None); params = list(model.parameters())
    phase = -1; run_loss = run_cells = run_solved = run_masked = 0.0; start = time.time(); seen = 0
    for step in range(a.steps):
        while phase < 0 or step >= pools[phase][0]:
            phase += 1; print(f"— step {step}: pool is now {pools[phase][2]}", flush=True)
        idx = pools[phase][1][torch.randint(0, len(pools[phase][1]), (a.batch,), generator=gen)]
        p, s = P[idx], S[idx]
        z = mask_for_training(p, s, gen, a.t_min).to(device); s = s.to(device)      # the one change
        for g in opt.param_groups:
            g['lr'] = lr_at(step, a.steps, a.lr)
        model.train()
        if a.full_memory:
            with autocast(amp):
                all_logits, _ = model(z, iters=a.iters)
            loss = looped_loss(all_logits, z, s)
            opt.zero_grad(); (loss * scaler.scale).backward(); loss, last = loss.item(), all_logits[-1].float()
        else:
            loss, last = backward_low_memory(model, z, s, a.iters, None, gen, amp, scaler.scale)
        scaler.step(opt, params)
        if not math.isfinite(loss):
            raise SystemExit(f"loss is {loss} at step {step + 1}: stopping. Try a lower --lr.")
        with torch.no_grad():
            out = torch.where(z != 0, z, last.argmax(-1) + 1); masked = z == 0
            run_loss += loss; run_cells += ((out == s) & masked).sum().item() / masked.sum().item()
            run_solved += (out == s).all(1).float().mean().item(); run_masked += masked.float().sum(1).mean().item(); seen += 1
        if (step + 1) % a.log_every == 0 or step + 1 == a.steps:
            el = time.time() - start
            print(f"Step {step + 1}/{a.steps} — Loss: {run_loss / seen:.4f} — masked cells right: {run_cells / seen * 100:.1f}% — "
                  f"grids completed: {run_solved / seen * 100:.1f}% — avg masked {run_masked / seen:.0f} — "
                  f"{(step + 1) * a.batch / el:.0f} puzzles/s — {el / 60:.0f} min elapsed, "
                  f"~{el / (step + 1) * (a.steps - step - 1) / 60:.0f} min left"
                  + (f" — skipped {scaler.skipped}" if scaler.skipped else ""), flush=True)
            run_loss = run_cells = run_solved = run_masked = 0.0; seen = 0
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            torch.save({'model': model.state_dict(), 'step': step + 1,
                        'info': f"looped masked diffusion, {step + 1}/{a.steps} steps x batch {a.batch}, "
                                f"{a.iters} training iterations, curriculum {a.curriculum}"}, a.out)
    print(f"Saved {a.out}")


def cmd_test(a):
    torch.manual_seed(0)
    ok = lambda name, cond: print(f"{'PASS' if cond else 'FAIL'}  {name}")
    sol = torch.tensor([[(3 * (r % 3) + r // 3 + c) % 9 + 1 for r in range(9) for c in range(9)]] * 2000)
    puz = torch.where(torch.rand(2000, 81) < 0.3, sol, torch.zeros_like(sol))
    z = mask_for_training(puz, sol, torch.Generator().manual_seed(1))
    ok("givens are never masked or changed", bool((z[puz != 0] == puz[puz != 0]).all()))
    ok("every revealed cell holds the solution digit", bool((z[z != 0] == sol[z != 0]).all()))
    ok("every training grid has at least one masked cell", int((z == 0).sum(1).min()) >= 1)
    frac = ((z == 0).sum(1).float() / (puz == 0).sum(1).float()).mean().item()
    ok(f"on average half the blank cells stay masked ({frac:.3f})", abs(frac - 0.5) < 0.03)
    z1 = mask_for_training(puz, sol, torch.Generator().manual_seed(1), t_min=1.0)
    ok("t = 1 leaves the puzzle untouched (the looped model's training)", bool((z1 == puz).all()))
    m = LoopedTransformer().eval(); out = solve_commit(m, puz[:8], iters=2, k=6)
    ok("commit sampler fills every cell and keeps the givens", bool((out != 0).all()) and bool((out[puz[:8] != 0] == puz[:8][puz[:8] != 0]).all()))
    ident = sol[:4].clone(); ident[:, :3] = 0                    # three blanks: commit sampler must stop after one round of k=4
    ok("commit sampler handles fewer blanks than k", bool((solve_commit(m, ident, iters=1, k=4) != 0).all()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    tr = sub.add_parser('train')
    tr.add_argument('--steps', type=int, default=8000); tr.add_argument('--batch', type=int, default=64)
    tr.add_argument('--iters', type=int, default=16); tr.add_argument('--lr', type=float, default=2e-3)
    tr.add_argument('--pool', type=int, default=1000000); tr.add_argument('--curriculum', default='reverse', choices=['reverse', 'none'])
    tr.add_argument('--t-min', type=float, default=0.0, help='lowest noise level; 1.0 reproduces the plain looped model')
    tr.add_argument('--log-every', type=int, default=100); tr.add_argument('--save-every', type=int, default=500)
    tr.add_argument('--train-file', default=HARD_TRAIN_PATH); tr.add_argument('--seed', type=int, default=0); tr.add_argument('--init', default=None)
    tr.add_argument('--amp', action='store_true'); tr.add_argument('--full-memory', action='store_true')
    tr.add_argument('--out', required=True); tr.set_defaults(fn=cmd_train)
    te = sub.add_parser('test'); te.set_defaults(fn=cmd_test)
    a = ap.parse_args(); a.fn(a)
