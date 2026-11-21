"""Train and evaluate the looped transformer (a port of Sotaku). Model code: sudoku/looped.py.

  python looped_sudoku.py test                                  # checks the port, no data needed
  python looped_sudoku.py bench                                 # 20 training steps, prints puzzles/second
  python looped_sudoku.py train --steps 8000 --batch 128 --out checkpoints/looped.pth
  python eval_by_difficulty.py checkpoints/looped.pth 500 --iters 128

Training follows Sotaku's recipe: every step runs the network `--iters` times (default 16) and averages
the loss over all of them; batches come from a reverse curriculum (hardest puzzles first, then
progressively easier ones are added); AdamW, warmup then cosine decay.
One Sotaku ingredient is off by default: starting 20% of batches from a later iteration (--late-prob).
"""
import argparse, math, os, time
import torch
from sudoku.data import HARD_TRAIN_PATH, load_hard
from sudoku.looped import LoopedTransformer, looped_loss, solve_looped, backward_low_memory

# Reverse curriculum: (fraction of training, minimum rating in the pool). Sotaku's 50K reference recipe.
PHASES = [(0.2, 51), (0.2, 11), (0.2, 1), (0.4, 0)]


def get_device():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')


def lr_at(step, total, peak, warmup_frac=0.028, min_ratio=0.01):
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def phase_pools(ratings, total_steps, curriculum):
    """-> list of (last step of phase, indices of eligible puzzles, label)"""
    phases = PHASES if curriculum == 'reverse' else [(1.0, 0)]
    pools, end = [], 0.0
    for frac, min_rating in phases:
        end += frac
        idx = torch.nonzero(ratings >= min_rating).squeeze(1)
        pools.append((round(end * total_steps), idx, f"rating {min_rating}+ ({len(idx):,} puzzles)"))
    return pools


def cmd_train(a):
    device = get_device(); print(f"Using device: {device}")
    torch.manual_seed(a.seed); gen = torch.Generator().manual_seed(a.seed)      # batches and dropout seeds
    P, S, R = load_hard(a.train_file, n=a.pool, min_rating=-1, with_rating=True)
    print(f"Training pool: {len(P):,} puzzles, all ratings   curriculum = {a.curriculum}   "
          f"{a.steps} steps x batch {a.batch} x {a.iters} iterations")
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    pools = phase_pools(R, a.steps, a.curriculum)
    model = LoopedTransformer().to(device)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=device, weights_only=True)['model'])
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95))
    horizons = [int(h) for h in a.late_horizons.split(',')]
    phase = -1; run_loss = run_cells = run_solved = 0.0; start = time.time(); seen = 0
    for step in range(a.steps):
        while phase < 0 or step >= pools[phase][0]:
            phase += 1; print(f"— step {step}: pool is now {pools[phase][2]}", flush=True)
        idx = pools[phase][1][torch.randint(0, len(pools[phase][1]), (a.batch,), generator=gen)]
        p, s = P[idx].to(device), S[idx].to(device)
        for g in opt.param_groups:
            g['lr'] = lr_at(step, a.steps, a.lr)
        model.train()
        state = None
        if a.late_prob > 0 and torch.rand(1, generator=gen).item() < a.late_prob:      # start from a later iteration
            with torch.no_grad():
                _, state = model(p, iters=horizons[torch.randint(0, len(horizons), (1,), generator=gen).item()])
        if a.full_memory:
            all_logits, _ = model(p, iters=a.iters, state=state)
            loss = looped_loss(all_logits, p, s)
            opt.zero_grad(); loss.backward(); loss, last = loss.item(), all_logits[-1]
        else:
            loss, last = backward_low_memory(model, p, s, a.iters, state, gen)
        opt.step()
        with torch.no_grad():
            out = torch.where(p != 0, p, last.argmax(-1) + 1); blank = p == 0
            run_loss += loss; run_cells += ((out == s) & blank).sum().item() / blank.sum().item()
            run_solved += (out == s).all(1).float().mean().item(); seen += 1
        if (step + 1) % a.log_every == 0 or step + 1 == a.steps:
            el = time.time() - start
            print(f"Step {step + 1}/{a.steps} — Loss: {run_loss / seen:.4f} — train cells: {run_cells / seen * 100:.1f}% — "
                  f"train solved: {run_solved / seen * 100:.1f}% — {(step + 1) * a.batch / el:.0f} puzzles/s — "
                  f"{el / 60:.0f} min elapsed, ~{el / (step + 1) * (a.steps - step - 1) / 60:.0f} min left", flush=True)
            run_loss = run_cells = run_solved = 0.0; seen = 0
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            torch.save({'model': model.state_dict(), 'step': step + 1,
                        'info': f"looped transformer, {step + 1}/{a.steps} steps x batch {a.batch}, "
                                f"{a.iters} training iterations, curriculum {a.curriculum}"}, a.out)
    print(f"Saved {a.out}")


def cmd_bench(a):
    """Time a few training steps at several batch sizes, smallest first, so a run can be sized.
    Each line is printed before its step starts: if one stalls for more than a couple of minutes the
    machine is out of memory and swapping; press Ctrl-C and use the last batch size that finished."""
    device = get_device(); print(f"Using device: {device}", flush=True)
    sizes = [a.batch] if a.batch else [32, 64, 128, 256]
    for batch in sizes:
        model = LoopedTransformer().to(device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sol = torch.tensor([[(3 * (r % 3) + r // 3 + c) % 9 + 1 for r in range(9) for c in range(9)]] * batch)
        puz = torch.where(torch.rand(batch, 81) < 0.3, sol, torch.zeros_like(sol)).to(device); sol = sol.to(device)
        times = []
        for i in range(a.steps + 1):
            print(f"  batch {batch}: step {i + 1}/{a.steps + 1} ...", end='', flush=True); t0 = time.time()
            if a.full_memory:
                loss = looped_loss(model(puz, iters=a.iters)[0], puz, sol); opt.zero_grad(); loss.backward(); loss.item()
            else:
                backward_low_memory(model, puz, sol, a.iters)
            opt.step()
            dt = time.time() - t0; print(f" {dt:.1f}s", flush=True)
            if i > 0:
                times.append(dt)
        rate = batch / (sum(times) / len(times))
        mem = f", {torch.mps.driver_allocated_memory() / 1e9:.1f} GB allocated" if device.type == 'mps' else ''
        print(f"batch {batch}, {a.iters} iterations: {rate:.0f} puzzles/s = {rate * 3600 / 1e6:.2f}M presentations per hour{mem}", flush=True)
        del model, opt


def cmd_test(a):
    torch.manual_seed(0)
    ok = lambda name, cond: print(f"{'PASS' if cond else 'FAIL'}  {name}")
    m = LoopedTransformer(); n = sum(p.numel() for p in m.parameters())
    ok(f"parameter count matches Sotaku's published 796,937 ({n:,})", n == 796937)
    sol = torch.tensor([[(3 * (r % 3) + r // 3 + c) % 9 + 1 for r in range(9) for c in range(9)]] * 4)
    puz = torch.where(torch.rand(4, 81) < 0.4, sol, torch.zeros_like(sol))
    out = solve_looped(m.eval(), puz, 3)
    ok("given cells are never changed", bool((out[puz != 0] == puz[puz != 0]).all()))
    logits, state = m(puz, iters=5); logits2, _ = m(puz, iters=2, state=m(puz, iters=3)[1])
    ok("5 iterations equal 3 iterations then 2 more from the saved state", torch.allclose(logits[-1], logits2[-1], atol=1e-5))
    ok("loss is finite and near ln 9 at initialisation", abs(looped_loss(logits, puz, sol).item() - math.log(9)) < 1.0)
    # the low-memory backward pass must give the same gradients as the ordinary one, with dropout on,
    # on the device training will use
    device = get_device(); m = LoopedTransformer().to(device).train(); puz, sol = puz.to(device), sol.to(device)
    gen = torch.Generator().manual_seed(1); seeds = torch.randint(0, 2 ** 31 - 1, (4,), generator=gen).tolist()
    m.zero_grad(); looped_loss(m(puz, iters=4, seeds=seeds)[0], puz, sol).backward()
    plain = torch.cat([p.grad.flatten() for p in m.parameters()]).cpu()
    loss, _ = backward_low_memory(m, puz, sol, 4, generator=torch.Generator().manual_seed(1))
    low = torch.cat([p.grad.flatten() for p in m.parameters()]).cpu()
    diff = (plain - low).abs().max().item() / plain.abs().max().item()
    ok(f"low-memory backward pass gives the same gradients on {device.type} (relative difference {diff:.1e})", diff < 1e-3)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    tr = sub.add_parser('train')
    tr.add_argument('--steps', type=int, default=8000); tr.add_argument('--batch', type=int, default=64)
    tr.add_argument('--iters', type=int, default=16); tr.add_argument('--lr', type=float, default=2e-3)
    tr.add_argument('--pool', type=int, default=1000000, help='training puzzles drawn from the hard training file')
    tr.add_argument('--curriculum', default='reverse', choices=['reverse', 'none'])
    tr.add_argument('--late-prob', type=float, default=0.0); tr.add_argument('--late-horizons', default='32,64,128,256,512')
    tr.add_argument('--log-every', type=int, default=100); tr.add_argument('--save-every', type=int, default=500)
    tr.add_argument('--train-file', default=HARD_TRAIN_PATH); tr.add_argument('--seed', type=int, default=0); tr.add_argument('--init', default=None)
    tr.add_argument('--out', required=True); tr.set_defaults(fn=cmd_train)
    tr.add_argument('--full-memory', action='store_true', help='ordinary backpropagation through all iterations (about 130 MB per puzzle in the batch)')
    be = sub.add_parser('bench'); be.add_argument('--batch', type=int, default=None, help='one batch size; default tries 32, 64, 128, 256')
    be.add_argument('--iters', type=int, default=16); be.add_argument('--steps', type=int, default=3)
    be.add_argument('--full-memory', action='store_true'); be.set_defaults(fn=cmd_bench)
    te = sub.add_parser('test'); te.set_defaults(fn=cmd_test)
    a = ap.parse_args(); a.fn(a)
