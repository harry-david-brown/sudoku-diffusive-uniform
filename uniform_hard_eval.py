import torch
import torch.nn as nn
import pandas as pd
import time
import random

HARD_DATASET_PATH = 'train.csv'


def load_hard_dataset(path, n=None, min_rating=50):
    df = pd.read_csv(path, dtype=str)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df[df['rating'] > min_rating].copy()
    print(f"Hard puzzles available (rating > {min_rating}): {len(df):,}")
    if n:
        df = df.iloc[:n]
    puzzles   = df['question'].tolist()
    solutions = df['answer'].tolist()
    ratings   = df['rating'].tolist()
    return puzzles, solutions, ratings


class SudokuDiffusion(nn.Module):
    def __init__(self, vocab_size=10, embed_dim=128, num_heads=4, num_layers=4, seq_len=81):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(embed_dim, 10)
        self.seq_len = seq_len

    def forward(self, x):
        positions = torch.arange(self.seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.transformer(x)
        return self.output(x)


def iterative_inference(puzzle_string, model, device, k=5):
    tokens = [torch.randint(1, 10, (1,)).item() if c in '0.' else int(c) for c in puzzle_string]
    x = torch.tensor([tokens], dtype=torch.long).to(device)
    still_unresolved = torch.tensor([c in '0.' for c in puzzle_string])
    with torch.no_grad():
        while still_unresolved.any():
            output = model(x)
            probs = torch.softmax(output[0], dim=-1)
            confidence, predicted = probs.max(dim=-1)
            confidence[~still_unresolved] = -1
            num_to_resolve = min(k, still_unresolved.sum().item())
            topk = confidence.topk(num_to_resolve).indices
            for pos in topk:
                x[0, pos] = predicted[pos]
                still_unresolved[pos] = False
    return ''.join(str(x[0, i].item()) for i in range(81))


def oneshot_inference(puzzle_string, model, device):
    tokens = [torch.randint(1, 10, (1,)).item() if c in '0.' else int(c) for c in puzzle_string]
    x = torch.tensor([tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(x)
    pred = output.argmax(dim=-1)[0].tolist()
    for i, c in enumerate(puzzle_string):
        if c not in '0.':
            pred[i] = int(c)
    return ''.join(str(d) for d in pred)


def check_validity(grid):
    digits = set('123456789')
    violations = []
    for r in range(9):
        if set(grid[r*9:(r+1)*9]) != digits: violations.append(f'Row {r+1}')
    for c in range(9):
        if set(grid[c::9]) != digits: violations.append(f'Col {c+1}')
    for br in range(3):
        for bc in range(3):
            box = [grid[(br*3+r)*9+(bc*3+c)] for r in range(3) for c in range(3)]
            if set(box) != digits: violations.append(f'Box ({br+1},{bc+1})')
    return violations


def evaluate(sample_puzzles, sample_solutions, sample_ratings, model, device, mode='oneshot', k=5):
    correct_cells   = 0
    correct_puzzles = 0
    total_cells     = 0
    total_violations = 0
    puzzles_with_violations = 0
    position_correct = torch.zeros(81)
    position_total   = torch.zeros(81)

    for puzzle_str, solution_str in zip(sample_puzzles, sample_solutions):
        if mode == 'oneshot':
            result = oneshot_inference(puzzle_str, model, device)
        else:
            result = iterative_inference(puzzle_str, model, device, k=k)

        unknown = [i for i, c in enumerate(puzzle_str) if c in '0.']
        correct_cells += sum(1 for i in unknown if result[i] == solution_str[i])
        total_cells   += len(unknown)

        for i in unknown:
            position_total[i] += 1
            if result[i] == solution_str[i]:
                position_correct[i] += 1

        v = check_validity(result)
        total_violations += len(v)
        if v: puzzles_with_violations += 1
        if result == solution_str: correct_puzzles += 1

    n = len(sample_puzzles)
    print(f"  Cell accuracy:         {correct_cells/total_cells*100:.2f}%")
    print(f"  Puzzle accuracy:       {correct_puzzles/n*100:.2f}%")
    print(f"  Puzzles w/ violations: {puzzles_with_violations}/{n}")
    print(f"  Avg violations:        {total_violations/n:.2f}")

    position_acc = (position_correct / position_total.clamp(min=1) * 100).tolist()
    print("\n  Per-position accuracy (unknown cells only):")
    for row in range(9):
        print('  ' + ' | '.join(f'{position_acc[row*9+col]:5.1f}' for col in range(9)))

    print("\n  By difficulty tier:")
    tiers = [(50, 100), (100, 200), (200, 465)]
    for low, high in tiers:
        tier_idx = [i for i, r in enumerate(sample_ratings) if low < float(r) <= high]
        if not tier_idx:
            continue
        t_correct = 0
        t_total   = 0
        t_violations = 0
        for i in tier_idx:
            puz = sample_puzzles[i]
            sol = sample_solutions[i]
            if mode == 'oneshot':
                res = oneshot_inference(puz, model, device)
            else:
                res = iterative_inference(puz, model, device, k=k)
            unk = [j for j, c in enumerate(puz) if c in '0.']
            t_correct += sum(1 for j in unk if res[j] == sol[j])
            t_total   += len(unk)
            t_violations += len(check_validity(res))
        print(f"    Rating {low}-{high} (n={len(tier_idx)}): "
              f"Cell acc {t_correct/t_total*100:.1f}%, "
              f"Avg violations {t_violations/len(tier_idx):.1f}")


# ── Setup ──────────────────────────────────────────────────────────────────────
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

puzzles, solutions, ratings = load_hard_dataset(HARD_DATASET_PATH, n=10000)
print(f"Loaded {len(puzzles):,} hard puzzles")
print(f"Rating range: {min(float(r) for r in ratings):.0f} – {max(float(r) for r in ratings):.0f}, "
      f"mean {sum(float(r) for r in ratings)/len(ratings):.1f}")

model = SudokuDiffusion().to(device)
model.load_state_dict(torch.load('sudoku_diffusion_uniform_500k.pth', map_location=device))
model.eval()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

sample_size = 1000
indices = random.sample(range(len(puzzles)), sample_size)
sample_puzzles   = [puzzles[i] for i in indices]
sample_solutions = [solutions[i] for i in indices]
sample_ratings   = [ratings[i] for i in indices]

# ── One-shot ───────────────────────────────────────────────────────────────────
print("\n=== Uniform Diffusion — Hard Puzzles (one-shot) ===")
evaluate(sample_puzzles, sample_solutions, sample_ratings, model, device, mode='oneshot')

# ── Iterative k=1 ─────────────────────────────────────────────────────────────
print("\n=== Uniform Diffusion — Hard Puzzles (iterative, k=1) ===")
start = time.time()
evaluate(sample_puzzles, sample_solutions, sample_ratings, model, device, mode='iterative', k=1)
print(f"\n  Time: {time.time()-start:.1f}s")

# ── k sweep ────────────────────────────────────────────────────────────────────
print("\n=== Iterative decoding sweep ===")
for k in [1, 5, 10, 20, 81]:
    correct = 0
    total_violations_k = 0
    start = time.time()
    for puzzle_str, solution_str in zip(sample_puzzles, sample_solutions):
        result = iterative_inference(puzzle_str, model, device, k=k) if k < 81 else oneshot_inference(puzzle_str, model, device)
        if result == solution_str: correct += 1
        total_violations_k += len(check_validity(result))
    elapsed = time.time() - start
    print(f"k={k:3d} — Accuracy: {correct/sample_size*100:.2f}% — "
          f"Avg violations: {total_violations_k/sample_size:.2f} — Time: {elapsed:.1f}s")