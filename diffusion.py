import torch
import torch.nn as nn
import pandas as pd
import time
from torch.utils.data import TensorDataset, DataLoader


def load_dataset(path, n=None):
    df = pd.read_csv(path, dtype=str)
    puzzles = df['quizzes'].tolist()
    solutions = df['solutions'].tolist()
    if n:
        puzzles = puzzles[:n]
        solutions = solutions[:n]
    X = torch.tensor([[int(c) for c in p] for p in puzzles], dtype=torch.long)
    Y = torch.tensor([[int(c) for c in s] for s in solutions], dtype=torch.long)
    return X, Y


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


def apply_uniform_noise(puzzles, solutions):
    unknown_cell = (puzzles == 0)
    rand_tensors = torch.rand(solutions.shape, device=solutions.device)
    rand_threshold = torch.rand(solutions.shape[0], 1, device=solutions.device).clamp(min=1/81)
    should_corrupt = unknown_cell & (rand_tensors < rand_threshold)
    corrupted = solutions.clone()
    corrupted[should_corrupt] = torch.randint(1, 10, (should_corrupt.sum(),), device=solutions.device)
    return corrupted, should_corrupt


# ── Setup ──────────────────────────────────────────────────────────────────────
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
puzzles, solutions = load_dataset('sudoku.csv', n=500000)
dataset = TensorDataset(puzzles, solutions)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
model = SudokuDiffusion().to(device)
#model.load_state_dict(torch.load('sudoku_diffusion_uniform_100k.pth', map_location=device))
#model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ── Training ───────────────────────────────────────────────────────────────────
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    start = time.time()
    for batch_puzzles, batch_solutions in loader:
        batch_puzzles = batch_puzzles.to(device)
        batch_solutions = batch_solutions.to(device)
        optimizer.zero_grad()
        corrupted, should_mask = apply_uniform_noise(batch_puzzles, batch_solutions)
        output = model(corrupted)
        loss = criterion(output[should_mask], batch_solutions[should_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    elapsed = time.time() - start
    print(f"Epoch {epoch+1}/{num_epochs} — Loss: {total_loss/len(loader):.4f} — {elapsed:.0f}s")

torch.save(model.state_dict(), 'sudoku_diffusion_uniform_500k.pth')
print("Model saved.")




# def iterative_inference_uniform(puzzle_string, model, device, k=5):
#     # givens fixed, unknowns initialized as random digits
#     tokens = [torch.randint(1, 10, (1,)).item() if c == '0' else int(c) for c in puzzle_string]
#     x = torch.tensor([tokens], dtype=torch.long).to(device)
    
#     # track which positions are still unresolved
#     still_unresolved = torch.tensor([c == '0' for c in puzzle_string])
    
#     model.eval()
#     with torch.no_grad():
#         while still_unresolved.any():
#             output = model(x)
#             probs = torch.softmax(output[0], dim=-1)
#             confidence, predicted = probs.max(dim=-1)
#             confidence[~still_unresolved] = -1
#             num_to_resolve = min(k, still_unresolved.sum().item())
#             topk_positions = confidence.topk(num_to_resolve).indices
#             for pos in topk_positions:
#                 x[0, pos] = predicted[pos]
#                 still_unresolved[pos] = False
    
#     return ''.join(str(x[0, i].item()) for i in range(81))




# # ── Iterative decoding benchmark ───────────────────────────────────────────────
# import random

# # sample 1000 puzzles for benchmarking
# sample_size = 1000
# indices = random.sample(range(len(puzzles)), sample_size)
# sample_puzzles = [''.join(str(c.item()) for c in puzzles[i]) for i in indices]
# sample_solutions = [''.join(str(c.item()) for c in solutions[i]) for i in indices]

# for k in [1, 5, 10, 20, 81]:
#     correct = 0
#     total_passes = 0
#     start = time.time()
#     for puzzle_str, solution_str in zip(sample_puzzles, sample_solutions):
#         # count unknown cells for pass calculation
#         num_unknown = puzzle_str.count('0')
#         passes = max(1, -(-num_unknown // k))  # ceiling division
#         total_passes += passes
#         result = iterative_inference(puzzle_str, model, device, k=k)
#         if result == solution_str:
#             correct += 1
#     elapsed = time.time() - start
#     print(f"k={k:3d} — Puzzle accuracy: {correct/sample_size*100:.2f}% — Avg passes: {total_passes/sample_size:.1f} — Time: {elapsed:.1f}s")