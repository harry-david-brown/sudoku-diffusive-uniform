import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.jump_chain = nn.Linear(embed_dim, 9)
        self.holding = nn.Linear(embed_dim, 1)
        self.seq_len = seq_len

    def forward(self, x):
        positions = torch.arange(self.seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.transformer(x)
        jump = self.jump_chain(x)
        hold = self.holding(x).squeeze(-1)
        return jump, hold


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
print(f"Using device: {device}")

puzzles, solutions = load_dataset('sudoku.csv', n=500000)
dataset = TensorDataset(puzzles, solutions)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SudokuDiffusion().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ── Training ───────────────────────────────────────────────────────────────────
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_jump_loss = 0
    total_hold_loss = 0
    start = time.time()

    for batch_puzzles, batch_solutions in loader:
        batch_puzzles = batch_puzzles.to(device)
        batch_solutions = batch_solutions.to(device)
        optimizer.zero_grad()

        corrupted, should_corrupt = apply_uniform_noise(batch_puzzles, batch_solutions)
        jump, hold = model(corrupted)

        # jump chain loss — cross entropy over corrupted positions
        # shift targets from 1-9 to 0-8
        jump_loss = F.cross_entropy(
            jump[should_corrupt],
            batch_solutions[should_corrupt] - 1
        )

        # holding distribution loss — BCE over all unknown cells
        hold_probs = torch.sigmoid(hold)
        unknown = (batch_puzzles == 0)
        targets = should_corrupt.float()
        holding_loss = F.binary_cross_entropy(
            hold_probs[unknown],
            targets[unknown]
        )

        loss = jump_loss + holding_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_jump_loss += jump_loss.item()
        total_hold_loss += holding_loss.item()

    elapsed = time.time() - start
    print(f"Epoch {epoch+1}/{num_epochs} — "
          f"Loss: {total_loss/len(loader):.4f} — "
          f"Jump: {total_jump_loss/len(loader):.4f} — "
          f"Hold: {total_hold_loss/len(loader):.4f} — "
          f"{elapsed:.0f}s")

torch.save(model.state_dict(), 'sudoku_diffusion_gidd_500k.pth')
print("Model saved.")



# ── Inference Function ───────────────────────────────────────────────────────────────────
def gidd_inference(puzzle_string, model, device, k=5, corrupt_threshold=0.5):
    # initialize — givens fixed, unknowns as random digits 1-9
    tokens = [torch.randint(1, 10, (1,)).item() if c == '0' else int(c) for c in puzzle_string]
    x = torch.tensor([tokens], dtype=torch.long).to(device)

    # track which positions are still unresolved
    still_unresolved = torch.tensor([c == '0' for c in puzzle_string])

    model.eval()
    with torch.no_grad():
        max_iters = 200  # safety limit
        iteration = 0
        while still_unresolved.any() and iteration < max_iters:
            jump, hold = model(x)

            # hold scores for unresolved positions
            hold_probs = torch.sigmoid(hold[0])  # [81]
            jump_probs = torch.softmax(jump[0], dim=-1)  # [81, 9]

            # mask resolved positions
            hold_probs_masked = hold_probs.clone()
            hold_probs_masked[~still_unresolved] = -1

            # pick top-k most confidently corrupted unresolved cells
            num_to_commit = min(k, still_unresolved.sum().item())
            topk = hold_probs_masked.topk(num_to_commit).indices

            for pos in topk:
                # predicted digit is argmax of jump chain + 1 (shift back to 1-9)
                predicted_digit = jump_probs[pos].argmax().item() + 1
                x[0, pos] = predicted_digit
                still_unresolved[pos] = False

            iteration += 1

    return ''.join(str(x[0, i].item()) for i in range(81))