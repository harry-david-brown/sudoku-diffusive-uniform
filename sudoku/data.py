"""Datasets, encoding and the difficulty bands used for reporting."""
import os
import numpy as np
import pandas as pd
import torch

# Easy set: Kaggle "1 million Sudoku games" (bryanpark/sudoku), columns quizzes,solutions.
EASY_PATH = 'sudoku.csv'
# Hard set: imone/sudoku-hard-v2 (a copy of sapientinc/sudoku-extreme), columns source,question,answer,rating.
# rating = number of backtracks a solver needs. Download with the Hugging Face datasets cache or place the
# csv files next to this repo as hard_train.csv / hard_test.csv.
_HF = os.path.expanduser('~/.cache/huggingface/hub/datasets--imone--sudoku-hard-v2/snapshots/'
                         '58942f96baeb572ca3127e2a9e9c70f330783d6b/')
HARD_TRAIN_PATH = 'hard_train.csv' if os.path.exists('hard_train.csv') else _HF + 'train.csv'
HARD_TEST_PATH = 'hard_test.csv' if os.path.exists('hard_test.csv') else _HF + 'test.csv'


def to_tensor(strings):
    """List of 81-character puzzle strings -> [N,81] int64, with '.' and '0' as 0."""
    buf = np.frombuffer(''.join(strings).encode(), dtype=np.uint8).reshape(len(strings), 81)
    arr = buf.astype(np.int64) - 48
    arr[(arr < 0) | (arr > 9)] = 0
    return torch.from_numpy(arr)


def load_easy(path=EASY_PATH, n=500000, skip=0):
    df = pd.read_csv(path, dtype=str, skiprows=range(1, skip + 1), nrows=n)
    return to_tensor(df['quizzes'].tolist()), to_tensor(df['solutions'].tolist())


def read_hard(path):
    df = pd.read_csv(path, dtype=str)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    return df


def load_hard(path, n=500000, min_rating=50, with_rating=False):
    """Puzzles rated above min_rating (-1 keeps all). A random subset if more than n qualify."""
    df = read_hard(path)
    df = df[df['rating'] > min_rating]
    if n and len(df) > n:
        df = df.sample(n=n, random_state=0)
    out = (to_tensor(df['question'].tolist()), to_tensor(df['answer'].tolist()))
    return out + (torch.tensor(df['rating'].values, dtype=torch.float32),) if with_rating else out


def load_data(name, n, split='train', skip=0, min_rating=50):
    if name == 'easy':
        return load_easy(n=n, skip=skip)
    if name == 'hard':
        return load_hard(HARD_TRAIN_PATH if split == 'train' else HARD_TEST_PATH, n=n, min_rating=min_rating)
    raise ValueError(name)


def difficulty_bands(df):
    """The reporting bands: (name, boolean mask over the hard test file)."""
    r, src = df['rating'], df['source']
    return [
        ('A easy kaggle, rating 0', src == 'puzzles0_kaggle'),
        ('B 17-clue, rating 0', (src == 'puzzles2_17_clue') & (r == 0)),
        ('C other, rating 0', (r == 0) & ~src.isin(['puzzles0_kaggle', 'puzzles2_17_clue'])),
        ('D rating 1-5', (r >= 1) & (r <= 5)),
        ('E rating 6-20', (r > 5) & (r <= 20)),
        ('F rating 21-50', (r > 20) & (r <= 50)),
        ('G rating 51-100', (r > 50) & (r <= 100)),
        ('H rating >100', r > 100),
    ]
