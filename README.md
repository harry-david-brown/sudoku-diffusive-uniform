# Neural Sudoku solvers

Small transformers (about 800k parameters) trained to solve Sudoku, as a testbed for one question: what does
a network need in order to solve constraint problems that require search, not just deduction?

Write-ups: [harrydavidbrown.com](https://harrydavidbrown.com).

## What is here

| | Model | File | Origin |
|---|---|---|---|
| **GIDD** | Masked / hybrid discrete diffusion. Fills the grid in rounds, most confident cells first. | `sudoku/gidd.py`, `gidd_sudoku.py` | Port of von Rütte et al., *Generalized Interpolating Discrete Diffusion* (ICML 2025), [reference code](https://github.com/dvruette/gidd) |
| **Looped** | One 4-layer transformer run many times over, feeding its own predictions back in. | `sudoku/looped.py`, `looped_sudoku.py` | Port of Cheng Lou's [Sotaku](https://github.com/chenglou/sotaku) |
| Two-head (archived) | An earlier detector-plus-corrector model. Not GIDD; kept for the record. | `legacy/two_head/` | Posts 3–5 |

```
sudoku/                 the library
  data.py               datasets, encoding, difficulty bands
  metrics.py            scoring
  gidd.py               noise schedule, model, loss, samplers
  looped.py             looped transformer
  checkpoint.py         load any checkpoint as a solver
gidd_sudoku.py          train / eval / test the GIDD model
looped_sudoku.py        train / bench / test the looped model
eval_by_difficulty.py   the standard report: results by difficulty band
first_error.py          diagnostic: model vs simple deduction rules
checkpoints/            trained weights
legacy/                 superseded code and weights
```

## Data

Neither dataset is in the repo.

- **Easy:** Kaggle [1 million Sudoku games](https://www.kaggle.com/datasets/bryanpark/sudoku). Save as `sudoku.csv`.
  30–36 givens, all solvable by simple deduction.
- **Hard:** [imone/sudoku-hard-v2](https://huggingface.co/datasets/imone/sudoku-hard-v2) (the same puzzles as
  `sapientinc/sudoku-extreme`). Each puzzle has a rating: the number of backtracks a solver needs. Save the two
  files as `hard_train.csv` and `hard_test.csv`, or leave them in the Hugging Face cache.

## How results are reported

"Hard" is not one thing. The hard test file runs from rating 0 to 385, and clue count is not difficulty
(17-clue puzzles are mostly rating 0). So every model is scored by band:

```
python eval_by_difficulty.py checkpoints/gidd_pu0_easyhard.pth 500
python eval_by_difficulty.py checkpoints/looped.pth 500 --iters 128
```

Percent of puzzles solved, 500 per band:

| Band | Givens | GIDD, easy-only | GIDD, easy → hard | GIDD, easy → hard, p_u = 0.2 |
|---|---|---|---|---|
| Easy Kaggle, rating 0 | 33.8 | 100.0 | 100.0 | 100.0 |
| Other, rating 0 | 25.5 | 41.6 | 83.4 | 66.6 |
| 17-clue, rating 0 | 17.0 | 1.8 | 22.2 | 6.6 |
| Rating 1–5 | 25.6 | 17.0 | 38.8 | 28.2 |
| Rating 6–20 | 25.0 | 0.8 | 2.0 | 1.2 |
| Rating 21–50 | 25.0 | 1.0 | 1.2 | 1.8 |
| Rating 51–100 | 24.8 | 0.6 | 0.6 | 1.2 |
| Rating > 100 | 24.0 | 0.0 | 1.2 | 0.6 |

The GIDD models stop where simple deduction stops (`first_error.py`). Sotaku's published weights, loaded
through `sudoku/looped.py`, solve 95–100% of every band at 128 iterations (40 puzzles per band). Those weights
are included as `checkpoints/external/sotaku_v2.pt` (MIT licence, © Cheng Lou) so the comparison can be rerun:

```
python eval_by_difficulty.py checkpoints/external/sotaku_v2.pt 500 --iters 1024
```

## Running

```
pip install -r requirements.txt

python gidd_sudoku.py test          # 21 checks of the GIDD math against the paper
python looped_sudoku.py test        # checks the Sotaku port

# GIDD: 500k easy puzzles, then fine-tune on hard ones
python gidd_sudoku.py train --pu 0.0 --out checkpoints/gidd_pu0_easy.pth
python gidd_sudoku.py train --pu 0.0 --data hard --init checkpoints/gidd_pu0_easy.pth --lr 1e-4 --out checkpoints/gidd_pu0_easyhard.pth

# Looped: size the run first, then train
python looped_sudoku.py bench
python looped_sudoku.py train --steps 8000 --batch 128 --out checkpoints/looped.pth
```

`--pu` is the share of random digits in the diffusion noise; 0 is plain masked diffusion.
Everything was developed on an Apple Silicon laptop (MPS).
