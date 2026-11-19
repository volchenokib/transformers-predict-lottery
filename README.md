# Can a Transformer Predict Lottery Draws?

We tested whether a small PyTorch Transformer could predict the next Sportloto 6/45 draw from the previous 20. The two-layer encoder was trained from scratch and compared with a fixed forecast using the six most frequent numbers in the training data and with uniform random selection.

## Result

On 2,189 held-out draws, the Transformer showed **no statistically significant advantage over random selection** (one-sided Monte Carlo test, 10,000 simulations, p = 0.883).

| Method | Mean matches per draw |
| --- | ---: |
| Transformer | 0.780 |
| Frequency baseline | 0.774 |
| Uniform random (expected) | 0.800 |

## Related work

Tao and Barkeshli (University of Maryland, College Park) showed that Transformers can predict sequences from certain pseudorandom number generators in a controlled setting. Our experiment uses historical lottery draws. [Learning Pseudorandom Numbers with Transformers (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/50eb39ab717507cccbe2b8590de32030-Abstract-Conference.html)

## Reproduce

Run from the root of a fresh checkout with [uv](https://docs.astral.sh/uv/). The commands use Python 3.12 and PyTorch 2.15.0 on CPU.

```bash
uv run --python 3.12 --with torch==2.15.0 python train.py --archive
uv run --python 3.12 --with torch==2.15.0 python evaluate.py
```

Training creates `checkpoints/best.pt` using the epoch with the lowest validation loss. Evaluation loads those weights and scores the held-out test split. The snapshot contains 14,613 draws in chronological order. After 20 context-only draws, the splits contain 10,215 training, 2,189 validation, and 2,189 test targets.

The [working snapshot](data/sportloto_6x45_4853-19465.csv) comes from [LotoDetective](https://lotodetective.ru/database). The 194 draws available in the downloaded [official Stoloto CSV](data/stoloto_6x45_19272-19465.csv) matched the snapshot. For exact split boundaries and run details, see the [experiment log](EXPERIMENT_LOG.md) and [protocol](EXPERIMENT_PLAN.md).
