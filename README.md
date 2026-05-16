# catalytic-residue-prediction

Predict catalytic residues in protein sequences.

## Setup

This project uses `pyproject.toml` and `uv`.

```bash
uv run python -m cli validate --csv data/residual_amino_database.csv
```

Run tests:

```bash
uv run --extra dev pytest
```

Run a baseline smoke test:

```bash
uv run python -m cli train-baseline \
  --model logistic \
  --csv data/residual_amino_database.csv \
  --epochs 1 \
  --radius 5 \
  --negative-ratio 5 \
  --max-train-positions 50000 \
  --max-eval-positions 50000
```

Available baseline models:

- `logistic`
- `random-forest`
- `xgboost`
- `mlp`

Run an embedding baseline with ESM2:

```bash
uv run --extra embeddings python -m cli train-embedding-baseline \
  --classifier logistic \
  --embedding-model facebook/esm2_t6_8M_UR50D \
  --csv data/residual_amino_database.csv \
  --negative-ratio 10 \
  --embedding-context-radius 2 \
  --embedding-context-mode concat \
  --max-train-positions 50000 \
  --max-eval-positions 50000
```

Per-residue embeddings are cached under `data/embedding_cache/`.
`--device auto` uses CUDA when available, otherwise CPU.
Use `--no-progress` to disable progress bars in non-interactive runs.
Use `--embedding-context-radius` to concatenate or average neighboring residue embeddings around each target position.

Project notes and dataset/model observations are tracked in `PROJECT_NOTES.md`.

Training runs append metrics and configuration to `results/experiments.csv` by default.
Use `--results-csv path/to/file.csv` to write a different table.
The result table also stores threshold-sweep metrics and top-k validation metrics for embedding runs.

Run a Squidly-inspired contrastive + BiLSTM experiment:

```bash
uv run --extra embeddings python -m cli train-squidly-like \
  --embedding-model facebook/esm2_t6_8M_UR50D \
  --csv data/residual_amino_database.csv \
  --max-contrastive-positions 50000 \
  --contrastive-epochs 3 \
  --contrastive-batches-per-epoch 500 \
  --classifier-epochs 5 \
  --max-train-sequences 1000 \
  --max-eval-positions 50000
```

This is a Squidly-like direction, not a full Squidly reproduction: it uses hard-negative contrastive residue pairs and a BiLSTM token classifier, but it does not yet use EC-number-informed pair mining or large ESM2 3B/15B embeddings.

Run an experiment suite:

```bash
uv run --extra embeddings python scripts/run_experiments.py \
  --profile full \
  --results-csv results/project_experiments.csv
```

Useful suite controls:

```bash
uv run --extra embeddings python scripts/run_experiments.py --list
uv run --extra embeddings python scripts/run_experiments.py --dry-run
uv run --extra embeddings python scripts/run_experiments.py --start-at embedding_mlp_single
uv run --extra embeddings python scripts/run_experiments.py --only squidly_like_small squidly_like_more_pairs
```
