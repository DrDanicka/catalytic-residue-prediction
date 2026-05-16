# Catalytic Residue Prediction Notes

## Dataset

- Source file: `data/residual_amino_database.csv`
- Columns observed: `Entry`, `Sequence`, `Residue`
- Rows: 106589 protein records plus header
- Sequence length summary from a quick scan:
  - min: 8
  - mean: about 419
  - max: 35213
- Catalytic residue annotations:
  - total annotated positive positions: about 178395
  - mean annotated positions per protein: about 1.67
  - max annotated positions in one protein: 23
- Validation run through `uv run python -m cli validate`:
  - total amino acid positions: 44751788
  - inferred index base: 0-based
  - out-of-bounds annotations after conversion: 0
  - sequences containing non-standard amino acid symbols: 467
  - duplicate `Entry` values: 0

## Important Data Assumptions

- `Residue` contains one or more residue indices separated by `|`.
- The file contains three annotations with index `0`:
  - `P34039`, length 19, residues `0|14`
  - `Q08258`, length 436, residues `0`
  - `P94469`, length 395, residues `0`
- Because index `0` exists and no annotation was found above the sequence length in the quick scan, the current code defaults to 0-based indexing when it infers indexing from the dataset.
- This should still be treated as a hypothesis until checked against the original annotation source.

## First Modeling Plan

The first implementation is a dependency-light baseline:

1. Parse the CSV with the Python standard library.
2. Validate sequence characters and residue indices.
3. Convert each protein into per-residue binary labels.
4. Split by protein entry, not by individual residue, to reduce leakage.
5. Train a window-based logistic regression baseline using NumPy.

This baseline is intentionally simple. Its job is to provide a reproducible reference point before adding protein language model embeddings.

## Metrics To Prefer

- PR-AUC
- ROC-AUC
- F1 at a selected threshold
- precision and recall
- top-k recall per protein

Accuracy is not useful as a headline metric because catalytic residues are rare compared with non-catalytic residues.

## Experiment Tracking

- Training commands append one row per run to `results/experiments.csv` by default.
- The table stores run id, timestamp, pipeline type, model/classifier, embedding model, split sizes, threshold, duration, training log, config JSON, and validation metrics.
- Use `--results-csv path/to/file.csv` to write a custom result table.
- Experiment suites can be run with `uv run --extra embeddings python scripts/run_experiments.py`.
- The full suite currently contains 15 configurations: window baselines, embedding baselines with multiple classifiers/context modes, and Squidly-like variants.

## Leakage Risks

- Random residue-level splitting would leak sequence context across train and test.
- Protein-level splitting is the minimum acceptable baseline.
- A stronger future split should cluster similar sequences and split by cluster or family.

## Model Notes

### Window Baseline Models

- Unit of prediction: one amino acid position.
- Features:
  - one-hot amino acid identity in a fixed window around the target residue
  - padding and unknown tokens for edges/non-standard residues
  - normalized relative position in the protein
  - local catalytic-prior features from amino acid physicochemical properties
- Negative examples are sampled from non-catalytic residues to control class imbalance.
- Implemented baseline model choices:
  - `logistic`: custom NumPy minibatch SGD logistic regression
  - `random-forest`: scikit-learn `RandomForestClassifier`
  - `xgboost`: XGBoost `XGBClassifier`
  - `mlp`: scikit-learn `MLPClassifier` with one hidden layer
- Training can be capped with `--max-train-positions` to keep tree and MLP baselines manageable.
- Smoke-test run:
  - command: `uv run python -m cli train-baseline --csv data/residual_amino_database.csv --epochs 1 --radius 5 --negative-ratio 5 --max-eval-positions 50000`
  - split sizes: 85153 train proteins, 10714 validation proteins, 10722 test proteins
  - validation positive rate in sampled evaluation positions: 0.004560
  - validation PR-AUC: 0.059169
  - validation ROC-AUC: 0.928811
  - validation F1 at threshold 0.5: 0.061718
- Additional small smoke tests passed for `random-forest`, `xgboost`, and `mlp` using capped train/evaluation samples.

The ROC-AUC is high in the smoke test, but PR-AUC is the more important number because positives are rare. Future work should tune thresholds and evaluate top-k recall per protein.

### Next Models

### Protein Language Model Embedding Baseline

- Implemented command: `uv run --extra embeddings python -m cli train-embedding-baseline`
- Default embedding model: `facebook/esm2_t6_8M_UR50D`
- The embedding pipeline:
  - normalizes non-standard amino acid symbols to `X`
  - chunks long proteins before passing them to the transformer
  - removes special-token embeddings and keeps one embedding per residue
  - caches per-protein embeddings in `data/embedding_cache/`
  - shows progress bars for embedding generation and matrix construction
  - uses CUDA when available, otherwise CPU; MPS is intentionally not used
  - can include local context with `--embedding-context-radius`
- Embedding context modes:
  - `concat`: concatenate embeddings from positions `i-r ... i+r`, with zero padding at sequence edges
  - `mean`: average available embeddings in the local context window
- Implemented classifiers over embeddings:
  - `logistic`
  - `random-forest`
  - `xgboost`
  - `mlp`
- Logistic regression and MLP use feature scaling before classification.
- A tiny end-to-end smoke test passed with ESM2, logistic regression, and capped train/evaluation positions.
- Tiny embedding smoke-test metrics are not meaningful because very small random evaluation samples can contain no positive residues. Use larger `--max-eval-positions` for real comparisons.
- Embedding runs now report:
  - fixed-threshold metrics
  - best threshold by F1 over a small threshold sweep
  - best precision threshold with non-zero recall
  - top-1/top-3/top-5/top-10 metrics over sampled validation positions grouped by protein

### Future Models

### Squidly-Like Experimental Pipeline

- Implemented command: `uv run --extra embeddings python -m cli train-squidly-like`
- Architecture:
  - ESM2 per-residue embeddings
  - contrastive projection MLP
  - hard-negative residue pair sampling, prioritizing catalytic/non-catalytic pairs of the same amino acid
  - BiLSTM token classifier over projected per-residue embeddings
  - weighted binary cross-entropy for catalytic residue imbalance
- This is inspired by Squidly, but it is not a full reproduction:
  - no EC-number-informed pair mining yet
  - no ESM2 3B/15B default
  - no five-model ensemble yet
  - no BLAST hybrid yet
- A tiny end-to-end smoke test passed with ESM2 8M, contrastive training, BiLSTM training, validation metrics, and experiment CSV logging.

### Future Models

- Try larger ESM2 models or ProtT5 embeddings.
- Add EC-number annotations if available and implement reaction-informed pair mining closer to Squidly schemes 2/3.
- Add a BLAST/DIAMOND homology-transfer baseline and ensemble it with ML predictions.
- Add model checkpoint saving for trained projection/classifier weights.
