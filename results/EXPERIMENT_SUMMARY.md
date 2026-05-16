# Experiment Summary

Source: `results/project_experiments.csv`.

Total runs: **15**.

All runs used protein-level train/validation/test splitting with `train_fraction=0.8`, `validation_fraction=0.1`, inferred `index_base=zero`, and validation on the natural class distribution sampled from validation proteins.

## Main Takeaways

- Best PR-AUC: **0.6276** from `embedding mlp ctx=2 concat`.
- Best threshold-tuned F1: **0.6298** from `embedding mlp ctx=2 concat` with precision **0.5965** and recall **0.6669**.
- Best fixed-threshold F1 at threshold 0.5: **0.3902** from `embedding mlp ctx=1 concat`.
- Best top-1 protein recall: **0.8940** from `embedding mlp ctx=1 concat`.
- The strongest family of models in this suite was **ESM2 embeddings + MLP**, especially with concatenated local embedding context.
- Squidly-like models improved over simple logistic/tree baselines and gave strong PR-AUC, but in this run they did not beat the embedding MLP context models. They likely need more tuning, larger ESM models, EC-aware pair mining, or longer training.

## Full Run Table

| # | Run | Pipeline | Model | Key config | Precision@0.5 | Recall@0.5 | F1@0.5 | PR-AUC | Best F1 | Best-F1 Precision | Best-F1 Recall | Top-1 Protein Recall | Top-5 Protein Recall |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `20260516T155400Z_33198c5d` | window | logistic | r=10, neg=10, train=100000 | 0.0297 | 0.7752 | 0.0571 | 0.0629 | 0.1345 | 0.1035 | 0.1922 |  |  |
| 2 | `20260516T155405Z_e0d94516` | window | mlp | r=10, neg=10, train=100000 | 0.1953 | 0.7516 | 0.3101 | 0.5512 | 0.5473 | 0.5986 | 0.5041 |  |  |
| 3 | `20260516T155412Z_6a68ca01` | window | random-forest | r=10, neg=10, train=100000 | 0.0218 | 0.9210 | 0.0427 | 0.1848 | 0.2067 | 0.5249 | 0.1287 |  |  |
| 4 | `20260516T155426Z_249e7849` | window | xgboost | r=10, neg=10, train=100000 | 0.0198 | 0.9406 | 0.0388 | 0.0978 | 0.1443 | 0.1629 | 0.1295 |  |  |
| 5 | `20260516T155438Z_7c42d6ef` | embedding | logistic | ctx=0:concat, train=50000 | 0.0354 | 0.9121 | 0.0682 | 0.1605 | 0.2496 | 0.2206 | 0.2875 | 0.6234 | 0.9355 |
| 6 | `20260516T155713Z_26e9ae89` | embedding | mlp | ctx=0:concat, train=50000 | 0.2204 | 0.8542 | 0.3504 | 0.5684 | 0.5793 | 0.5204 | 0.6531 | 0.8651 | 0.9737 |
| 7 | `20260516T160004Z_56978f75` | embedding | xgboost | ctx=0:concat, train=50000 | 0.0441 | 0.9047 | 0.0841 | 0.3179 | 0.3357 | 0.2731 | 0.4357 | 0.7167 | 0.9500 |
| 8 | `20260516T160252Z_aee6307c` | embedding | mlp | ctx=1:concat, train=50000 | 0.2516 | 0.8689 | 0.3902 | 0.6101 | 0.6223 | 0.6579 | 0.5904 | 0.8940 | 0.9703 |
| 9 | `20260516T160528Z_6cfd3ae3` | embedding | mlp | ctx=2:concat, train=50000 | 0.2328 | 0.8721 | 0.3675 | 0.6276 | 0.6298 | 0.5965 | 0.6669 | 0.8855 | 0.9737 |
| 10 | `20260516T160815Z_711c5225` | embedding | mlp | ctx=2:mean, train=50000 | 0.1833 | 0.7296 | 0.2930 | 0.3270 | 0.4115 | 0.3872 | 0.4389 | 0.7727 | 0.9364 |
| 11 | `20260516T161048Z_a7b1fabb` | embedding | xgboost | ctx=2:mean, train=50000 | 0.0263 | 0.8176 | 0.0509 | 0.1658 | 0.2485 | 0.2674 | 0.2321 | 0.5505 | 0.8626 |
| 12 | `20260516T161333Z_b7aab990` | squidly_like | contrastive_projection_bilstm | posw=100.0, neg=10, seq=1000, maxlen=1500 | 0.1318 | 0.8583 | 0.2285 | 0.4745 | 0.4842 | 0.5171 | 0.4552 | 0.8134 | 0.9618 |
| 13 | `20260516T163500Z_f45041fc` | squidly_like | contrastive_projection_bilstm | posw=100.0, neg=20, seq=1000, maxlen=1500 | 0.1054 | 0.8664 | 0.1879 | 0.3811 | 0.3759 | 0.2546 | 0.7174 | 0.7990 | 0.9644 |
| 14 | `20260516T170155Z_8f39819e` | squidly_like | contrastive_projection_bilstm | posw=50.0, neg=10, seq=1000, maxlen=1500 | 0.1246 | 0.8241 | 0.2165 | 0.5094 | 0.5139 | 0.4662 | 0.5725 | 0.7761 | 0.9542 |
| 15 | `20260516T172350Z_a7b08f06` | squidly_like | contrastive_projection_bilstm | posw=100.0, neg=10, seq=1000, maxlen=2500 | 0.0686 | 0.9007 | 0.1275 | 0.4537 | 0.5235 | 0.5388 | 0.5090 | 0.7837 | 0.9491 |

## Interpretation

### Window Baselines

- The window MLP was the only strong pure sequence-window baseline: fixed-threshold F1 **0.3101**, PR-AUC **0.5512**, best threshold-tuned F1 **0.5473**.
- Window logistic, random forest, and XGBoost had high recall at threshold 0.5 but weak precision and low fixed-threshold F1. Their high `best_precision` values are not very meaningful when recall is near zero.

### Embedding Baselines

- ESM2 embedding + MLP was the best overall approach in this suite.
- Adding concatenated local context helped: `ctx=1 concat` gave the best fixed-threshold F1 (**0.3902**) and best top-1 protein recall (**0.8940**).
- `ctx=2 concat` gave the best PR-AUC (**0.6276**) and best threshold-tuned F1 (**0.6298**), with precision **0.5965** and recall **0.6669** at the best F1 threshold.
- Mean-pooled context underperformed concatenated context in this run.

### Squidly-Like Runs

- The best Squidly-like PR-AUC was **0.5094** with lower positive weight (`positive_weight=50`).
- The best Squidly-like threshold-tuned F1 was **0.5235** for the longer-sequence variant (`max_sequence_length=2500`).
- Squidly-like models are promising but currently below the best embedding MLP context models. This is expected because this implementation does not yet use EC-aware pair mining, larger ESM2 3B/15B embeddings, model ensembling, or BLAST hybridization.

## Recommended Model From This Suite

Use the embedding MLP with concatenated context radius 2 as the current best project model:

```bash
uv run --extra embeddings python -m cli train-embedding-baseline \
  --classifier mlp \
  --embedding-model facebook/esm2_t6_8M_UR50D \
  --csv data/residual_amino_database.csv \
  --negative-ratio 10 \
  --epochs 50 \
  --learning-rate 0.001 \
  --mlp-hidden-units 128 \
  --embedding-context-radius 2 \
  --embedding-context-mode concat \
  --max-train-positions 50000 \
  --max-eval-positions 300000
```

For operating threshold, prefer the threshold associated with `validation_best_f1_threshold` from the run rather than the default 0.5 threshold.

## Next Steps

- Add model checkpoint saving for the best run.
- Add final held-out test-set evaluation separate from validation.
- Try `facebook/esm2_t12_35M_UR50D` and possibly `facebook/esm2_t30_150M_UR50D` if hardware allows.
- Improve Squidly-like pair mining with EC-number annotations if available.
- Add BLAST/DIAMOND homology transfer and ensemble it with the ML model.
