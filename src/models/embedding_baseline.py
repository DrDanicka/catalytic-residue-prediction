from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from data.labeling import make_labels
from data.load import ProteinRecord
from features.embeddings import CachedProteinEmbedder, ProteinEmbeddingConfig
from models.baseline import IndexBase, ProbabilityModel, SklearnProbabilityModel, limit_training_sample, sample_positions


EmbeddingClassifierName = Literal["logistic", "random-forest", "xgboost", "mlp"]


class EmbeddingTrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    classifier: EmbeddingClassifierName = "logistic"
    embedding_model: str = "facebook/esm2_t6_8M_UR50D"
    embedding_cache_dir: Path = Path("data/embedding_cache")
    device: str = "auto"
    max_residues_per_chunk: int = Field(default=900, ge=1)
    negative_ratio: int = Field(default=10, ge=1)
    max_train_positions: int | None = Field(default=50_000, ge=1)
    max_eval_positions: int = Field(default=50_000, ge=1)
    epochs: int = Field(default=20, ge=1)
    batch_size: int = Field(default=512, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)
    n_estimators: int = Field(default=100, ge=1)
    max_depth: int | None = Field(default=None, ge=1)
    mlp_hidden_units: int = Field(default=128, ge=1)
    embedding_context_radius: int = Field(default=0, ge=0)
    embedding_context_mode: Literal["concat", "mean"] = "concat"
    n_jobs: int = 1
    seed: int = 13
    show_progress: bool = True

    def embedding_config(self) -> ProteinEmbeddingConfig:
        return ProteinEmbeddingConfig(
            model_name=self.embedding_model,
            device=self.device,
            cache_dir=self.embedding_cache_dir,
            max_residues_per_chunk=self.max_residues_per_chunk,
            show_progress=self.show_progress,
        )


def evaluation_positions(
    records: list[ProteinRecord],
    index_base: IndexBase,
    max_positions: int,
    seed: int,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    all_positions: list[tuple[int, int]] = []
    labels: list[int] = []

    for record_index, record in enumerate(records):
        record_labels = make_labels(record, index_base)
        for position, label in enumerate(record_labels):
            all_positions.append((record_index, position))
            labels.append(int(label))

    y = np.asarray(labels, dtype=np.int8)
    if len(all_positions) <= max_positions:
        return all_positions, y

    rng = np.random.default_rng(seed)
    selected = rng.choice(np.arange(len(all_positions)), size=max_positions, replace=False)
    return [all_positions[index] for index in selected], y[selected]


def unique_record_indices(positions: list[tuple[int, int]]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for record_index, _ in positions:
        if record_index not in seen:
            seen.add(record_index)
            ordered.append(record_index)
    return ordered


def build_embedding_matrix(
    records: list[ProteinRecord],
    positions: list[tuple[int, int]],
    embedder: CachedProteinEmbedder,
    description: str,
    show_progress: bool,
    context_radius: int,
    context_mode: Literal["concat", "mean"],
) -> np.ndarray:
    if not positions:
        raise ValueError("No positions selected for embedding matrix")

    embeddings_by_record: dict[int, np.ndarray] = {}
    record_indices = unique_record_indices(positions)
    record_iter = tqdm(
        record_indices,
        desc=f"{description}: embeddings",
        unit="protein",
        disable=not show_progress,
    )
    for record_index in record_iter:
        record = records[record_index]
        embeddings_by_record[record_index] = embedder.embed_record(record.entry, record.sequence)

    first_embedding = embeddings_by_record[positions[0][0]]
    feature_size = embedding_feature_size(first_embedding.shape[1], context_radius, context_mode)
    x = np.zeros((len(positions), feature_size), dtype=np.float32)
    position_iter = tqdm(
        enumerate(positions),
        total=len(positions),
        desc=f"{description}: matrix",
        unit="pos",
        disable=not show_progress,
    )
    for row, (record_index, position) in position_iter:
        x[row] = embedding_position_features(
            embeddings_by_record[record_index],
            position,
            context_radius,
            context_mode,
        )
    return x


def embedding_feature_size(embedding_size: int, context_radius: int, context_mode: Literal["concat", "mean"]) -> int:
    if context_mode == "mean":
        return embedding_size
    return embedding_size * (context_radius * 2 + 1)


def embedding_position_features(
    embeddings: np.ndarray,
    position: int,
    context_radius: int,
    context_mode: Literal["concat", "mean"],
) -> np.ndarray:
    if context_radius == 0:
        return embeddings[position]

    start = max(0, position - context_radius)
    end = min(len(embeddings), position + context_radius + 1)
    if context_mode == "mean":
        return np.mean(embeddings[start:end], axis=0).astype(np.float32)

    embedding_size = embeddings.shape[1]
    features = np.zeros((context_radius * 2 + 1, embedding_size), dtype=np.float32)
    for feature_row, sequence_position in enumerate(range(position - context_radius, position + context_radius + 1)):
        if 0 <= sequence_position < len(embeddings):
            features[feature_row] = embeddings[sequence_position]
    return features.reshape(-1)


def train_embedding_classifier(
    x: np.ndarray,
    y: np.ndarray,
    config: EmbeddingTrainingConfig,
) -> tuple[ProbabilityModel, list[str]]:
    if config.classifier == "logistic":
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=config.epochs,
            random_state=config.seed,
        )
        model = make_pipeline(StandardScaler(), classifier)
        model.fit(x, y.astype(np.int8))
        return SklearnProbabilityModel(estimator=model), [f"iterations={classifier.n_iter_[0]}"]

    if config.classifier == "random-forest":
        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            class_weight="balanced_subsample",
            n_jobs=config.n_jobs,
            random_state=config.seed,
        )
        model.fit(x, y.astype(np.int8))
        return SklearnProbabilityModel(estimator=model), [
            f"n_estimators={config.n_estimators}",
            f"max_depth={config.max_depth}",
        ]

    if config.classifier == "xgboost":
        from xgboost import XGBClassifier

        positives = max(float(np.sum(y == 1)), 1.0)
        negatives = max(float(np.sum(y == 0)), 1.0)
        model = XGBClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth or 6,
            learning_rate=config.learning_rate,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=negatives / positives,
            n_jobs=config.n_jobs,
            random_state=config.seed,
        )
        model.fit(x, y.astype(np.int8))
        return SklearnProbabilityModel(estimator=model), [
            f"n_estimators={config.n_estimators}",
            f"max_depth={config.max_depth or 6}",
            f"scale_pos_weight={negatives / positives:.4f}",
        ]

    if config.classifier == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=(config.mlp_hidden_units,),
            max_iter=config.epochs,
            batch_size=config.batch_size,
            learning_rate_init=config.learning_rate,
            random_state=config.seed,
            early_stopping=True,
            n_iter_no_change=3,
        )
        model = make_pipeline(StandardScaler(), classifier)
        model.fit(x, y.astype(np.int8))
        return SklearnProbabilityModel(estimator=model), [
            f"iterations={classifier.n_iter_}",
            f"loss={classifier.loss_:.4f}",
            f"hidden_units={config.mlp_hidden_units}",
        ]

    raise ValueError(f"Unsupported embedding classifier: {config.classifier}")


def train_embedding_baseline(
    train_records: list[ProteinRecord],
    validation_records: list[ProteinRecord],
    index_base: IndexBase,
    config: EmbeddingTrainingConfig,
) -> tuple[ProbabilityModel, list[str], list[tuple[int, int]], np.ndarray, np.ndarray]:
    train_positions, train_y = sample_positions(train_records, index_base, config.negative_ratio, config.seed)
    train_positions, train_y = limit_training_sample(
        train_positions,
        train_y,
        config.max_train_positions,
        config.seed,
    )
    validation_positions, validation_y = evaluation_positions(
        validation_records,
        index_base,
        config.max_eval_positions,
        config.seed,
    )

    embedder = CachedProteinEmbedder(config.embedding_config())
    if config.show_progress:
        print(f"train_positions: {len(train_positions)}")
        print(f"validation_positions: {len(validation_positions)}")
        print(f"train_embedding_proteins: {len(unique_record_indices(train_positions))}")
        print(f"validation_embedding_proteins: {len(unique_record_indices(validation_positions))}")
    train_x = build_embedding_matrix(
        train_records,
        train_positions,
        embedder,
        "train",
        config.show_progress,
        config.embedding_context_radius,
        config.embedding_context_mode,
    )
    model, training_log = train_embedding_classifier(train_x, train_y, config)
    validation_x = build_embedding_matrix(
        validation_records,
        validation_positions,
        embedder,
        "validation",
        config.show_progress,
        config.embedding_context_radius,
        config.embedding_context_mode,
    )
    validation_scores = model.predict_proba(validation_x)
    return model, training_log, validation_positions, validation_y, validation_scores
