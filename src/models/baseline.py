from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from data.labeling import make_labels
from data.load import ProteinRecord
from features.windows import residue_window_features, window_feature_size
from models.logistic import LogisticRegressionSGD


IndexBase = Literal["zero", "one"]
BaselineModelName = Literal["logistic", "random-forest", "xgboost", "mlp"]


class ProbabilityModel(Protocol):
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        pass


class SklearnProbabilityModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    estimator: Any

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probabilities = self.estimator.predict_proba(x)
        return probabilities[:, 1]


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: BaselineModelName = "logistic"
    radius: int = Field(default=10, ge=0)
    negative_ratio: int = Field(default=10, ge=1)
    epochs: int = Field(default=5, ge=1)
    batch_size: int = Field(default=512, ge=1)
    learning_rate: float = Field(default=0.05, gt=0)
    seed: int = 13
    max_train_positions: int | None = Field(default=250_000, ge=1)
    max_eval_positions: int = Field(default=250_000, ge=1)
    n_estimators: int = Field(default=100, ge=1)
    max_depth: int | None = Field(default=None, ge=1)
    mlp_hidden_units: int = Field(default=64, ge=1)
    n_jobs: int = 1


def sample_positions(
    records: list[ProteinRecord],
    index_base: IndexBase,
    negative_ratio: int,
    seed: int,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    rng = np.random.default_rng(seed)
    sampled: list[tuple[int, int]] = []
    labels: list[int] = []

    for record_index, record in enumerate(records):
        record_labels = make_labels(record, index_base)
        positive_positions = np.flatnonzero(record_labels == 1)
        negative_positions = np.flatnonzero(record_labels == 0)
        if len(positive_positions) == 0:
            continue

        negative_count = min(len(negative_positions), len(positive_positions) * negative_ratio)
        sampled_negatives = rng.choice(negative_positions, size=negative_count, replace=False)

        for position in positive_positions:
            sampled.append((record_index, int(position)))
            labels.append(1)
        for position in sampled_negatives:
            sampled.append((record_index, int(position)))
            labels.append(0)

    order = rng.permutation(len(sampled))
    sampled = [sampled[index] for index in order]
    y = np.asarray([labels[index] for index in order], dtype=np.float32)
    return sampled, y


def build_matrix(records: list[ProteinRecord], positions: list[tuple[int, int]], radius: int) -> np.ndarray:
    x = np.zeros((len(positions), window_feature_size(radius)), dtype=np.float32)
    for row, (record_index, position) in enumerate(positions):
        x[row] = residue_window_features(records[record_index].sequence, position, radius)
    return x


def limit_training_sample(
    positions: list[tuple[int, int]],
    y: np.ndarray,
    max_train_positions: int | None,
    seed: int,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    if max_train_positions is None or len(positions) <= max_train_positions:
        return positions, y

    rng = np.random.default_rng(seed)
    selected = rng.choice(np.arange(len(positions)), size=max_train_positions, replace=False)
    return [positions[index] for index in selected], y[selected]


def train_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    config: TrainingConfig,
) -> tuple[LogisticRegressionSGD, list[str]]:
    model = LogisticRegressionSGD.initialize(x.shape[1], seed=config.seed)
    rng = np.random.default_rng(config.seed)
    positive_weight = max(1.0, config.negative_ratio / 2)
    training_log: list[str] = []

    for _ in range(config.epochs):
        order = rng.permutation(len(y))
        epoch_losses: list[float] = []
        for start in range(0, len(y), config.batch_size):
            batch = order[start : start + config.batch_size]
            loss = model.fit_batch(x[batch], y[batch], config.learning_rate, positive_weight=positive_weight)
            epoch_losses.append(loss)
        training_log.append(f"loss={float(np.mean(epoch_losses)):.4f}")

    return model, training_log


def train_random_forest(x: np.ndarray, y: np.ndarray, config: TrainingConfig) -> tuple[SklearnProbabilityModel, list[str]]:
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


def train_mlp(x: np.ndarray, y: np.ndarray, config: TrainingConfig) -> tuple[SklearnProbabilityModel, list[str]]:
    model = MLPClassifier(
        hidden_layer_sizes=(config.mlp_hidden_units,),
        max_iter=config.epochs,
        batch_size=config.batch_size,
        learning_rate_init=config.learning_rate,
        random_state=config.seed,
        early_stopping=True,
        n_iter_no_change=3,
    )
    model.fit(x, y.astype(np.int8))
    return SklearnProbabilityModel(estimator=model), [
        f"iterations={model.n_iter_}",
        f"loss={model.loss_:.4f}",
        f"hidden_units={config.mlp_hidden_units}",
    ]


def train_xgboost(x: np.ndarray, y: np.ndarray, config: TrainingConfig) -> tuple[SklearnProbabilityModel, list[str]]:
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


def train_baseline(
    records: list[ProteinRecord],
    index_base: IndexBase,
    config: TrainingConfig,
) -> tuple[ProbabilityModel, list[str]]:
    positions, y = sample_positions(records, index_base, config.negative_ratio, config.seed)
    positions, y = limit_training_sample(positions, y, config.max_train_positions, config.seed)
    x = build_matrix(records, positions, config.radius)

    if config.model_name == "logistic":
        return train_logistic_regression(x, y, config)
    if config.model_name == "random-forest":
        return train_random_forest(x, y, config)
    if config.model_name == "xgboost":
        return train_xgboost(x, y, config)
    if config.model_name == "mlp":
        return train_mlp(x, y, config)

    raise ValueError(f"Unsupported baseline model: {config.model_name}")


def evaluation_sample(
    records: list[ProteinRecord],
    index_base: IndexBase,
    radius: int,
    max_positions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_positions: list[tuple[int, int]] = []
    labels: list[int] = []
    for record_index, record in enumerate(records):
        record_labels = make_labels(record, index_base)
        for position, label in enumerate(record_labels):
            all_positions.append((record_index, position))
            labels.append(int(label))

    rng = np.random.default_rng(seed)
    if len(all_positions) > max_positions:
        selected = rng.choice(np.arange(len(all_positions)), size=max_positions, replace=False)
        positions = [all_positions[index] for index in selected]
        y = np.asarray([labels[index] for index in selected], dtype=np.int8)
    else:
        positions = all_positions
        y = np.asarray(labels, dtype=np.int8)
    x = build_matrix(records, positions, radius)
    return x, y
