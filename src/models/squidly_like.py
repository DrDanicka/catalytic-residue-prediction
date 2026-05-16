from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from tqdm.auto import tqdm

from data.labeling import make_labels
from data.load import ProteinRecord
from features.embeddings import CachedProteinEmbedder, ProteinEmbeddingConfig
from models.baseline import IndexBase, limit_training_sample, sample_positions
from models.embedding_baseline import evaluation_positions, unique_record_indices


class SquidlyLikeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    embedding_model: str = "facebook/esm2_t6_8M_UR50D"
    embedding_cache_dir: Path = Path("data/embedding_cache")
    device: str = "auto"
    max_residues_per_chunk: int = Field(default=900, ge=1)
    negative_ratio: int = Field(default=10, ge=1)
    max_contrastive_positions: int = Field(default=50_000, ge=2)
    contrastive_epochs: int = Field(default=3, ge=0)
    contrastive_batches_per_epoch: int = Field(default=500, ge=1)
    contrastive_batch_size: int = Field(default=512, ge=2)
    projection_dim: int = Field(default=128, ge=1)
    projection_hidden_dim: int = Field(default=512, ge=1)
    projection_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    lstm_hidden_dim: int = Field(default=128, ge=1)
    lstm_layers: int = Field(default=2, ge=1)
    lstm_dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    classifier_epochs: int = Field(default=5, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)
    max_train_sequences: int = Field(default=1000, ge=1)
    max_eval_positions: int = Field(default=50_000, ge=1)
    max_sequence_length: int = Field(default=1500, ge=1)
    positive_weight: float = Field(default=100.0, gt=0)
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


@dataclass(frozen=True)
class ResidueExample:
    record_index: int
    position: int
    label: int
    amino_acid: str


def resolve_torch_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def create_projection_model(input_dim: int, config: SquidlyLikeConfig) -> Any:
    import torch.nn as nn

    return nn.Sequential(
        nn.Dropout(config.projection_dropout),
        nn.Linear(input_dim, config.projection_hidden_dim),
        nn.ReLU(),
        nn.Linear(config.projection_hidden_dim, max(config.projection_hidden_dim // 2, config.projection_dim)),
        nn.ReLU(),
        nn.Linear(max(config.projection_hidden_dim // 2, config.projection_dim), config.projection_dim),
    )


def create_lstm_classifier(config: SquidlyLikeConfig) -> Any:
    import torch.nn as nn

    class ResidueBiLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dropout = config.lstm_dropout if config.lstm_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=config.projection_dim,
                hidden_size=config.lstm_hidden_dim,
                num_layers=config.lstm_layers,
                dropout=dropout,
                bidirectional=True,
                batch_first=True,
            )
            self.output = nn.Linear(config.lstm_hidden_dim * 2, 1)

        def forward(self, x):
            encoded, _ = self.lstm(x)
            return self.output(encoded).squeeze(-1)

    return ResidueBiLSTM()


def residue_examples(
    records: list[ProteinRecord],
    positions: list[tuple[int, int]],
    y: np.ndarray,
) -> list[ResidueExample]:
    return [
        ResidueExample(
            record_index=record_index,
            position=position,
            label=int(label),
            amino_acid=records[record_index].sequence[position],
        )
        for (record_index, position), label in zip(positions, y, strict=True)
    ]


def build_residue_embedding_matrix(
    records: list[ProteinRecord],
    examples: list[ResidueExample],
    embedder: CachedProteinEmbedder,
    show_progress: bool,
) -> np.ndarray:
    embeddings_by_record: dict[int, np.ndarray] = {}
    record_indices = unique_record_indices([(example.record_index, example.position) for example in examples])
    for record_index in tqdm(record_indices, desc="contrastive: embeddings", unit="protein", disable=not show_progress):
        record = records[record_index]
        embeddings_by_record[record_index] = embedder.embed_record(record.entry, record.sequence)

    first = embeddings_by_record[examples[0].record_index]
    x = np.zeros((len(examples), first.shape[1]), dtype=np.float32)
    for row, example in enumerate(tqdm(examples, desc="contrastive: matrix", unit="pos", disable=not show_progress)):
        x[row] = embeddings_by_record[example.record_index][example.position]
    return x


def pair_indices(examples: list[ResidueExample], batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_label: dict[int, list[int]] = {0: [], 1: []}
    by_label_aa: dict[tuple[int, str], list[int]] = {}
    amino_acids: dict[str, dict[int, list[int]]] = {}
    for index, example in enumerate(examples):
        by_label[example.label].append(index)
        by_label_aa.setdefault((example.label, example.amino_acid), []).append(index)
        amino_acids.setdefault(example.amino_acid, {0: [], 1: []})[example.label].append(index)

    left: list[int] = []
    right: list[int] = []
    pair_labels: list[int] = []
    half = batch_size // 2

    for _ in range(half):
        label = int(rng.integers(0, 2))
        anchor_index = int(rng.choice(by_label[label]))
        anchor = examples[anchor_index]
        candidates = by_label_aa.get((label, anchor.amino_acid), by_label[label])
        if len(candidates) > 1:
            candidate = anchor_index
            while candidate == anchor_index:
                candidate = int(rng.choice(candidates))
        else:
            candidate = int(rng.choice(by_label[label]))
        left.append(anchor_index)
        right.append(candidate)
        pair_labels.append(1)

    hard_negative_amino_acids = [aa for aa, groups in amino_acids.items() if groups[0] and groups[1]]
    for _ in range(batch_size - half):
        if hard_negative_amino_acids:
            amino_acid = str(rng.choice(hard_negative_amino_acids))
            negative = int(rng.choice(amino_acids[amino_acid][0]))
            positive = int(rng.choice(amino_acids[amino_acid][1]))
        else:
            negative = int(rng.choice(by_label[0]))
            positive = int(rng.choice(by_label[1]))
        left.append(negative)
        right.append(positive)
        pair_labels.append(-1)

    order = rng.permutation(len(left))
    return (
        np.asarray(left, dtype=np.int64)[order],
        np.asarray(right, dtype=np.int64)[order],
        np.asarray(pair_labels, dtype=np.float32)[order],
    )


def train_contrastive_projection(
    records: list[ProteinRecord],
    index_base: IndexBase,
    embedder: CachedProteinEmbedder,
    config: SquidlyLikeConfig,
    torch_device: str,
) -> tuple[Any, list[str]]:
    import torch
    import torch.nn.functional as functional

    torch.set_grad_enabled(True)
    positions, y = sample_positions(records, index_base, config.negative_ratio, config.seed)
    positions, y = limit_training_sample(positions, y, config.max_contrastive_positions, config.seed)
    examples = residue_examples(records, positions, y)
    x = build_residue_embedding_matrix(records, examples, embedder, config.show_progress)
    torch.set_grad_enabled(True)
    projection = create_projection_model(x.shape[1], config).to(torch_device)

    if config.contrastive_epochs == 0:
        return projection, ["contrastive_skipped=true"]

    optimizer = torch.optim.AdamW(projection.parameters(), lr=config.learning_rate)
    x_tensor = torch.from_numpy(x).to(torch_device)
    rng = np.random.default_rng(config.seed)
    training_log: list[str] = []

    for epoch in range(config.contrastive_epochs):
        losses: list[float] = []
        iterator = range(config.contrastive_batches_per_epoch)
        for _ in tqdm(iterator, desc=f"contrastive epoch {epoch + 1}", unit="batch", disable=not config.show_progress):
            left, right, pair_y = pair_indices(examples, config.contrastive_batch_size, rng)
            left_projection = projection(x_tensor[left])
            right_projection = projection(x_tensor[right])
            target = torch.from_numpy(pair_y).to(torch_device)
            loss = functional.cosine_embedding_loss(left_projection, right_projection, target, margin=0.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        training_log.append(f"contrastive_epoch_{epoch + 1}_loss={float(np.mean(losses)):.4f}")

    return projection, training_log


def selected_sequence_indices(records: list[ProteinRecord], max_sequences: int, max_sequence_length: int, seed: int) -> list[int]:
    candidates = [index for index, record in enumerate(records) if len(record.sequence) <= max_sequence_length]
    rng = np.random.default_rng(seed)
    if len(candidates) <= max_sequences:
        return candidates
    selected = rng.choice(np.asarray(candidates), size=max_sequences, replace=False)
    return [int(index) for index in selected]


def project_sequence(embedding: np.ndarray, projection: Any, torch_device: str) -> Any:
    import torch

    with torch.no_grad():
        x = torch.from_numpy(embedding.astype(np.float32)).to(torch_device)
        return projection(x).detach()


def train_lstm_classifier(
    records: list[ProteinRecord],
    index_base: IndexBase,
    embedder: CachedProteinEmbedder,
    projection: Any,
    config: SquidlyLikeConfig,
    torch_device: str,
) -> tuple[Any, list[str]]:
    import torch
    import torch.nn as nn

    torch.set_grad_enabled(True)
    classifier = create_lstm_classifier(config).to(torch_device)
    optimizer = torch.optim.AdamW(list(projection.parameters()) + list(classifier.parameters()), lr=config.learning_rate)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.positive_weight, device=torch_device))
    sequence_indices = selected_sequence_indices(records, config.max_train_sequences, config.max_sequence_length, config.seed)
    training_log: list[str] = []

    projection.train()
    classifier.train()
    for epoch in range(config.classifier_epochs):
        losses: list[float] = []
        iterator = tqdm(sequence_indices, desc=f"lstm epoch {epoch + 1}", unit="seq", disable=not config.show_progress)
        for record_index in iterator:
            record = records[record_index]
            labels = make_labels(record, index_base).astype(np.float32)
            embedding = embedder.embed_record(record.entry, record.sequence)
            torch.set_grad_enabled(True)
            projected = projection(torch.from_numpy(embedding).to(torch_device)).unsqueeze(0)
            targets = torch.from_numpy(labels).to(torch_device).unsqueeze(0)
            logits = classifier(projected)
            loss = loss_function(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        training_log.append(f"lstm_epoch_{epoch + 1}_loss={float(np.mean(losses)):.4f}")
    return classifier, training_log


def evaluate_squidly_like(
    records: list[ProteinRecord],
    index_base: IndexBase,
    embedder: CachedProteinEmbedder,
    projection: Any,
    classifier: Any,
    config: SquidlyLikeConfig,
    torch_device: str,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    import torch

    positions, y = evaluation_positions(records, index_base, config.max_eval_positions, config.seed)
    record_indices = unique_record_indices(positions)
    scores_by_record: dict[int, np.ndarray] = {}
    projection.eval()
    classifier.eval()
    with torch.no_grad():
        for record_index in tqdm(record_indices, desc="squidly-like: validation", unit="protein", disable=not config.show_progress):
            record = records[record_index]
            embedding = embedder.embed_record(record.entry, record.sequence)
            projected = project_sequence(embedding, projection, torch_device).unsqueeze(0)
            logits = classifier(projected).squeeze(0)
            scores_by_record[record_index] = torch.sigmoid(logits).detach().cpu().numpy()

    scores = np.asarray([scores_by_record[record_index][position] for record_index, position in positions], dtype=np.float32)
    return positions, y, scores


def train_squidly_like(
    train_records: list[ProteinRecord],
    validation_records: list[ProteinRecord],
    index_base: IndexBase,
    config: SquidlyLikeConfig,
) -> tuple[list[str], list[tuple[int, int]], np.ndarray, np.ndarray, str]:
    torch_device = resolve_torch_device(config.device)
    embedder = CachedProteinEmbedder(config.embedding_config())
    projection, contrastive_log = train_contrastive_projection(train_records, index_base, embedder, config, torch_device)
    classifier, lstm_log = train_lstm_classifier(train_records, index_base, embedder, projection, config, torch_device)
    positions, y, scores = evaluate_squidly_like(validation_records, index_base, embedder, projection, classifier, config, torch_device)
    return contrastive_log + lstm_log, positions, y, scores, torch_device
