from pathlib import Path

import numpy as np
import pytest

from features.embeddings import CachedProteinEmbedder, ProteinEmbedder
from features.embeddings import ProteinEmbeddingConfig, embedding_cache_path, normalize_sequence_for_embedding
from models.embedding_baseline import embedding_position_features


def test_normalize_sequence_for_embedding_replaces_nonstandard_symbols():
    assert normalize_sequence_for_embedding("ACDUBZ*") == "ACDXXXX"


def test_embedding_cache_path_includes_sanitized_model_name():
    config = ProteinEmbeddingConfig(
        model_name="facebook/esm2_t6_8M_UR50D",
        cache_dir=Path("cache"),
    )

    path = embedding_cache_path("P12345", "ACDE", config)

    assert path.parent == Path("cache/facebook_esm2_t6_8M_UR50D")
    assert path.name.startswith("P12345_")
    assert path.suffix == ".npz"


def test_cached_embedder_recomputes_corrupted_cache_file(tmp_path):
    config = ProteinEmbeddingConfig(cache_dir=tmp_path)
    path = embedding_cache_path("P12345", "ACDE", config)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    embedder = CachedProteinEmbedder(config)

    class DummyEmbedder:
        def embed_sequence(self, sequence: str) -> np.ndarray:
            return np.ones((len(sequence), 3), dtype=np.float32)

    embedder._embedder = DummyEmbedder()

    embedding = embedder.embed_record("P12345", "ACDE")

    assert embedding.shape == (4, 3)
    with np.load(path) as data:
        assert data["embedding"].shape == (4, 3)


def test_auto_device_uses_cuda_or_cpu_only(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    device = ProteinEmbedder.__new__(ProteinEmbedder)._resolve_device("auto")

    assert device == "cpu"


def test_embedding_position_features_concatenates_context_with_edge_padding():
    embeddings = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

    features = embedding_position_features(embeddings, position=0, context_radius=1, context_mode="concat")

    assert features.tolist() == [0, 0, 1, 2, 3, 4]


def test_embedding_position_features_averages_available_context():
    embeddings = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

    features = embedding_position_features(embeddings, position=1, context_radius=1, context_mode="mean")

    assert features.tolist() == [3, 4]
