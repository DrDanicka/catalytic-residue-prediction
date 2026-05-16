from hashlib import sha256
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile
import zlib

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from tqdm.auto import tqdm


STANDARD_OR_UNKNOWN_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYX")
CACHE_READ_ERRORS = (EOFError, BadZipFile, OSError, ValueError, zlib.error)


class ProteinEmbeddingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str = "facebook/esm2_t6_8M_UR50D"
    device: str = "auto"
    cache_dir: Path = Path("data/embedding_cache")
    max_residues_per_chunk: int = Field(default=900, ge=1)
    show_progress: bool = True


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)


def sequence_hash(sequence: str) -> str:
    return sha256(sequence.encode("utf-8")).hexdigest()


def normalize_sequence_for_embedding(sequence: str) -> str:
    sequence = sequence.strip().upper()
    return "".join(amino_acid if amino_acid in STANDARD_OR_UNKNOWN_AMINO_ACIDS else "X" for amino_acid in sequence)


def embedding_cache_path(entry: str, sequence: str, config: ProteinEmbeddingConfig) -> Path:
    model_dir = config.cache_dir / sanitize_model_name(config.model_name)
    return model_dir / f"{entry}_{sequence_hash(sequence)[:16]}.npz"


class ProteinEmbedder:
    def __init__(self, config: ProteinEmbeddingConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.tokenizer, self.model = self._load_model(config.model_name)

    def _resolve_device(self, device: str) -> str:
        import torch

        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self, model_name: str):
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(self.device)
        model.eval()
        return tokenizer, model

    def embed_sequence(self, sequence: str) -> np.ndarray:
        normalized = normalize_sequence_for_embedding(sequence)
        chunks = [
            normalized[start : start + self.config.max_residues_per_chunk]
            for start in range(0, len(normalized), self.config.max_residues_per_chunk)
        ]
        chunk_iter = tqdm(
            chunks,
            desc="Embedding chunks",
            unit="chunk",
            leave=False,
            disable=not self.config.show_progress or len(chunks) <= 1,
        )
        chunk_embeddings = [self._embed_chunk(chunk) for chunk in chunk_iter]
        embeddings = np.concatenate(chunk_embeddings, axis=0).astype(np.float32)
        if embeddings.shape[0] != len(sequence):
            raise ValueError(f"Embedding length mismatch: got {embeddings.shape[0]}, expected {len(sequence)}")
        return embeddings

    def _embed_chunk(self, sequence: str) -> np.ndarray:
        import torch

        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        special_tokens_mask = inputs.pop("special_tokens_mask")[0].detach().cpu().numpy().astype(bool)
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state[0].detach().cpu().numpy()
        residue_embeddings = hidden[~special_tokens_mask]
        if residue_embeddings.shape[0] != len(sequence):
            residue_embeddings = hidden[1 : 1 + len(sequence)]
        if residue_embeddings.shape[0] != len(sequence):
            raise ValueError(f"Chunk embedding length mismatch: got {residue_embeddings.shape[0]}, expected {len(sequence)}")
        return residue_embeddings


class CachedProteinEmbedder:
    def __init__(self, config: ProteinEmbeddingConfig) -> None:
        self.config = config
        self._embedder: ProteinEmbedder | None = None

    def embed_record(self, entry: str, sequence: str) -> np.ndarray:
        path = embedding_cache_path(entry, sequence, self.config)
        if path.exists():
            try:
                with np.load(path) as data:
                    return data["embedding"].astype(np.float32)
            except CACHE_READ_ERRORS:
                path.unlink(missing_ok=True)

        path.parent.mkdir(parents=True, exist_ok=True)
        embedding = self.embedder.embed_sequence(sequence)
        self._write_cache_file(path, entry, sequence, embedding)
        return embedding

    def _write_cache_file(self, path: Path, entry: str, sequence: str, embedding: np.ndarray) -> None:
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
                temporary_path = Path(handle.name)
                np.savez_compressed(
                    handle,
                    embedding=embedding,
                    entry=entry,
                    sequence_hash=sequence_hash(sequence),
                    model_name=self.config.model_name,
                )
            temporary_path.replace(path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink(missing_ok=True)

    @property
    def embedder(self) -> ProteinEmbedder:
        if self._embedder is None:
            self._embedder = ProteinEmbedder(self.config)
        return self._embedder
