from collections.abc import Iterable
import hashlib

from data.load import ProteinRecord


def stable_fraction(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / 2**64


def split_records(
    records: Iterable[ProteinRecord],
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 13,
) -> tuple[list[ProteinRecord], list[ProteinRecord], list[ProteinRecord]]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1")

    train: list[ProteinRecord] = []
    validation: list[ProteinRecord] = []
    test: list[ProteinRecord] = []
    validation_cutoff = train_fraction + validation_fraction

    for record in records:
        fraction = stable_fraction(record.entry, seed)
        if fraction < train_fraction:
            train.append(record)
        elif fraction < validation_cutoff:
            validation.append(record)
        else:
            test.append(record)
    return train, validation, test
