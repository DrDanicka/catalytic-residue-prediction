from typing import Literal

import numpy as np

from data.load import ProteinRecord

IndexBase = Literal["auto", "zero", "one"]


def infer_index_base(records: list[ProteinRecord]) -> Literal["zero", "one"]:
    has_zero = any(index == 0 for record in records for index in record.residues)
    if has_zero:
        return "zero"
    return "one"


def residue_indices_to_zero_based(
    record: ProteinRecord,
    index_base: Literal["zero", "one"],
) -> tuple[int, ...]:
    if index_base == "zero":
        return record.residues
    return tuple(index - 1 for index in record.residues)


def make_labels(record: ProteinRecord, index_base: Literal["zero", "one"]) -> np.ndarray:
    labels = np.zeros(len(record.sequence), dtype=np.int8)
    for index in residue_indices_to_zero_based(record, index_base):
        if 0 <= index < len(labels):
            labels[index] = 1
    return labels
