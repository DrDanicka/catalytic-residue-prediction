from typing import Literal

from pydantic import BaseModel, ConfigDict

from data.labeling import infer_index_base, residue_indices_to_zero_based
from data.load import ProteinRecord

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


class DatasetReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    protein_count: int
    total_residue_count: int
    positive_count: int
    min_sequence_length: int
    mean_sequence_length: float
    max_sequence_length: int
    max_annotations_per_protein: int
    inferred_index_base: Literal["zero", "one"]
    zero_index_annotations: int
    out_of_bounds_annotations: int
    nonstandard_sequence_count: int
    duplicate_entry_count: int


def validate_records(records: list[ProteinRecord]) -> DatasetReport:
    if not records:
        raise ValueError("No records loaded")

    index_base = infer_index_base(records)
    lengths = [len(record.sequence) for record in records]
    entries = [record.entry for record in records]
    zero_index_annotations = sum(index == 0 for record in records for index in record.residues)
    out_of_bounds = 0
    nonstandard_sequence_count = 0

    for record in records:
        if any(amino_acid not in STANDARD_AMINO_ACIDS for amino_acid in record.sequence):
            nonstandard_sequence_count += 1
        for index in residue_indices_to_zero_based(record, index_base):
            if index < 0 or index >= len(record.sequence):
                out_of_bounds += 1

    return DatasetReport(
        protein_count=len(records),
        total_residue_count=sum(lengths),
        positive_count=sum(len(record.residues) for record in records),
        min_sequence_length=min(lengths),
        mean_sequence_length=sum(lengths) / len(lengths),
        max_sequence_length=max(lengths),
        max_annotations_per_protein=max(len(record.residues) for record in records),
        inferred_index_base=index_base,
        zero_index_annotations=zero_index_annotations,
        out_of_bounds_annotations=out_of_bounds,
        nonstandard_sequence_count=nonstandard_sequence_count,
        duplicate_entry_count=len(entries) - len(set(entries)),
    )
