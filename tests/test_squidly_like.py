import numpy as np

from data.load import ProteinRecord
from models.squidly_like import ResidueExample, pair_indices, selected_sequence_indices


def test_pair_indices_contains_positive_and_negative_pairs():
    examples = [
        ResidueExample(record_index=0, position=0, label=0, amino_acid="H"),
        ResidueExample(record_index=0, position=1, label=0, amino_acid="H"),
        ResidueExample(record_index=1, position=0, label=1, amino_acid="H"),
        ResidueExample(record_index=1, position=1, label=1, amino_acid="H"),
    ]

    _, _, labels = pair_indices(examples, batch_size=8, rng=np.random.default_rng(1))

    assert set(labels.tolist()) == {-1.0, 1.0}


def test_selected_sequence_indices_respects_max_sequence_length():
    records = [
        ProteinRecord(entry="short", sequence="ACDE", residues=(0,)),
        ProteinRecord(entry="long", sequence="ACDEFGHIK", residues=(0,)),
    ]

    selected = selected_sequence_indices(records, max_sequences=10, max_sequence_length=4, seed=1)

    assert selected == [0]

