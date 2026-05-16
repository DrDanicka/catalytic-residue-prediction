import numpy as np

from features.amino_acids import (
    PAD_TOKEN,
    UNK_TOKEN,
    VOCAB_INDEX,
    amino_acid_properties,
)


def window_feature_size(radius: int) -> int:
    window_length = radius * 2 + 1
    return window_length * len(VOCAB_INDEX) + 1 + 6


def residue_window_features(sequence: str, position: int, radius: int) -> np.ndarray:
    features = np.zeros(window_feature_size(radius), dtype=np.float32)
    offset = 0
    vocab_size = len(VOCAB_INDEX)

    for window_position in range(position - radius, position + radius + 1):
        if window_position < 0 or window_position >= len(sequence):
            token = PAD_TOKEN
        else:
            amino_acid = sequence[window_position]
            token = amino_acid if amino_acid in VOCAB_INDEX else UNK_TOKEN
        features[offset + VOCAB_INDEX[token]] = 1.0
        offset += vocab_size

    if len(sequence) > 1:
        features[offset] = position / (len(sequence) - 1)
    offset += 1
    features[offset : offset + 6] = amino_acid_properties(sequence[position])
    return features
