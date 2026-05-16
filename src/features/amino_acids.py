AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
VOCAB = [PAD_TOKEN, UNK_TOKEN, *AMINO_ACIDS]
VOCAB_INDEX = {token: index for index, token in enumerate(VOCAB)}

HYDROPHOBIC = set("AILMFWYV")
POLAR = set("STNQCY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
AROMATIC = set("FWYH")
SMALL = set("ACDGSTV")


def amino_acid_properties(amino_acid: str) -> tuple[float, ...]:
    return (
        float(amino_acid in HYDROPHOBIC),
        float(amino_acid in POLAR),
        float(amino_acid in POSITIVE),
        float(amino_acid in NEGATIVE),
        float(amino_acid in AROMATIC),
        float(amino_acid in SMALL),
    )
