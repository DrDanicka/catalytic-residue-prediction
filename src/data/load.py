import csv
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator


class ProteinRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    entry: str
    sequence: str
    residues: tuple[int, ...]

    @field_validator("entry")
    @classmethod
    def validate_entry(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("entry cannot be empty")
        return value

    @field_validator("sequence")
    @classmethod
    def normalize_sequence(cls, value: str) -> str:
        value = value.strip().upper()
        if not value:
            raise ValueError("sequence cannot be empty")
        return value


def parse_residue_indices(value: str) -> tuple[int, ...]:
    value = value.strip()
    if not value:
        return ()
    return tuple(int(part) for part in value.split("|") if part != "")


def load_records(csv_path: str | Path) -> list[ProteinRecord]:
    path = Path(csv_path)
    records: list[ProteinRecord] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Entry", "Sequence", "Residue"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"Expected CSV columns {sorted(required)}, got {reader.fieldnames}")

        for row_number, row in enumerate(reader, start=2):
            entry = (row.get("Entry") or "").strip()
            sequence = (row.get("Sequence") or "").strip().upper()
            residue_text = row.get("Residue") or ""
            try:
                residues = parse_residue_indices(residue_text)
            except ValueError as exc:
                raise ValueError(f"Invalid Residue value at CSV row {row_number}: {residue_text!r}") from exc
            try:
                records.append(ProteinRecord(entry=entry, sequence=sequence, residues=residues))
            except ValidationError as exc:
                raise ValueError(f"Invalid protein record at CSV row {row_number}") from exc
    return records
