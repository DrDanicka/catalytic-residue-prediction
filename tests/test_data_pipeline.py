from data.labeling import infer_index_base, make_labels
from data.load import ProteinRecord
from data.splits import split_records
from data.validate import validate_records


def test_infers_zero_based_when_zero_annotation_exists():
    records = [ProteinRecord(entry="p1", sequence="ACDE", residues=(0, 2))]

    assert infer_index_base(records) == "zero"
    assert make_labels(records[0], "zero").tolist() == [1, 0, 1, 0]


def test_validation_counts_out_of_bounds_after_index_base_conversion():
    records = [ProteinRecord(entry="p1", sequence="ACDE", residues=(0, 4))]

    report = validate_records(records)

    assert report.inferred_index_base == "zero"
    assert report.out_of_bounds_annotations == 1


def test_split_records_is_stable():
    records = [ProteinRecord(entry=f"p{index}", sequence="ACDE", residues=(0,)) for index in range(20)]

    first = split_records(records, seed=7)
    second = split_records(records, seed=7)

    assert first == second
