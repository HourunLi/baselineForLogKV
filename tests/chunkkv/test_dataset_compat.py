import hashlib
import json

import pytest

from chunkkv.dataset_compat import compatible_list_features


datasets = pytest.importorskip("datasets")
pa = pytest.importorskip("pyarrow")
pytestmark = pytest.mark.skipif(hasattr(datasets, "List"), reason="Regression targets datasets<4")


def test_new_list_cache_metadata_reads_without_rewriting(tmp_path):
    string = {"dtype": "string", "_type": "Value"}
    integer = {"dtype": "int64", "_type": "Value"}
    modern_features = {
        "texts": {"feature": string, "id": "text-list", "_type": "List"},
        "records": {"feature": {"text": string, "_type": integer}, "_type": "List"},
        "nested": {"feature": {"feature": string, "_type": "List"}, "_type": "List"},
        "fixed": {"feature": integer, "length": 2, "id": "pair", "_type": "List"},
        "labels": {
            "feature": {"names": ["no", "yes"], "_type": "ClassLabel"},
            "_type": "List",
        },
    }
    expected = datasets.Features({
        "texts": datasets.Sequence(datasets.Value("string"), id="text-list"),
        "records": [{"text": datasets.Value("string"), "_type": datasets.Value("int64")}],
        "nested": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
        "fixed": datasets.Sequence(datasets.Value("int64"), length=2, id="pair"),
        "labels": datasets.Sequence(datasets.ClassLabel(names=["no", "yes"])),
    })
    rows = [
        {
            "texts": ["alpha", "beta"],
            "records": [{"text": "record", "_type": 3}],
            "nested": [["nested"], []],
            "fixed": [1, 2],
            "labels": [0, 1],
        },
        {"texts": [], "records": [], "nested": None, "fixed": [3, 4], "labels": []},
    ]
    info_path = tmp_path / "dataset_info.json"
    info_path.write_text(json.dumps({"features": modern_features}), encoding="utf-8")
    arrow_path = tmp_path / "data.arrow"
    schema = expected.arrow_schema.with_metadata({
        b"huggingface": json.dumps({"info": {"features": modern_features}}).encode("utf-8")
    })
    table = pa.Table.from_pylist(rows, schema=schema)
    with pa.OSFile(str(arrow_path), "wb") as sink:
        with pa.ipc.new_stream(sink, schema) as writer:
            writer.write_table(table)
    original_hashes = {path: hashlib.sha256(path.read_bytes()).digest() for path in (info_path, arrow_path)}
    original_descriptor = datasets.Features.__dict__["from_dict"]

    with pytest.raises(ValueError, match="Feature type 'List' not found"):
        datasets.DatasetInfo.from_directory(str(tmp_path))
    with pytest.raises(ValueError, match="Feature type 'List' not found"):
        datasets.Dataset.from_file(str(arrow_path))

    with compatible_list_features():
        assert datasets.Features.from_dict(dic=modern_features) == expected
        info = datasets.DatasetInfo.from_directory(str(tmp_path))
        # Exercise both embedded Arrow metadata alone and the DatasetInfo path.
        for dataset in (
            datasets.Dataset.from_file(str(arrow_path)),
            datasets.Dataset.from_file(str(arrow_path), info=info),
        ):
            assert dataset.to_list() == rows
            assert dataset.features == expected
            assert dataset.features.arrow_schema.equals(expected.arrow_schema, check_metadata=True)
        assert info.features == expected
        assert info.features["labels"].feature.names == ["no", "yes"]
        assert info.features["fixed"].length == 2
        assert info.features["fixed"].id == "pair"

    assert datasets.Features.__dict__["from_dict"] is original_descriptor
    assert {path: hashlib.sha256(path.read_bytes()).digest() for path in original_hashes} == original_hashes


@pytest.mark.parametrize(
    ("feature", "message"),
    [
        ({"_type": "UnknownFeature"}, "Feature type 'UnknownFeature' not found"),
        (
            {"feature": {"text": {"dtype": "string", "_type": "Value"}}, "length": 2, "_type": "List"},
            "Cannot read fixed-length List of structs",
        ),
    ],
)
def test_unsupported_metadata_errors_and_restores_descriptor(feature, message):
    original_descriptor = datasets.Features.__dict__["from_dict"]
    with pytest.raises(ValueError, match=message):
        with compatible_list_features():
            datasets.Features.from_dict({"column": feature})
    assert datasets.Features.__dict__["from_dict"] is original_descriptor
    assert datasets.Features.from_dict({"column": {"dtype": "string", "_type": "Value"}}) == {
        "column": datasets.Value("string")
    }
