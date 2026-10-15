"""Read datasets 4.x List metadata using the existing offline datasets 3.x runtime."""

from contextlib import contextmanager


def legacy_list_metadata(value):
    """Preserve list-of-struct layout; legacy Sequence(dict) would transpose it."""
    if isinstance(value, list):
        return [legacy_list_metadata(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: legacy_list_metadata(item) for key, item in value.items()}
    if value.get("_type") != "List":
        return result
    feature = result["feature"]
    if isinstance(feature, dict) and not isinstance(feature.get("_type"), str):
        if result.get("length", -1) != -1:
            raise ValueError(
                "Cannot read fixed-length List of structs with datasets<4 without changing its schema; "
                "prepare this dataset in the evaluation environment."
            )
        return [feature]
    return {**result, "_type": "Sequence"}


@contextmanager
def compatible_list_features():
    """Adapt both dataset_info.json and Arrow feature metadata, without rewriting either."""
    import datasets

    if hasattr(datasets, "List"):
        yield
        return
    features = datasets.Features
    original = features.__dict__["from_dict"]

    @classmethod
    def from_dict(cls, dic):
        return original.__func__(cls, legacy_list_metadata(dic))

    features.from_dict = from_dict
    try:
        yield
    finally:
        features.from_dict = original
