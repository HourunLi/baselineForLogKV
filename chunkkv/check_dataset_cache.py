"""Find offline dataset-cache entries that the evaluation's `datasets` cannot read.

Offline `load_dataset(path, name)` reuses the newest `<config>/<version>/<hash>` entry by directory
modification time (`datasets.packaged_modules.cache`). An entry prepared by `datasets>=4` stores list
columns as feature type `List` (and, from 4.7, mixed dict columns as `Json`); once it is the newest entry
of a config, `datasets<4` fails with "Feature type 'List' not found" (or 'Json') before any task or
model code runs. Reporting changes nothing;
`--quarantine` moves such entries aside (never deletes them) so the older readable entry is used again.
With `--convert`, a config that has no readable entry gets a datasets 3.x copy of its selected entry
and the original stays in the quarantine. `List` and 3.x `Sequence`/`[...]` describe the same Arrow
types, so those columns keep their Arrow data and only the feature metadata is rewritten; `Json`
columns (JSON text) are decoded into the struct 3.x itself would store. Exits with 1 while any dataset
config would still fail offline loading.
"""

import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys

from .evaluate import configure_environment
from .train import sha256


class Unconvertible(ValueError):
    """The entry's Arrow data has no feature description in the installed `datasets`."""


def scan(root):
    """Entries per dataset config, oldest first, with the error the installed `datasets` raises on each."""
    from datasets import DatasetInfo

    configs = defaultdict(list)
    # <dataset>/<config>/<version>/<hash>; downloads/ holds raw files, never cache entries.
    found = (info.parent for dataset in Path(root).iterdir() if dataset.is_dir() and dataset.name != "downloads"
             for info in dataset.glob("*/*/*/dataset_info.json"))
    for entry in found:
        try:
            DatasetInfo.from_directory(str(entry))  # the call that fails inside the offline cache builder
            error = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}".splitlines()[0].split(". Available feature types")[0]
        configs[entry.parent.parent].append((entry.stat().st_mtime, entry, error))
    return {config: sorted(entries, key=lambda item: item[0]) for config, entries in configs.items()}


def _is_struct(feature):
    # The rule `generate_from_dict` uses: a dict without a string `_type` is a struct of named fields.
    return isinstance(feature, dict) and not isinstance(feature.get("_type"), str)


def legacy_features(feature, known):
    """Spell a datasets>=4 feature dict with the installed (3.x) types that give the same Arrow type."""
    if isinstance(feature, list):
        return [legacy_features(item, known) for item in feature]
    if _is_struct(feature):
        return {key: legacy_features(value, known) for key, value in feature.items()}
    kind = feature["_type"]
    if kind == "List":
        inner = legacy_features(feature["feature"], known)
        length = feature.get("length", -1)
        if _is_struct(inner):
            if length != -1:
                raise Unconvertible("fixed-length list of structs")
            return [inner]  # 3.x: a Python list of a struct; Sequence(struct) would be a struct of lists
        return {"feature": inner, **({"length": length} if length != -1 else {}), "_type": "Sequence"}
    if kind not in known:
        raise Unconvertible(f"feature type '{kind}'")
    if "feature" in feature:  # LargeList, Sequence
        return {**feature, "feature": legacy_features(feature["feature"], known)}
    return feature


def _contains_json(feature):
    if isinstance(feature, list):
        return any(_contains_json(item) for item in feature)
    if _is_struct(feature):
        return any(_contains_json(value) for value in feature.values())
    return feature["_type"] == "Json" or ("feature" in feature and _contains_json(feature["feature"]))


def _decode_json(feature, value):
    """A column value with every `Json` leaf (stored as JSON text) parsed back into Python objects."""
    if value is None:
        return None
    if isinstance(feature, list):
        return [_decode_json(feature[0], item) for item in value]
    if _is_struct(feature):
        return {key: _decode_json(sub, value.get(key)) for key, sub in feature.items()}
    if feature["_type"] == "Json":
        return json.loads(value) if isinstance(value, str) else value
    if "feature" in feature:
        return [_decode_json(feature["feature"], item) for item in value]
    return value


def _without_nulls(value):
    # 3.x stores decoded dicts as structs over every key seen, so an absent key reads back as None.
    if isinstance(value, dict):
        return {key: _without_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_nulls(item) for item in value]
    return value


def _read_table(path):
    import pyarrow as pa

    return pa.ipc.open_stream(pa.memory_map(str(path))).read_all()


def conversion_plan(entry):
    """3.x features for `entry` and its Arrow split files; raises Unconvertible if the data would change.

    Columns without `Json` keep their Arrow data and only get the 3.x spelling of their features.
    A `Json` column (datasets>=4.7 stores mixed-key dicts as JSON text) is decoded; its 3.x feature is
    the type 3.x itself gives such data, a struct over every key, which is what loading the source
    files with datasets 3.x produces.
    """
    import pyarrow as pa
    from datasets import Features
    from datasets.features.features import _FEATURE_TYPES

    info = json.loads((entry / "dataset_info.json").read_text(encoding="utf-8"))
    original = info["features"]
    json_columns = [name for name, feature in original.items() if _contains_json(feature)]
    features = {name: feature if name in json_columns else legacy_features(feature, set(_FEATURE_TYPES))
                for name, feature in original.items()}
    arrows = []
    for path in sorted(entry.iterdir()):
        if path.is_dir():
            raise Unconvertible(f"unexpected directory {path.name}")
        # cache-*.arrow hold map/filter results; datasets recomputes them when missing.
        if path.suffix == ".arrow" and not path.name.startswith("cache-"):
            arrows.append(path.name)
    if not arrows:
        raise Unconvertible("no Arrow split files")
    if json_columns:
        tables = [_read_table(entry / name) for name in arrows]
        for column in json_columns:
            try:
                values = [_decode_json(original[column], value)
                          for table in tables for value in table.column(column).to_pylist()]
                inferred = pa.array(values).type
            except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
                raise Unconvertible(f"decoded JSON of {column} has no single Arrow type: {exc}") from exc
            features[column] = Features.from_arrow_schema(pa.schema([pa.field(column, inferred)])).to_dict()[column]
    expected = Features.from_dict(features).arrow_schema.remove_metadata()
    for name in arrows:
        with pa.memory_map(str(entry / name)) as source:
            schema = pa.ipc.open_stream(source).schema.remove_metadata()
        if schema.names != expected.names or any(
                not schema.field(column).equals(expected.field(column))
                for column in schema.names if column not in json_columns):
            raise Unconvertible(f"{name} stores a different Arrow type than its translated features")
    return {"info": info, "original": original, "features": features, "arrows": arrows,
            "json_columns": json_columns}


def _with_features(schema, features):
    metadata = dict(schema.metadata or {})
    huggingface = json.loads(metadata.get(b"huggingface", b"{}"))
    huggingface.setdefault("info", {})["features"] = features
    metadata[b"huggingface"] = json.dumps(huggingface).encode()
    return metadata


def convert(source, destination, plan):
    """Write entry `source` to `destination` with 3.x features, then check that every row reads back."""
    import pyarrow as pa
    from datasets import Dataset, DatasetInfo, Features

    features, arrows, json_columns = plan["features"], plan["arrows"], plan["json_columns"]
    expected = Features.from_dict(features)
    destination.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        target = destination / path.name
        if path.name in arrows and not json_columns:
            with pa.memory_map(str(path)) as stream:
                reader = pa.ipc.open_stream(stream)
                schema = reader.schema.with_metadata(_with_features(reader.schema, features))
                with pa.OSFile(str(target), "wb") as sink, pa.ipc.new_stream(sink, schema) as writer:
                    for batch in reader:
                        writer.write_batch(pa.RecordBatch.from_arrays(batch.columns, schema=schema))
        elif path.name in arrows:
            table = _read_table(path)
            schema = expected.arrow_schema.with_metadata(_with_features(table.schema, features))
            columns = [pa.array([_decode_json(plan["original"][field.name], value)
                                 for value in table.column(field.name).to_pylist()], type=field.type)
                       if field.name in json_columns else table.column(field.name) for field in schema]
            with pa.OSFile(str(target), "wb") as sink, pa.ipc.new_stream(sink, schema) as writer:
                writer.write_table(pa.Table.from_arrays(columns, schema=schema))
        elif path.name == "dataset_info.json":
            target.write_text(json.dumps({**plan["info"], "features": features}, indent=2, sort_keys=True),
                              encoding="utf-8")
        elif path.suffix != ".arrow":
            shutil.copy2(path, target)
    if DatasetInfo.from_directory(str(destination)).features != expected:
        raise RuntimeError(f"Converted dataset_info.json in {destination} does not read back")
    for name in arrows:
        if Dataset.from_file(str(destination / name)).features != expected:
            raise RuntimeError(f"Converted {name} does not read back with the translated features")
        old, new = _read_table(source / name), _read_table(destination / name)
        unchanged = old.num_rows == new.num_rows and all(
            _without_nulls([_decode_json(plan["original"][column], value)
                            for value in old.column(column).to_pylist()])
            == _without_nulls(new.column(column).to_pylist())
            if column in json_columns else old.column(column).equals(new.column(column))
            for column in old.column_names)
        if not unchanged:
            raise RuntimeError(f"Converted {name} does not hold the original rows")
    return {name: {"original_sha256": sha256(source / name), "converted_sha256": sha256(destination / name)}
            for name in arrows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, help="default: the HF_DATASETS_CACHE used by chunkkv.evaluate")
    parser.add_argument("--quarantine", type=Path,
                        help="move unreadable entries here, for configs that keep a readable entry")
    parser.add_argument("--convert", action="store_true",
                        help="with --quarantine: give configs without a readable entry a datasets 3.x copy "
                             "of their selected entry (same Arrow data, rewritten feature metadata)")
    args = parser.parse_args()
    configure_environment()  # the evaluator's cache root, set before datasets reads it
    import datasets
    import datasets.config

    root = (args.cache_dir or Path(datasets.config.HF_DATASETS_CACHE)).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"No dataset cache at {root}")
    target = args.quarantine and args.quarantine.expanduser().resolve()
    if target and target.is_relative_to(root):
        parser.error("--quarantine must be outside the dataset cache")
    if args.convert and not target:
        parser.error("--convert keeps the original entries in --quarantine DIR")
    print(f"datasets {datasets.__version__}, offline cache {root}")
    blocked, unresolved, movable, conversions = 0, 0, [], []
    for config, entries in sorted(scan(root).items()):
        unreadable = [entry for _, entry, error in entries if error]
        if not unreadable:
            continue
        readable = [entry for _, entry, error in entries if not error]
        modified, newest, error = entries[-1]
        name = config.relative_to(root)
        if error:
            blocked += 1
            print(f"{name}: offline load selects {newest.relative_to(config)} "
                  f"(modified {datetime.fromtimestamp(modified):%Y-%m-%d %H:%M:%S}) -> {error}")
        else:
            print(f"{name}: newest entry is readable; {len(unreadable)} older unreadable entries")
        if readable:
            print(f"  readable entry kept: {readable[-1].relative_to(config)}")
            movable += unreadable
            continue
        try:
            if "Feature type" not in error:
                raise Unconvertible(error)
            plan = conversion_plan(newest)
        except (ValueError, KeyError, OSError) as exc:
            unresolved += 1
            print(f"  no readable entry, no offline conversion ({exc}): "
                  "prepare this config with the evaluation environment (datasets<4)")
            continue
        decoded = f"; decodes Json columns {plan['json_columns']}" if plan["json_columns"] else ""
        print(f"  no readable entry; convertible offline with --quarantine DIR --convert "
              f"(rows unchanged{decoded})")
        if args.convert:
            movable += unreadable
            conversions.append((newest, plan))
        else:
            unresolved += 1
    if target:
        for entry in movable:
            destination = target / entry.relative_to(root)
            if destination.exists():
                raise FileExistsError(f"Quarantine already holds {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(entry), str(destination))
            print(f"moved {entry} -> {destination}")
        log = target / "conversions.json"
        records = json.loads(log.read_text(encoding="utf-8")) if log.is_file() else []
        for entry, plan in conversions:
            original = target / entry.relative_to(root)
            # Build outside the cache so a half-written entry can never be selected.
            staging = target / ".converting" / entry.relative_to(root)
            files = convert(original, staging, plan)
            shutil.move(str(staging), str(entry))
            os.utime(entry)  # newest entry of its config, so offline loading selects it
            records.append({"entry": str(entry), "original": str(original), "datasets": datasets.__version__,
                            "converted_at": datetime.now().isoformat(timespec="seconds"),
                            "features": plan["features"], "decoded_json_columns": plan["json_columns"],
                            "files": files})
            log.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"converted {original} -> {entry}")
        staging_root = target / ".converting"
        if staging_root.is_dir() and not any(path.is_file() for path in staging_root.rglob("*")):
            shutil.rmtree(staging_root)  # only the emptied staging directories remain
        blocked = unresolved
    elif movable:
        print(f"{len(movable)} unreadable entries can be moved aside with --quarantine DIR")
    print(f"{blocked} dataset configs fail offline loading" if blocked
          else "Every dataset config selects an entry this datasets version can read")
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
