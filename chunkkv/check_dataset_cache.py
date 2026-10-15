"""Find offline dataset-cache entries that the evaluation's `datasets` cannot read.

Offline `load_dataset(path, name)` reuses the newest `<config>/<version>/<hash>` entry by directory
modification time (`datasets.packaged_modules.cache`). An entry prepared by `datasets>=4` stores list
columns as feature type `List` (and, from 4.7, mixed dict columns as `Json`); once it is the newest entry
of a config, `datasets<4` fails with "Feature type 'List' not found" (or 'Json') before any task or
model code runs. Reporting changes nothing;
`--quarantine` moves such entries aside (never deletes them) so the older readable entry is used again.
With `--convert`, a config that has no readable entry gets a datasets 3.x copy of its selected entry:
the Arrow data is identical (`List` and `Sequence`/`[...]` describe the same Arrow types), only the
feature metadata is rewritten, and the original stays in the quarantine. Exits with 1 while any
dataset config would still fail offline loading.
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


def conversion_plan(entry):
    """Translated features and Arrow split files of `entry`; raises Unconvertible if the data differs."""
    import pyarrow as pa
    from datasets import Features
    from datasets.features.features import _FEATURE_TYPES

    info = json.loads((entry / "dataset_info.json").read_text(encoding="utf-8"))
    features = legacy_features(info["features"], set(_FEATURE_TYPES))
    expected = Features.from_dict(features).arrow_schema.remove_metadata()
    arrows = []
    for path in sorted(entry.iterdir()):
        if path.is_dir():
            raise Unconvertible(f"unexpected directory {path.name}")
        # cache-*.arrow hold map/filter results; datasets recomputes them when missing.
        if path.suffix != ".arrow" or path.name.startswith("cache-"):
            continue
        with pa.memory_map(str(path)) as source:
            schema = pa.ipc.open_stream(source).schema
        if not schema.remove_metadata().equals(expected):
            raise Unconvertible(f"{path.name} stores a different Arrow type than its translated features")
        arrows.append(path.name)
    if not arrows:
        raise Unconvertible("no Arrow split files")
    return info, features, arrows


def convert(source, destination, info, features, arrows):
    """Copy entry `source` to `destination` with 3.x feature metadata, then check every batch is unchanged."""
    import pyarrow as pa
    from datasets import Dataset, DatasetInfo, Features

    destination.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        target = destination / path.name
        if path.name in arrows:
            with pa.memory_map(str(path)) as stream:
                reader = pa.ipc.open_stream(stream)
                metadata = dict(reader.schema.metadata or {})
                huggingface = json.loads(metadata.get(b"huggingface", b"{}"))
                huggingface.setdefault("info", {})["features"] = features
                metadata[b"huggingface"] = json.dumps(huggingface).encode()
                schema = reader.schema.with_metadata(metadata)
                with pa.OSFile(str(target), "wb") as sink, pa.ipc.new_stream(sink, schema) as writer:
                    for batch in reader:
                        writer.write_batch(pa.RecordBatch.from_arrays(batch.columns, schema=schema))
        elif path.name == "dataset_info.json":
            target.write_text(json.dumps({**info, "features": features}, indent=2, sort_keys=True), encoding="utf-8")
        elif path.suffix != ".arrow":
            shutil.copy2(path, target)
    expected = Features.from_dict(features)
    if DatasetInfo.from_directory(str(destination)).features != expected:
        raise RuntimeError(f"Converted dataset_info.json in {destination} does not read back")
    for name in arrows:
        if Dataset.from_file(str(destination / name)).features != expected:
            raise RuntimeError(f"Converted {name} does not read back with the translated features")
        with pa.memory_map(str(source / name)) as old, pa.memory_map(str(destination / name)) as new:
            old_batches, new_batches = list(pa.ipc.open_stream(old)), list(pa.ipc.open_stream(new))
            if len(old_batches) != len(new_batches) or not all(
                    a.equals(b, check_metadata=False) for a, b in zip(old_batches, new_batches)):
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
        print("  no readable entry; convertible offline with --quarantine DIR --convert (Arrow data unchanged)")
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
            files = convert(original, staging, *plan)
            shutil.move(str(staging), str(entry))
            os.utime(entry)  # newest entry of its config, so offline loading selects it
            records.append({"entry": str(entry), "original": str(original), "datasets": datasets.__version__,
                            "converted_at": datetime.now().isoformat(timespec="seconds"),
                            "features": plan[1], "files": files})
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
