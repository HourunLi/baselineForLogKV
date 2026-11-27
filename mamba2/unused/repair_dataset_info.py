"""Rebuild a truncated ``dataset_info.json`` in an offline Hugging Face datasets cache.

Offline, ``datasets`` reads the ``dataset_info.json`` of every cached copy of the
requested config; one that is cut short (an interrupted copy or upload) breaks
the load with ``JSONDecodeError: Expecting property name enclosed in double
quotes``. eval.py names such files (``[hf-cache] {"event": "dataset_info_unreadable", ...}``).

This tool keeps every complete value the truncated file still holds and drops
the one being written at the cut. What is missing comes from the cache directory
itself: split row counts, bytes and shard lengths from the arrow files next to
it, features from their schema metadata. Other lost keys come from a readable
sibling config of the same dataset (``--template`` to choose one); lost download
records (unused offline) become null. A recovered row count that disagrees with
the arrow files stops the repair. It writes nothing without ``--write``; with it,
the original is kept as ``dataset_info.json.truncated``.

    python unused/repair_dataset_info.py <cache>/cais___mmlu/world_religions/0.0.0/<hash>/dataset_info.json
    python unused/repair_dataset_info.py <same path> --write --check
    python unused/repair_dataset_info.py --scan /home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/hf_cache \\
        [--write --check]   # every unreadable file under the cache (majob.sh / eval.sh: REPAIR_HF_CACHE=1)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Keys that describe one config's data; never copied from another config.
_CONFIG_KEYS = {"config_name", "splits", "dataset_size", "size_in_bytes", "download_checksums", "download_size"}


def close_truncated(text):
    """The longest prefix of `text` that ends after a complete value, closed into a JSON object.

    Returns (object, kept characters, depth). `depth` counts the containers still
    open at the cut: the last member at each of those levels is the value that
    was being written, so it may be incomplete. Raises ValueError when nothing
    usable remains.
    """
    stack, cuts, in_string, escaped = [], [], False, False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack:
                break
            stack.pop()
            if not stack:
                return json.loads(text[:i + 1]), i + 1, 0
        elif ch == "," and stack:
            # Everything before this comma is complete at this depth.
            cuts.append((i, "".join(reversed(stack))))
    for cut, closers in reversed(cuts):
        try:
            value = json.loads(text[:cut] + closers)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, cut, len(closers)
    raise ValueError("no complete value survives in the truncated file")


def _last_key(container):
    return next(reversed(container)) if isinstance(container, dict) and container else None


def arrow_splits(directory, dataset_name):
    """({split: (rows per shard, bytes, sharded)}, features) from ``<dataset_name>-<split>[-SSSSS-of-NNNNN].arrow``.

    `features` is what ``datasets`` stored in the arrow schema metadata (None if absent).
    """
    import pyarrow as pa

    pattern = re.compile(re.escape(dataset_name) + r"-(?P<split>.+?)(?:-(?P<shard>\d{5})-of-(?P<total>\d{5}))?\.arrow")
    shards, features = {}, None
    for path in sorted(Path(directory).iterdir()):
        match = pattern.fullmatch(path.name)
        if match:
            with pa.memory_map(str(path)) as source:
                table = pa.ipc.open_stream(source).read_all()
            index = int(match["shard"]) if match["shard"] else 0
            shards.setdefault(match["split"], []).append((index, table.num_rows, table.nbytes, match["total"]))
            metadata = (table.schema.metadata or {}).get(b"huggingface")
            if metadata and features is None:
                features = json.loads(metadata).get("info", {}).get("features")
    out = {}
    for split, parts in shards.items():
        parts.sort()
        total = parts[0][3]
        if total is not None and (len(parts) != int(total) or [p[0] for p in parts] != list(range(int(total)))):
            raise ValueError(f"split {split!r} is missing shards: found {[p[0] for p in parts]} of {total}")
        out[split] = ([p[1] for p in parts], sum(p[2] for p in parts), total is not None)
    return out, features


def _same_features(a, b):
    if a is None or b is None:
        return False
    try:
        from datasets import Features

        return Features.from_dict(a) == Features.from_dict(b)
    except ImportError:
        return a == b
    except Exception:  # A malformed spelling is simply not the same.
        return False


def sibling_template(path):
    """A readable dataset_info.json of another config of the same dataset, same version if possible."""
    path = Path(path)
    version, dataset_root = path.parent.parent.name, path.parent.parent.parent.parent
    candidates = sorted(dataset_root.glob("*/*/*/dataset_info.json"),
                        key=lambda p: (p.parent.parent.name != version, str(p)))
    for candidate in candidates:
        if candidate.parent == path.parent:
            continue
        try:
            return candidate, json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return None, None


def rebuild(path, template_path=None):
    """(repaired info, report lines) for the truncated dataset_info.json at `path`."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        json.loads(text)
        return None, [f"{path}: valid JSON, nothing to repair"]
    except json.JSONDecodeError as exc:
        report = [f"{path}: {exc} ({len(text)} characters)"]
    info, kept, depth = close_truncated(text)
    # The member being written at the cut may be incomplete: keep a partial
    # `splits` (its entries are checked against the arrow files below) and drop
    # anything else, which is then rebuilt like a lost key.
    partial = _last_key(info) if depth >= 2 else None
    if partial is not None and partial != "splits":
        del info[partial]
        report.append(f"dropped the incomplete {partial!r}")
    elif partial == "splits" and depth >= 4:
        entry = info["splits"][_last_key(info["splits"])]
        field = _last_key(entry)
        if field is not None:
            del entry[field]
    report.append(f"kept {kept} of {len(text)} characters; complete keys: {sorted(info)}")
    config = path.parent.parent.parent.name
    if info.setdefault("config_name", config) != config:
        raise ValueError(f"config_name {info['config_name']!r} does not match the cache directory {config!r}")
    if template_path is not None:
        template = json.loads(Path(template_path).read_text(encoding="utf-8"))
    else:
        template_path, template = sibling_template(path)
    if template is not None:
        report.append(f"template for lost keys: {template_path}")
    dataset_name = info.get("dataset_name") or (template or {}).get("dataset_name")
    if not dataset_name:
        dataset_name = path.parent.parent.parent.parent.name.split("___")[-1]
    on_disk, features = arrow_splits(path.parent, dataset_name)
    if not on_disk:
        raise ValueError(f"no {dataset_name}-<split>.arrow files next to {path}")
    if features is not None:
        if "features" not in info:
            # The sibling's spelling when it means the same features (as a complete file would read).
            same = template is not None and _same_features(template.get("features"), features)
            info["features"] = template["features"] if same else features
            report.append("features: from the arrow schema metadata")
        elif not _same_features(info["features"], features):
            report.append("warning: recovered features differ from the arrow schema metadata")

    splits = info.get("splits")
    if not isinstance(splits, dict):
        splits = info["splits"] = {}
    missing = sorted(set(splits) - set(on_disk))
    if missing:
        raise ValueError(f"splits {missing} are described but have no arrow files")
    # Recovered splits keep their order; lost ones follow the template's order.
    known = list(splits) + [name for name in (template or {}).get("splits") or {} if name not in splits]
    order = {name: i for i, name in enumerate(known)}
    on_disk = dict(sorted(on_disk.items(), key=lambda item: (order.get(item[0], len(order)), item[0])))
    for split, (lengths, nbytes, sharded) in on_disk.items():
        entry = splits.setdefault(split, {})
        rows = sum(lengths)
        if entry.get("num_examples") is not None and entry["num_examples"] != rows:
            raise ValueError(f"split {split!r}: recovered num_examples {entry['num_examples']} != {rows} rows on disk")
        filled = [key for key in ("name", "num_bytes", "num_examples", "shard_lengths", "dataset_name")
                  if key not in entry]
        entry.setdefault("name", split)
        entry.setdefault("num_bytes", nbytes)
        entry["num_examples"] = rows
        entry.setdefault("shard_lengths", lengths if sharded else None)
        entry.setdefault("dataset_name", dataset_name)
        if filled:
            report.append(f"split {split}: filled {filled} from the arrow files ({rows} rows)")

    lost = [key for key in ("download_checksums", "download_size") if key not in info]
    for key in lost:
        info[key] = None
    if lost:
        report.append(f"{lost}: lost, set to null (download records, unused offline)")
    if "dataset_size" not in info:
        info["dataset_size"] = sum(entry["num_bytes"] for entry in info["splits"].values())
        report.append("dataset_size: sum of split bytes")
    if "size_in_bytes" not in info:
        info["size_in_bytes"] = info["dataset_size"] + (info.get("download_size") or 0)
        report.append("size_in_bytes: dataset_size + download_size")
    copied = []
    for key, value in (template or {}).items():
        if key not in info and key not in _CONFIG_KEYS:
            info[key] = value
            copied.append(key)
    if copied:
        report.append(f"copied from the template: {copied}")
    if template is not None:
        info = {key: info[key] for key in [k for k in template if k in info] + [k for k in info if k not in template]}
    try:
        import datasets

        datasets.DatasetInfo.from_dict(info)
    except ImportError:
        report.append("datasets is not installed; the result was only checked as JSON")
    return info, report


def scan(root):
    for path in sorted(Path(root).glob("*/*/*/*/dataset_info.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            yield path, exc


def check(path):
    """Load the repaired config offline from its cache root, as eval.py does."""
    path = Path(path).resolve()
    hash_dir = path.parent
    config, dataset_dir, root = hash_dir.parent.parent.name, hash_dir.parent.parent.parent, hash_dir.parents[3]
    import datasets

    if not datasets.config.HF_DATASETS_OFFLINE:
        raise RuntimeError("set HF_DATASETS_OFFLINE=1 before datasets is imported (main() does)")
    name = "/".join(dataset_dir.name.split("___"))
    loaded = datasets.load_dataset(name, config, cache_dir=str(root))
    return name, config, {split: len(part) for split, part in loaded.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="truncated dataset_info.json files")
    parser.add_argument("--scan", metavar="CACHE_ROOT",
                        help="find unreadable dataset_info.json files under a datasets cache and repair them too")
    parser.add_argument("--template", help="readable dataset_info.json of a sibling config for keys the prefix lost")
    parser.add_argument("--write", action="store_true", help="replace the file (the original becomes *.truncated)")
    parser.add_argument("--check", action="store_true", help="after --write, load the config offline from the cache")
    args = parser.parse_args()
    if args.check:
        # Before anything imports datasets: the check must read the cache, never the Hub.
        os.environ["HF_DATASETS_OFFLINE"] = os.environ["HF_HUB_OFFLINE"] = "1"
    paths = list(args.paths)
    if args.scan:
        found = list(scan(args.scan))
        for path, exc in found:
            print(f"{path}: {type(exc).__name__}: {exc}")
        print(f"{len(found)} unreadable dataset_info.json under {args.scan}")
        paths += [str(path) for path, _ in found if str(path) not in paths]
    status = 0
    for path in paths:
        try:
            info, report = rebuild(path, args.template)
        except (OSError, ValueError) as exc:
            print(f"{path}: cannot repair: {exc}", file=sys.stderr)
            status = 1
            continue
        print("\n".join(report))
        if info is None:
            continue
        text = json.dumps(info, indent=2, ensure_ascii=False)
        if not args.write:
            print(text)
            print("dry run: nothing written (add --write)")
            continue
        path = Path(path)
        backup = path.with_name(path.name + ".truncated")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        temporary = path.with_name(path.name + ".repairing")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != info:
            raise RuntimeError(f"{path} does not read back as written")
        print(f"wrote {path} ({len(text)} characters); original kept as {backup}")
        if args.check:
            try:
                name, config, rows = check(path)
            except Exception as exc:  # Report and go on; the exit status says it failed.
                print(f"{path}: offline load still fails: {type(exc).__name__}: {exc}", file=sys.stderr)
                status = 1
                continue
            print(f"offline load_dataset({name!r}, {config!r}): {rows}")
    return status


if __name__ == "__main__":
    sys.exit(main())
