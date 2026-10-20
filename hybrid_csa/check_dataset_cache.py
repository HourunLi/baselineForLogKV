"""Find offline dataset-cache entries the evaluation's `datasets` cannot read, or undo a quarantine.

Offline `load_dataset(path, name)` parses `dataset_info.json` of every `<config>/<version>/<hash>` entry
(`datasets.packaged_modules.cache`), then reuses the newest by directory modification time. Two kinds of
entry therefore fail it:

* `dataset_info.json` missing or not valid JSON. This breaks loading of its config whatever the entry's
  age. datasets writes the file as one line; a multi-line one was written by another tool (earlier
  versions of this tool wrote multi-line converted copies). Such entries are reported and never moved.
* feature types the installed `datasets` lacks (datasets 4 writes `List`, 4.7 adds `Json`). This matters
  only for the newest entry and means the installed `datasets` is older than the one that wrote the cache
  (see `EVAL_DATASETS_MIN`), not that the cache is wrong.

Reporting changes nothing. `--compare DIR` relates broken entries to the same paths in quarantine
directory DIR and checks each conversion DIR records against the recorded SHA-256 of the original files.
`--quarantine DIR` moves entries of the second kind aside for configs that keep a readable one;
`--restore DIR` moves everything in DIR back to where it came from. Nothing is ever deleted. Exits with 1
while any dataset config would still fail offline loading or a recorded conversion lacks its original files.
"""

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

from .evaluate import EVAL_DATASETS_MIN, configure_environment
from .train import sha256

# Earlier versions of this tool wrote converted copies of some entries and recorded them here.
CONVERSIONS = "conversions.json"
CONVERTED_COPIES = "_converted_copies"


def read_problem(entry):
    """None if the installed `datasets` reads `entry`, else ("json" | "features", message); see above."""
    from datasets import DatasetInfo

    try:
        json.loads((entry / "dataset_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return "json", f"{type(exc).__name__}: {exc}"
    try:
        DatasetInfo.from_directory(str(entry))  # the call that fails inside the offline cache builder
    except Exception as exc:
        return "features", f"{type(exc).__name__}: {exc}".splitlines()[0].split(". Available feature types")[0]
    return None


def scan(root):
    """Entries per dataset config, oldest first, with why the installed `datasets` cannot read each."""
    configs = defaultdict(list)
    # Every <dataset>/<config>/<version>/<hash> directory, as the cache builder globs them, with or without
    # dataset_info.json; downloads/ holds raw files, never cache entries.
    for dataset in Path(root).iterdir():
        if dataset.is_dir() and dataset.name != "downloads":
            for entry in dataset.glob("*/*/*"):
                if entry.is_dir():
                    configs[entry.parent.parent].append((entry.stat().st_mtime, entry, read_problem(entry)))
    return {config: sorted(entries, key=lambda item: item[0]) for config, entries in configs.items()}


def _facts(path):
    data = path.read_bytes()
    lines = data.count(b"\n") + 1
    modified = datetime.fromtimestamp(path.stat().st_mtime)
    return data, (f"{len(data)} bytes, {lines} lines, modified {modified:%Y-%m-%d %H:%M:%S}, "
                  f"sha256 {hashlib.sha256(data).hexdigest()[:16]}")


def _valid_json(data):
    try:
        json.loads(data)
    except ValueError:
        return False
    return True


def describe(root, entry, compare=None):
    """Facts about an entry whose dataset_info.json does not parse, and the same path under `compare`."""
    info = entry / "dataset_info.json"
    if not info.is_file():
        return [f"no dataset_info.json; the directory holds {sorted(path.name for path in entry.iterdir())[:8]}"]
    data, facts = _facts(info)
    lines = [f"dataset_info.json: {facts} (datasets writes it as one line)"]
    text = data.decode("utf-8", "replace")
    try:
        json.loads(text)
    except ValueError as exc:
        position = getattr(exc, "pos", len(text))
        lines.append(f"text before the error: {text[max(0, position - 80):position]!r}")
    if compare:
        relative = entry.relative_to(root)
        for other in (compare / relative, compare / CONVERTED_COPIES / relative):
            path = other / "dataset_info.json"
            if path.is_file():
                other_data, other_facts = _facts(path)
                relation = ("identical to the broken file" if other_data == data
                            else f"the broken file is its first {len(data)} bytes" if other_data.startswith(data)
                            else "different from the broken file")
                validity = "valid" if _valid_json(other_data) else "not valid"
                lines.append(f"{path}: {other_facts}, {validity} JSON; {relation}")
    return lines


def check_conversions(root, compare):
    """Compare each conversion recorded in `compare` with the cache; returns how many lack their originals."""
    log = compare / CONVERSIONS
    if not log.is_file():
        print(f"{compare}: no {CONVERSIONS}, so no recorded conversions to check")
        return 0
    problems = 0
    for record in json.loads(log.read_text(encoding="utf-8")):
        relative = None
        for path, base in ((record["entry"], root), (record["original"], compare)):
            try:
                relative = Path(path).relative_to(base)
                break
            except ValueError:
                continue
        if relative is None:
            print(f"conversion {record['entry']}: neither under {root} nor recorded under {compare}")
            problems += 1
            continue
        entry = root / relative
        states = []
        for name, hashes in record["files"].items():
            digest = sha256(entry / name) if (entry / name).is_file() else None
            states.append("original" if digest == hashes["original_sha256"]
                          else "converted copy" if digest == hashes["converted_sha256"]
                          else "missing" if digest is None else "matches neither original nor copy")
        info = entry / "dataset_info.json"
        data = info.read_bytes() if info.is_file() else None
        # The original was written by datasets, as one line; the converted copy has one line per key.
        lines = None if data is None else data.strip().count(b"\n") + 1
        one_line = lines == 1 and _valid_json(data)
        described = ("missing" if data is None else
                     f"{'valid' if _valid_json(data) else 'NOT valid'} JSON, {lines} lines")
        files = ", ".join(f"{name} {state}" for name, state in zip(record["files"], states))
        restored = one_line and all(state == "original" for state in states)
        print(f"conversion {relative}: {files}; dataset_info.json {described}"
              + ("" if restored else " <- not the original entry"))
        problems += not restored
    return problems


def restore(root, quarantine):
    """Move every quarantined entry back; a converted copy in its place goes to DIR/_converted_copies."""
    log = quarantine / CONVERSIONS
    converted = {record["entry"]: record for record in json.loads(log.read_text(encoding="utf-8"))} \
        if log.is_file() else {}
    entries = sorted(info.parent for info in quarantine.glob("*/*/*/*/dataset_info.json")
                     if not info.relative_to(quarantine).parts[0].startswith(("_", ".")))
    moves = []
    # Check every destination first, so a refusal leaves the cache exactly as it was.
    for entry in entries:
        relative = entry.relative_to(quarantine)
        destination = root / relative
        copy = None
        if destination.exists():
            record = converted.get(str(destination))
            if record is None or any(sha256(destination / name) != hashes["converted_sha256"]
                                     for name, hashes in record["files"].items()):
                raise FileExistsError(f"{destination} exists and is not a recorded converted copy; nothing moved")
            copy = quarantine / CONVERTED_COPIES / relative
            if copy.exists():
                raise FileExistsError(f"{copy} already exists; nothing moved")
        moves.append((entry, destination, copy))
    for entry, destination, copy in moves:
        if copy is not None:
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(destination), str(copy))
            print(f"moved converted copy {destination} -> {copy}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(entry), str(destination))  # a rename keeps the entry's own modification time
        print(f"restored {entry} -> {destination}")
    return len(moves)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, help="default: the HF_DATASETS_CACHE used by hybrid_csa.evaluate")
    parser.add_argument("--compare", type=Path,
                        help="a quarantine directory: relate broken entries to the same paths in it and check "
                             "every conversion it records against the original files")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--quarantine", type=Path,
                        help="move unreadable entries here, for configs that keep a readable entry")
    action.add_argument("--restore", type=Path, help="move all entries quarantined in this directory back")
    args = parser.parse_args()
    configure_environment()  # the evaluator's cache root, set before datasets reads it
    import datasets
    import datasets.config

    root = (args.cache_dir or Path(datasets.config.HF_DATASETS_CACHE)).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"No dataset cache at {root}")
    target = (args.quarantine or args.restore) and (args.quarantine or args.restore).expanduser().resolve()
    compare = args.compare and args.compare.expanduser().resolve()
    if any(directory and directory.is_relative_to(root) for directory in (target, compare)):
        parser.error("The quarantine directory must be outside the dataset cache")
    if compare and not compare.is_dir():
        parser.error(f"No quarantine directory at {compare}")
    minimum = ".".join(map(str, EVAL_DATASETS_MIN))
    print(f"datasets {datasets.__version__} (the cache needs >={minimum}), offline cache {root}")
    if args.restore:
        if not target.is_dir():
            parser.error(f"No quarantine directory at {target}")
        print(f"restored {restore(root, target)} entries from {target}")
        target = None
    blocked, unresolved, movable = 0, 0, []
    for config, entries in sorted(scan(root).items()):
        name = config.relative_to(root)
        broken = [entry for _, entry, problem in entries if problem and problem[0] == "json"]
        for _, entry, problem in entries:
            if problem and problem[0] == "json":
                print(f"{name}: {entry.relative_to(config)} -> {problem[1]}")
                for line in describe(root, entry, compare):
                    print(f"  {line}")
        if broken:
            print("  datasets parses dataset_info.json of every entry of this config before choosing one, so "
                  "loading it fails until this entry is repaired; --quarantine leaves it in place")
        # Among entries whose dataset_info.json parses, offline loading selects the newest.
        usable = [(modified, entry, problem and problem[1]) for modified, entry, problem in entries
                  if not (problem and problem[0] == "json")]
        unreadable = [entry for _, entry, error in usable if error]
        readable = [entry for _, entry, error in usable if not error]
        if unreadable:
            modified, newest, error = usable[-1]
            if error:
                print(f"{name}: offline load selects {newest.relative_to(config)} "
                      f"(modified {datetime.fromtimestamp(modified):%Y-%m-%d %H:%M:%S}) -> {error}")
            else:
                print(f"{name}: newest entry is readable; {len(unreadable)} older unreadable entries")
            if readable:
                print(f"  readable entry kept: {readable[-1].relative_to(config)}")
                movable += unreadable
            else:
                print(f"  no readable entry: datasets {datasets.__version__} is older than the version that wrote it")
        blocked += bool(broken or unreadable and usable[-1][2])
        unresolved += bool(broken or unreadable and not readable)
    if target:
        for entry in movable:
            destination = target / entry.relative_to(root)
            if destination.exists():
                raise FileExistsError(f"Quarantine already holds {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(entry), str(destination))
            print(f"moved {entry} -> {destination}")
        blocked = unresolved
    elif movable:
        print(f"{len(movable)} unreadable entries can be moved aside with --quarantine DIR")
    problems = check_conversions(root, compare) if compare else 0
    print(f"{blocked} dataset configs fail offline loading" if blocked
          else "Every dataset config selects an entry this datasets version can read")
    if problems:
        print(f"{problems} recorded conversions do not hold the original entry")
    return 1 if blocked or problems else 0


if __name__ == "__main__":
    sys.exit(main())
