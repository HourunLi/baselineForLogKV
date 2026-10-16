"""Find offline dataset-cache entries that the evaluation's `datasets` cannot read, or undo a quarantine.

Offline `load_dataset(path, name)` reuses the newest `<config>/<version>/<hash>` entry by directory
modification time (`datasets.packaged_modules.cache`). An entry written by a newer `datasets` can use
feature types an older one lacks (datasets 4 writes `List`, 4.7 adds `Json`), and then offline task
loading fails with "Feature type ... not found" before any model call. That means the installed
`datasets` is older than the one that wrote the cache (see `EVAL_DATASETS_MIN`), not that the cache is
wrong. Reporting changes nothing. `--quarantine DIR` moves unreadable entries aside for
configs that keep a readable one; `--restore DIR` moves everything in DIR back to where it came from.
Nothing is ever deleted. Exits with 1 while any dataset config would still fail offline loading.
"""

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

from .evaluate import EVAL_DATASETS_MIN, configure_environment
from .train import sha256

# Earlier versions of this tool wrote converted copies of some entries and recorded them here.
CONVERSIONS = "conversions.json"
CONVERTED_COPIES = "_converted_copies"


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
    parser.add_argument("--cache-dir", type=Path, help="default: the HF_DATASETS_CACHE used by chunkkv.evaluate")
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
    if target and target.is_relative_to(root):
        parser.error("The quarantine directory must be outside the dataset cache")
    minimum = ".".join(map(str, EVAL_DATASETS_MIN))
    print(f"datasets {datasets.__version__} (the cache needs >={minimum}), offline cache {root}")
    if args.restore:
        if not target.is_dir():
            parser.error(f"No quarantine directory at {target}")
        print(f"restored {restore(root, target)} entries from {target}")
        target = None
    blocked, unresolved, movable = 0, 0, []
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
        else:
            unresolved += 1
            print(f"  no readable entry: datasets {datasets.__version__} is older than the version that wrote it")
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
    print(f"{blocked} dataset configs fail offline loading" if blocked
          else "Every dataset config selects an entry this datasets version can read")
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
