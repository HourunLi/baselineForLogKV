"""Find offline dataset-cache entries that the evaluation's `datasets` cannot read.

Offline `load_dataset(path, name)` reuses the newest `<config>/<version>/<hash>` entry by directory
modification time (`datasets.packaged_modules.cache`). An entry prepared by `datasets>=4` stores list
columns as feature type `List` (and, from 4.7, mixed dict columns as `Json`); once it is the newest entry
of a config, `datasets<4` fails with "Feature type 'List' not found" (or 'Json') before any task or
model code runs. Reporting changes nothing;
`--quarantine` moves such entries aside (never deletes them) so the older readable entry is used again.
Exits with 1 while any dataset config would still fail offline loading.
"""

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import shutil
import sys

from .evaluate import configure_environment


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, help="default: the HF_DATASETS_CACHE used by chunkkv.evaluate")
    parser.add_argument("--quarantine", type=Path,
                        help="move unreadable entries here, for configs that keep a readable entry")
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
    print(f"datasets {datasets.__version__}, offline cache {root}")
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
            print("  no readable entry: prepare this config with the evaluation environment (datasets<4)")
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
