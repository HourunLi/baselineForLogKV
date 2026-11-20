"""Read PPL inputs on mounts that store symlinks as ordinary path-text files."""
from __future__ import annotations

import os
from pathlib import Path


def file_state(path):
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        head = stream.read(128)
        stream.seek(max(0, size - 4))
        footer = stream.read(4)
    return size, head, footer


def describe(path, state=None):
    size, head, footer = state if state is not None else file_state(path)
    return f"path={Path(path).resolve()} size={size} head={head!r} footer={footer!r}"


def resolve_input(path, report=None):
    """Follow real or flattened links read-only, without changing a shared file from eval ranks."""
    source = Path(path).expanduser()
    current = Path(os.path.abspath(source))
    seen = set()
    for _ in range(32):
        if current in seen:
            raise ValueError(f"PPL input link cycle: input={source}, repeated target={current}")
        seen.add(current)
        if current.is_symlink():
            target = os.readlink(current)
        else:
            try:
                state = file_state(current)
            except OSError as error:
                raise FileNotFoundError(f"PPL input missing or unreadable: input={source}, target={current}; "
                                        "prepare ordinary files with unused/fetch_long_eval_data.py") from error
            if report is not None:
                report(f"[PPL input] {describe(current, state)}")
            size, _, _ = state
            if size > 4096:
                return current
            try:
                text = current.read_bytes().decode("utf-8").strip(" \t\r\n\0")
            except UnicodeDecodeError:
                return current
            if not (text.startswith(("/", "./", "../")) and "\n" not in text and "\0" not in text):
                return current
            target = text
        following = Path(target)
        current = Path(os.path.abspath(following if following.is_absolute() else current.parent / following))
        if report is not None:
            report(f"[PPL link] input={source} target={current}")
    raise ValueError(f"PPL input has more than 32 links: {source}")


def whole_parquet(path):
    try:
        size, head, footer = file_state(path)
        return size >= 12 and head[:4] == footer == b"PAR1"
    except OSError:
        return False


def open_parquet(path):
    import pyarrow.parquet as pq

    def report(message):
        print(message, flush=True)

    try:
        actual = resolve_input(path, report=report)
    except (OSError, ValueError) as error:
        raise ValueError(f"{error}. Set LONG_PPL_DATA/LONG_PPL_BOOKS to the ordinary folders printed by "
                         "unused/fetch_long_eval_data.py; evaluation does not download missing data.") from error
    state = file_state(actual)
    size, head, footer = state
    if not (size >= 12 and head[:4] == footer == b"PAR1"):
        raise ValueError(f"PPL input is not a whole Parquet file: input={path}; {describe(actual, state)}. "
                         "Expected PAR1 at both ends. Redownload/materialize with unused/fetch_long_eval_data.py "
                         "and export its ordinary LONG_PPL_DATA/LONG_PPL_BOOKS folders.")
    try:
        return pq.ParquetFile(actual)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot open PPL Parquet: input={path}; {describe(actual, state)}; "
                         "run unused/fetch_long_eval_data.py to verify/repair the source") from error


def is_hub_snapshot(path):
    parts = Path(path).expanduser().parts
    return any(part == "snapshots" and i > 0 and parts[i - 1].startswith("datasets--")
               for i, part in enumerate(parts))
