"""Native feature support and isolated transform caches for offline evaluation."""

from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version


def require_dataset_runtime():
    instruction = (
        "ChunkKV evaluation requires datasets==4.8.4 and pyarrow>=21 for native List/Json caches. "
        "Install with: python -m pip install -e '.[chunkkv-eval]'"
    )
    try:
        supported = version("datasets") == "4.8.4" and int(version("pyarrow").split(".")[0]) >= 21
    except PackageNotFoundError as exc:
        raise RuntimeError(instruction) from exc
    if not supported:
        raise RuntimeError(instruction)


@contextmanager
def isolated_dataset_transforms():
    import datasets

    # Keep load_dataset's offline source cache; map/filter use per-process temp files.
    caching_enabled = datasets.is_caching_enabled()
    datasets.disable_caching()
    try:
        yield
    finally:
        if caching_enabled:
            datasets.enable_caching()
