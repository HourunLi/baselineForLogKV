"""Shared Hugging Face / RULER cache settings for offline evaluation.

The cache root, offline flags and RULER JSON locations match SemanticLogKV's
evaluator so both methods read the same task data. HF libraries capture these
variables when first imported, so ``configure_hf_environment`` must run before
``datasets``, ``transformers`` or ``lm_eval`` is imported.

RULER QA tasks fetch two JSON files by URL; ``install_ruler_cache`` serves them
from ``$RULER_CACHE_DIR`` (same file names as the reference cache) and fails
clearly when offline and missing. TLS verification is left untouched.
``install_dataset_alias_fallback`` reads the same shared dataset cache entries
when a mount cannot resolve the ``namespace___name`` alias.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_HF_CACHE_BASE = "/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache"

RULER_URL_FILES = {
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json": "dev-v2.0.json",
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json": "hotpot_dev_distractor_v1.json",
}


def configure_hf_environment(base: str = DEFAULT_HF_CACHE_BASE, offline: bool = True) -> dict:
    values = dict(
        HF_HOME=base, HF_DATASETS_CACHE=f"{base}/hf_cache", HF_EVALUATE_CACHE=f"{base}/evaluate",
        HF_MODULES_CACHE=f"{base}/modules", HUGGINGFACE_HUB_CACHE=f"{base}/hub", HF_HUB_CACHE=f"{base}/hub",
        RULER_CACHE_DIR=f"{base}/ruler_cache", NLTK_DATA=f"{base}/nltk_data",
        HF_HUB_OFFLINE="1" if offline else "0", HF_DATASETS_OFFLINE="1" if offline else "0",
        HF_DATASETS_IN_MEMORY_MAX_SIZE="0", HF_DATASETS_TRUST_REMOTE_CODE="1",
        TOKENIZERS_PARALLELISM="false", HF_ALLOW_CODE_EVAL="1",
    )
    os.environ.update(values)
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    return values


class _CachedResponse:
    status_code = 200
    headers = {"Content-Type": "application/json"}

    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


def _cached_json(url: str):
    name = RULER_URL_FILES.get(url)
    if name is None:
        return None
    path = Path(os.environ.get("RULER_CACHE_DIR", "")) / name
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    if os.environ.get("HF_DATASETS_OFFLINE") == "1":
        raise RuntimeError(f"Offline evaluation needs {name} in {path.parent}")
    return None


def install_ruler_cache() -> None:
    """Serve known RULER URLs from the local cache for every ``requests`` call path."""
    import requests

    if getattr(requests, "_kv_baseline_ruler_cache", False):
        return
    original_request, original_get = requests.Session.request, requests.get

    def request(self, method, url, **kwargs):
        data = _cached_json(url)
        return _CachedResponse(data) if data is not None else original_request(self, method, url, **kwargs)

    def get(url, **kwargs):
        data = _cached_json(url)
        return _CachedResponse(data) if data is not None else original_get(url, **kwargs)

    requests.Session.request = request
    requests.get = get
    requests._kv_baseline_ruler_cache = True


def follow_dataset_cache_alias(original_load):
    """Wrap ``datasets.load_dataset`` for shared offline caches whose aliases do not resolve.

    Some mounts keep ``<cache>/<namespace>___<name>`` as a link to the legacy
    ``<cache>/<name>`` directory that this process cannot follow. When an
    offline ``namespace/name`` load misses, load the same cache entry under the
    legacy name instead; every other failure is re-raised unchanged.
    """
    def load_dataset(*args, **kwargs):
        try:
            return original_load(*args, **kwargs)
        except (ConnectionError, FileNotFoundError):
            import datasets.config as config
            from datasets.naming import camelcase_to_snakecase
            from datasets.packaged_modules import _PACKAGED_DATASETS_MODULES

            path = args[0] if args else kwargs.get("path")
            if not (config.HF_DATASETS_OFFLINE and isinstance(path, str) and path.count("/") == 1):
                raise
            namespace, name = path.split("/")
            root = Path(kwargs.get("cache_dir") or config.HF_DATASETS_CACHE).expanduser()
            alias = root / f"{namespace}___{camelcase_to_snakecase(name)}"
            target = root / camelcase_to_snakecase(name)
            try:
                is_alias = (os.path.lexists(alias) and target.is_dir()
                            and (not alias.is_dir() or os.path.samefile(alias, target)))
            except OSError:
                is_alias = False
            if not is_alias or name in _PACKAGED_DATASETS_MODULES or os.path.exists(name):
                raise
            print(f"[hf-cache] {path}: cache alias {alias} is unusable, loading {target} as {name!r}", flush=True)
            if args:
                return original_load(name, *args[1:], **kwargs)
            return original_load(**dict(kwargs, path=name))

    load_dataset._kv_baseline_alias = True
    return load_dataset


def install_dataset_alias_fallback() -> None:
    import datasets

    if not getattr(datasets.load_dataset, "_kv_baseline_alias", False):
        datasets.load_dataset = follow_dataset_cache_alias(datasets.load_dataset)
