"""Offline ``datasets`` loads skip the ``<hash>.incomplete`` folders an unfinished download leaves.

Offline (HF_DATASETS_OFFLINE=1), ``datasets`` finds a cached config by globbing
``<cache>/<namespace>___<name>/<config>/<version>/*`` and reading the ``dataset_info.json`` of every match
(``datasets.packaged_modules.cache.cache._find_hash_in_cache``). That glob also matches ``<hash>.incomplete``,
the folder a builder fills before moving it to ``<hash>``; the file is written only at the end. While a
download is still running (the OBS bucket refuses renames, so the final move copies the folder and then deletes
it, which takes a while), or after one was killed, every offline load of the config fails with
``FileNotFoundError: .../<hash>.incomplete/dataset_info.json``, whether ``<hash>`` is complete or missing.
Online loads never take such a folder for a finished cache; with this patch the offline lookup skips it as
well. When nothing else is cached for the config, the error names the unfinished folders. Applied by
models.cam.evaluate before any task loads data.

Copied from SemanticLogKV (HourunLi/LogKV niahLogKV@4615501, ``litgpt/hf_cache_patch.py``, Apache-2.0); only
this docstring and the hint in the error message differ. Both projects read the same offline cache.
"""
from __future__ import annotations

import glob
import inspect
import os
import types

SUFFIX = ".incomplete"


def finished(paths) -> list:
    """``paths`` without the folders a builder is still filling (or left unfinished)."""
    return [p for p in paths if not os.fspath(p).rstrip("/").endswith(SUFFIX)]


def leftovers(dataset_name: str, config_id: str | None = None, cache_dir: str | None = None) -> list[str]:
    """The ``.incomplete`` folders in ``dataset_name``'s offline cache (of ``config_id`` only, if given)."""
    import datasets.config
    from datasets.naming import camelcase_to_snakecase

    parts = dataset_name.split("/")
    parts[-1] = camelcase_to_snakecase(parts[-1])
    root = os.path.join(os.path.expanduser(str(cache_dir or datasets.config.HF_DATASETS_CACHE)), "___".join(parts))
    return sorted(p for p in glob.glob(os.path.join(root, config_id or "*", "*", "*" + SUFFIX)) if os.path.isdir(p))


def apply_patch() -> None:
    try:
        from datasets.packaged_modules.cache import cache
    except ImportError:  # a datasets release without the offline cache builder
        return
    if getattr(cache, "_skips_incomplete", False):
        return
    find = cache._find_hash_in_cache
    signature = inspect.signature(find)
    # The module uses glob only to list cached folders; the copy leaves out unfinished ones.
    cache.glob = types.SimpleNamespace(**vars(glob))
    cache.glob.glob = lambda *args, **kwargs: finished(glob.glob(*args, **kwargs))

    def find_finished(*args, **kwargs):
        try:
            return find(*args, **kwargs)
        except ValueError as error:  # "Couldn't find cache for <dataset> for config <config>"
            call = signature.bind(*args, **kwargs).arguments
            name, config = call.get("dataset_name"), call.get("config_name")
            config_kwargs, features = call.get("config_kwargs"), call.get("custom_features")
            config_id = None
            if config or config_kwargs or features:  # as datasets names the config's folder
                import datasets

                config_id = datasets.BuilderConfig(config or "default").create_config_id(
                    config_kwargs=config_kwargs or {}, custom_features=features)
            left = leftovers(name, config_id, call.get("cache_dir")) if isinstance(name, str) else []
            if not left:
                raise
            raise ValueError(
                f"{error}\n{name}" + (f" config '{config_id}'" if config_id else "")
                + " is not completely downloaded: the cache holds only the unfinished folder(s) "
                + ", ".join(left) + " of a download still running or one that was interrupted. Wait for the running "
                "one, or download it again online (SemanticLogKV's data download for the long-context evals, "
                "unused/fetch_long_eval_data.py there); a folder no download is writing may be deleted."
            ) from error

    cache._find_hash_in_cache = find_finished
    cache._skips_incomplete = True
