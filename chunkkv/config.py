"""Project-owned experiment configuration without importing the model runtime."""

import os
from pathlib import Path
import re

import yaml


BASELINE_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = BASELINE_ROOT / "chunkkv/_vendor/semanticlogkv"
DEFAULT_EXPERIMENT_CONFIG = BASELINE_ROOT / "configs/chunkkv/experiment_qwen1.7b_32k.yaml"
INPUT_PATHS = ("ckpt_dir", "resume_dir", "data_dir", "dataset_dir", "tokenizer_dir", "data_mix_yaml",
               "log_kv_semantic_s_h_path")


def require_project_file(path):
    """Experiment/training YAMLs used by entry points must belong to this project."""
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(BASELINE_ROOT):
        raise ValueError(f"Configs must live inside this project ({BASELINE_ROOT}): {path}")
    return path


def load_yaml_chain(path, root=None):
    """Load shallow ``config:`` inheritance, recording its inputs; parents stay in the top YAML's folder."""
    path = Path(path).resolve(strict=True)
    root = path.parent if root is None else root
    if not path.is_relative_to(root):
        raise ValueError(f"Config inheritance leaves {root}: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    files = [path]
    if "config" in config:
        parent, parent_files = load_yaml_chain(path.parent / config.pop("config"), root)
        config, files = {**parent, **config}, parent_files + files
    return config, files


def expand_env_vars(value):
    """Preserve the reference's ${VAR}, ${VAR-default}, and ${VAR:-default} semantics."""
    if isinstance(value, str):
        def replace(match):
            inner = match.group(1)
            separator = ":-" if ":-" in inner else "-"
            name, default = inner.split(separator, 1) if separator in inner else (inner, "")
            return os.environ.get(name, default)
        return re.sub(r"\$\{([^}^{]+)\}", replace, value)
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_env_vars(item) for item in value)
    return value


def normalize_input_paths(config):
    config = dict(config)
    for key in INPUT_PATHS:
        if config.get(key):
            value = os.path.expanduser(str(config[key]))
            # Absolute checkpoint strings retain their trailing slash: evaluation cache keys use it.
            config[key] = value if Path(value).is_absolute() else str((BASELINE_ROOT / value).resolve())
    return config


def load_experiment_config(path=DEFAULT_EXPERIMENT_CONFIG):
    config, _ = load_yaml_chain(path)
    return normalize_input_paths(expand_env_vars(config))
