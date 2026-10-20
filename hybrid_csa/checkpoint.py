"""Adapters are bound to the exact Base weights and architecture they were trained on."""

import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def base_identity(checkpoint_dir):
    path = Path(checkpoint_dir).resolve(strict=True)
    names = ("lit_model.pth", "model_config.yaml", "config.json", "tokenizer.json", "tokenizer.model",
             "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "generation_config.json")
    if not (path / "lit_model.pth").is_file() or not (path / "model_config.yaml").is_file():
        raise ValueError("The original Base needs lit_model.pth and model_config.yaml")
    return {name: sha256(path / name) for name in names if (path / name).is_file()}


def _model_base_identity(model, directory):
    # Identity is expensive (several GB): cache only within this loaded model/run.
    key = str(Path(directory).resolve(strict=True))
    cached = getattr(model, "_base_identity", None)
    if cached is None or cached[0] != key:
        cached = (key, base_identity(directory))
        model._base_identity = cached
    return cached[1]


def save_adapter(model, path, base_checkpoint_dir, training_record):
    path = Path(path).resolve()
    if not isinstance(training_record.get("optimizer_steps"), int) or training_record["optimizer_steps"] < 1:
        raise ValueError("An adapter must record positive optimizer_steps")
    if training_record.get("stage_steps", {}).get(model.stage, 0) < 1:
        raise ValueError("An adapter must record actual steps for its saved stage")
    # JSON-compatible records keep the artifact inspectable with weights_only=True.
    record = json.loads(json.dumps(training_record))
    payload = {"format_version": 1, "method": "Qwen3-Hybrid-CSA4", "settings": model.settings.as_dict(),
               "stage": model.stage, "base": _model_base_identity(model, base_checkpoint_dir),
               "training": record, "adapter": model.adapter_state_dict()}
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids silently replacing a different trained artifact.
    with path.open("xb") as handle:
        torch.save(payload, handle)
    return {key: value for key, value in payload.items() if key != "adapter"}


def load_adapter(model, path, base_checkpoint_dir, require_sparse=True):
    path = Path(path).resolve(strict=True)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_payload(payload, model.settings.as_dict(), _model_base_identity(model, base_checkpoint_dir),
                     require_sparse)
    model.load_adapter_state_dict(payload["adapter"])
    model.set_stage(payload["stage"])
    return {"path": str(path), "sha256": sha256(path),
            **{key: value for key, value in payload.items() if key != "adapter"}}


def validate_payload(payload, settings, base, require_sparse=True):
    """Shared checks for the launcher handoff and actual model loading."""
    if payload.get("format_version") != 1 or payload.get("method") != "Qwen3-Hybrid-CSA4":
        raise ValueError("Unsupported HybridCSA adapter format")
    expected, found = settings, payload["settings"]
    # These only bound temporary memory; changing them must preserve model semantics.
    execution = {"query_chunk_size", "prefill_chunk_size"}
    if {k: v for k, v in expected.items() if k not in execution} != {k: v for k, v in found.items() if k not in execution}:
        raise ValueError("Adapter architecture settings do not match the evaluation configuration")
    if payload["base"] != base:
        raise ValueError("Adapter Base weight/config/tokenizer fingerprints do not match")
    training = payload.get("training", {})
    if training.get("optimizer_steps", 0) < 1:
        raise ValueError("Untrained adapter: no optimizer steps recorded")
    if require_sparse and (payload["stage"] != "sparse" or training.get("stage_steps", {}).get("sparse", 0) < 1):
        raise ValueError("Formal CSA evaluation requires an adapter trained in the sparse stage")
    if not payload.get("adapter"):
        raise ValueError("Adapter state is empty")
