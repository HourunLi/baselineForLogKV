"""Optional dense-CPT control: the bundled trainer and local experiment with LogKV attention off.

ChunkKV is training-free; its main row uses the original Base. This entry only
separates the effect of SemanticLogKV's CPT corpus and budget from the effect of
its compressed attention. It runs the bundled reference ``demo.py`` on the
project-owned experiment YAML (Base ``ckpt_dir``, data, batch, steps, optimizer, schedule,
FSDP and checkpoint format) but keeps standard causal attention, then ChunkKV is
applied at evaluation: ``chunkkv.evaluate --model <save_path>``.

``--prepare`` runs once on the first node and writes ``resolved_train.yaml`` and
``chunkkv_train_manifest.json`` to ``save_path``. The torchrun step re-resolves
the same YAML on every rank and refuses to train if it no longer matches.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import yaml

from .config import (
    BASELINE_ROOT, RUNTIME_ROOT, expand_env_vars, load_yaml_chain, normalize_input_paths, require_project_file,
)

DEFAULT_CONFIG = BASELINE_ROOT / "configs/chunkkv/dense_cpt_qwen1.7b_32k.yaml"
# The control trains the same weights with ordinary attention and evaluates through
# chunkkv.evaluate. Semantic clusters are off so demo.py installs no routing replay.
FORCED = {"run_eval": "", "save_ckpt": True, "resume_dir": None, "log_kv_semantic_clusters": False}
CONTROLLED = {"config", "ckpt_dir", "save_path", "tensorboard_root", "merged_state_staging_dir", "expid"}
MANIFEST = "chunkkv_train_manifest.json"
RESOLVED = "resolved_train.yaml"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute(path, base):
    path = Path(os.path.expanduser(str(path)))
    return (path if path.is_absolute() else base / path).resolve()


def resolve_training_config(train_config):
    """Return the exact demo.py config for the dense control and its provenance."""
    train_config = Path(train_config).resolve(strict=True)
    spec = expand_env_vars(yaml.safe_load(train_config.read_text(encoding="utf-8")) or {})
    unknown = sorted(set(spec) - {"experiment_config", "save_path", "expid", "overrides"})
    if unknown or "experiment_config" not in spec or "save_path" not in spec:
        raise ValueError(f"{train_config} needs experiment_config and save_path; unknown keys: {unknown}")
    overrides = spec.get("overrides") or {}
    rejected = sorted(set(overrides) & (CONTROLLED | set(FORCED)))
    if rejected:
        raise ValueError(f"These training keys are fixed by the dense control: {rejected}")
    source, source_files = load_yaml_chain(_absolute(spec["experiment_config"], train_config.parent))
    source = normalize_input_paths(expand_env_vars(source))
    if not source.get("ckpt_dir"):
        raise ValueError("The local experiment must name its original Base ckpt_dir")
    save_path = _absolute(spec["save_path"], BASELINE_ROOT)
    outputs = {"save_path": save_path,
               "tensorboard_root": save_path.with_name(save_path.name + "_tensorboard"),
               "merged_state_staging_dir": save_path.with_name(save_path.name + "_fsdp_staging")}
    protected = [_absolute(source[key], BASELINE_ROOT) for key in
                 ("ckpt_dir", "save_path", "tensorboard_root", "merged_state_staging_dir") if source.get(key)]
    protected.append(Path("/data/lihourun/semanticLogKV"))
    for name, path in outputs.items():
        if (path.is_relative_to(RUNTIME_ROOT) or RUNTIME_ROOT.is_relative_to(path)
                or any(path.is_relative_to(p) or p.is_relative_to(path) for p in protected)):
            raise ValueError(f"{name}={path} must be outside the bundled runtime and away from the Base and SemanticLogKV run")
    resolved = {**normalize_input_paths({**source, **overrides}), **FORCED,
                **{key: str(value) for key, value in outputs.items()},
                "expid": spec.get("expid") or f"{source.get('expid', 'semanticlogkv')}-dense-control"}
    runtime_files = sorted(RUNTIME_ROOT.rglob("*.py")) + [RUNTIME_ROOT / "SOURCE.json"]
    record = {
        "manifest_version": 2,
        "purpose": "dense CPT control: bundled trainer, local experiment and standard causal attention",
        "train_config": {str(train_config): sha256(train_config)},
        "experiment_config_files": {str(path): sha256(path) for path in source_files},
        "base_checkpoint_dir": str(_absolute(source["ckpt_dir"], BASELINE_ROOT)),
        "forced": FORCED, "overrides": overrides,
        "trainer_code": {str(path.relative_to(RUNTIME_ROOT)): sha256(path) for path in runtime_files},
        "control_code": {path.name: sha256(path) for path in (Path(__file__), Path(__file__).with_name("config.py"))},
        "resolved_config": resolved,
    }
    return resolved, record


def prepare(train_config):
    """Write the resolved config and provenance once; refuse to mix two different runs."""
    resolved, record = resolve_training_config(train_config)
    save_path = Path(resolved["save_path"])
    manifest = save_path / MANIFEST
    if manifest.is_file():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if {k: v for k, v in previous.items() if k != "base_weight_sha256"} != record:
            raise ValueError(f"{save_path} belongs to a different dense-control configuration")
        print(f"Resuming the prepared dense control in {save_path}")
        return resolved
    if save_path.exists() and any(save_path.iterdir()):
        raise FileExistsError(f"{save_path} is not empty and was not prepared for this control")
    base = Path(record["base_checkpoint_dir"]) / "lit_model.pth"
    record["base_weight_sha256"] = sha256(base)
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / RESOLVED).write_text(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=True))
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared dense control in {save_path} (Base SHA-256 {record['base_weight_sha256']})")
    return resolved


# demo.py prints ``kv_cache.B`` right after enabling LogKV training.
DENSE_CACHE_MARKER = SimpleNamespace(B="none (dense control, LogKV training disabled)")


def make_dense_gpt(base):
    """Same GPT and checkpoint layout; the LogKV training switch leaves causal attention on."""
    class DenseControlGPT(base):
        def enable_log_kv_training(self, *args, **kwargs):
            self.disable_log_kv_training()
            for block in self.transformer.h:
                block.attn.kv_cache = DENSE_CACHE_MARKER

    return DenseControlGPT


def load_trainer():
    """Private instance of the bundled trainer with its dense-control model override."""
    spec = importlib.util.spec_from_file_location(
        "chunkkv._vendor.semanticlogkv._chunkkv_dense_trainer", RUNTIME_ROOT / "demo.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    module.GPT = make_dense_gpt(module.GPT)
    return module


def run(train_config):
    resolved, record = resolve_training_config(train_config)
    save_path = Path(resolved["save_path"])
    if int(os.environ.get("RANK", "0")) == 0:
        manifest = save_path / MANIFEST
        if not manifest.is_file():
            raise FileNotFoundError(f"Run `python -m chunkkv.train --prepare` first: missing {manifest}")
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if {k: v for k, v in previous.items() if k != "base_weight_sha256"} != record:
            raise ValueError("Training inputs changed after --prepare; prepare a new save_path")
    trainer = load_trainer()
    # Each rank reads its own copy, so ranks never wait on shared-disk visibility.
    with tempfile.TemporaryDirectory() as directory:
        config = Path(directory) / RESOLVED
        config.write_text(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=True))
        previous_dir = Path.cwd()
        try:
            os.chdir(BASELINE_ROOT)
            trainer.main(config=str(config))
        finally:
            os.chdir(previous_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepare", action="store_true", help="resolve, hash and record inputs (single process)")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    # Formal runs read only project-owned YAMLs (no other project's experiment chain).
    _, record = resolve_training_config(require_project_file(args.config))
    for path in record["experiment_config_files"]:
        require_project_file(path)
    if args.print_config:
        resolved, _ = resolve_training_config(args.config)
        print(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=True))
    elif args.prepare:
        prepare(args.config)
    else:
        run(args.config)


if __name__ == "__main__":
    main()
