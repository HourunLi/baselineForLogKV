"""Export the exact original LitGPT Base checkpoint to local HF format.

No model is downloaded. ``ckpt_dir`` of this project's experiment YAML selects
the source Base (the SemanticLogKV CPT starting point); CPT outputs never do.
The LitGPT config class comes from this project's private runtime snapshot and
the tensor mapping from ``models.qwen3.convert_lit_to_hf``; no other checkout
is imported or read.
"""

import argparse
import hashlib
import json
import shutil
import warnings
from pathlib import Path

import torch
import yaml
from transformers import Qwen3Config, Qwen3ForCausalLM

from . import convert_lit_to_hf
from .config import BASELINE_ROOT, DEFAULT_EXPERIMENT_CONFIG, load_experiment_config, require_project_file

MANIFEST_VERSION = 2
TOKENIZER_NAMES = (
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
    "chat_template.jinja", "generation_config.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_prepared_base(path: Path) -> dict:
    """Fail before training if converted weights or tokenizer provenance drifted."""
    path = Path(path).expanduser().resolve()
    manifest_path = path / "semanticlogkv_base_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing prepared Base manifest: {manifest_path}; run models.qwen3.prepare_base first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    provenance = {key: value for key, value in manifest.items()
                  if key not in ("prepared_weight_files", "prepared_config_sha256")}
    if config.get("semanticlogkv_base") != provenance:
        raise ValueError("Prepared Base config provenance disagrees with its manifest")
    # Version 1 manifests came from the earlier converter that imported a sibling LitGPT
    # checkout; their tensors and hashes are verified the same way.
    if manifest.get("manifest_version") not in (1, MANIFEST_VERSION):
        raise ValueError("Unsupported prepared Base manifest version")
    if sha256_file(path / "config.json") != manifest["prepared_config_sha256"]:
        raise ValueError("Prepared Base config hash mismatch")
    for group in ("prepared_weight_files", "tokenizer_files"):
        if not manifest.get(group):
            raise ValueError(f"Prepared Base manifest has no {group}")
        for name, expected in manifest[group].items():
            actual_path = path / name
            if not actual_path.is_file():
                raise FileNotFoundError(f"Missing prepared Base file: {actual_path}")
            if sha256_file(actual_path) != expected:
                raise ValueError(f"Prepared Base file hash mismatch: {actual_path}")
    return manifest


def validate_config(lit_config, hf_config: Qwen3Config) -> None:
    """Refuse configs that change weight interpretation or transformer math."""
    pairs = {
        "hidden_size": (lit_config.n_embd, hf_config.hidden_size),
        "intermediate_size": (lit_config.intermediate_size, hf_config.intermediate_size),
        "num_hidden_layers": (lit_config.n_layer, hf_config.num_hidden_layers),
        "num_attention_heads": (lit_config.n_head, hf_config.num_attention_heads),
        "num_key_value_heads": (lit_config.n_query_groups, hf_config.num_key_value_heads),
        "head_dim": (lit_config.head_size, hf_config.head_dim),
        "vocab_size": (lit_config.padded_vocab_size, hf_config.vocab_size),
        "rms_norm_eps": (lit_config.norm_eps, hf_config.rms_norm_eps),
        "rope_theta": (lit_config.rope_base, hf_config.rope_theta),
    }
    mismatches = [f"{name}: LitGPT={a}, HF={b}" for name, (a, b) in pairs.items() if a != b]
    if mismatches:
        raise ValueError("Source configs disagree: " + "; ".join(mismatches))
    if (lit_config.mlp_class_name != "LLaMAMLP" or lit_config.norm_class_name != "RMSNorm"
            or not lit_config.norm_qk or lit_config.parallel_residual
            or lit_config.bias or lit_config.attn_bias or lit_config.lm_head_bias
            or lit_config.rotary_percentage != 1.0 or lit_config.rope_interleave
            or lit_config.rope_condense_ratio != 1 or lit_config.rope_adjustments
            or hf_config.rope_scaling or hf_config.attention_bias
            or hf_config.hidden_act != "silu" or hf_config.use_sliding_window):
        raise ValueError("Expected the standard dense Qwen3 Base architecture without RoPE extensions")


def build_hf_model(lit_config, hf_config: Qwen3Config, lit_weights: dict, converter):
    """Preserve every source tensor and fail on missing, extra or mismatched keys."""
    validate_config(lit_config, hf_config)
    state = {}
    converter(lit_config, state, lit_weights)
    if hf_config.tie_word_embeddings:
        # tie_weights() must never silently replace a distinct source LM head.
        if not torch.equal(state["model.embed_tokens.weight"], state["lm_head.weight"]):
            raise ValueError("HF config ties embeddings but the LitGPT source LM head differs")
    with torch.device("meta"):
        model = Qwen3ForCausalLM(hf_config)
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    return model.eval()


def prepare_base(
    experiment_config: Path = DEFAULT_EXPERIMENT_CONFIG,
    output: Path = BASELINE_ROOT / "outputs/semanticlogkv_base",
) -> dict:
    from models._vendor.semanticlogkv.litgpt.config import Config

    config_path = Path(experiment_config).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    converter = convert_lit_to_hf.copy_weights_qwen_3
    experiment = load_experiment_config(config_path)
    if not experiment.get("ckpt_dir"):
        raise ValueError(f"Experiment config has no original Base ckpt_dir: {config_path}")
    source = Path(experiment["ckpt_dir"]).resolve()
    required = ("lit_model.pth", "model_config.yaml", "config.json", "tokenizer_config.json")
    for name in required:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Missing original Base file: {source / name}")
    if not any((source / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        raise FileNotFoundError(f"No tokenizer.json or tokenizer.model in {source}")
    with (source / "config.json").open(encoding="utf-8") as stream:
        if json.load(stream).get("model_type") != "qwen3":
            raise ValueError("The source HF config must identify model_type=qwen3")
    with (source / "model_config.yaml").open(encoding="utf-8") as stream:
        model_config = yaml.safe_load(stream)
    if model_config is None:
        raise ValueError(f"{source / 'model_config.yaml'} is empty")
    # An empty experiment schedule has no effect on dense Base conversion.
    # Keep the source file untouched so provenance hashes still cover it.
    if "research_breakpoint_schedule" in model_config:
        schedule = model_config.pop("research_breakpoint_schedule")
        if schedule not in (None, [], {}):
            raise ValueError(
                "Non-empty research_breakpoint_schedule in source model_config.yaml; "
                "its semantics must be checked before converting this checkpoint as a dense Base"
            )
        warnings.warn("Ignoring empty research_breakpoint_schedule for dense Base conversion", stacklevel=2)
    lit_config = Config(**model_config)
    hf_config = Qwen3Config.from_pretrained(source, local_files_only=True)
    source_weights = torch.load(source / "lit_model.pth", map_location="cpu", weights_only=True, mmap=True)
    if "model" in source_weights:
        source_weights = source_weights["model"]
    model = build_hf_model(lit_config, hf_config, source_weights, converter)

    tokenizer_paths = [source / name for name in TOKENIZER_NAMES if (source / name).is_file()]
    if (source / "chat_templates").is_dir():
        tokenizer_paths.extend(sorted((source / "chat_templates").glob("*.jinja")))
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "source_checkpoint_dir": str(source),
        "source_weight_sha256": sha256_file(source / "lit_model.pth"),
        "source_config_files": {
            name: sha256_file(source / name) for name in ("model_config.yaml", "config.json")
        },
        "tokenizer_files": {
            str(path.relative_to(source)): sha256_file(path) for path in tokenizer_paths
        },
        "experiment_config": str(config_path),
        "experiment_config_sha256": sha256_file(config_path),
        "source_model_name": lit_config.name,
        "converter": "models.qwen3.convert_lit_to_hf.copy_weights_qwen_3 (from LitGPT 0.5.12)",
        "converter_sha256": sha256_file(Path(convert_lit_to_hf.__file__)),
    }
    # HF config round trips this field into the DMS student's config and export.
    model.config.semanticlogkv_base = manifest
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output, safe_serialization=True)
    for path in tokenizer_paths:
        destination = output / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    # Keep output hashes outside config.json: placing them in the saved model's
    # own config would make a self-referential manifest.
    manifest = {**manifest, "prepared_config_sha256": sha256_file(output / "config.json"), "prepared_weight_files": {
        path.name: sha256_file(path) for path in sorted(output.glob("*.safetensors"))
    }}
    (output / "semanticlogkv_base_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG,
                        help="Project-owned experiment YAML whose ckpt_dir is the original Base")
    parser.add_argument("--output", type=Path, default=BASELINE_ROOT / "outputs/semanticlogkv_base")
    args = parser.parse_args()
    manifest = prepare_base(require_project_file(args.experiment_config), args.output)
    print(f"Prepared exact Base weights in {args.output.resolve()}")
    print(f"Original lit_model.pth SHA-256: {manifest['source_weight_sha256']}")


if __name__ == "__main__":
    main()
