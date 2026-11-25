"""Qwen3 backbone transfer and standalone HAttention checkpoint export."""
import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import save_torch_state_dict

from .model import HAttentionLM, HConfig, Mixer


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def base_identity(folder):
    index = folder / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    files += ["config.json", "tokenizer.json", "tokenizer_config.json"]
    return {name: file_hash(folder / name) for name in files}


def load_teacher(folder):
    from transformers import Qwen3ForCausalLM

    model, info = Qwen3ForCausalLM.from_pretrained(
        folder, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"incomplete Qwen3 teacher: {info}")
    return model.eval().requires_grad_(False)


def convert(teacher, context):
    c = teacher.config
    if (c.model_type != "qwen3" or c.hidden_act != "silu" or c.tie_word_embeddings
            or c.num_attention_heads * c.head_dim != c.hidden_size):
        raise ValueError("expected a dense, untied Qwen3 Base with matching head and hidden dimensions")
    cfg = HConfig(c.hidden_size, c.num_attention_heads, c.num_hidden_layers, c.vocab_size,
                  c.num_key_value_heads, c.head_dim, c.head_dim, 1, 4, c.rms_norm_eps, False, True,
                  trained_levels=(context - 1).bit_length() + 1, block_size=context,
                  student=True, mlp_width=c.intermediate_size)
    with torch.device("meta"):
        student = HAttentionLM(cfg)
    student.backbone.embeddings.load_state_dict(teacher.model.embed_tokens.state_dict(), assign=True)
    student.backbone.norm_f.load_state_dict(teacher.model.norm.state_dict(), assign=True)
    student.lm_head.load_state_dict(teacher.lm_head.state_dict(), assign=True)
    for block, source in zip(student.backbone.layers, teacher.model.layers):
        block.mixer = Mixer(cfg)
        # Spread initial decay timescales; dt=softplus(1) would erase most distant history at step zero.
        with torch.no_grad():
            dt = torch.empty(cfg.num_heads).uniform_(-6.9, -2.3).exp()
            # Start the new residual branch at zero; its first update fits the teacher output scale.
            block.mixer.out_proj.weight.zero_()
            block.mixer.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        block.mixer.to(dtype=teacher.dtype)
        for dest, origin in ((block.mixer_norm, source.input_layernorm),
                             (block.mlp_norm, source.post_attention_layernorm), (block.mlp, source.mlp)):
            dest.load_state_dict(origin.state_dict(), assign=True)
    return student


def checkpoint_report(folder, *, verify_all=False):
    report = json.loads((folder / "training.json").read_text())
    files = report["files_sha256"]
    if verify_all:
        index = folder / "model.safetensors.index.json"
        required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
        if index.exists():
            required.add(index.name)
            required.update(json.loads(index.read_text())["weight_map"].values())
        else:
            required.add("model.safetensors")
        if not required.issubset(files) or set(files) != set(report["file_stats"]):
            raise ValueError(f"incomplete checkpoint manifest: {folder}")
    for name, digest in files.items():
        if Path(name).name != name:
            raise ValueError(f"invalid checkpoint asset name: {name}")
        path = folder / name
        st = path.stat()
        if (verify_all or [st.st_size, st.st_mtime_ns] != report["file_stats"][name]) and file_hash(path) != digest:
            raise ValueError(f"checkpoint asset changed since export: {path}")
    return report


def export_checkpoint(out, cfg, state, tokenizer_dir, report):
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        folder = Path(tmp) / "checkpoint"
        folder.mkdir()
        save_torch_state_dict({k: v.detach().to(device="cpu", dtype=torch.bfloat16) for k, v in state.items()},
                              folder, max_shard_size="2GB")
        (folder / "config.json").write_text(json.dumps(dict(model_type="qwen3_hattention", **asdict(cfg)), indent=2))
        for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "special_tokens_map.json",
                     "vocab.json", "merges.txt", "added_tokens.json", "tokenizer.model"):
            source = tokenizer_dir / name
            if source.exists():
                shutil.copy2(source, folder / name)
        if not all((folder / f).is_file() for f in ("tokenizer.json", "tokenizer_config.json")):
            raise FileNotFoundError(f"incomplete Qwen tokenizer in {tokenizer_dir}")
        report = {**report, "files_sha256": {p.name: file_hash(p) for p in sorted(folder.iterdir())}}
        report["file_stats"] = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in folder.iterdir()}
        (folder / "training.json").write_text(json.dumps(report, indent=2))
        folder.rename(out)
        print(f"[checkpoint] saved stage={report.get('stage')} steps={report.get('actual_steps', 0)} "
              f"path={out.resolve()}", flush=True)
