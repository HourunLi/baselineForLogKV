"""Generate HOMER relative-position biases from a frozen HF Llama or Qwen3 decoder."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
from pathlib import Path
import random
import tempfile
from unittest.mock import patch

import torch
import yaml


@dataclass
class Settings:
    model: str
    output: str
    length: int
    data: str = "Salesforce/wikitext"
    samples: int = 100
    seed: int = 42
    dtype: str = "bfloat16"
    device: str = "cuda"
    yarn_factor: float = 1
    original_context: int = 32768
    reference_bias: str | None = None
    holdout_samples: int = 10
    protected_prefix: int = 1
    protected_suffix: int = 1

    def validate(self):
        if self.samples < 2 or self.length < 2 or self.original_context < 1 or self.yarn_factor < 1:
            raise ValueError("Invalid sample count, length or rotary settings")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported dtype")
        if self.reference_bias and (self.holdout_samples < 2 or min(self.protected_prefix, self.protected_suffix) < 0
                                    or self.protected_prefix + self.protected_suffix + 2 > self.length):
            raise ValueError("Comparison requires held-out samples and at least two unprotected tokens")


def digest(path):
    with Path(path).open("rb") as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
        return h.hexdigest()


def weight_files(path):
    for stem, extension in (("model", "safetensors"), ("pytorch_model", "bin")):
        single = path / f"{stem}.{extension}"
        index = path / (single.name + ".index.json")
        if single.is_file():
            return [single]
        if index.is_file():
            return [path / p for p in sorted(set(json.loads(index.read_text())["weight_map"].values()))]
    raise FileNotFoundError(f"{path}: expected Hugging Face weights or their shard index")


def last_row(q, k):
    """Batch 1. Return FP32 significance in distance order, starting at self."""
    groups = q.size(1) // k.size(1)
    last = q[0, :, -1].reshape(k.size(1), groups, k.size(-1))
    logits = torch.bmm(last, k[0].transpose(1, 2)) / (q.size(-1) ** 0.5)
    return logits.amax(dim=(0, 1)).float().flip([0])


@contextmanager
def observe(model, rows):
    # In transformers 4.51 the native SDPA receives post-normalization, post-RoPE Q/K.
    # Keep the normal SDPA implementation and mask preparation for the actual forward.
    module = inspect.getmodule(model.layers[0].self_attn)
    interfaces = dict(module.ALL_ATTENTION_FUNCTIONS)
    sdpa = interfaces["sdpa"]

    def recorded(layer, q, k, v, mask, **kwargs):
        if rows[layer.layer_idx] is not None:
            raise RuntimeError("Duplicate attention observation")
        rows[layer.layer_idx] = last_row(q, k)
        return sdpa(layer, q, k, v, mask, **kwargs)

    with patch.object(module, "ALL_ATTENTION_FUNCTIONS", {**interfaces, "sdpa": recorded}):
        yield


@torch.inference_mode()
def scores(model, ids):
    rows = [None] * len(model.layers)
    with observe(model, rows):
        model(input_ids=ids, use_cache=False, output_attentions=False)
    if any(row is None for row in rows):
        raise RuntimeError("Native SDPA did not expose every decoder layer")
    values = torch.stack(rows).cpu()
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite attention scores")
    return values


def windows(split, tokenizer, cfg, name, heldout):
    # Reserve the final quarter of source rows even when comparison is disabled.
    # Calibration segments can overlap, but never overlap the comparison corpus.
    boundary = 3 * len(split) // 4
    lo, hi = (boundary, len(split)) if heldout else (0, boundary)
    starts, row_ids, tokens = [], [], []
    for row in range(lo, hi, 256):
        texts = split[row:min(row + 256, hi)]["text"]
        ids = tokenizer([x + "\n" for x in texts], add_special_tokens=False)["input_ids"]
        for index, (text, encoded) in enumerate(zip(texts, ids), row):
            if text.strip() and encoded:
                starts.append(len(tokens))
                row_ids.append(index)
                tokens.extend(encoded)
    end = bisect_right(starts, len(tokens) - cfg.length)
    n = cfg.holdout_samples if heldout else cfg.samples
    count = n // 2 + int(name == "validation" and n % 2 == 1)
    if end < count:
        raise ValueError(f"{name}: {end} eligible starts for {count} windows of length {cfg.length}")
    rng = random.Random(f"{cfg.seed}/{name}/{heldout}")
    for i in rng.sample(range(end), count):
        start = starts[i]
        ids = torch.tensor(tokens[start:start + cfg.length], dtype=torch.long)
        yield ids, dict(split=name, start_row=row_ids[i], token_offset_in_partition=start,
                        length=len(ids), token_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest())


def reference_table(path, layers, length):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if len(obj) != layers:
        raise ValueError("Reference bias layer count differs from the model")
    curves = [torch.as_tensor(obj[i]) for i in range(layers)]
    if any(curve.ndim != 1 or curve.numel() < length for curve in curves):
        raise ValueError("Expected one 1-D distance curve per layer covering the requested range")
    table = torch.stack([curve[:length] for curve in curves]).float()
    if table.shape != (layers, length) or not torch.isfinite(table).all():
        raise ValueError("Reference does not cover the requested distances with finite values")
    return table


def run(cfg):
    import transformers
    from datasets import load_dataset, load_from_disk
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    cfg.validate()
    out = Path(cfg.output)
    if out.exists():
        raise FileExistsError(f"Select a fresh output directory: {out}")
    path = Path(cfg.model).resolve()
    files = weight_files(path)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    if config.model_type not in ("llama", "qwen3"):
        raise ValueError("Expected Llama or Qwen3")
    if cfg.yarn_factor > 1:
        config.max_position_embeddings = int(cfg.original_context * cfg.yarn_factor)
        config.rope_scaling = dict(rope_type="yarn", factor=float(cfg.yarn_factor), beta_fast=32, beta_slow=1,
                                   original_max_position_embeddings=cfg.original_context)
    if cfg.length > config.max_position_embeddings:
        raise ValueError("Calibration length exceeds configured model window")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    other = reference_table(cfg.reference_bias, config.num_hidden_layers, cfg.length) if cfg.reference_bias else None
    model, info = AutoModel.from_pretrained(path, config=config, attn_implementation="sdpa", local_files_only=True,
                                            torch_dtype=getattr(torch, cfg.dtype), output_loading_info=True)
    if info["missing_keys"] or info["mismatched_keys"]:
        raise ValueError(f"Incomplete backbone weights: {info}")
    model.eval().to(cfg.device)
    data = load_from_disk(cfg.data) if Path(cfg.data).is_dir() else load_dataset(cfg.data, "wikitext-103-raw-v1")
    bias = torch.zeros(config.num_hidden_layers, cfg.length)
    manifest = dict(settings=asdict(cfg), model_path=str(path), model_type=config.model_type,
                    weight_files={p.name: digest(p) for p in files}, config_sha256=digest(path/"config.json"),
                    tokenizer_files={p.name: digest(p) for p in path.iterdir() if p.name in
                                     ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json",
                                      "merges.txt", "special_tokens_map.json", "added_tokens.json")},
                    rope_scaling=config.rope_scaling, rope_theta=config.rope_theta,
                    transformers=transformers.__version__, torch=str(torch.__version__),
                    fingerprints={k: data[k]._fingerprint for k in ("validation", "test")},
                    calibration=[], heldout=[])
    for split in manifest["fingerprints"]:
        for ids, record in windows(data[split], tokenizer, cfg, split, False):
            bias += scores(model, ids[None].to(cfg.device))
            manifest["calibration"].append(record)
    bias /= len(manifest["calibration"])
    manifest["samples_per_distance"] = len(manifest["calibration"])
    if other is not None:
        agreement = torch.zeros(len(bias))
        candidates = slice(cfg.protected_suffix, cfg.length - cfg.protected_prefix)
        keep = (cfg.length - cfg.protected_suffix - cfg.protected_prefix) // 2
        for split in manifest["fingerprints"]:
            for ids, record in windows(data[split], tokenizer, cfg, split, True):
                raw = scores(model, ids[None].to(cfg.device))
                a = (raw - bias)[:, candidates].topk(keep).indices
                b = (raw - other)[:, candidates].topk(keep).indices
                retained = torch.zeros_like(raw[:, candidates], dtype=torch.bool).scatter_(1, a, True)
                agreement += retained.gather(1, b).float().mean(1)
                manifest["heldout"].append(record)
        manifest["comparison"] = dict(reference_sha256=digest(cfg.reference_bias),
                                      rmse_per_layer=(bias-other).square().mean(1).sqrt().tolist(),
                                      kept_set_overlap_per_layer=(agreement / len(manifest["heldout"])).tolist())
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp)/"result"
        stage.mkdir()
        torch.save(bias, stage/"bias.pt")
        manifest["bias_sha256"] = digest(stage/"bias.pt")
        (stage/"manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(out)
    print(out / "bias.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    values = yaml.safe_load(args.config.read_text())
    if values.pop("protocol") != "homer_calibration":
        parser.error("Expected protocol: homer_calibration")
    cfg = Settings(**values)
    cfg.validate()
    if args.print_config:
        print(json.dumps(asdict(cfg), indent=2))
    else:
        run(cfg)


if __name__ == "__main__":
    main()
