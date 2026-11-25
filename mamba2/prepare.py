"""Prepare the Qwen3-8B-Base warm start from an ordinary local LitGPT checkpoint."""
import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from litgpt.config import Config
from litgpt.model import GPT
from mamba_model import MambaLM, Mamba2Mixer


def specifications(config):
    with torch.device("meta"):
        dense = GPT(config).state_dict()
        mamba = MambaLM(config).state_dict()
    return dense, mamba


def counts(dense, mamba):
    return {"qwen_parameters": sum(t.numel() for t in dense.values()),
            "mamba_parameters": sum(t.numel() for t in mamba.values()),
            "copied_parameters": sum(t.numel() for name, t in mamba.items() if ".mixer." not in name),
            "new_mixer_parameters": sum(t.numel() for name, t in mamba.items() if ".mixer." in name)}


def hf_names(config):
    names = {"lm_head.weight": "lm_head.weight", "model.embed_tokens.weight": "transformer.wte.weight",
             "model.norm.weight": "transformer.ln_f.weight"}
    layer_names = {"input_layernorm": "norm_1", "post_attention_layernorm": "norm_2",
                   "mlp.up_proj": "mlp.fc_2", "mlp.gate_proj": "mlp.fc_1", "mlp.down_proj": "mlp.proj"}
    for i in range(config.n_layer):
        for src, dst in layer_names.items():
            names[f"model.layers.{i}.{src}.weight"] = f"transformer.h.{i}.{dst}.weight"
    return names


def shared_weights(config, base, dense, expected):
    lit_path = base / "lit_model.pth"
    if lit_path.is_file():
        source = torch.load(lit_path, mmap=True, map_location="cpu", weights_only=True)
        source = source.get("model", source)
        if source.keys() != dense.keys():
            raise ValueError(f"Unexpected Qwen checkpoint keys: {sorted(source.keys() ^ dense.keys())[:10]}")
        for name, t in source.items():
            if t.shape != dense[name].shape or not t.is_floating_point():
                raise ValueError(f"Invalid Qwen tensor {name}: {t.shape} {t.dtype}")
        return {k: source[k] for k in expected if ".mixer." not in k}

    from safetensors import safe_open
    hf = json.loads((base / "config.json").read_text())
    required = {"model_type": "qwen3", "num_hidden_layers": config.n_layer, "hidden_size": config.n_embd,
                "num_key_value_heads": config.n_query_groups, "num_attention_heads": config.n_head,
                "intermediate_size": config.intermediate_size, "vocab_size": config.padded_vocab_size,
                "tie_word_embeddings": False}
    for key, value in required.items():
        if hf.get(key) != value:
            raise ValueError(f"Expected HF {key}={value}, found {hf.get(key)}")
    index = base / "model.safetensors.index.json"
    mapping = hf_names(config)
    files = json.loads(index.read_text())["weight_map"] if index.exists() else dict.fromkeys(mapping, "model.safetensors")
    shards = {}
    for name in mapping:
        shard = files[name]
        if Path(shard).name != shard:
            raise ValueError(f"Invalid Qwen shard name: {shard}")
        shards.setdefault(shard, []).append(name)
    weights = {}
    for shard, names in shards.items():
        with safe_open(base / shard, framework="pt", device="cpu") as handle:
            for name in names:
                dst = mapping[name]
                tensor = handle.get_tensor(name)
                if tensor.shape != expected[dst].shape or not tensor.is_floating_point():
                    raise ValueError(f"Invalid HF tensor {name}: {tensor.shape} {tensor.dtype}")
                weights[dst] = tensor
    return weights


def convert(config, base, out, seed):
    if out.exists():
        raise FileExistsError(f"Select a new output directory: {out}")
    if not (base / "tokenizer.json").is_file():
        raise FileNotFoundError(base / "tokenizer.json")
    dense, expected = specifications(config)
    weights = shared_weights(config, base, dense, expected)
    torch.manual_seed(seed)
    for layer in range(config.n_layer):
        mixer = Mamba2Mixer(config).to(dtype=weights["transformer.wte.weight"].dtype)
        weights.update({f"transformer.h.{layer}.mixer.{k}": v for k, v in mixer.state_dict().items()})
    if weights.keys() != expected.keys():
        raise ValueError("Incomplete Mamba-2 warm start")
    report = {**counts(dense, expected), "base": str(base.resolve()), "seed": seed, "stage": "before_cpt"}
    out.mkdir(parents=True, exist_ok=False)
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json",
                 "generation_config.json", "merges.txt", "vocab.json"):
        if (base / name).is_file():
            shutil.copyfile(base / name, out / name)
    torch.save(weights, out / "lit_model.pth")
    (out / "init_report.json").write_text(json.dumps(report, indent=2))
    (out / "model_config.yaml").write_text(yaml.safe_dump(asdict(config)))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--count-only", action="store_true")
    args = p.parse_args()
    cfg = Config.from_name("Qwen/Qwen3-8B-Base", block_size=32768, mamba2=True)
    if args.count_only:
        report = counts(*specifications(cfg))
    else:
        if not args.base or not args.out:
            p.error("--base and --out are required for conversion")
        report = convert(cfg, args.base, args.out, args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
