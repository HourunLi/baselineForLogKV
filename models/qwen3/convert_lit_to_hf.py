# Adapted from LitGPT 0.5.12, litgpt/scripts/convert_lit_checkpoint.py (copy_weights_qwen_3).
# Copyright Lightning AI. Licensed under the Apache License 2.0.
# Local changes: dense (LLaMAMLP) Qwen3 only; no lazy-tensor saver; unknown source keys raise.
"""LitGPT -> Hugging Face tensor names for the dense Qwen3 Base used by SemanticLogKV."""

import re

WEIGHT_MAP = {
    "transformer.wte.weight": "model.embed_tokens.weight",
    "transformer.h.{}.norm_1.weight": "model.layers.{}.input_layernorm.weight",
    "transformer.h.{}.norm_2.weight": "model.layers.{}.post_attention_layernorm.weight",
    "transformer.h.{}.attn.proj.weight": "model.layers.{}.self_attn.o_proj.weight",
    "transformer.h.{}.attn.norm_q.weight": "model.layers.{}.self_attn.q_norm.weight",
    "transformer.h.{}.attn.norm_k.weight": "model.layers.{}.self_attn.k_norm.weight",
    "transformer.ln_f.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
    "transformer.h.{}.mlp.fc_1.weight": "model.layers.{}.mlp.gate_proj.weight",
    "transformer.h.{}.mlp.fc_2.weight": "model.layers.{}.mlp.up_proj.weight",
    "transformer.h.{}.mlp.proj.weight": "model.layers.{}.mlp.down_proj.weight",
}


def layer_template(name, num_matches=1):
    found = re.findall(r"\.(\d+)\.", name)
    if not found:
        return name, -1
    return re.sub(r"\.(\d+)\.", ".{}.", name, count=num_matches), *(int(x) for x in found[:num_matches])


def copy_weights_qwen_3(config, state_dict, lit_weights):
    """Fill ``state_dict`` with HF names; the fused QKV is stored as [Q..., K..., V...]."""
    if config.mlp_class_name != "LLaMAMLP":
        raise NotImplementedError("Only the dense Qwen3 LLaMAMLP layout is supported")
    for from_name, param in lit_weights.items():
        template, *ids = layer_template(from_name, num_matches=2)
        if from_name.endswith(".attn.qkv.weight"):
            q, k, v = param.split((config.n_head * config.head_size,
                                   config.n_query_groups * config.head_size,
                                   config.n_query_groups * config.head_size))
            for part, tensor in zip(("q_proj", "k_proj", "v_proj"), (q, k, v)):
                state_dict[f"model.layers.{ids[0]}.self_attn.{part}.weight"] = tensor
            continue
        if template not in WEIGHT_MAP:
            raise KeyError(f"Unexpected LitGPT tensor for dense Qwen3: {from_name}")
        state_dict[WEIGHT_MAP[template].format(*ids)] = param
