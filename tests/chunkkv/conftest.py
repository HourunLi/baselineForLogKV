import copy

import pytest
import torch

from litgpt.config import Config
from litgpt.model import GPT


def tiny_config(**overrides):
    """Qwen3-shaped LitGPT config: GQA, QK-norm, RMSNorm, LLaMA MLP, full RoPE."""
    values = dict(name="tiny-qwen3", block_size=128, vocab_size=64, padded_vocab_size=64, n_layer=4,
                  n_head=4, n_query_groups=2, n_embd=32, head_size=8, intermediate_size=64, norm_qk=True,
                  mlp_class_name="LLaMAMLP", norm_class_name="RMSNorm", parallel_residual=False,
                  bias=False, rope_base=10000)
    return Config(**{**values, **overrides})


@pytest.fixture
def gpt_pair():
    """A float32 reference GPT and an identical copy for the ChunkKV wrapper."""
    torch.manual_seed(0)
    reference = GPT(tiny_config()).eval()
    return reference, copy.deepcopy(reference)
