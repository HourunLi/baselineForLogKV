import json
from pathlib import Path

import pytest
import torch
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (LlamaConfig, LlamaModel, PreTrainedTokenizerFast,
                          Qwen3Config, Qwen3Model)

from homer.calibrate import Settings, last_row, run


def test_statistic_matches_expanded_query_heads():
    torch.manual_seed(117)
    q = torch.randn(1, 6, 17, 8)
    k = torch.randn(1, 2, 17, 8)
    dense = q @ k.repeat_interleave(3, dim=1).transpose(-2, -1) / 8**0.5
    torch.testing.assert_close(last_row(q, k), dense[0, :, -1].amax(0).flip([0]))


@pytest.mark.parametrize("config_type,model_type", [(Qwen3Config, Qwen3Model), (LlamaConfig, LlamaModel)])
def test_calibration_pipeline(tmp_path, config_type, model_type):
    torch.manual_seed(7)
    source = tmp_path/"model"
    cfg = config_type(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, vocab_size=8, max_position_embeddings=64)
    model = model_type(cfg).eval()
    model.save_pretrained(source)
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "the": 1, "cat": 2, "sat": 3}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]").save_pretrained(source)
    data_path = tmp_path/"wikitext"
    DatasetDict({k: Dataset.from_dict({"text": ["the cat sat", "the cat"] * 24})
                 for k in ("test", "validation")}).save_to_disk(data_path)
    settings = Settings(str(source), str(tmp_path/"out"), 8, data=str(data_path), samples=6,
                        device="cpu", dtype="float32", holdout_samples=4)
    run(settings)
    ref = Path(settings.output)/"bias.pt"
    bias = torch.load(ref, weights_only=True)
    assert bias.shape == (2, 8) and torch.isfinite(bias).all()
    settings.reference_bias, settings.output = str(ref), str(tmp_path/"repeat")
    run(settings)
    manifest = json.loads((Path(settings.output)/"manifest.json").read_text())
    assert manifest["comparison"]["kept_set_overlap_per_layer"] == [1, 1]
    assert manifest["comparison"]["rmse_per_layer"] == [0, 0]
    assert manifest["samples_per_distance"] == 6 and len(manifest["heldout"]) == 4
    assert max(x["start_row"] for x in manifest["calibration"]) < min(x["start_row"] for x in manifest["heldout"])
