"""Inference-only adapter for the GLiClass Edge uni-encoder checkpoint.

Input layout and pooling follow Knowledgator/GLiClass (Apache-2.0). This small
adapter uses the project's existing Transformers encoder, PyTorch, safetensors
and tokenizers, without importing the GLiClass package or installing dependencies.
"""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from torch import nn
from transformers import AutoConfig, AutoModel


class _Projector(nn.Module):
    def __init__(self, encoder_size, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(encoder_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, encoder_size)

    def forward(self, x):
        return self.linear_2(torch.nn.functional.gelu(self.linear_1(x)))


class _Scorer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * hidden_size, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, text, labels):
        text = text[:, None].expand_as(labels)
        return self.mlp(torch.cat((text, labels), dim=-1)).squeeze(-1)


class LocalGLiClass(nn.Module):
    def __init__(self, config):
        super().__init__()
        supported = dict(architecture_type="uni-encoder", scorer_type="mlp", pooling_strategy="first",
                         class_token_pooling="average", embed_class_token=True, normalize_features=False,
                         extract_text_features=False, use_segment_embeddings=True)
        for key, expected in supported.items():
            if config.get(key) != expected:
                raise ValueError(f"unsupported GLiClass checkpoint setting {key}={config.get(key)!r}")
        if any(config.get(k, False) for k in ("use_lstm", "layer_wise", "squeeze_layers")):
            raise ValueError("GLiClass adapter supports the Edge inference architecture only")
        if config.get("projector_hidden_act") != "gelu" or config.get("encoder_layer_id", -1) != -1:
            raise ValueError("unsupported GLiClass activation/encoder layer")
        encoder = dict(config["encoder_config"])
        model_type = encoder.pop("model_type")
        if model_type != "modernbert":
            raise ValueError("GLiClass Edge adapter requires its ModernBERT encoder")
        # Avoid optional Flash Attention and automatic torch.compile; no pretrained
        # model is fetched. Both APIs exist in the original supported 4.51.x range.
        encoder["reference_compile"] = False
        self.encoder_model = AutoModel.from_config(
            AutoConfig.for_model(model_type, **encoder), attn_implementation="sdpa")
        size = encoder["hidden_size"]
        self.text_projector = _Projector(size, config["hidden_size"])
        self.classes_projector = _Projector(size, config["hidden_size"])
        self.scorer = _Scorer(size)
        self.segment_embeddings = nn.Embedding(3, size)
        self.logit_scale = nn.Parameter(torch.tensor(config["logit_scale_init_value"]))
        self.class_token_id, self.text_token_id = config["class_token_index"], config["text_token_index"]

    def forward(self, input_ids, attention_mask, num_labels):
        label_mask = input_ids.eq(self.class_token_id)
        if not bool(label_mask.sum(-1).eq(num_labels).all()):
            raise ValueError("GLiClass labels were truncated or reserved label markers appeared in source text")
        positions = torch.arange(input_ids.size(1), device=input_ids.device)[None]
        separator = input_ids.eq(self.text_token_id).int().argmax(-1)
        segments = positions.ge(separator[:, None]).long()
        embeddings = self.encoder_model.get_input_embeddings()(input_ids) + self.segment_embeddings(segments)
        hidden = self.encoder_model(inputs_embeds=embeddings, attention_mask=attention_mask,
                                    return_dict=True).last_hidden_state
        starts = label_mask.nonzero(as_tuple=True)[1].reshape(input_ids.size(0), num_labels)
        ends = torch.cat((starts[:, 1:], attention_mask.sum(-1, keepdim=True)), dim=-1)
        prefix = torch.cat((torch.zeros_like(hidden[:, :1]), hidden.cumsum(1)), dim=1)
        # extract_text_features=False in the published checkpoint: the final label
        # pools to the valid sequence end, matching the upstream inference formula.
        gather = lambda index: prefix.gather(1, index[..., None].expand(-1, -1, hidden.size(-1)))
        labels = (gather(ends) - gather(starts)) / (ends - starts)[..., None].clamp_min(1)
        return self.scorer(self.text_projector(hidden[:, 0]), self.classes_projector(labels))


class LocalGLiClassPipeline:
    def __init__(self, model_dir, *, device="cpu", max_length=1024):
        root = Path(model_dir)
        config = json.loads((root / "config.json").read_text())
        self.model = LocalGLiClass(config)
        state = load_file(str(root / "model.safetensors"), device="cpu")
        if any(not name.startswith("model.") for name in state):
            raise ValueError("unexpected GLiClass checkpoint namespace")
        self.model.load_state_dict({name.removeprefix("model."): value for name, value in state.items()}, strict=True)
        self.device = torch.device(device)
        self.model.eval().requires_grad_(False).to(self.device)
        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        for token, expected in (("<<LABEL>>", config["class_token_index"]), ("<<SEP>>", config["text_token_index"])):
            if self.tokenizer.token_to_id(token) != expected:
                raise ValueError(f"GLiClass tokenizer/config disagree about {token}")
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=config["pad_token_id"], pad_token="[PAD]")
        self.prompt_first = config["prompt_first"]

    @torch.inference_mode()
    def __call__(self, texts, labels, *, threshold=0.0, batch_size=32, prompt=None):
        if isinstance(texts, str):
            texts = [texts]
        prefix = "".join("<<LABEL>>" + label for label in labels) + "<<SEP>>" + (prompt or "")
        results = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer.encode_batch([
                prefix + text if self.prompt_first else text + prefix for text in texts[start:start + batch_size]])
            ids = torch.tensor([entry.ids for entry in encoded], device=self.device)
            mask = torch.tensor([entry.attention_mask for entry in encoded], device=self.device)
            probabilities = self.model(ids, mask, len(labels)).sigmoid().cpu().tolist()
            results.extend([[dict(label=label, score=score) for label, score in zip(labels, row)
                             if score >= threshold] for row in probabilities])
        return results
