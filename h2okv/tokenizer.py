"""Tokenizer with the same special-token policy as SemanticLogKV's evaluator.

Reimplemented here (no import from that project) so prompts, BOS insertion and
EOS stripping produce identical token ids: special ids come from
``tokenizer_config.json`` then ``generation_config.json``; ``encode`` adds BOS
only when requested or when the checkpoint uses BOS, and drops a BOS/EOS the
backend added on its own.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch


def _load_json_tolerant(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Some published generation configs carry trailing commas.
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", text))


class Tokenizer:
    def __init__(self, checkpoint_dir: Path | str) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise NotADirectoryError(f"The checkpoint directory does not exist: {checkpoint_dir}")
        self.model_name = checkpoint_dir.stem
        self.use_bos = self.check_if_bos_token_used(checkpoint_dir)
        self.bos_id = None
        self.eos_id = None

        if (vocabulary_path := checkpoint_dir / "tokenizer.json").is_file():
            from tokenizers import Tokenizer as HFTokenizer

            self.processor = HFTokenizer.from_file(str(vocabulary_path))
            self.backend = "huggingface"
            if (config_path := checkpoint_dir / "tokenizer_config.json").is_file():
                config = json.loads(config_path.read_text(encoding="utf-8"))
                bos_token, eos_token = config.get("bos_token"), config.get("eos_token")
                if isinstance(bos_token, dict):
                    bos_token = bos_token.get("content")
                if isinstance(eos_token, dict):
                    eos_token = eos_token.get("content")
                self.bos_id = self.token_to_id(bos_token) if bos_token is not None else None
                self.eos_id = self.token_to_id(eos_token) if eos_token is not None else None
            if (generation_path := checkpoint_dir / "generation_config.json").is_file():
                config = _load_json_tolerant(generation_path)
                if self.bos_id is None:
                    self.bos_id = config.get("bos_token_id")
                if self.eos_id is None:
                    self.eos_id = config.get("eos_token_id")
        elif (vocabulary_path := checkpoint_dir / "tokenizer.model").is_file():
            from sentencepiece import SentencePieceProcessor

            self.processor = SentencePieceProcessor(model_file=str(vocabulary_path))
            self.backend = "sentencepiece"
            self.bos_id = self.processor.bos_id()
            self.eos_id = self.processor.eos_id()
        else:
            raise NotImplementedError(f"No tokenizer.json or tokenizer.model under {checkpoint_dir}")

        self.apply_decoding_fix = None
        if (config_path := checkpoint_dir / "tokenizer_config.json").is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.apply_decoding_fix = "LlamaTokenizer" in config.get("tokenizer_class", "")

    @property
    def vocab_size(self) -> int:
        if self.backend == "huggingface":
            return self.processor.get_vocab_size(with_added_tokens=False)
        return self.processor.vocab_size()

    def token_to_id(self, token: str) -> int:
        id_ = (self.processor.token_to_id(token) if self.backend == "huggingface"
               else self.processor.piece_to_id(token))
        if id_ is None:
            raise ValueError(f"token {token!r} not found in the collection.")
        return id_

    @staticmethod
    def check_if_bos_token_used(checkpoint_dir: Path) -> bool:
        if not (config_path := checkpoint_dir / "tokenizer_config.json").is_file():
            return False
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if checkpoint_dir.stem.startswith(("Meta-Llama-3", "Llama-3")):
            return True
        if checkpoint_dir.stem.startswith("SmolLM2") and checkpoint_dir.name.endswith("Instruct"):
            return True
        if "add_bos_token" in config:
            return config["add_bos_token"]
        return config.get("tokenizer_class") == "LlamaTokenizer"

    def encode(self, string: str, device: torch.device | None = None, bos: bool | None = None,
               eos: bool = False, max_length: int = -1) -> torch.Tensor:
        tokens = (self.processor.encode(string).ids if self.backend == "huggingface"
                  else self.processor.encode(string))
        if bos or (bos is None and self.use_bos):
            if self.bos_id is None:
                raise NotImplementedError("This tokenizer does not have a defined bos token.")
            if not tokens or tokens[0] != self.bos_id:
                tokens = [self.bos_id] + tokens
        elif tokens and tokens[0] == self.bos_id:
            tokens = tokens[1:]
        if eos and (not tokens or tokens[-1] != self.eos_id):
            tokens = tokens + [self.eos_id]
        elif tokens and tokens[-1] == self.eos_id:
            tokens = tokens[:-1]
        if max_length > 0:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        if len(tokens) == 1 and self.apply_decoding_fix:
            dummy_token_id = 33  # \x1e
            dummy_token = self.processor.decode([dummy_token_id])
            if dummy_token != "\x1e":
                dummy_token_id = 165
                dummy_token = self.processor.decode([dummy_token_id])
            return self.processor.decode([dummy_token_id] + tokens)[len(dummy_token):]
        return self.processor.decode(tokens)
