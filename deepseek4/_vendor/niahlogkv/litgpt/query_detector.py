"""Optional, request-level GLiClass task-span detection. No LM or benchmark labels are used."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from time import perf_counter

DEFAULT_MODEL_DIR = "/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/gliclass-instruct-edge"

LABELS = (
    "current question or instruction addressed to the assistant",
    "quoted question, dialogue, historical task or example inside the document",
    "ordinary document content",
)
TASK_PROMPT = (
    "Classify only the text between <candidate> and </candidate>, using its surrounding context. "
    "Identify a current request, question, instruction or memory cue for the assistant. "
    "Questions and instructions merely quoted in source documents are not current requests."
)
# General linguistic candidates, independent of benchmark templates and absolute position.
_REQUEST = re.compile(
    r"\b(?:what|which|who|whom|whose|where|when|why|how|please|"
    r"answer|summari[sz]e|explain|describe|compare|list|extract|identify|find|"
    r"calculate|compute|return|provide|write|translate|remember|recall|"
    r"keep in mind|focus on|tell me|give me|can you|could you|would you)\b"
    r"|[?？]|请|回答|总结|概括|解释|提取|列出|找出|计算|记住|注意|关注|是什么|为什么|如何|哪里",
    re.IGNORECASE,
)
_BOUNDARY = re.compile(r"(?:[.!?][\"'”’)\]]*(?=\s|$)|[。！？]|[\r\n]+)")


@dataclass(frozen=True)
class TaskSpan:
    start: int
    end: int
    score: float


def candidates(text: str, max_chars: int = 768, use_rules: bool = True):
    """Yield bounded, position-preserving sentences throughout the prompt, including its tail."""
    start = 0
    for stop in chain((m.end() for m in _BOUNDARY.finditer(text)), (len(text),)):
        # Split long unpunctuated runs too: no implicit first/last-window restriction.
        while start < stop:
            end = min(start + max_chars, stop)
            if end < stop:
                split = text.rfind(" ", start + max_chars // 2, end)
                if split > start:
                    end = split + 1
            lo, hi = start, end
            while lo < hi and text[lo].isspace():
                lo += 1
            while hi > lo and text[hi - 1].isspace():
                hi -= 1
            if lo < hi and (not use_rules or _REQUEST.search(text[lo:hi])):
                yield lo, hi
            start = end


def validate_local_model(path: str | Path) -> Path:
    """Require regular local artifacts, never Hub IDs, symlinks or remote-code models."""
    root = Path(path).expanduser().absolute()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"query detector needs a local ordinary directory: {root}")
    for p in root.rglob("*"):
        if p.is_symlink():
            raise ValueError(f"query detector bundle contains a symlink: {p}")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"missing local GLiClass artifact: {root / name}")
    config = json.loads((root / "config.json").read_text())
    if config.get("architecture_type") != "uni-encoder" or not isinstance(config.get("encoder_config"), dict):
        raise ValueError("query detector requires a self-contained GLiClass uni-encoder config")
    return root


class QueryDetector:
    """Load once per eval rank; scan once per visible generation request, before LM prefill."""

    def __init__(self, model_dir, *, device="cpu", threshold=0.8, batch_size=32,
                 max_chars=512, context_chars=128, max_spans=8, use_rules=True, pipeline=None):
        started = perf_counter()
        if not 0 <= threshold <= 1 or batch_size < 1 or max_chars < 32 or context_chars < 0 or max_spans < 1:
            raise ValueError("invalid query detector threshold/batch/length/span limits")
        self.threshold, self.batch_size = float(threshold), int(batch_size)
        self.max_chars, self.context_chars, self.max_spans = int(max_chars), int(context_chars), int(max_spans)
        self.use_rules = bool(use_rules)
        self.calls = self.scanned = self.selected = 0
        self.seconds = 0.0
        self.load_seconds = 0.0
        self.revision = None
        if pipeline is not None:  # Dependency injection for deterministic, offline tests.
            self.pipeline = pipeline
            return
        root = validate_local_model(model_dir)
        if (root / "bundle.json").is_file():
            self.revision = json.loads((root / "bundle.json").read_text()).get("revision")
        from deepseek4._vendor.niahlogkv.litgpt.gliclass_local import LocalGLiClassPipeline
        self.pipeline = LocalGLiClassPipeline(root, device=device)
        self.load_seconds = perf_counter() - started

    def detect(self, text: str) -> list[TaskSpan]:
        started = perf_counter()
        accepted = []
        batch, bounds = [], []

        def classify():
            if not batch:
                return
            results = self.pipeline(batch, list(LABELS), threshold=0.0, batch_size=self.batch_size,
                                    prompt=TASK_PROMPT)
            if len(results) != len(bounds):
                raise RuntimeError("GLiClass returned an unexpected number of predictions")
            for (lo, hi), result in zip(bounds, results):
                scores = {r["label"]: float(r["score"]) for r in result}
                positive = scores.get(LABELS[0], 0.0)
                if math.isfinite(positive) and positive >= self.threshold and positive > max(
                        scores.get(label, 0.0) for label in LABELS[1:]):
                    accepted.append(TaskSpan(lo, hi, positive))
            self.scanned += len(bounds)
            batch.clear()
            bounds.clear()

        for lo, hi in candidates(text, self.max_chars, self.use_rules):
            context = self.context_chars
            batch.append(text[max(0, lo - context):lo] + "\n<candidate>" + text[lo:hi] +
                         "</candidate>\n" + text[hi:hi + context])
            bounds.append((lo, hi))
            if len(batch) >= self.batch_size:
                classify()
        classify()
        # Reject-all is valid. Bound the number of cues independently of prompt length.
        selected = sorted(sorted(accepted, key=lambda s: (-s.score, s.start))[:self.max_spans],
                          key=lambda s: s.start)
        merged = []
        for span in selected:
            if merged and not text[merged[-1].end:span.start].strip():
                prev = merged.pop()
                span = TaskSpan(prev.start, span.end, max(prev.score, span.score))
            merged.append(span)
        self.calls += 1
        self.selected += len(merged)
        self.seconds += perf_counter() - started
        return merged

    def stats(self):
        return dict(calls=self.calls, candidates=self.scanned, spans=self.selected, seconds=self.seconds,
                    load_seconds=self.load_seconds, model_revision=self.revision)


def cue_ids(text, spans, tokenizer):
    """Select original LM token IDs by character offsets; no detector vocabulary is used."""
    if not spans:
        return []
    if getattr(tokenizer, "backend", None) == "huggingface":
        encoded = tokenizer.processor.encode(text)
        ordered = iter(spans)
        span = next(ordered, None)
        ids = []
        for token, (lo, hi) in zip(encoded.ids, encoded.offsets):
            while span is not None and span.end <= lo:
                span = next(ordered, None)
            if span is None:
                break
            if hi > lo and lo < span.end and hi > span.start:
                ids.append(token)
        return ids
    # SentencePiece/dummy tokenizers may not expose source offsets. Include preceding
    # whitespace to preserve word-prefix variants when encoding complete selected sentences.
    ids = []
    for span in spans:
        # Include preceding whitespace for byte-level BPE word variants.
        start = span.start
        while start and text[start - 1].isspace():
            start -= 1
        part = tokenizer.encode(text[start:span.end], bos=False)
        ids.extend(part.tolist() if hasattr(part, "tolist") else part)
    return ids
