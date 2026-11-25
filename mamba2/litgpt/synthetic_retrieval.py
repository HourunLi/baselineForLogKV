"""Synthetic retrieval documents for the learned retention heads (niahLogKV).

``insert_facts`` puts short facts (a key and a value of one of several kinds:
numbers, decimals, codes, uuids, words, names, dates) into a block of
training token ids at random sentence boundaries and appends questions that
ask for some of them, with their answers, at the end. The block keeps its
length: the haystack's own tail is dropped. The returned mask marks the fact
tokens; the heads learn to score those spans above the rest (labels), and
CPT on such blocks teaches the LM to answer from what the exact pool kept.

Templates never use RULER's wording ("special magic", "One of the ...",
"mentioned in the provided text"), so NIAH evaluates unseen sentences.
Keys and word values come from the tokenizer's own single-token words.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

import numpy as np

KINDS = ("number", "decimal", "code", "uuid", "word", "name", "date")
FACT_TEMPLATES = (
    "The {key} code is {value}.",
    "Remember that the {key} is {value}.",
    "For the record, {key} = {value}.",
    "Note: the {key} was set to {value}.",
    "Our {key} is {value}; please keep it.",
    "{Key}: {value}.",
    "The value stored under {key} is {value}.",
    "I was told the {key} is {value}.",
    "Keep in mind that {key} corresponds to {value}.",
    "Write this down - {key}: {value}.",
    "The {key} turned out to be {value}.",
    "Last time we checked, the {key} read {value}.",
    "Everyone agreed the {key} should be {value}.",
    "In the ledger, {key} is listed as {value}.",
)
QUESTION_TEMPLATES = (
    "Question: What is the {key}?\nAnswer: {value}",
    "Q: What was the {key}?\nA: {value}",
    "What is {key} set to? {value}",
    "Recall the {key}: {value}",
    "The {key} given earlier is {value}",
)
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")
# Fallbacks when a vocabulary has too few single-token words (tests, small tokenizers).
_WORDS = ("amber", "basin", "candle", "delta", "ember", "falcon", "garnet", "harbor", "island", "jasper", "kettle",
          "lantern", "meadow", "nectar", "orchid", "pepper", "quartz", "raven", "saddle", "timber", "umber", "velvet",
          "willow", "yonder", "zephyr", "anchor", "breeze", "copper", "dune", "fable")
_NAMES = ("Alder", "Brenna", "Corwin", "Dalia", "Emrys", "Fiona", "Garrick", "Helena", "Idris", "Juno", "Kestrel",
          "Lorne", "Marisol", "Niall", "Oriel", "Perrin", "Quinlan", "Rowena", "Soren", "Talia")
_WORD_RE = re.compile(r"^ [a-z]{4,10}$")
_NAME_RE = re.compile(r"^ [A-Z][a-z]{3,9}$")


@dataclass
class Fact:
    key: str
    value: str
    text: str
    question: str


class RetrievalKit:
    """Word lists, an encoder and sentence-boundary ids for one tokenizer."""

    def __init__(self, texts, encode, boundary_ids, kinds=KINDS) -> None:
        words = sorted({t[1:] for t in texts if _WORD_RE.match(t)})
        names = sorted({t[1:] for t in texts if _NAME_RE.match(t)})
        self.words = words if len(words) >= 200 else list(_WORDS)
        self.names = names if len(names) >= 100 else list(_NAMES)
        self.encode = encode
        self.boundary = np.zeros(max(boundary_ids, default=0) + 1, dtype=bool)
        self.boundary[list(boundary_ids)] = True
        unknown = set(kinds) - set(KINDS)
        if unknown or not kinds:
            raise ValueError(f"synthetic value kinds must be a non-empty subset of {KINDS}, got {kinds}")
        self.kinds = tuple(kinds)

    @classmethod
    def from_tokenizer(cls, tokenizer, kinds=KINDS) -> "RetrievalKit":
        return cls(tokenizer.gamma_literal_texts, lambda s: tokenizer.encode(s, bos=False).tolist(),
                   tokenizer.alpha_span_boundary_ids, kinds)

    def _word(self, rng) -> str:
        return self.words[rng.integers(len(self.words))]

    def _name(self, rng) -> str:
        return self.names[rng.integers(len(self.names))]

    def key(self, rng) -> str:
        form = rng.integers(3)
        if form == 0:
            return f"{self._word(rng)}-{self._word(rng)}"
        if form == 1:
            return f"{self._word(rng)} {self._word(rng)}"
        return f"{self._name(rng)}'s {self._word(rng)}"

    def value(self, rng, kind: str) -> str:
        if kind == "number":
            digits = int(rng.integers(3, 11))
            return str(int(rng.integers(1, 10))) + "".join(str(d) for d in rng.integers(0, 10, digits - 1))
        if kind == "decimal":
            return f"{int(rng.integers(1, 10_000))}.{int(rng.integers(0, 100)):02d}"
        if kind == "code":
            if rng.integers(2):
                letters = "".join(chr(65 + c) for c in rng.integers(0, 26, 2))
                return f"{letters}-{int(rng.integers(100, 1_000_000))}"
            return "".join("0123456789abcdef"[c] for c in rng.integers(0, 16, int(rng.integers(6, 13))))
        if kind == "uuid":
            return str(uuid.UUID(int=int.from_bytes(rng.bytes(16), "big"), version=4))
        if kind == "word":
            return " ".join(self._word(rng) for _ in range(int(rng.integers(1, 3))))
        if kind == "name":
            return f"{self._name(rng)} {self._name(rng)}"
        if kind == "date":
            year, month, day = int(rng.integers(1900, 2030)), int(rng.integers(12)), int(rng.integers(1, 29))
            return f"{MONTHS[month]} {day}, {year}" if rng.integers(2) else f"{year}-{month + 1:02d}-{day:02d}"
        raise ValueError(kind)

    def fact(self, rng, used_keys=()) -> Fact:
        key = self.key(rng)
        while key in used_keys:
            key = self.key(rng)
        value = self.value(rng, self.kinds[rng.integers(len(self.kinds))])
        template = FACT_TEMPLATES[rng.integers(len(FACT_TEMPLATES))]
        text = template.format(key=key, Key=key[:1].upper() + key[1:], value=value)
        question = QUESTION_TEMPLATES[rng.integers(len(QUESTION_TEMPLATES))].format(key=key, value=value)
        return Fact(key, value, text, question)


def insert_facts(ids, rng, kit: RetrievalKit, *, facts=(2, 8), questions=(1, 3), snap=256, return_facts=False):
    """(ids, fact mask) of the same length as ``ids`` (int64 / bool numpy arrays).

    Each fact starts right after a sentence boundary when one occurs within
    ``snap`` tokens of its random position. Questions for a random subset of
    the facts, with their answers, end the block. With ``return_facts`` the
    facts in block order and the asked ones are returned too.
    """
    ids = np.asarray(ids, dtype=np.int64)
    n = len(ids)
    count = int(rng.integers(facts[0], facts[1] + 1))
    chosen: list[Fact] = []
    for _ in range(count):
        chosen.append(kit.fact(rng, {f.key for f in chosen}))
    asked = sorted(rng.choice(count, min(count, int(rng.integers(questions[0], questions[1] + 1))), replace=False))
    tail = np.asarray(kit.encode("\n\n" + "\n\n".join(chosen[i].question for i in asked)), dtype=np.int64)
    fact_ids = [np.asarray(kit.encode(" " + f.text), dtype=np.int64) for f in chosen]
    keep = n - len(tail) - sum(map(len, fact_ids))
    if keep < n // 4:
        raise ValueError(f"a block of {n} tokens is too short for {count} facts and their questions")
    hay = ids[:keep]
    ends = np.flatnonzero(kit.boundary[np.minimum(hay, len(kit.boundary) - 1)] & (hay < len(kit.boundary))) + 1
    points = []
    for p in rng.integers(0, keep + 1, count):
        j = np.searchsorted(ends, p)
        points.append(int(ends[j]) if j < len(ends) and ends[j] - p <= snap else int(p))
    order = np.argsort(points, kind="stable")
    pieces, mask, at = [], [], 0
    for i in order:
        pieces += [hay[at:points[i]], fact_ids[i]]
        mask += [np.zeros(points[i] - at, dtype=bool), np.ones(len(fact_ids[i]), dtype=bool)]
        at = points[i]
    pieces += [hay[at:], tail]
    mask += [np.zeros(keep - at, dtype=bool), np.zeros(len(tail), dtype=bool)]
    out, out_mask = np.concatenate(pieces), np.concatenate(mask)
    assert len(out) == n
    if return_facts:
        return out, out_mask, [chosen[i] for i in order], [chosen[i] for i in asked]
    return out, out_mask
