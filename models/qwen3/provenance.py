"""File hashes and the check that a prepared/evaluated model descends from the comparison Base."""

import hashlib
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_base_source(provenance, tokenizer_dir):
    """Fail closed when a model cannot be tied to the comparison Base."""
    if not provenance or not provenance.get("tokenizer_files"):
        raise ValueError("Missing semanticlogkv_base provenance; prepare the shared Base first")
    source = Path(tokenizer_dir).resolve()
    if file_sha256(source / "lit_model.pth") != provenance["source_weight_sha256"]:
        raise ValueError("Prepared Base weights differ from the SemanticLogKV comparison checkpoint")
    for name, expected in provenance.get("source_config_files", {}).items():
        if file_sha256(source / name) != expected:
            raise ValueError(f"Comparison Base config differs from the prepared source: {name}")
    for name, expected in provenance["tokenizer_files"].items():
        if file_sha256(source / name) != expected:
            raise ValueError(f"Comparison tokenizer differs from the prepared Base tokenizer: {name}")
