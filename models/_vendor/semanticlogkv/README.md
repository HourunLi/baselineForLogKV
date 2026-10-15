# the CaM baseline bundled evaluation runtime

This private snapshot supplies the LitGPT model, tokenizer, generation loop and the
SemanticLogKV evaluation protocol (`eval.py`) used by the CaM baseline. All executable code is
part of this project; no sibling checkout or installed `litgpt` package is imported.
`SOURCE.json` records the SemanticLogKV source commit, the original file SHA-256 hashes
and the bundled hashes; `LICENSE.md` preserves Apache-2.0.

Only the import closure of `eval.py` is kept (including delayed imports for GPU
kernels and checkpoint conversion). The SemanticLogKV training driver (`demo.py`),
its data pipeline (`data.py`), LitGPT's public API, CLI and dataset catalog are omitted.
Local changes only isolate imports in the `models._vendor.semanticlogkv` namespace, preserve old
prompt-style class paths, and let `CAM_HF_CACHE` choose the evaluation cache root.
Model math, request handling, generation and scoring are retained from the source.

Updates are explicit: inspect upstream changes, port the required changes here,
refresh the source and bundled hashes, and rerun this project's tests. Never use
`sys.path` injection or a fallback to another checkout.
