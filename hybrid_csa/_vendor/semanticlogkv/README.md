# HybridCSA bundled runtime

This private snapshot supplies the LitGPT model, dense CPT driver, and SemanticLogKV
evaluation protocol used by HybridCSA. All executable code is part of this project;
no sibling checkout or installed `litgpt` package is imported. `SOURCE.json` records
the source commit and file SHA-256 hashes, and `LICENSE.md` preserves Apache-2.0.

The snapshot contains the import closure of `demo.py`, `eval.py`, `data.py` and
`utils.py`, including delayed imports for GPU kernels and checkpoint conversion.
LitGPT's public API, CLI, dataset catalog and unrelated commands are omitted.
Local changes only isolate imports, preserve old prompt-style class paths, and
let `HYBRID_CSA_HF_CACHE` choose the evaluation cache root. Model math, training loops,
request handling and scoring are retained from the recorded source.

Updates are explicit: inspect upstream changes, port the required changes here,
refresh the source and bundled hashes, and run HybridCSA's focused tests. Never use
`sys.path` injection or a fallback to another checkout. Existing checkpoints use
tensor state dictionaries and YAML configs; Python-object model pickles are not
a supported checkpoint format.
