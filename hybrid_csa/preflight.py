"""Read-only checks in the interpreter that will actually run each phase."""

import argparse
from contextlib import redirect_stdout
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "eval"), required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--hybrid-csa-config", type=Path)
    args = parser.parse_args()
    import torch

    try:
        torch.cuda.init()
    except Exception as error:
        raise RuntimeError(f"CUDA initialisation failed in {sys.executable}: {error}; "
                           f"torch CUDA={torch.version.cuda}; "
                           f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')}") from error
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This baseline requires CUDA BF16 support")
    for name in ("lit_model.pth", "model_config.yaml", "tokenizer_config.json"):
        if not (args.base / name).is_file():
            raise FileNotFoundError(args.base / name)
    packages = ("torch", "lightning", "jsonargparse", "tokenizers", "PyYAML", "safetensors")
    with redirect_stdout(sys.stderr):
        if args.phase == "eval":
            from .evaluate import load_evaluator
            load_evaluator()
            packages += ("lm-eval", "datasets", "pyarrow", "transformers", "pandas", "xlsxwriter")
        else:
            import litdata
            import pyarrow
            from ._vendor.semanticlogkv.litgpt.tokenizer import Tokenizer
            Tokenizer(args.base)
            packages += ("litdata", "pyarrow")
    result = {"python_executable": sys.executable, "python": sys.version,
              "versions": {name: importlib.metadata.version(name) for name in packages},
              "installed_packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
              "visible_gpus": torch.cuda.device_count(), "torch_cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0)}
    if args.adapter:
        from .checkpoint import base_identity, sha256, validate_payload
        from .evaluate import load_hybrid_csa_options
        payload = torch.load(args.adapter, map_location="cpu", weights_only=True)
        validate_payload(payload, load_hybrid_csa_options(args.hybrid_csa_config), base_identity(args.base))
        result["adapter"] = {"path": str(args.adapter.resolve()), "sha256": sha256(args.adapter),
                             "stage": payload["stage"], "training": payload["training"]}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
