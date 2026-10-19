"""Check the actual APIs needed by the official CUDA implementation."""

import argparse
import re
from importlib.metadata import PackageNotFoundError, version

# Training and evaluation both run in SemanticLogKV's environment (torch218, transformers 4.51.3) without
# changing its packages: Qwen3 needs transformers>=4.51, dms.transformers_compat supplies the newer APIs DMS
# uses, and dms.training.loop trains without HF Trainer/accelerate.
TRANSFORMERS_MIN = (4, 51)


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def check_environment(training_only: bool = False) -> dict[str, str | None]:
    import torch
    from torch.nn.attention.flex_attention import AuxRequest, flex_attention  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("DMS training/inference requires an NVIDIA CUDA GPU; CPU/MPS/NPU are unsupported.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This recipe requires a GPU with bfloat16 support.")
    transformers = _version("transformers")
    if transformers is None or tuple(int(p) for p in re.findall(r"\d+", transformers)[:2]) < TRANSFORMERS_MIN:
        raise RuntimeError(f"DMS needs transformers>=4.51 (Qwen3), found {transformers}; "
                           "run in SemanticLogKV's environment without changing its packages")
    versions = {name: _version(name) for name in ("torch", "transformers", "datasets")}
    if not training_only:
        from flash_attn import flash_attn_with_kvcache  # noqa: F401

        versions["flash-attn"] = _version("flash-attn")
    versions["gpu"] = torch.cuda.get_device_name()
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-only", action="store_true", help="Do not require FlashAttention decoding")
    args = parser.parse_args()
    for name, value in check_environment(args.training_only).items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
