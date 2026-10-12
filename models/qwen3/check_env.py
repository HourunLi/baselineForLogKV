"""Check the actual APIs needed by the official CUDA implementation."""

import argparse
from importlib.metadata import version


def check_environment(training_only: bool = False) -> dict[str, str]:
    import torch
    from torch.nn.attention.flex_attention import AuxRequest, flex_attention  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("DMS training/inference requires an NVIDIA CUDA GPU; CPU/MPS/NPU are unsupported.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This recipe requires a GPU with bfloat16 support.")
    if version("transformers") != "4.57.3":
        raise RuntimeError("Install the pinned dependencies: pip install -e .")
    versions = {name: version(name) for name in ("torch", "transformers", "datasets", "accelerate")}
    if not training_only:
        from flash_attn import flash_attn_with_kvcache  # noqa: F401

        versions["flash-attn"] = version("flash-attn")
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
