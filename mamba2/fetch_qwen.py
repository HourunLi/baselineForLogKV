"""Fetch ordinary HF weight/tokenizer files using LogKV's asset-download policy."""
import argparse
from pathlib import Path

from unused.fetch_long_eval_data import patch_requests
from unused.long_eval_download import download_plain
from unused.fetch_progress import ProgressLines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--revision", default="main")
    p.add_argument("--verify-ssl", action="store_true")
    args = p.parse_args()
    patch_requests(args.verify_ssl)
    from huggingface_hub import HfApi
    repo = "Qwen/Qwen3-8B-Base"
    info = HfApi().model_info(repo, revision=args.revision, files_metadata=True, timeout=60)
    names = [f.rfilename for f in info.siblings if "/" not in f.rfilename and
             (f.rfilename.endswith(".safetensors") or f.rfilename in {
                 "model.safetensors.index.json", "config.json", "tokenizer.json", "tokenizer_config.json",
                 "vocab.json", "merges.txt", "generation_config.json", "special_tokens_map.json"})]
    if not {"config.json", "tokenizer.json", "model.safetensors.index.json"}.issubset(names):
        raise ValueError(f"Incomplete Qwen repository at {info.sha}")
    args.out.mkdir(parents=True, exist_ok=True)
    with ProgressLines(args.out / "download.log") as log:
        for name in sorted(names):
            download_plain(repo, name, args.out, log, model_info=info)


if __name__ == "__main__":
    main()
