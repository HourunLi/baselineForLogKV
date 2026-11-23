"""Download the published native HAttention checkpoint as verified ordinary files."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from .unused.fetch_long_eval_data import patch_requests

REPO = "hanguo/log-linear-attention"
REVISION = "29374fef386e0b8dcce9bfbd14cbcee8334954a1"
FILES = ("config.json", "generation_config.json", "special_tokens_map.json", "tokenizer.json",
         "tokenizer_config.json", "LICENSE", "model.safetensors")


def download(folder, verify_ssl):
    patch_requests(verify_ssl)
    import requests

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    response = requests.get(f"{endpoint}/api/models/{REPO}/revision/{REVISION}", params={"blobs": "true"}, timeout=60)
    response.raise_for_status()
    info = response.json()
    if info["sha"] != REVISION:
        raise ValueError("unexpected checkpoint revision")
    entries = {item["rfilename"]: item for item in info["siblings"]}
    folder.mkdir(parents=True, exist_ok=True)
    verified_files = {}
    for name in FILES:
        remote = "mamba-2/" + name
        item = entries[remote]
        path = folder / name
        if path.is_symlink():
            raise ValueError(f"expected an ordinary file: {path}")
        size = item["size"]
        def valid():
            if not path.is_file() or path.stat().st_size != size:
                return False
            lfs = item.get("lfs")
            digest = hashlib.sha256() if lfs else hashlib.sha1(f"blob {size}\0".encode())
            with path.open("rb") as source:
                for block in iter(lambda: source.read(4 << 20), b""):
                    digest.update(block)
            return digest.hexdigest() == (lfs["sha256"] if lfs else item["blobId"])
        if not valid():
            url = f"{endpoint}/{REPO}/resolve/{REVISION}/{remote}"
            with requests.get(url, stream=True, timeout=(60, 300)) as response:
                response.raise_for_status()
                with path.open("wb") as dest:
                    for block in response.iter_content(4 << 20):
                        dest.write(block)
            if not valid():
                raise IOError(f"incomplete or corrupt download: {path}")
        stat = path.stat()
        verified_files[name] = {"size": size, "mtime_ns": stat.st_mtime_ns,
                                "digest": item.get("lfs", {}).get("sha256") or item["blobId"]}
        print(f"verified {path} ({size} bytes)", flush=True)

    (folder / "hattention_source.json").write_text(json.dumps({
        "repository": REPO, "revision": REVISION, "subfolder": "mamba-2", "files": verified_files
    }, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--verify-ssl", action="store_true")
    args = p.parse_args()
    download(args.folder, args.verify_ssl)
