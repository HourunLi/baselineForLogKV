"""Download the published native HAttention checkpoint as verified ordinary files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from threading import Event, Thread
from time import monotonic

from .unused.fetch_long_eval_data import patch_requests

REPO = "hanguo/log-linear-attention"
REVISION = "29374fef386e0b8dcce9bfbd14cbcee8334954a1"
FILES = ("config.json", "generation_config.json", "special_tokens_map.json", "tokenizer.json",
         "tokenizer_config.json", "LICENSE", "model.safetensors")


class TransferProgress:
    """Print newline-delimited updates, including when a network read stalls."""

    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.received = 0
        self.stop = Event()
        self.thread = Thread(target=self.monitor, daemon=True)

    def report(self, previous, started):
        count = self.received
        speed = (count - previous) / max(monotonic() - started, 1e-9)
        remaining = max(self.size - count, 0)
        eta = "0s" if not remaining else (f"{remaining / speed:.0f}s" if speed else "estimating")
        print(f"{self.name}  {100 * count / max(self.size, 1):.1f}%  "
              f"{count / 2**20:.2f}/{self.size / 2**20:.2f} MiB  "
              f"{speed / 2**20:.2f} MiB/s  ETA {eta}", flush=True)
        return count

    def monitor(self):
        previous, started = 0, self.started
        while not self.stop.wait(5):
            previous = self.report(previous, started)
            started = monotonic()

    def __enter__(self):
        self.started = monotonic()
        self.report(0, self.started)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        self.thread.join()
        self.report(0, self.started)


def download(folder, verify_ssl):
    patch_requests(verify_ssl)
    import requests

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    print(f"Fetching checkpoint metadata from {endpoint} ...", flush=True)
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
            print(f"Checking file digest: {path} ({size / 2**20:.2f} MiB)", flush=True)
            lfs = item.get("lfs")
            digest = hashlib.sha256() if lfs else hashlib.sha1(f"blob {size}\0".encode())
            with path.open("rb") as source:
                for block in iter(lambda: source.read(4 << 20), b""):
                    digest.update(block)
            return digest.hexdigest() == (lfs["sha256"] if lfs else item["blobId"])
        if not valid():
            url = f"{endpoint}/{REPO}/resolve/{REVISION}/{remote}"
            with TransferProgress(name, size) as progress:
                with requests.get(url, stream=True, timeout=(60, 300)) as response:
                    response.raise_for_status()
                    with path.open("wb") as dest:
                        for block in response.iter_content(256 << 10):
                            dest.write(block)
                            progress.received += len(block)
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
