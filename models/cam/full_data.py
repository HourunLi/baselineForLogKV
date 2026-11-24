"""Resolve the full CaM run's offline assets in the same way on each node and worker."""
import importlib.metadata
import os
from pathlib import Path
import sys


def ppl_sources(length):
    home = Path(os.environ["HF_HOME"])
    ultra = length >= 524288
    sources = {}
    for key, folder in (("LONG_PPL_DATA", "long_ppl_data/emozilla--pg19-test"),
                        ("LONG_PPL_BOOKS", "long_ppl_books/emozilla--pg19")):
        default = home / folder
        if key == "LONG_PPL_BOOKS":
            large = home / "long_ppl_books_1m/emozilla--pg19"
            if large.is_dir() and (ultra or not default.exists()):
                default = large
        value = (os.getenv(key + "_1M") if ultra else None) or os.getenv(key) or default
        sources[key] = str(Path(value).expanduser().absolute())
    return sources


def check_full_data(model, tokenizer, generator):
    import datasets
    import nltk
    import torch
    from transformers import AutoTokenizer

    from models._vendor.niahlogkv import eval as reference
    from models._vendor.niahlogkv.eval_tasks.babilong_long import babilong_long_utils as babi
    from models._vendor.niahlogkv.eval_tasks.long_ppl import long_ppl_utils as ppl
    from models._vendor.niahlogkv.litgpt import ruler_patch as ruler

    versions = {p: importlib.metadata.version(p) for p in ("torch", "transformers", "lm-eval", "datasets", "pyarrow")}
    print(f"Full CaM: Python {sys.version.split()[0]}, {versions}, CUDA {torch.version.cuda}, "
          f"visible GPUs {torch.cuda.device_count()}, HF_HOME={os.environ['HF_HOME']}", flush=True)
    problems = []

    def check(label, action, *args, **kwargs):
        try:
            return action(*args, **kwargs)
        except (OSError, ValueError, RuntimeError, LookupError) as error:
            problems.append(f"{label}: {error}")

    def asset(path):
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(path)

    checkpoint = Path(reference._resolve_checkpoint_dir(model))
    check("Gamma CPT", reference._checkpoint_file_for_loading, checkpoint / "lit_model.pth")
    for directory in {tokenizer, generator}:
        check("Qwen tokenizer", asset, Path(directory) / "tokenizer.json")
    for file in ("dev-v2.0.json", "hotpot_dev_distractor_v1.json"):
        check("RULER", asset, Path(os.environ["RULER_CACHE_DIR"]) / file)
    essay = ruler._essay_file()
    if essay.is_file() or os.getenv("RULER_DATASET_CACHE_DIR"):
        if essay.is_symlink():
            problems.append(f"RULER essays: expected an ordinary file at {essay}")
        check("RULER essays", asset, essay)
    else:
        check("RULER essays (ordinary file or offline HF dataset)", datasets.load_dataset,
              "baber/paul_graham_essays", split="train")
    check("RULER CWE", ruler._read_cwe_wordlist, ruler._cwe_wordlist_file())
    check("NLTK", nltk.data.find, "tokenizers/punkt_tab/english/")
    for task in ("longbench2_1m", "infinitebench_256k"):
        check(task, reference._check_task_data, [task])
    for length in ("32k", "64k", "128k", "256k", "512k", "1m"):
        for i in range(1, 6):
            options = dict(length=length, qa_split=f"qa{i}")
            if length == "256k":
                options["dataset"] = "RMT-team/babilong"
            if check(f"BABILong {length}/qa{i}", babi.load_dataset, **options) is None:
                break

    # Check enough length for both corpora without materializing the 100-document selection.
    lengths_needed = {}
    for length in (262144, 1048576):
        for key, path in ppl_sources(length).items():
            print(f"{key} ({length // 1024}K): {path}", flush=True)
            lengths_needed[path] = max(lengths_needed.get(path, 0), length)
    encoder = check("RULER/PPL tokenizer", AutoTokenizer.from_pretrained, generator, local_files_only=True)
    for path, minimum in lengths_needed.items():
        check("PPL input", ppl.validate_inputs, path)
        if encoder is not None and Path(path).exists():
            docs = check("PPL length", ppl.select_documents, path, encoder, 1,
                         concat=os.getenv("LONG_PPL_CONCAT") == "1", longest=minimum)
            if docs == []:
                problems.append(f"PPL: no document with >= {minimum} tokens in {path}; "
                                "set LONG_PPL_DATA_1M / LONG_PPL_BOOKS_1M to the matching reference corpus")
    if problems:
        raise SystemExit("Full CaM offline inputs unavailable on this node:\n" + "\n".join(problems))
