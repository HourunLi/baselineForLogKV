"""RULER 离线补丁 — 运行时 monkey-patch，不修改 conda 系统包。

在 eval.py 中 `from lm_eval import evaluator` 之后导入本模块即可：

    from lm_eval import evaluator
    from litgpt.ruler_patch import apply_patch  # noqa: I001
    apply_patch()

原理：
1. 全局 monkey-patch requests.Session.request，强制 verify=False（对抗代理 SSL）
2. 劫持 lm_eval.tasks.ruler.qa_utils.download_json，优先从本地缓存加载
3. 安装 import hook，确保 qa_utils 被延迟导入时也能被 patch
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import ssl
import string
import sys
from functools import cache, partial
from pathlib import Path

LONG_RULER_MIN = 524288
RULER_CWE_WORDS_URL = (
    "https://media.githubusercontent.com/media/NVIDIA/RULER/533f013bba67c21a5824d48f416c0f68df0e399d/"
    "scripts/data/synthetic/json/english_words.json"
)
RULER_CWE_WORDS_SHA256 = "affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca"


def _cwe_wordlist_file() -> Path:
    return Path(_get_ruler_cache_dir()).expanduser() / "english_words.json"


def _verify_cwe_wordlist(payload: bytes, path: Path) -> None:
    if hashlib.sha256(payload).hexdigest() != RULER_CWE_WORDS_SHA256:
        raise ValueError(f"RULER CWE word list SHA256 mismatch: {path}")


def _read_cwe_wordlist(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"RULER CWE requires the NVIDIA word list as an ordinary file: {path}")
    payload = path.read_bytes()
    _verify_cwe_wordlist(payload, path)
    return payload


@cache
def _cwe_vocabulary(path: Path) -> tuple[str, ...]:
    return tuple(json.loads(_read_cwe_wordlist(path)).values())


def _essay_file() -> Path:
    return Path(os.environ.get("RULER_ESSAYS_FILE") or
                Path(_get_ruler_cache_dir()) / "paul_graham_essays.txt").expanduser()


def _expand_essay(haystack, tokenizer, target: int):
    """Cyclically repeat only the distractor background; the upstream generator inserts each needle once."""
    if not isinstance(haystack, list) or not haystack:
        raise ValueError("RULER essay background is empty")
    count_tokens = partial(tokenizer, add_special_tokens=False, truncation=False, verbose=False)
    tokens = len(count_tokens(" ".join(haystack))["input_ids"])
    if not tokens:
        raise ValueError("RULER essay background tokenizes to zero tokens")
    repeats = max(1, math.ceil((target + 2048) / tokens))
    expanded = haystack if repeats == 1 else haystack * repeats
    if repeats > 1:
        tokens = len(count_tokens(" ".join(expanded))["input_ids"])
    while tokens < target + 1024:
        repeats += 1
        expanded = haystack * repeats
        tokens = len(count_tokens(" ".join(expanded))["input_ids"])
    return expanded, repeats, tokens


def _fit_ruler_prompt(render, tokenizer, target, estimate, low, high, suffix, reserve, task):
    """Fit complete entries to the rendered prompt budget with bounded tokenization."""
    count = max(low, estimate if high is None else min(estimate, high))
    best = None
    for _ in range(24):
        if high is not None and low > high:
            break
        prompt, outputs = render(count)
        length = len(tokenizer(prompt + suffix, truncation=False).input_ids) + reserve
        if length <= target:
            if best is None or length > best[2]:
                best = prompt, outputs, length, count
            if length >= target - 1024:
                return best
            low = count + 1
        else:
            high = count - 1
        guess = round(count * (target - 512) / max(1, length))
        if high is None:
            count = max(low, min(guess, count * 2))
        else:
            count = guess if low <= guess <= high else (low + high) // 2
    if best is not None and best[2] >= target - 4096:
        return best
    actual = best[2] if best is not None else "no sample within budget"
    raise ValueError(f"RULER {task} could not fit {target} tokens (best: {actual})")


def _generate_sized_ruler_rows(namespace, render, *, tokenizer, target, reserve, num_samples, low, task,
                               estimate=None, high=None, seed=42, numpy_rng=None):
    """Fit a rendered example and reuse its background count to size the next example."""
    if not num_samples:
        return []
    tokenizer = partial(tokenizer, verbose=False)
    rng = random.Random(seed)
    python_state = random.getstate()
    numpy_state = numpy_rng.get_state() if numpy_rng is not None else None
    rows = []
    try:
        for index in namespace["tqdm"](range(num_samples), desc=f"Generating {task} Samples | {target}"):
            sample_seed = rng.getrandbits(32)

            def make_prompt(count):
                # Upstream renderers use module-level RNGs. Reuse the sample's seed
                # across sizing attempts and restore the caller's RNGs afterwards.
                random.seed(sample_seed)
                if numpy_rng is not None:
                    numpy_rng.seed(sample_seed)
                prompt, answers, prefix = render(count)
                return prompt + " " + prefix, (prompt, answers, prefix)

            if estimate is None:
                probe = max(low, 512)
                text, _ = make_prompt(probe)
                length = len(tokenizer(text, truncation=False).input_ids)
                estimate = round(probe * (target - reserve) / max(1, length))
            _, (prompt, answers, prefix), length, estimate = _fit_ruler_prompt(
                make_prompt, tokenizer, target, estimate, low, high, "", reserve, f"{task} sample {index}",
            )
            rows.append(dict(index=index, input=prompt, outputs=answers, length=length,
                             max_length=target, gen_prefix=prefix))
        return rows
    finally:
        random.setstate(python_state)
        if numpy_rng is not None:
            numpy_rng.set_state(numpy_state)


def _generate_long_niah(namespace, *, haystack, TOKENIZER, max_seq_length, type_haystack, type_needle_k,
                        type_needle_v, template, num_samples, tokens_to_generate, num_needle_v, num_needle_k,
                        num_needle_q, remove_newline_tab, random_seed, essay_tokens):
    num_needle_k = max(num_needle_k, num_needle_q)
    estimate = round(len(haystack) * (max_seq_length - tokens_to_generate) / essay_tokens) if essay_tokens else None

    def render(count):
        prompt, answers, query = namespace["generate_input_output"](
            count, haystack, type_haystack=type_haystack, num_needle_k=num_needle_k,
            type_needle_k=type_needle_k, num_needle_v=num_needle_v, type_needle_v=type_needle_v,
            template=template, num_needle_q=num_needle_q, random_seed=random_seed,
        )
        if remove_newline_tab:
            prompt = " ".join(prompt.split())
        if num_needle_q * num_needle_v == 1:
            prefix = f"The special magic {type_needle_v[:-1]} for {query} mentioned in the provided text is"
        else:
            prefix = f"The special magic {type_needle_v} for {query} mentioned in the provided text are"
        return prompt, answers, prefix

    return _generate_sized_ruler_rows(
        namespace, render, tokenizer=TOKENIZER, target=max_seq_length, reserve=tokens_to_generate,
        num_samples=num_samples, low=max(1, num_needle_k * num_needle_v),
        task=f"NIAH ({type_haystack})", estimate=estimate,
        high=len(haystack) if type_haystack == "essay" else None, seed=random_seed,
    )


def _generate_long_vt(namespace, *, tokenizer, num_samples, max_seq_length, num_chains, num_hops, add_fewshot,
                      tokens_to_generate, icl_example, remove_newline_tab):
    config = namespace["CONFIG"]["variable_tracking"]
    question_start = config["template"][:20]
    answer_start = config["answer_prefix"].split("{num_v}", 1)[0]
    example = None
    if add_fewshot and icl_example is not None:
        example = icl_example["input"] + " " + " ".join(icl_example["outputs"]) + "\n\n"

    def render(count):
        prompt, answers = namespace["generate_input_output"](
            count, num_chains, num_hops, is_icl=add_fewshot and icl_example is None,
        )
        if example is not None:
            cutoff = prompt.index(question_start)
            prompt = prompt[:cutoff] + namespace["randomize_icl"](example) + "\n\n" + prompt[cutoff:]
        if remove_newline_tab:
            prompt = " ".join(prompt.split())
        cutoff = prompt.rindex(answer_start)
        prefix = prompt[cutoff:].strip()
        return (prompt[:cutoff] if icl_example is not None else prompt), answers, prefix

    return _generate_sized_ruler_rows(
        namespace, render, tokenizer=tokenizer, target=max_seq_length, reserve=tokens_to_generate,
        num_samples=num_samples, low=max(1, num_hops + 2), task="VT", numpy_rng=namespace["np"].random,
    )


def _generate_long_qa(namespace, *, tokenizer, docs, qas, max_seq_length, num_samples,
                      tokens_to_generate, pre_samples, remove_newline_tab):
    """Fit each question with distinct distractors while keeping its source documents."""
    template = namespace["TEMPLATE"]
    document_prompt = namespace["DOCUMENT_PROMPT"]
    seed = namespace["SEED"]
    prefix = namespace["CONFIG"]["answer_prefix"]
    suffix = " " + prefix  # target_delimiter inherited from niah_single_1.yaml

    def make_input(question, doc_ids):
        context = "\n\n".join(document_prompt.format(i=i, document=docs[doc_id])
                              for i, doc_id in enumerate(doc_ids, 1))
        prompt = template.format(context=context, query=question)
        return " ".join(prompt.split()) if remove_newline_tab else prompt

    if num_samples == 0:
        return []
    probe = random.Random(seed).sample(range(len(docs)), min(128, len(docs)))
    probe_length = len(tokenizer(make_input(qas[pre_samples]["query"], probe) + suffix,
                                 truncation=False).input_ids) + tokens_to_generate
    estimate = round(len(probe) * max_seq_length / probe_length)

    rows = []
    for index in namespace["tqdm"](range(num_samples), desc=f"Generating QA Samples | {max_seq_length}"):
        source_index = index + pre_samples
        qa = qas[source_index]
        required = list(dict.fromkeys(qa["context"]))
        related = list(dict.fromkeys(qa.get("more_context", ())))
        required_set = set(required)
        related = [doc_id for doc_id in related if doc_id not in required_set]
        excluded = required_set.union(related)
        distractors = [doc_id for doc_id in range(len(docs)) if doc_id not in excluded]
        rng = random.Random(seed + source_index)
        rng.shuffle(related)
        rng.shuffle(distractors)
        candidates = required + related + distractors

        def render(count):
            selected = candidates[:count]
            random.Random(seed).shuffle(selected)
            return make_input(qa["query"], selected), qa["outputs"]

        prompt, outputs, actual, _ = _fit_ruler_prompt(
            render, tokenizer, max_seq_length, estimate, len(required), len(candidates),
            suffix, tokens_to_generate, f"QA sample {source_index}",
        )
        rows.append(dict(index=index, input=prompt, outputs=outputs, length=actual,
                         max_length=max_seq_length, gen_prefix=prefix))
    return rows


def _generate_long_cwe(namespace, *, num_samples, max_seq_length, tokenizer,
                       remove_newline_tab, tokens_to_generate):
    """Use NVIDIA's expanded word pool and original 30/3 occurrence counts."""
    config = namespace["CONFIG"]
    vocab = _cwe_vocabulary(_cwe_wordlist_file())
    prefix = config["answer_prefix"].strip()
    suffix = "\n\n" + prefix  # cwe.yaml target_delimiter
    rng = random.Random(42 + max_seq_length)

    def format_words(words):
        return " ".join(f"{i}. {word}" for i, word in enumerate(words, 1))

    def make_input(words, common_repeats, uncommon_repeats, shuffle_seed):
        entries = words[:10] * common_repeats + words[10:] * uncommon_repeats
        random.Random(shuffle_seed).shuffle(entries)
        prompt = config["template"].format(context=format_words(entries), query="").strip()
        return " ".join(prompt.split()) if remove_newline_tab else prompt

    probe = rng.sample(vocab, 4096)
    probe_length = len(tokenizer(make_input(probe, 30, 3, 42) + suffix, truncation=False).input_ids)
    estimate = round(len(probe) * max_seq_length / probe_length)
    rows = []
    for index in namespace["tqdm"](range(num_samples), desc=f"Generating CWE Samples | {max_seq_length}"):
        seed = rng.getrandbits(64)
        shuffle_seed = rng.getrandbits(64)

        def render(count):
            words = random.Random(seed).sample(vocab, count)
            return make_input(words, 30, 3, shuffle_seed), words[:10]

        prompt, outputs, actual, _ = _fit_ruler_prompt(
            render, tokenizer, max_seq_length, estimate, 11, len(vocab), suffix, tokens_to_generate, "CWE",
        )
        example = rng.sample(sorted(namespace["WORDS"]), 40)
        example_prompt = make_input(example, 10, 3, rng.getrandbits(64))
        rows.append(dict(index=index, input=prompt,
                         input_example=example_prompt + config["answer_prefix"] + format_words(example[:10]),
                         outputs=outputs, length=actual, max_length=max_seq_length, gen_prefix=prefix))
    return rows


def _generate_long_fwe(namespace, *, num_samples, max_seq_length, tokenizer, vocab_size,
                       coded_wordlen, alpha, remove_newline_tab, tokens_to_generate):
    """Keep the upstream vocabulary and Zipf counts, fitting the word-count parameter per sample."""
    import numpy as np
    from scipy.special import zeta

    config = namespace["CONFIG"]
    prefix = config["answer_prefix"].strip()
    vocab_size = max_seq_length // 50 if vocab_size == -1 else vocab_size
    if not 4 <= vocab_size <= len(string.ascii_lowercase) ** coded_wordlen or alpha <= 1:
        raise ValueError("RULER FWE needs at least four distinct codes and alpha > 1")
    weights = np.arange(1, vocab_size + 1) ** -float(alpha)
    normalizer = zeta(alpha)
    rng = random.Random(42 + max_seq_length)
    estimate = max_seq_length // 2
    rows = []
    for index in namespace["tqdm"](range(num_samples), desc=f"Generating FWE Samples | {max_seq_length}"):
        words = set()
        while len(words) < vocab_size:
            words.update("".join(rng.choices(string.ascii_lowercase, k=coded_wordlen))
                         for _ in range(vocab_size - len(words)))
        vocab = sorted(words)
        random.Random(42).shuffle(vocab)
        vocab[0] = "..."

        def render(count):
            entries = []
            for word, occurrences in zip(vocab, (count * weights / normalizer).astype(int)):
                entries.extend([word] * occurrences)
            random.Random(42).shuffle(entries)
            prompt = config["template"].format(context=" ".join(entries), query="").strip()
            return (" ".join(prompt.split()) if remove_newline_tab else prompt), vocab[1:4]

        prompt, outputs, actual, _ = _fit_ruler_prompt(
            render, tokenizer, max_seq_length, estimate, 1, None, " " + prefix, tokens_to_generate, "FWE",
        )
        rows.append(dict(index=index, input=prompt, outputs=outputs, length=actual,
                         max_length=max_seq_length, gen_prefix=prefix))
    return rows


def _patch_long_ruler_module(module) -> None:
    """Keep <=256K recipes; cap million-token datasets and make their actual lengths observable."""
    import functools
    import inspect

    entry = next((name for name in ("generate_samples", "sys_vartrack_w_noise_random",
                                    "sys_word_pair_random", "sys_kwext") if hasattr(module, name)), None)
    original = getattr(module, entry) if entry else None
    if original is not None and not getattr(original, "_logkv_long_ruler", False):
        signature = inspect.signature(original)
        module_name = module.__name__.rsplit(".", 1)[-1] if entry == "generate_samples" else ""

        @functools.wraps(original)
        def generate(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            length = int(values.get("max_seq_length", 0))
            if length < LONG_RULER_MIN:
                return original(*args, **kwargs)
            cap = int(os.environ.get("RULER_LONG_SAMPLES", "100"))
            if cap < 1:
                raise ValueError("RULER_LONG_SAMPLES must be positive")
            if "num_samples" in values:
                values["num_samples"] = min(int(values["num_samples"]), cap)
            repeats = 1
            essay_tokens = None
            if values.get("type_haystack") == "essay":
                values["haystack"], repeats, essay_tokens = _expand_essay(
                    values["haystack"], values["TOKENIZER"], length,
                )
            protocol = "extended-v1"
            if entry == "sys_word_pair_random":
                cwe_args = {key: value for key, value in values.items() if key != "incremental"}
                rows = _generate_long_cwe(original.__globals__, **cwe_args)
                protocol = "extended-cwe-nvidia-v1"
            elif entry == "sys_kwext":
                rows = _generate_long_fwe(original.__globals__, **values)
                protocol = "extended-fwe-fit-v1"
            elif module_name == "qa_utils":
                qa_args = {key: value for key, value in values.items() if key != "incremental"}
                rows = _generate_long_qa(original.__globals__, **qa_args)
                protocol = "extended-qa-fit-v1"
            elif module_name == "prepare_niah":
                niah_args = {key: value for key, value in values.items() if key != "incremental"}
                rows = _generate_long_niah(original.__globals__, **niah_args, essay_tokens=essay_tokens)
                protocol = "extended-niah-fit-v1"
            elif entry == "sys_vartrack_w_noise_random":
                vt_args = {key: value for key, value in values.items() if key != "incremental"}
                rows = _generate_long_vt(original.__globals__, **vt_args)
                protocol = "extended-vt-fit-v1"
            else:
                rows = original(*bound.args, **bound.kwargs)
            for row in rows:
                actual = int(row.get("length", 0))
                if actual < length - 4096:
                    raise ValueError(f"RULER requested {length} tokens but generated only {actual}; "
                                     "the background/document pool cannot fill this length")
                row["logkv_ruler_protocol"] = protocol
                row["logkv_haystack_repeats"] = repeats
            return rows

        generate._logkv_long_ruler = True
        setattr(module, entry, generate)
    original_haystack = getattr(module, "get_haystack", None)
    if original_haystack is not None and not getattr(original_haystack, "_logkv_plain_essay", False):
        @functools.wraps(original_haystack)
        def get_haystack(type_haystack, *args, **kwargs):
            path = _essay_file()
            if type_haystack == "essay" and path.is_file():
                if path.is_symlink():
                    raise ValueError(f"RULER_ESSAYS_FILE must be an ordinary file: {path}")
                return path.read_text(encoding="utf-8").split()
            return original_haystack(type_haystack, *args, **kwargs)
        get_haystack._logkv_plain_essay = True
        module.get_haystack = get_haystack


def _patch_ruler_module(module) -> None:
    if module.__name__.rsplit(".", 1)[-1] == "qa_utils":
        _patch_qa_utils_module(module)
    _patch_long_ruler_module(module)


def _patch_yaml_function_imports() -> None:
    """lm-eval also executes task utils under bare names, bypassing normal Python import hooks."""
    import functools
    from types import SimpleNamespace
    import lm_eval.utils as utils
    from litgpt.ruler_dataset_cache import cached_builder

    original = utils.import_function
    if getattr(original, "_logkv_ruler_import", False):
        return

    @functools.wraps(original)
    def import_function(*args, **kwargs):
        function = original(*args, **kwargs)
        namespace = getattr(function, "__globals__", {})
        filename = str(namespace.get("__file__", "")).replace("\\", "/")
        if "/lm_eval/tasks/ruler/" in filename:
            proxy = SimpleNamespace(**namespace)
            _patch_ruler_module(proxy)
            for name in ("generate_samples", "get_haystack", "sys_vartrack_w_noise_random",
                         "sys_word_pair_random", "sys_kwext", "download_json"):
                if name in namespace and hasattr(proxy, name):
                    namespace[name] = getattr(proxy, name)
        return cached_builder(function, filename)

    import_function._logkv_ruler_import = True
    utils.import_function = import_function


def _get_ruler_cache_dir() -> str:
    return os.environ.get(
        "RULER_CACHE_DIR",
        os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "ruler_cache",
        ),
    )


_URL_TO_FILENAME = {
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json": "dev-v2.0.json",
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json": "hotpot_dev_distractor_v1.json",
}


def _load_json_local(filename: str) -> dict | None:
    local_path = os.path.join(_get_ruler_cache_dir(), filename)
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _download_and_cache(url: str, filename: str) -> dict:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    import requests as req

    resp = req.get(url, verify=False, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    cache_dir = _get_ruler_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, filename)
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data


@cache
def _patched_download_json(url: str) -> dict:
    offline = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
    filename = _URL_TO_FILENAME.get(url)

    if filename:
        data = _load_json_local(filename)
        if data is not None:
            return data

    if offline:
        raise RuntimeError(
            f"HF_DATASETS_OFFLINE=1 且 RULER 本地缓存未找到。"
            f"需要将 {filename or url.split('/')[-1]} 放到 {_get_ruler_cache_dir()}/"
        )

    if filename:
        return _download_and_cache(url, filename)

    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    import requests as req
    resp = req.get(url, verify=False, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _patch_qa_utils_module(qa_utils) -> None:
    if getattr(qa_utils, "_ruler_patched", False):
        return
    qa_utils.download_json = _patched_download_json
    qa_utils._ruler_patched = True


class _FakeResponse:
    """模拟 requests.Response，用于从本地缓存返回数据。"""

    def __init__(self, data: dict):
        self._data = data
        self.status_code = 200
        self.headers = {"Content-Type": "application/json"}

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


def _ssl_bypass_enabled() -> bool:
    """是否全局关闭 SSL 验证。默认开启（对抗内网代理的自签证书）；
    设 ``RULER_SSL_NO_VERIFY=0`` 可恢复正常证书校验（本地缓存拦截不受影响）。
    注意：开启时影响评测进程内**所有** requests HTTPS 请求，属于有意的全局行为。"""
    return os.environ.get("RULER_SSL_NO_VERIFY", "1") != "0"


def _patch_requests_ssl() -> None:
    """全局 patch requests：拦截已知 RULER URL 直接返回本地缓存；
    并在 ``RULER_SSL_NO_VERIFY != 0`` 时强制所有 HTTPS 请求跳过 SSL 验证。"""
    import requests

    if getattr(requests.Session, "_ssl_patched", False):
        return

    _original_request = requests.Session.request

    def _patched_request(self, method, url, **kwargs):
        if _ssl_bypass_enabled():
            kwargs.setdefault("verify", False)
        filename = _URL_TO_FILENAME.get(url)
        if filename:
            data = _load_json_local(filename)
            if data is not None:
                return _FakeResponse(data)
        return _original_request(self, method, url, **kwargs)

    requests.Session.request = _patched_request
    requests.Session._ssl_patched = True  # type: ignore[attr-defined]

    _original_get = requests.get

    def _patched_get(url, **kwargs):
        filename = _URL_TO_FILENAME.get(url)
        if filename:
            data = _load_json_local(filename)
            if data is not None:
                return _FakeResponse(data)
        if _ssl_bypass_enabled():
            kwargs.setdefault("verify", False)
        return _original_get(url, **kwargs)

    if not getattr(requests, "_get_patched", False):
        requests.get = _patched_get
        requests._get_patched = True  # type: ignore[attr-defined]


class _QaUtilsImportHook:
    """Meta path finder：当 lm_eval.tasks.ruler.qa_utils 被导入时自动 patch。
    同时兼容旧式 find_module 和新式 find_spec API。"""

    _TARGET = "lm_eval.tasks.ruler.qa_utils"
    _TARGETS = tuple("lm_eval.tasks.ruler." + name for name in (
        "qa_utils", "prepare_niah", "niah_utils", "vt_utils", "cwe_utils", "fwe_utils"))

    def find_module(self, fullname, path=None):
        if fullname in self._TARGETS:
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            mod = sys.modules[fullname]
            _patch_ruler_module(mod)
            return mod
        sys.meta_path.remove(self)
        try:
            mod = importlib.import_module(fullname)
        finally:
            sys.meta_path.insert(0, self)
        _patch_ruler_module(mod)
        return mod

    def find_spec(self, fullname, path, target=None):
        """新式 API (PEP 451)，确保 Python 3.10+ 也能触发。"""
        if fullname not in self._TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None:
            return None
        orig_exec = spec.loader.exec_module if hasattr(spec.loader, "exec_module") else None
        if orig_exec:
            def _exec_module_wrapper(module):
                orig_exec(module)
                _patch_ruler_module(module)
            spec.loader.exec_module = _exec_module_wrapper
        return spec


def apply_patch() -> None:
    """安装所有补丁：SSL 绕过（可用 RULER_SSL_NO_VERIFY=0 关闭）+ qa_utils patch + import hook"""

    # 1. SSL 全局绕过（urllib/ssl 层），受 RULER_SSL_NO_VERIFY 开关控制
    if _ssl_bypass_enabled():
        ssl._create_default_https_context = ssl._create_unverified_context
        os.environ.setdefault("CURL_CA_BUNDLE", "")
        os.environ.setdefault("REQUESTS_CA_BUNDLE", "")

    # 2. 全局 patch requests.Session.request（缓存拦截始终生效；verify=False 受开关控制）
    try:
        _patch_requests_ssl()
    except Exception:
        pass

    # 3. 禁用 urllib3 SSL 警告（仅在绕过开启时需要）
    if _ssl_bypass_enabled():
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass

    # 4. 尝试直接 patch qa_utils（如果已经被导入）
    qa_utils = sys.modules.get("lm_eval.tasks.ruler.qa_utils")
    if qa_utils is not None:
        _patch_qa_utils_module(qa_utils)
    else:
        # 尝试主动导入
        try:
            qa_utils = importlib.import_module("lm_eval.tasks.ruler.qa_utils")
            _patch_qa_utils_module(qa_utils)
        except (ImportError, ModuleNotFoundError, Exception):
            pass

    # 5. 安装 import hook，确保后续延迟导入也能被 patch
    if not any(isinstance(f, _QaUtilsImportHook) for f in sys.meta_path):
        sys.meta_path.insert(0, _QaUtilsImportHook())
    for name in _QaUtilsImportHook._TARGETS:
        if name in sys.modules:
            _patch_ruler_module(sys.modules[name])
    _patch_yaml_function_imports()
