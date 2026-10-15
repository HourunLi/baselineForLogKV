# Vendored for the CaM baseline; local namespace changes are recorded in SOURCE.json.
"""RULER 离线补丁 — 运行时 monkey-patch，不修改 conda 系统包。

在 eval.py 中 `from lm_eval import evaluator` 之后导入本模块即可：

    from lm_eval import evaluator
    from models._vendor.semanticlogkv.litgpt.ruler_patch import apply_patch  # noqa: I001
    apply_patch()

原理：
1. 全局 monkey-patch requests.Session.request，强制 verify=False（对抗代理 SSL）
2. 劫持 lm_eval.tasks.ruler.qa_utils.download_json，优先从本地缓存加载
3. 安装 import hook，确保 qa_utils 被延迟导入时也能被 patch
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import ssl
import sys
from functools import cache
from pathlib import Path


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

    def find_module(self, fullname, path=None):
        if fullname == self._TARGET:
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            mod = sys.modules[fullname]
            _patch_qa_utils_module(mod)
            return mod
        sys.meta_path.remove(self)
        try:
            mod = importlib.import_module(fullname)
        finally:
            sys.meta_path.insert(0, self)
        _patch_qa_utils_module(mod)
        return mod

    def find_spec(self, fullname, path, target=None):
        """新式 API (PEP 451)，确保 Python 3.10+ 也能触发。"""
        if fullname != self._TARGET:
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
                _patch_qa_utils_module(module)
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
