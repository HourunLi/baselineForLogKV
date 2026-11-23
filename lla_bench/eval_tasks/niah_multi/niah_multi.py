"""RULER 多针 NIAH：针数可配的 multivalue / multiquery。

lm-eval 0.4.9 自带的 ``niah_multivalue`` / ``niah_multiquery`` 固定 4 根针。这里复用其样本
生成、模板与评分（逐样本找回针的比例），只让各任务 YAML 的 ``dataset_kwargs`` 指定针数与
生成预留。eval.py 用 ``TaskManager(include_path=eval_tasks)`` 注册这些任务。
"""

from __future__ import annotations

import logging

from lm_eval.tasks.ruler.common_utils import (  # noqa: F401  aggregate_metrics/process_results 供 YAML 引用
    DEFAULT_SEQ_LENGTHS,
    aggregate_metrics,
    get_tokenizer,
    process_results,
)
from lm_eval.tasks.ruler.niah_utils import TEMPLATE, download_dataset
from lm_eval.tasks.ruler.prepare_niah import generate_input_output, generate_samples, get_haystack

eval_logger = logging.getLogger(__name__)


def niah_multi(
    *,
    num_needle_k: int = 1,
    num_needle_v: int = 1,
    num_needle_q: int = 1,
    tokens_to_generate: int = 128,
    type_haystack: str = "essay",
    type_needle_k: str = "words",
    type_needle_v: str = "numbers",
    **kwargs,
):
    """需要 --metadata 的 ``pretrained`` 与 ``max_seq_lengths``；``niah_num_samples`` 可选，默认 500/长度。

    ``tokens_to_generate`` 必须等于 YAML 的 ``max_gen_toks``：样本按它预留生成空间，否则
    eval.py 在 prompt + max_gen_toks 超过模型长度时会从左截掉 prompt，连同靠前的针。
    """
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    num_samples = int(kwargs.pop("niah_num_samples", 500))
    tokenizer = get_tokenizer(**kwargs)
    haystack = get_haystack(type_haystack=type_haystack)
    num_needle_k = max(num_needle_k, num_needle_q)
    num_needles = num_needle_k * num_needle_v
    needle_kwargs = dict(
        type_haystack=type_haystack,
        type_needle_k=type_needle_k,
        type_needle_v=type_needle_v,
        num_needle_k=num_needle_k,
        num_needle_v=num_needle_v,
        num_needle_q=num_needle_q,
    )
    # 连只含针与问题的 prompt 都放不下时，lm-eval 的逐样本重试会无限循环；
    # 按最短 prompt 留 25% 余量（键长随机）判断，跳过这些长度。
    probe, _, _ = generate_input_output(
        0 if type_haystack == "essay" else num_needles, haystack, template=TEMPLATE, **needle_kwargs
    )
    min_length = int(1.25 * len(tokenizer(probe).input_ids)) + tokens_to_generate

    def samples_for(seq: int) -> list[dict]:
        if seq < min_length:
            eval_logger.warning(
                f"跳过 {seq}：放不下 {num_needles} 根针、问题与 {tokens_to_generate} 生成预留（约需 {min_length}）。"
            )
            return []
        samples = generate_samples(
            haystack,
            max_seq_length=seq,
            template=TEMPLATE,
            num_samples=num_samples,
            tokens_to_generate=tokens_to_generate,
            TOKENIZER=tokenizer,
            **needle_kwargs,
        )
        # essay 草堆用尽时 lm-eval 只截短样本，分数仍记在 seq 名下；正常不足量不超过一次增量。
        longest = max(s["length"] for s in samples)
        if seq - longest > max(seq // 10, 4096):
            eval_logger.warning(
                f"{type_haystack} 草堆不足 {seq} tokens：最长样本仅 {longest}（含生成预留），"
                f"该长度下的分数对应更短的上下文。"
            )
        return samples

    return download_dataset(samples_for(seq) for seq in seq_lengths)
