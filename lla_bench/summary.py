"""Export the pinned niahLogKV paper table from actual evaluation logs or JSON reports."""
import argparse
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .long_eval_plan import INFINITEBENCH_MAX, LENGTHS, ruler_tasks

ORDERED_COLUMNS = [
    "Common Sense Reasoning (ACC)", "recall-intensive tasks", "LongBench", "LongBench_e",
    "BoolQ", "PIQA", "SIQA", "HellaSwag", "WinoGrande", "ARC-e", "ARC-c", "OpenBookQA", "AVG_CSR",
    "ceval-valid", "ifeval", "mmlu", "truthfulqa_gen", "truthfulqa_mc1", "truthfulqa_mc2", "AVG_General",
    "SWDE", "FDA", "TriviaQA", "NQ", "DROP", "SQUAD", "AVG_Recall",
    "gov_report", "hotpotqa", "lcc", "Qasper", "MFQA_zh", "MFQA_en", "musique", "narrativeqa",
    "passage_count", "passage_retrieval_en", "longbench_repobench-p", "samsum", "triviaqa",
    "trec", "lsht", "multi_news", "qmsum", "dureader", "2wikimqa", "vcsum", "AVG_LB", "AVG_LB_wo",
    "2wikimqa_e", "gov_report_e", "hotpotqa_e", "lcc_e", "multi_news_e", "MFQA_en_e",
    "passage_count_e", "passage_retrieval_en_e", "qasper_e", "repobench-p_e", "samsum_e",
    "trec_e", "triviaqa_e", "AVG_LBe", "AVG_LBe_wo", "AVG_all",
    "niah_single_1", "niah_single_2", "niah_single_3", "AVG_NIAH",
    # RULER 多针（lm-eval 自带）与 cue 适用边界的提示扰动（eval_tasks/niah_cue），均取 32K。
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multivalue", "niah_multiquery", "AVG_NIAH_multi",
    "niah_single_1_qend", "niah_single_1_preamble", "niah_single_1_paraphrase",
    "niah_single_2_qend", "niah_single_2_preamble", "niah_single_2_paraphrase",
    "niah_single_3_qend", "niah_single_3_preamble", "niah_single_3_paraphrase", "AVG_NIAH_cue",
    # niahLogKV 泛化：值为随机英文词、不含数字的针（eval_tasks/niah_cue/*_words），取 32K。
    "niah_single_2_words", "niah_single_2_words_qend", "niah_single_2_words_paraphrase", "AVG_NIAH_words",
]
LONG_LENGTHS = {str(tokens): name for name, tokens in LENGTHS.items()}
LONG_NIAH = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
             "niah_multikey_3", "niah_multivalue", "niah_multiquery"]
BLOCK_LENGTHS = tuple(LENGTHS)
BABILONG_LENGTHS = PPL_LENGTHS = LONGBENCH2_LENGTHS = TURN_LENGTHS = BLOCK_LENGTHS
TURN_TASKS = [f"niah_turns_{n}" for n in TURN_LENGTHS]
# 书的困惑度曲线的截断点：每个配置长度只到它自己的长度（long_ppl_books_32k 为 8k、16k、32k）。
BOOK_POINT_LENGTHS = {"8k": 8192, "16k": 16384, **LENGTHS}
BOOK_POINTS = {n: [point for point, tokens in BOOK_POINT_LENGTHS.items() if tokens <= length]
               for n, length in LENGTHS.items()}
# InfiniteBench 的十个任务（eval_tasks/infinitebench，32K-256K），AVG_InfiniteBench 为十个的平均（论文的 Avg）。
INFINITEBENCH = ["passkey", "number_string", "kv_retrieval", "longbook_sum_eng", "longbook_qa_eng", "longbook_choice_eng",
                 "longbook_qa_chn", "longdialogue_qa_eng", "code_debug", "math_find"]
INFINITEBENCH_LENGTHS = tuple(n for n, length in LENGTHS.items() if length <= INFINITEBENCH_MAX)
# LongBench v2 的 20 个子任务（lm-eval 的 longbench2_*，各长度的组 longbench2_<长度> 都是这 20 个，日志里子任务名不带长度，
# 按它所在的组记到该长度），LongBench2_<长度> 是组的总分（按题数加权，即全部题目的正确率，lm-eval 报告的那个）。
LONGBENCH2_SUBTASKS = ["academic_multi", "academic_single", "agent_history", "code", "detective", "dialogue_history",
                       "event_order", "fin_multi", "fin_single", "govt_multi", "govt_single", "graph", "legal_multi",
                       "legal_single", "lit_single", "many_shot", "news_multi", "table", "translate", "user_guide"]
LONGBENCH2_GROUP = re.compile(rf"longbench2_({'|'.join(LONGBENCH2_LENGTHS)})")
RULER_AT = {n: ruler_tasks(length) for n, length in LENGTHS.items()}


# 长文本列先按长度分块（32K 主线之后是 32K、64K、128K、256K、512K、1M），块内再按任务分组、
# 每组的平均紧跟在它平均的任务之后。只列评测会产出结果的列，各块的顺序相同：
#   RULER 单针 niah_single_1-3、AVG_NIAH；多针 niah_multikey_1-3、niah_multivalue、niah_multiquery、AVG_NIAH_multi；
#   其余 ruler_vt、ruler_cwe、ruler_fwe、ruler_qa_squad（至 128K）、ruler_qa_hotpot、AVG_RULER（该长度全部 RULER
#   任务的平均：13 个，256K 起为 12 个）；BABILong qa1-qa5、AVG_babilong；
#   LongBench v2 的 20 个子任务 longbench2_<子任务>_<长度>、LongBench2_<长度>（组的总分，按题数加权）；
#   同一份缓存上逐轮问 niah_turns_<长度>_turn1-4、niah_turns_<长度>（四轮平均）；
#   书的困惑度曲线 long_ppl_books_<长度>_<截断长度>（截断长度 8k 起到该长度，同一批书，越低越好），之后是同长度的
#   long_ppl_<长度>（PG19 测试集文档末尾的困惑度，越低越好）；
#   32K–256K 的 InfiniteBench 十个任务及 AVG_InfiniteBench_<长度>。
# 32K 块的捞针列与 32K 主线的 niah_single_1 等是同一份结果，重复列出。都不计入 AVG_all。新列加在所属任务组里、它的平均
# 之前，并更新 tests/parse_lmeval_table_columns.txt（测试检查各块顺序一致、各长度的列连在一起、平均紧跟其任务）。
def _length_block(n):
    """One length's columns: grouped by task, each average right after the tasks it averages."""
    single, multi = LONG_NIAH[:3], LONG_NIAH[3:]
    rest = [t for t in RULER_AT[n] if t not in LONG_NIAH]
    columns = [f"{t}_{n}" for t in single] + [f"AVG_NIAH_{n}"]  # RULER: the single needles
    columns += [f"{t}_{n}" for t in multi] + [f"AVG_NIAH_multi_{n}"]  # the multi-needle tasks
    columns += [f"{t}_{n}" for t in rest] + [f"AVG_RULER_{n}"]  # the rest; AVG_RULER: all of RULER at n
    columns += [f"babilong_qa{i}_{n}" for i in range(1, 6)] + [f"AVG_babilong_{n}"]
    columns += [f"longbench2_{t}_{n}" for t in LONGBENCH2_SUBTASKS] + [f"LongBench2_{n}"]  # the group's total last
    columns += [f"niah_turns_{n}_turn{i}" for i in range(1, 5)] + [f"niah_turns_{n}"]  # niah_turns: the four turns' mean
    columns += [f"long_ppl_books_{n}_{q}" for q in BOOK_POINTS[n]] + [f"long_ppl_{n}"]
    if n in INFINITEBENCH_LENGTHS:
        columns += [f"infinitebench_{t}_{n}" for t in INFINITEBENCH] + [f"AVG_InfiniteBench_{n}"]
    return columns


for _length in BLOCK_LENGTHS:
    ORDERED_COLUMNS += _length_block(_length)
NIAH_MULTI = ["niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multivalue", "niah_multiquery"]
NIAH_CUE = [f"niah_single_{n}_{v}" for n in (1, 2, 3) for v in ("qend", "preamble", "paraphrase")]
NIAH_WORDS = ["niah_single_2_words", "niah_single_2_words_qend", "niah_single_2_words_paraphrase"]

RAW_TO_TARGET_MAP = {
    "boolq": "BoolQ", "piqa": "PIQA", "social_iqa": "SIQA", "hellaswag": "HellaSwag",
    "winogrande": "WinoGrande", "arc_easy": "ARC-e", "arc_challenge": "ARC-c", "openbookqa": "OpenBookQA",
    "longbench_gov_report": "gov_report", "longbench_hotpotqa": "hotpotqa", "longbench_lcc": "lcc",
    "longbench_qasper": "Qasper", "longbench_multifieldqa_zh": "MFQA_zh", "longbench_multifieldqa_en": "MFQA_en",
    "longbench_musique": "musique", "longbench_narrativeqa": "narrativeqa", "longbench_passage_count": "passage_count",
    "longbench_passage_retrieval_en": "passage_retrieval_en", "longbench_samsum": "samsum",
    "longbench_triviaqa": "triviaqa", "longbench_trec": "trec", "longbench_lsht": "lsht",
    "longbench_multi_news": "multi_news", "longbench_qmsum": "qmsum", "longbench_dureader": "dureader",
    "longbench_2wikimqa": "2wikimqa", "longbench_vcsum": "vcsum",
    "longbench_2wikimqa_e": "2wikimqa_e", "longbench_gov_report_e": "gov_report_e", "longbench_hotpotqa_e": "hotpotqa_e",
    "longbench_lcc_e": "lcc_e", "longbench_multi_news_e": "multi_news_e", "longbench_multifieldqa_en_e": "MFQA_en_e",
    "longbench_passage_count_e": "passage_count_e", "longbench_passage_retrieval_en_e": "passage_retrieval_en_e",
    "longbench_qasper_e": "qasper_e", "longbench_repobench-p_e": "repobench-p_e", "longbench_samsum_e": "samsum_e",
    "longbench_trec_e": "trec_e", "longbench_triviaqa_e": "triviaqa_e",
    **{f"longbench2_{n}": f"LongBench2_{n}" for n in LONGBENCH2_LENGTHS},
}

TARGET_METRIC_MAP = {
    "BoolQ": "acc", "PIQA": "acc", "SIQA": "acc", "HellaSwag": "acc_norm",
    "WinoGrande": "acc", "ARC-e": "acc", "ARC-c": "acc_norm", "OpenBookQA": "acc_norm",
    "ceval-valid": "acc", "mmlu": "acc", "ifeval": "prompt_level_strict_acc",
    "truthfulqa_gen": "rouge1_max", "truthfulqa_mc1": "acc", "truthfulqa_mc2": "acc",
    "SWDE": "f1", "FDA": "f1", "TriviaQA": "exact_match", "NQ": "exact_match", "DROP": "f1", "SQUAD": "exact_match",
    "niah_single_1": "32768", "niah_single_2": "32768", "niah_single_3": "32768",
    **{task: "32768" for task in NIAH_MULTI + NIAH_CUE + NIAH_WORDS},
    **{f"babilong_qa{i}_{n}": "acc" for i in range(1, 6) for n in BABILONG_LENGTHS},
    **{f"long_ppl_{n}": "ppl" for n in PPL_LENGTHS},
    **{f"LongBench2_{n}": "acc" for n in LONGBENCH2_LENGTHS},
    **{t: "acc" for t in TURN_TASKS},
    **{f"long_ppl_books_{n}_{q}": "ppl" for n in BLOCK_LENGTHS for q in BOOK_POINTS[n]},
}
for col in ORDERED_COLUMNS:
    if col not in TARGET_METRIC_MAP and not col.startswith(("AVG", "Common", "recall", "LongBench")):
        TARGET_METRIC_MAP[col] = "score"

# ==============================================================================
# 2. 定义各模块的平均值计算范围
# ==============================================================================
AVG_GROUPS = {
    "AVG_CSR": ["BoolQ", "PIQA", "SIQA", "HellaSwag", "WinoGrande", "ARC-e", "ARC-c", "OpenBookQA"],
    "AVG_General": ["ceval-valid", "ifeval", "mmlu", "truthfulqa_gen", "truthfulqa_mc1", "truthfulqa_mc2"],
    "AVG_Recall": ["SWDE", "FDA", "TriviaQA", "NQ", "DROP", "SQUAD"],
    "AVG_LB": ["gov_report", "hotpotqa", "lcc", "Qasper", "MFQA_zh", "MFQA_en", "musique", "narrativeqa", "passage_count", "passage_retrieval_en", "longbench_repobench-p", "samsum", "triviaqa", "trec", "lsht", "multi_news", "qmsum", "dureader", "2wikimqa", "vcsum"],
    # 注：如果你需要计算 w/o（比如去掉代码或篇章检索），可以在下面的列表里删掉对应的名称
    "AVG_LB_wo": ["gov_report", "hotpotqa", "Qasper", "MFQA_zh", "MFQA_en", "musique", "narrativeqa", "samsum", "triviaqa", "trec", "lsht", "multi_news", "qmsum", "dureader", "2wikimqa", "vcsum"],
    "AVG_LBe": ["2wikimqa_e", "gov_report_e", "hotpotqa_e", "lcc_e", "multi_news_e", "MFQA_en_e", "passage_count_e", "passage_retrieval_en_e", "qasper_e", "repobench-p_e", "samsum_e", "trec_e", "triviaqa_e"],
    "AVG_NIAH": ["niah_single_1", "niah_single_2", "niah_single_3"],
    "AVG_NIAH_multi": NIAH_MULTI,
    "AVG_NIAH_cue": NIAH_CUE,
    "AVG_NIAH_words": NIAH_WORDS,
    **{f"AVG_NIAH_{n}": [f"niah_single_{k}_{n}" for k in (1, 2, 3)] for n in BLOCK_LENGTHS},
    **{f"AVG_NIAH_multi_{n}": [f"{t}_{n}" for t in NIAH_MULTI] for n in BLOCK_LENGTHS},
    **{f"AVG_RULER_{n}": [f"{t}_{n}" for t in RULER_AT[n]] for n in BLOCK_LENGTHS},
    **{f"AVG_babilong_{n}": [f"babilong_qa{i}_{n}" for i in range(1, 6)] for n in BABILONG_LENGTHS},
    **{f"AVG_InfiniteBench_{n}": [f"infinitebench_{t}_{n}" for t in INFINITEBENCH] for n in INFINITEBENCH_LENGTHS},
}
# 每个平均及其任务都是表头里的列（否则 build_table 会在表尾多出一列、或悄悄少平均一项）。
assert all(c in ORDERED_COLUMNS for avg, tasks in AVG_GROUPS.items() for c in (avg, *tasks)), \
    [c for avg, tasks in AVG_GROUPS.items() for c in (avg, *tasks) if c not in ORDERED_COLUMNS]
# 新增的诊断组不计入 AVG_all，旧结果与新结果的总平均保持可比。
DIAGNOSTIC_GROUPS = {"AVG_NIAH_multi", "AVG_NIAH_cue", "AVG_NIAH_words",
                     *(f"AVG_NIAH_{n}" for n in BLOCK_LENGTHS), *(f"AVG_NIAH_multi_{n}" for n in BLOCK_LENGTHS),
                     *(f"AVG_RULER_{n}" for n in BLOCK_LENGTHS), *(f"AVG_babilong_{n}" for n in BABILONG_LENGTHS),
                     *(f"AVG_InfiniteBench_{n}" for n in INFINITEBENCH_LENGTHS)}

# ==============================================================================
# 3. 解析与计算核心代码
# ==============================================================================
TABLE_HEADER = "|Tasks|Version|Filter|n-shot|Metric| |Value| |Stderr|"
TABLE_ROW = re.compile(r"\|(?:[^|]*\|){9}\s*$")  # lm-eval 结果表的一行：九列，十个竖线


def table_lines(text):
    """``text`` 里所有 lm-eval 结果表的数据行：粘贴的表，或整份 eval 日志（每个阶段一张表，行首可能有前缀）。
    去掉表头（Tasks / Groups）与分隔行，保留 lm-eval 的九列。"""
    rows = []
    for line in text.splitlines():
        match = TABLE_ROW.search(line.rstrip())
        if not match:
            continue
        row = match.group(0).strip()
        first = row.split("|")[1].strip()
        if first in ("Tasks", "Groups") or set(row) <= set("|-: "):
            continue
        rows.append(row)
    return rows


def parse_single_model(model_name, markdown_text):
    rows = table_lines(markdown_text)
    if not rows: return {"Model": model_name}

    # 全按字符串读：只有 RULER 行的表，Metric 列（65536 等）也要和 LONG_LENGTHS 的键比较。
    df = pd.read_csv(io.StringIO('\n'.join([TABLE_HEADER] + rows)), sep='|', dtype=str, keep_default_na=False)
    df = df.iloc[:, 1:-1]
    df.columns = df.columns.str.strip()
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    df['Tasks'] = df['Tasks'].replace('', np.nan).ffill()
    row_data = {"Model": model_name}
    longbench2_length = None  # the length of the longbench2_<length> group whose subtasks follow

    for _, row in df.iterrows():
        listed = str(row['Tasks'])
        raw_task = re.sub(r"^-\s*", "", listed)  # a group's subtasks are listed as " - name"
        metric = row['Metric']
        value = row['Value']
        if not listed.startswith("-"):
            group = LONGBENCH2_GROUP.fullmatch(raw_task)
            longbench2_length = group.group(1) if group else None
        elif longbench2_length and raw_task.startswith("longbench2_") and metric == "acc":
            # LongBench v2's subtasks carry no length: they belong to the group above them.
            if raw_task[len("longbench2_"):] in LONGBENCH2_SUBTASKS:
                try:
                    row_data[f"{raw_task}_{longbench2_length}"] = float(value)
                except ValueError:
                    pass
            continue

        target_task = RAW_TO_TARGET_MAP.get(raw_task, raw_task)
        expected_metric = TARGET_METRIC_MAP.get(target_task)
        if target_task in TURN_TASKS and re.fullmatch(r"turn\d", metric):  # niah_turns: one column per turn
            try:
                row_data[f"{target_task}_{metric}"] = float(value)
            except ValueError:
                pass
            continue
        ruler_length = LONG_LENGTHS.get(metric)
        if target_task in RULER_AT.get(ruler_length, ()):
            try:
                row_data[f"{target_task}_{ruler_length}"] = float(value)
            except ValueError:
                pass
            if metric != "32768":
                continue  # 32K 的捞针结果另外还填进 32K 主线的 niah_single_1 等列（下面的通用匹配）

        if expected_metric and metric == expected_metric:
            try:
                row_data[target_task] = float(value)
            except ValueError:
                pass # 忽略非数字
    return row_data


def build_table(outputs):
    """``{模型名: 结果表或日志文本}`` -> 一行一个模型、列为 ORDERED_COLUMNS 的表，含各组平均与 AVG_all。"""
    # 1. 组合数据为 DataFrame
    final_df = pd.DataFrame([parse_single_model(m, text) for m, text in outputs.items()])
    final_df = final_df.reindex(columns=['Model'] + ORDERED_COLUMNS) # 对齐格式

    # 2. ================= 自动计算各组平均值 =================
    scores = final_df[ORDERED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    # lm-eval reports TruthfulQA ROUGE F1 on a 0–100 scale.
    scores["truthfulqa_gen"] = scores["truthfulqa_gen"] / 100
    for avg_col, sub_tasks in AVG_GROUPS.items():
        # 选出有实际成绩的列，利用 pd.to_numeric 把可能混入的字符转为 NaN 并求均值
        valid_cols = [t for t in sub_tasks if t in final_df.columns]
        final_df[avg_col] = scores[valid_cols].mean(axis=1).round(4)

    # 计算全局汇总平均 AVG_all (综合所有分组子任务的基础列)
    all_base_tasks = list(set([task for group, group_tasks in AVG_GROUPS.items() if group not in DIAGNOSTIC_GROUPS
                               for task in group_tasks]))
    valid_base_tasks = [t for t in all_base_tasks if t in final_df.columns]
    final_df['AVG_all'] = scores[valid_base_tasks].mean(axis=1).round(4)
    return final_df


def read_table(path):
    if path.suffix.lower() == ".json":
        from lm_eval.utils import make_table

        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload["results"]
        if not isinstance(results, dict) or "results" not in results:
            raise ValueError(f"Expected an eval_results_<timestamp>.json report: {path}")
        table = make_table(results)
    else:
        table = path.read_text(encoding="utf-8")
    if not table_lines(table):
        raise ValueError(f"No lm-eval result table in {path}")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", action="append", required=True, metavar="NAME_OR_FILE",
                        help="model label followed by its JSON reports or logs; repeat for another model")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    outputs = {}
    for label, *paths in args.model:
        if not paths or label in outputs:
            parser.error("each --model needs a unique label and at least one input file")
        outputs[label] = "\n".join(read_table(Path(path)) for path in paths)
    frame = build_table(outputs)
    frame.to_csv(args.output or sys.stdout, index=False, na_rep="")


if __name__ == "__main__":
    main()
