#!/bin/bash

# Eval-only companion to majob.sh. It uses the same runtime environment and
# logKV argument extraction, but never launches CPT training.
#
# Usage:
#   bash eval.sh <training_yaml_or_eval_yaml> [benchmark_csv|none] [niah_csv|none]
#   (niah_csv defaults to every NIAH task: DEFAULT_NIAH_BENCHMARKS below)
#   NIAH only: bash eval.sh <yaml> none   or   NIAH_ONLY=1 bash eval.sh <yaml>
#   Several lengths (a YAML with eval_lengths, e.g. exp/qwen1.7b-32k/eval_long_word_first.yaml; long_eval_plan.py):
#     [EVAL_LENGTHS=32k,128k] [EVAL_SUITES=ruler,babilong,ppl,books,turns,longbench2,infinitebench] [YARN_MODE=fixed] [NIAH_ONLY=1] \
#         bash eval.sh <yaml>
#   Everything (the 32K main line, then every length; exp/qwen1.7b-32k/eval_full_<method>.yaml, eval_main: true):
#     [LIMIT=2] [EVAL_LENGTHS=none] [EVAL_SUITES=...] bash eval.sh exp/qwen1.7b-32k/eval_full_word_first.yaml
#
# Examples:
#   bash eval.sh exp/qwen1.7b-32k/arc.yaml
#   bash eval.sh exp/qwen1.7b-32k/arc.yaml piqa none
#   DIAG_ARGS="--log_kv_diag_mode baseline --log_kv_diag_exact_from_layer 21" \
#       bash eval.sh exp/qwen1.7b-32k/diag.yaml niah_single_1 none

if [ -f /home/ma-user/anaconda3/bin/activate ]; then
    if ! source /home/ma-user/anaconda3/bin/activate torch218; then
        echo "ERROR: failed to activate torch218."
        exit 1
    fi
fi


# Set caches before Python imports Hugging Face and freezes its cache paths.
export HF_HOME=${HF_HOME:-/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache}
export LONGBENCH2_DIR=${LONGBENCH2_DIR:-${HF_HOME}/longbench2}
export BABILONG_DIR=${BABILONG_DIR:-${HF_HOME}/babilong}
export HF_DATASETS_CACHE="${HF_HOME}/hf_cache"
export HF_EVALUATE_CACHE="${HF_HOME}/evaluate"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export RULER_CACHE_DIR="${HF_HOME}/ruler_cache"
export NLTK_DATA="${HF_HOME}/nltk_data"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1

PYTHON_BIN=$(python -c 'import sys; print(sys.executable)') || exit 1
echo "Eval Python: ${PYTHON_BIN}"
echo "LongBench v2 data directory (ordinary files): ${LONGBENCH2_DIR}"

export CUDA_DEVICE_MAX_CONNECTIONS=32
export CUDNN_LOGERR_DBG=1
export CUDNN_LOGDEST_DBG=stderr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

export NVTE_WITH_USER_CUDA=1
export CUDNN_PATH=/usr
export NVTE_CUDA_INCLUDE_DIR=/usr/local/cuda-12.8/include

export NCCL_NVLS_ENABLE=0
export NCCL_NET_PLUGIN=none
# IB/RoCE ACK 超时是指数而非秒数：4.096us x 2^N，网卡只取低 5 bit（有效 1..31）。
# 12000 截成 uint8 是 224，低 5 bit 为 0 = 无限等待：尾包丢失后永不重传，NCCL
# 内核以 100% 利用率空等，表现为随机卡死。22 约 17s 重传一次，重试耗尽后报错。
export NCCL_IB_TIMEOUT=22
# LogKV 路由决策与注意力重叠（侧流）。先在目标 GPU 上运行
# python unused/check_route_overlap.py，输出 bitwise_equal=true 后再设为 1。
export LOGKV_ROUTE_OVERLAP=${LOGKV_ROUTE_OVERLAP:-0}
export NCCL_NET_GDR_LEVEL=2
export NCCL_MIN_NCHANNELS=4

# Detect GPUs visible to this job (including CUDA_VISIBLE_DEVICES); ModelArts
# allocation is an upper bound. GPUS_PER_NODE may explicitly select fewer GPUs.
GPUS_PER_NODE=$("${PYTHON_BIN}" - <<'PY_GPU_COUNT'
import os
import sys

import torch

visible = torch.cuda.device_count()
if visible < 1:
    raise SystemExit("ERROR: no CUDA GPUs are visible; check CUDA_VISIBLE_DEVICES and the CUDA runtime.")
try:
    allocated = int(os.environ.get("MA_NUM_GPUS") or visible)
    available = min(visible, allocated)
    selected = int(os.environ.get("GPUS_PER_NODE") or available)
except ValueError:
    raise SystemExit("ERROR: GPUS_PER_NODE and MA_NUM_GPUS must be positive integers.")
if allocated < 1 or not 1 <= selected <= available:
    raise SystemExit(
        f"ERROR: GPUS_PER_NODE={selected} must be within 1..{available} "
        f"(visible={visible}, MA_NUM_GPUS={allocated})."
    )
print(f"GPU detection: visible={visible}, MA_NUM_GPUS={allocated}, using={selected}", file=sys.stderr)
print(selected)
PY_GPU_COUNT
) || exit 1
NUM_NODES=${MA_NUM_HOSTS:-${GROUP_WORLD_SIZE:-1}}
MASTER_ADDR=${MASTER_ADDR:-localhost}
if [ -n "${MA_VJ_NAME:-}" ]; then
    MASTER_ADDR="${MA_VJ_NAME}-${MA_TASK_NAME}-${MA_MASTER_INDEX:-0}.${MA_VJ_NAME}"
fi
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${VC_TASK_INDEX:-${NODE_RANK:-${GROUP_RANK:-0}}}
MAIN_EVAL_MASTER_PORT=${MAIN_EVAL_MASTER_PORT:-${EVAL_MASTER_PORT:-${MASTER_PORT}}}
NIAH_EVAL_MASTER_PORT=${NIAH_EVAL_MASTER_PORT:-${NIAH_MASTER_PORT:-$((MASTER_PORT + 1))}}

for name in GPUS_PER_NODE NUM_NODES NODE_RANK MASTER_PORT MAIN_EVAL_MASTER_PORT NIAH_EVAL_MASTER_PORT; do
    value=${!name}
    if [[ -z "${value}" || "${value}" == *[!0-9]* ]]; then
        echo "ERROR: ${name} must be a non-negative integer, got '${value}'"
        exit 1
    fi
done

if [ "${NUM_NODES}" -gt 1 ] && [ "${MASTER_ADDR}" = "localhost" ]; then
    echo "ERROR: NUM_NODES=${NUM_NODES} but MASTER_ADDR=localhost."
    echo "Multi-node torchrun needs a master address reachable from every node."
    exit 1
fi

export LITGPT_EXPECTED_WORLD_SIZE=${LITGPT_EXPECTED_WORLD_SIZE:-$((NUM_NODES * GPUS_PER_NODE))}

echo "Starting eval-only job: Node ${NODE_RANK} / ${NUM_NODES}"
echo "Expected world size: ${LITGPT_EXPECTED_WORLD_SIZE} (= ${NUM_NODES} nodes x ${GPUS_PER_NODE} gpus)"
echo "Main eval rendezvous: ${MASTER_ADDR}:${MAIN_EVAL_MASTER_PORT}"
echo "NIAH eval rendezvous: ${MASTER_ADDR}:${NIAH_EVAL_MASTER_PORT}"

if [ -z "$1" ]; then
    echo "ERROR: missing YAML config."
    echo "Usage: bash $0 <training_yaml_or_eval_yaml> [benchmark_csv|none] [niah_csv|none]"
    exit 1
fi

CONFIG_FILE=$1
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: config file not found: ${CONFIG_FILE}"
    exit 1
fi

echo "Reading eval settings from ${CONFIG_FILE}"

CONFIG_EXPORTS=$("${PYTHON_BIN}" - "${CONFIG_FILE}" <<'EOF'
import json
import os
import re
import shlex
import sys

import yaml


def expand_string(text: str) -> str:
    def replace(match):
        inner = match.group(1)
        if ":-" in inner:
            name, default = inner.split(":-", 1)
            return os.environ.get(name) or default
        elif "-" in inner:
            name, default = inner.split("-", 1)
        else:
            name, default = inner, ""
        return os.environ.get(name, default)

    return re.sub(r"\$\{([^}^{]+)\}", replace, text)


def expand(value):
    if isinstance(value, str):
        return expand_string(value)
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value


def load(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "config" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg.pop("config"))
        base = load(base_path)
        cfg = {**base, **cfg}
    return expand(cfg)


def emit(name, value):
    if value is None:
        value = ""
    print(f"{name}={shlex.quote(str(value))}")


def value_or(name, default):
    value = cfg.get(name)
    return default if value is None else value


cfg = load(sys.argv[1])
arch_name = cfg.get("arch_name", "Qwen/Qwen3-0.6B-Base")
save_dir = cfg.get("save_path") or cfg.get("checkpoint_dir")
if not save_dir:
    raise SystemExit("Config must define save_path (training YAML) or checkpoint_dir (eval YAML).")

ckpt_dir = cfg.get("ckpt_dir") or os.path.join("checkpoints", arch_name)
resume_dir = cfg.get("resume_dir")
tokenizer_dir = cfg.get("tokenizer_dir")
# eval_tag: methods evaluated on one checkpoint (baseline, word_first, query, ...) each get their own folder,
# <save_path>/evaluate/<eval_tag>, instead of sharing <save_path>/evaluate; an explicit output_path wins.
output_path = cfg.get("output_path") or os.path.join(save_dir, "evaluate", *([str(cfg["eval_tag"])] if cfg.get("eval_tag") else []))
benchmark = cfg.get("benchmark") or ""

try:
    max_steps = int(float(cfg.get("max_steps", 0) or 0))
except (TypeError, ValueError):
    max_steps = 0
try:
    num_epochs = int(float(cfg.get("num_epochs", 10) or 0))
except (TypeError, ValueError):
    num_epochs = 0

candidates = []
for path in (tokenizer_dir, resume_dir, ckpt_dir, os.path.join("checkpoints", arch_name)):
    if path and path not in candidates:
        candidates.append(path)

emit("SAVE_DIR", save_dir)
emit("EVAL_ARCH_NAME", cfg.get("arch_name") or "")
emit("EVAL_OUTPUT_DIR", output_path)
emit("CONFIG_BENCHMARKS", benchmark)
emit("MAX_STEPS", max_steps)
emit("NUM_EPOCHS", num_epochs)
emit("TOKENIZER_CANDIDATES", ":".join(candidates))
# Long-context evaluation: the RULER / NIAH context lengths, the NIAH task list, and model config
# overrides (e.g. block_size and YaRN rope_adjustments) handed to eval.py as JSON.
lengths = cfg.get("eval_max_seq_lengths") or [1024, 2048, 4096, 8192, 16384, 32768]
emit("EVAL_MAX_SEQ_LENGTHS", json.dumps([int(x) for x in lengths]))
niah = cfg.get("niah_benchmark")
emit("CONFIG_NIAH_BENCHMARKS", ",".join(niah) if isinstance(niah, (list, tuple)) else (niah or ""))
overrides = cfg.get("config_overrides")
emit("CONFIG_OVERRIDES", json.dumps(overrides) if overrides else "")
emit("EVAL_LIMIT", cfg.get("limit") or "")  # samples per task (lm-eval's --limit), every stage
# Data of the repository's long-context tasks (eval_tasks/long_ppl, eval_tasks/babilong_long).
emit("CONFIG_LONG_PPL_DATA", cfg.get("long_ppl_data") or "")
emit("CONFIG_LONG_PPL_BOOKS", cfg.get("long_ppl_books") or "")  # long_ppl_books_<L>: books of >= 256K tokens
# Several lengths in one job (long_eval_plan.py): a list such as [32k, 64k, 128k, 256k].
emit("CONFIG_EVAL_LENGTHS", ",".join(str(x) for x in (cfg.get("eval_lengths") or [])))
emit("CONFIG_EVAL_SUITES", ",".join(str(x) for x in (cfg.get("eval_suites") or [])))
emit("CONFIG_BABILONG_DATASET", cfg.get("babilong_dataset") or "")
emit("CONFIG_INFINITEBENCH_DIR", cfg.get("infinitebench_dir") or "")  # eval_tasks/infinitebench
# The full evaluation (eval_main: true): the 32K main line, then the length passes with their own samples per task.
emit("CONFIG_EVAL_MAIN", "1" if cfg.get("eval_main") else "")
emit("CONFIG_LONG_LIMIT", cfg.get("long_limit") or "")
EOF
)
CONFIG_STATUS=$?
if [ ${CONFIG_STATUS} -ne 0 ]; then
    echo "ERROR: failed to parse config ${CONFIG_FILE}"
    exit ${CONFIG_STATUS}
fi
eval "${CONFIG_EXPORTS}"
# An environment variable wins over the YAML; eval.py's tasks read these.
LONG_PPL_DATA=${LONG_PPL_DATA:-${CONFIG_LONG_PPL_DATA}}
LONG_PPL_BOOKS=${LONG_PPL_BOOKS:-${CONFIG_LONG_PPL_BOOKS}}
BABILONG_DATASET=${BABILONG_DATASET:-${CONFIG_BABILONG_DATASET}}
INFINITEBENCH_DIR=${INFINITEBENCH_DIR:-${CONFIG_INFINITEBENCH_DIR}}
if [ -n "${LONG_PPL_DATA}" ]; then export LONG_PPL_DATA; fi
if [ -n "${LONG_PPL_BOOKS}" ]; then export LONG_PPL_BOOKS; fi
if [ -n "${BABILONG_DATASET}" ]; then export BABILONG_DATASET; fi
if [ -n "${INFINITEBENCH_DIR}" ]; then export INFINITEBENCH_DIR; fi

# Several lengths in one job (eval_lengths in the YAML, or EVAL_LENGTHS=32k,128k): one pass per length with its
# own window and YaRN factor (YARN_MODE=per_length, the default, or fixed), tasks chosen by EVAL_SUITES and
# NIAH_ONLY=1; see long_eval_plan.py. Each pass writes to <output_path>/long<L> (fixed: yarn<F>/long<L>).
# The full evaluation (eval_main: true in the YAML, or EVAL_MAIN=1) runs the 32K main line first (the default task
# lists, main and NIAH stage, the YAML's limit, into <output_path>), then the passes (long_limit samples per task),
# which leave out what the main line measured; EVAL_LENGTHS=none keeps the main line only.
# An explicit 512K/1M-only selection uses those passes and their standard suites by default.
DEFAULT_EVAL_MAIN=${CONFIG_EVAL_MAIN}
DEFAULT_EVAL_SUITES=${CONFIG_EVAL_SUITES}
if [[ ${EVAL_LENGTHS,,} =~ ^[[:space:],]*(512k|1m)([[:space:],]+(512k|1m))*[[:space:],]*$ ]]; then
    DEFAULT_EVAL_MAIN=0
    DEFAULT_EVAL_SUITES=${CONFIG_EVAL_SUITES:-ruler,babilong,books,longbench2}
fi
EVAL_LENGTHS=${EVAL_LENGTHS:-${CONFIG_EVAL_LENGTHS}}
EVAL_SUITES=${EVAL_SUITES:-${DEFAULT_EVAL_SUITES}}
case "$(printf '%s' "${EVAL_LENGTHS}" | tr '[:upper:]' '[:lower:]')" in
    none|off|false|0) EVAL_LENGTHS="" ;;
esac
EVAL_MAIN=${EVAL_MAIN:-${DEFAULT_EVAL_MAIN}}
if [ -n "${EVAL_LENGTHS}" ] && { [ -n "${2:-}" ] || [ -n "${3:-}" ]; }; then
    echo "ERROR: ${CONFIG_FILE} evaluates several lengths (EVAL_LENGTHS=${EVAL_LENGTHS}): choose tasks with EVAL_SUITES or NIAH_ONLY=1, not task-list arguments."
    exit 1
fi

checkpoint_exists() {
    [ -f "$1" ] || { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }
}

checkpoint_finished() {
    "${PYTHON_BIN}" - "${SAVE_DIR}" "${MAX_STEPS}" "${NUM_EPOCHS}" <<'EOF'
import os
import sys

import yaml


save_dir = sys.argv[1]
try:
    max_steps = int(float(sys.argv[2] or 0))
except (TypeError, ValueError):
    max_steps = 0
try:
    num_epochs = int(float(sys.argv[3] or 0))
except (TypeError, ValueError):
    num_epochs = 0

meta_path = os.path.join(save_dir, "checkpoint_meta.yaml")
if not os.path.exists(meta_path):
    sys.exit(0)

with open(meta_path, encoding="utf-8") as f:
    meta = yaml.safe_load(f) or {}

if meta.get("in_progress", False):
    sys.exit(1)
if meta.get("finished", False):
    sys.exit(0)

try:
    global_step = int(meta.get("global_step", -1))
except (TypeError, ValueError):
    global_step = -1
try:
    completed_epochs = int(meta.get("completed_epochs", -1))
except (TypeError, ValueError):
    completed_epochs = -1
try:
    data_epoch = int(meta.get("data_epoch", -1))
except (TypeError, ValueError):
    data_epoch = -1

epoch_reached = completed_epochs >= num_epochs if num_epochs > 0 and completed_epochs >= 0 else False
if not epoch_reached and num_epochs > 0 and data_epoch > num_epochs:
    epoch_reached = True
step_reached = max_steps > 0 and global_step >= max_steps

sys.exit(0 if epoch_reached or step_reached or (max_steps <= 0 and num_epochs <= 0) else 1)
EOF
}

checkpoint_step_label() {
    "${PYTHON_BIN}" - "${SAVE_DIR}" "${MAX_STEPS}" "${NUM_EPOCHS}" <<'EOF'
import os
import sys

import yaml


save_dir, max_steps, num_epochs = sys.argv[1], sys.argv[2], sys.argv[3]
meta_path = os.path.join(save_dir, "checkpoint_meta.yaml")
if not os.path.exists(meta_path):
    print("legacy/no-meta")
    raise SystemExit
with open(meta_path, encoding="utf-8") as f:
    meta = yaml.safe_load(f) or {}
print(
    f"step={meta.get('global_step', 'unknown')}/{max_steps}, "
    f"epoch={meta.get('completed_epochs', 'unknown')}/{num_epochs} "
    f"(current={meta.get('data_epoch', 'unknown')}), "
    f"finished={meta.get('finished', False)}, "
    f"in_progress={meta.get('in_progress', False)}"
)
EOF
}

has_tokenizer() {
    [ -n "$1" ] && { [ -f "$1/tokenizer.json" ] || [ -f "$1/tokenizer.model" ]; }
}

TOKENIZER_SOURCE=""
IFS=':' read -r -a TOKENIZER_CANDIDATE_ARRAY <<< "${TOKENIZER_CANDIDATES}"
for TOK_DIR in "${TOKENIZER_CANDIDATE_ARRAY[@]}"; do
    if has_tokenizer "${TOK_DIR}"; then
        TOKENIZER_SOURCE="${TOK_DIR}"
        break
    fi
done

EVAL_LIMIT=${LIMIT:-${EVAL_LIMIT}}  # LIMIT=2 overrides the YAML's samples per task for one run (e.g. a memory check)
# The length passes' samples per task: long_limit (the full evaluation), else the YAML's limit.
LONG_EVAL_LIMIT=${LIMIT:-${LONG_EVAL_LIMIT:-${CONFIG_LONG_LIMIT:-${EVAL_LIMIT}}}}
# What the main line runs with (the length passes bring their own overrides and limit).
MAIN_ARGS=()
if [ -n "${CONFIG_OVERRIDES}" ]; then
    MAIN_ARGS+=(--config_overrides "${CONFIG_OVERRIDES}")
fi
if [ -n "${EVAL_LIMIT}" ]; then
    MAIN_ARGS+=(--limit "${EVAL_LIMIT}")
fi
DIAG_ARGS=${DIAG_ARGS:-}
TOKENIZER_ARGS=""
if [ -n "${TOKENIZER_SOURCE}" ]; then
    TOKENIZER_ARGS="--tokenizer_dir ${TOKENIZER_SOURCE}"
    echo "Tokenizer source: ${TOKENIZER_SOURCE}"
else
    echo "WARNING: tokenizer not found in candidates: ${TOKENIZER_CANDIDATES}"
    echo "If eval.py fails to load a tokenizer, set tokenizer_dir in the YAML."
fi

echo "Checkpoint: ${SAVE_DIR}"
echo "Output dir: ${EVAL_OUTPUT_DIR}"
echo "Mamba-2 eval: fixed-size recurrent state; prefill chunk ${LOGKV_EVAL_PREFILL_CHUNK:-2048}"
if [ -n "${DIAG_ARGS}" ]; then
    echo "Extra eval args: ${DIAG_ARGS}"
fi

if ! checkpoint_exists "${SAVE_DIR}/lit_model.pth"; then
    echo "ERROR: checkpoint not found: ${SAVE_DIR}/lit_model.pth"
    exit 1
fi

if [ -z "${TOKENIZER_SOURCE}" ]; then
    echo "Set QWEN_BASE to the original Qwen tokenizer directory." >&2
    exit 1
fi

# REPAIR_HF_CACHE=1: node 0 first repairs unreadable dataset_info.json files in the shared HF cache
# (eval.py's HF_DATASETS_CACHE), keeping originals as *.truncated. Its barrier file comes after, so the
# other nodes start evaluating only once the repair is done.
HF_DATASETS_ROOT=${HF_DATASETS_ROOT:-/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/hf_cache}
if [ "${REPAIR_HF_CACHE:-0}" = "1" ] && [ "${NODE_RANK}" = "0" ]; then
    echo "Node 0: repairing unreadable dataset_info.json under ${HF_DATASETS_ROOT}"
    if ! "${PYTHON_BIN}" unused/repair_dataset_info.py --scan "${HF_DATASETS_ROOT}" --write --check; then
        echo "ERROR: HF cache repair failed (see above); not starting eval."
        exit 1
    fi
fi

# One barrier folder per job (MA_VJ_NAME names the ModelArts job, MASTER_ADDR elsewhere): evals of the same
# checkpoint may run at once, and node 0 of one must not count or remove another's node files, nor a
# failed run's leftovers.
BARRIER_DIR="${EVAL_OUTPUT_DIR}/.eval_only_barrier_d/${MA_VJ_NAME:-${MASTER_ADDR}}"
mkdir -p "${BARRIER_DIR}"
touch "${BARRIER_DIR}/node_${NODE_RANK}"

CUR_COUNT=0
LOOP_ITER=0
while [ "${CUR_COUNT}" -lt "${NUM_NODES}" ]; do
    CUR_COUNT=$(ls -1 "${BARRIER_DIR}"/node_* 2>/dev/null | wc -l)
    LOOP_ITER=$((LOOP_ITER + 1))
    echo "Barrier check ${LOOP_ITER}: ${CUR_COUNT}/${NUM_NODES} nodes ready"
    sleep 5
done

echo "All nodes are ready. Starting eval.py"
sleep 30

DEFAULT_BENCHMARKS="boolq,piqa,social_iqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"
DEFAULT_BENCHMARKS="${DEFAULT_BENCHMARKS},mmlu,ceval-valid,ifeval,truthfulqa_gen,truthfulqa_mc1,truthfulqa_mc2"
DEFAULT_BENCHMARKS="${DEFAULT_BENCHMARKS},longbench_2wikimqa,longbench_dureader,longbench_gov_report,longbench_hotpotqa"
DEFAULT_BENCHMARKS="${DEFAULT_BENCHMARKS},longbench_lcc,longbench_lsht,longbench_multi_news,longbench_multifieldqa_en,longbench_multifieldqa_zh"
DEFAULT_BENCHMARKS="${DEFAULT_BENCHMARKS},longbench_musique,longbench_narrativeqa,longbench_passage_count,longbench_passage_retrieval_en"
DEFAULT_BENCHMARKS="${DEFAULT_BENCHMARKS},longbench_qasper,longbench_qmsum,longbench_repobench-p,longbench_samsum,longbench_trec,longbench_triviaqa,longbench_vcsum"
DEFAULT_BENCHMARKS="${DEFAULT_BENCHMARKS},longbench_2wikimqa_e,longbench_gov_report_e,longbench_hotpotqa_e,longbench_lcc_e,longbench_multi_news_e"
DEFAULT_BENCHMARKS="${DEFAULT_BENCHMARKS},longbench_multifieldqa_en_e,longbench_passage_count_e,longbench_passage_retrieval_en_e,longbench_qasper_e"
DEFAULT_BENCHMARKS="${DEFAULT_BENCHMARKS},longbench_repobench-p_e,longbench_samsum_e,longbench_trec_e,longbench_triviaqa_e"
# At the native 32K: several questions on one cache (eval_tasks/niah_turns) and the book perplexity curve
# (eval_tasks/long_ppl, data LONG_PPL_BOOKS). Optional in the default list: eval.py leaves them out, with a
# warning, for a checkpoint whose window is shorter (a 4K run) or when LONG_PPL_BOOKS is not set.
DEFAULT_OPTIONAL_BENCHMARKS="niah_turns_32k,long_ppl_books_32k"
DEFAULT_BENCHMARKS="${DEFAULT_BENCHMARKS},${DEFAULT_OPTIONAL_BENCHMARKS}"

BENCHMARKS=${2:-}
if [ -z "${BENCHMARKS}" ]; then
    BENCHMARKS=${CONFIG_BENCHMARKS:-}
fi
export EVAL_OPTIONAL_TASKS=  # tasks asked for by name must run
if [ -z "${BENCHMARKS}" ]; then
    BENCHMARKS=${DEFAULT_BENCHMARKS}
    export EVAL_OPTIONAL_TASKS=${DEFAULT_OPTIONAL_BENCHMARKS}
fi

# Every needle-in-a-haystack eval: RULER single needles, lm-eval's multi-needle tasks and the
# cue-boundary prompt variants in eval_tasks/niah_cue. Pass none to skip, or a comma list for a subset.
DEFAULT_NIAH_BENCHMARKS="niah_single_1,niah_single_2,niah_single_3"
DEFAULT_NIAH_BENCHMARKS="${DEFAULT_NIAH_BENCHMARKS},niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery"
DEFAULT_NIAH_BENCHMARKS="${DEFAULT_NIAH_BENCHMARKS},niah_single_1_qend,niah_single_1_preamble,niah_single_1_paraphrase"
DEFAULT_NIAH_BENCHMARKS="${DEFAULT_NIAH_BENCHMARKS},niah_single_2_qend,niah_single_2_preamble,niah_single_2_paraphrase"
DEFAULT_NIAH_BENCHMARKS="${DEFAULT_NIAH_BENCHMARKS},niah_single_3_qend,niah_single_3_preamble,niah_single_3_paraphrase"
# niahLogKV: word-valued needles (no digit), the generalisation check for learned retention heads.
DEFAULT_NIAH_BENCHMARKS="${DEFAULT_NIAH_BENCHMARKS},niah_single_2_words,niah_single_2_words_qend,niah_single_2_words_paraphrase"
NIAH_BENCHMARKS=${3:-${CONFIG_NIAH_BENCHMARKS:-${DEFAULT_NIAH_BENCHMARKS}}}

# Task lists lose whitespace and stray commas; none / skip / off / false / 0 or nothing left skips that
# stage instead of handing eval.py an empty --benchmark. NIAH_ONLY=1 skips the main stage.
normalize_benchmarks() {
    local value
    value=$(printf '%s' "$1" | tr -d '[:space:]' | tr -s ',' | sed -e 's/^,//' -e 's/,$//')
    case "$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')" in
        ""|none|skip|off|false|0) echo none ;;
        *) echo "${value}" ;;
    esac
}
BENCHMARKS=$(normalize_benchmarks "${BENCHMARKS}")
NIAH_BENCHMARKS=$(normalize_benchmarks "${NIAH_BENCHMARKS}")
if [ "${NIAH_ONLY:-0}" = "1" ]; then
    BENCHMARKS=none
fi
for TASK_LIST in "${BENCHMARKS}" "${NIAH_BENCHMARKS}"; do
    case "${TASK_LIST}" in
        -*) echo "ERROR: task list '${TASK_LIST}' starts with '-': put eval.py options in DIAG_ARGS, not in the task arguments."
            exit 1 ;;
    esac
done
# The length passes (long_eval_plan.py), each merging its window and YaRN into the YAML's overrides; after the main
# line (EVAL_MAIN=1) they leave out its tasks and its NIAH tasks at its lengths.
LONG_EVAL_PASSES=()
if [ -n "${EVAL_LENGTHS}" ]; then
    LONG_EVAL_SKIP=""
    if [ "${EVAL_MAIN}" = "1" ]; then
        LONG_EVAL_SKIP=$("${PYTHON_BIN}" long_eval_plan.py measured "${BENCHMARKS}" "${NIAH_BENCHMARKS}" "${EVAL_MAX_SEQ_LENGTHS}") || exit 1
    fi
    LONG_EVAL_PLAN=$("${PYTHON_BIN}" long_eval_plan.py "${EVAL_LENGTHS}" "${EVAL_SUITES:-}" "${YARN_MODE:-per_length}" \
        "${NIAH_ONLY:-0}" "${CONFIG_OVERRIDES}" "${EVAL_OUTPUT_DIR}" "${LONG_EVAL_SKIP}") || exit 1
    mapfile -t LONG_EVAL_PASSES <<< "${LONG_EVAL_PLAN}"
    echo "Long-context passes (YARN_MODE=${YARN_MODE:-per_length}, EVAL_SUITES=${EVAL_SUITES:-all}, main line first: ${EVAL_MAIN:-0}):"
    printf '  %s\n' "${LONG_EVAL_PASSES[@]}"
fi
# The main stage, then the NIAH stage, with the current task lists, lengths, output dir and PASS_ARGS.
run_stages() {
    echo "Main benchmarks: ${BENCHMARKS}"
    echo "NIAH benchmarks: ${NIAH_BENCHMARKS}"
    # RULER / NIAH context lengths: the YAML's eval_max_seq_lengths (default 1K-32K).
    META='{"pretrained": "'"${TOKENIZER_SOURCE}"'", "max_seq_lengths": '"${EVAL_MAX_SEQ_LENGTHS}"'}'
    MAIN_METADATA_ARGS=()
    # Tasks that build their samples or cut their prompts with the tokenizer (RULER, long_ppl, InfiniteBench) need
    # the metadata.
    if [ "${BENCHMARKS}" != "none" ] && [[ "${BENCHMARKS}" == *niah_* || "${BENCHMARKS}" == *ruler* || "${BENCHMARKS}" == *long_ppl* \
        || "${BENCHMARKS}" == *infinitebench* ]]; then
        MAIN_METADATA_ARGS=(--metadata "${META}")
    fi

    if [ "${BENCHMARKS}" != "none" ] && [ -n "${BENCHMARKS}" ]; then
        "${PYTHON_BIN}" -m torch.distributed.run \
            --nnodes=${NUM_NODES} \
            --nproc_per_node=${GPUS_PER_NODE} \
            --node_rank=${NODE_RANK} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${MAIN_EVAL_MASTER_PORT} \
            eval.py \
            --checkpoint_dir "${SAVE_DIR}" \
            --benchmark "${BENCHMARKS}" \
            --output_path "${EVAL_OUTPUT_DIR}" \
            "${MAIN_METADATA_ARGS[@]}" \
            "${PASS_ARGS[@]}" \
            ${DIAG_ARGS} \
            ${TOKENIZER_ARGS}

        EVAL_STATUS=$?
        if [ ${EVAL_STATUS} -ne 0 ]; then
            echo "ERROR: main eval failed with exit code ${EVAL_STATUS}"
            exit ${EVAL_STATUS}
        fi
    else
        echo "Skipping main benchmark eval."
    fi

    if [ "${NIAH_BENCHMARKS}" != "none" ] && [ -n "${NIAH_BENCHMARKS}" ]; then
        "${PYTHON_BIN}" -m torch.distributed.run \
            --nnodes=${NUM_NODES} \
            --nproc_per_node=${GPUS_PER_NODE} \
            --node_rank=${NODE_RANK} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${NIAH_EVAL_MASTER_PORT} \
            eval.py \
            --checkpoint_dir "${SAVE_DIR}" \
            --benchmark "${NIAH_BENCHMARKS}" \
            --metadata "${META}" \
            --output_path "${EVAL_OUTPUT_DIR}" \
            "${PASS_ARGS[@]}" \
            ${DIAG_ARGS} \
            ${TOKENIZER_ARGS}

        EVAL_STATUS=$?
        if [ ${EVAL_STATUS} -ne 0 ]; then
            echo "ERROR: NIAH eval failed with exit code ${EVAL_STATUS}"
            exit ${EVAL_STATUS}
        fi
    else
        echo "Skipping NIAH eval."
    fi
}

if [ -z "${EVAL_LENGTHS}" ] || [ "${EVAL_MAIN}" = "1" ]; then
    PASS_ARGS=("${MAIN_ARGS[@]}")
    if [ -n "${EVAL_LENGTHS}" ]; then
        echo "===== 32K main line: ${EVAL_OUTPUT_DIR} ====="
    fi
    run_stages
fi
for PASS in "${LONG_EVAL_PASSES[@]}"; do
    [ -n "${PASS}" ] || continue
    IFS='|' read -r EVAL_OUTPUT_DIR PASS_OVERRIDES EVAL_MAX_SEQ_LENGTHS BENCHMARKS NIAH_BENCHMARKS PASS_OPTIONAL <<< "${PASS}"
    export EVAL_OPTIONAL_TASKS=${PASS_OPTIONAL}
    PASS_ARGS=(--config_overrides "${PASS_OVERRIDES}")
    if [ -n "${LONG_EVAL_LIMIT}" ]; then
        PASS_ARGS+=(--limit "${LONG_EVAL_LIMIT}")
    fi
    echo "===== Long-context pass: ${EVAL_OUTPUT_DIR} | ${PASS_OVERRIDES} ====="
    run_stages
done

if [ ${NODE_RANK} -eq 0 ] && [ -d "${BARRIER_DIR}" ]; then
    rm -rf "${BARRIER_DIR}"
fi

echo "Eval-only job finished."
