#!/usr/bin/env bash
# Baseline evaluation. Run once per node, as in niahLogKV.
set -e
cd "$(dirname "$0")"
source scripts/hattention_env.sh
print_only=0
if [[ ${1:-} == --print-config ]]; then
    print_only=1
    shift
fi
CONFIG_FILE=${1:-configs/nfini_eval.yaml}
export NFINI_CHECKPOINT_DIR=${NFINI_CHECKPOINT_DIR:-${NFINI_OUTPUT_ROOT:-checkpoints/nfini-attention}/cpt}
eval_settings=$("${PYTHON_BIN}" - "${CONFIG_FILE}" <<'PYCONFIG'
import json, os, shlex, sys
import yaml
from pathlib import Path
from lla_bench.utils import expand_env_vars
cfg = expand_env_vars(yaml.safe_load(Path(sys.argv[1]).read_text()))
from importlib.metadata import version
for package, expected in (("lm-eval", "0.4.11"), ("datasets", "4.8.4")):
    actual = version(package)
    if actual != expected:
        raise SystemExit(f"{package} {actual}, expected {expected} for the pinned evaluation protocol")
def emit(name, value):
    print(name+'='+shlex.quote(str(value)))
emit("METHOD", cfg.get("method", "hattention"))
emit("SAVE_DIR", cfg["checkpoint_dir"])
emit("EVAL_OUTPUT_DIR", cfg["output_path"])
emit("CONFIG_EVAL_LENGTHS", ",".join(cfg["eval_lengths"]))
emit("CONFIG_EVAL_SUITES", ",".join(cfg.get("eval_suites") or []))
emit("CONFIG_EVAL_MAIN", int(cfg["eval_main"]))
emit("CONFIG_LONG_LIMIT", cfg["long_limit"])
emit("EVAL_MAX_SEQ_LENGTHS", json.dumps(cfg["eval_max_seq_lengths"]))
emit("CONFIG_OVERRIDES", json.dumps({"block_size": 32768}))
emit("CONFIG_BENCHMARKS", cfg.get("benchmark") or "")
emit("CONFIG_NIAH_BENCHMARKS", cfg.get("niah_benchmark") or "")
PYCONFIG
)
eval "${eval_settings}"
# Several lengths in one job (eval_lengths in the YAML, or EVAL_LENGTHS=32k,128k): one pass per length with its
# own window and YaRN factor (YARN_MODE=per_length, the default, or fixed), tasks chosen by EVAL_SUITES and
# NIAH_ONLY=1; see long_eval_plan.py. Each pass writes to <output_path>/long<L> (fixed: yarn<F>/long<L>).
# The full evaluation (eval_main: true in the YAML, or EVAL_MAIN=1) runs the 32K main line first (the default task
# lists, main and NIAH stage, the YAML's limit, into <output_path>), then the passes (long_limit samples per task),
# which leave out what the main line measured; EVAL_LENGTHS=none keeps the main line only.
# An explicit 512K/1M-only selection uses those passes and their standard suites by default.
DEFAULT_EVAL_MAIN=${CONFIG_EVAL_MAIN}
DEFAULT_EVAL_SUITES=${CONFIG_EVAL_SUITES}
if [[ $(printf '%s' "${EVAL_LENGTHS:-}" | tr '[:upper:]' '[:lower:]') =~ ^[[:space:],]*(512k|1m)([[:space:],]+(512k|1m))*[[:space:],]*$ ]]; then
    DEFAULT_EVAL_MAIN=0
    DEFAULT_EVAL_SUITES=${CONFIG_EVAL_SUITES:-ruler,babilong,books,longbench2}
fi
if [[ ${METHOD} == nfini-attention && -z ${CONFIG_EVAL_SUITES} ]]; then
    DEFAULT_EVAL_SUITES=ruler,babilong,ppl,books,turns,longbench2
    suite_lengths=$(printf '%s' "${EVAL_LENGTHS:-${CONFIG_EVAL_LENGTHS}}" | tr '[:upper:] ' '[:lower:],')
    case ",${suite_lengths}," in
        *,32k,*|*,64k,*|*,128k,*|*,256k,*) DEFAULT_EVAL_SUITES+=,infinitebench ;;
    esac
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


planner=lla_bench.long_eval_plan
if [[ ${METHOD} == nfini-attention ]]; then
    planner=nfini_attention.long_eval_plan
    export LOGKV_EVAL_PREFILL_CHUNK=${LOGKV_EVAL_PREFILL_CHUNK:-2048}
elif [[ ${YARN_MODE:-per_length} != per_length ]]; then
    echo "The native SSM uses no rotary embeddings. Fixed YaRN runs would duplicate the same model." >&2
    exit 1
fi
MAIN_ARGS=(--config_overrides "${CONFIG_OVERRIDES}")
EVAL_LIMIT=${LIMIT:-}
LONG_EVAL_LIMIT=${LIMIT:-${LONG_EVAL_LIMIT:-${CONFIG_LONG_LIMIT}}}
[[ -z "${EVAL_LIMIT}" ]] || MAIN_ARGS+=(--limit "${EVAL_LIMIT}")
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
    if [[ ${METHOD} != nfini-attention ]]; then export EVAL_OPTIONAL_TASKS=${DEFAULT_OPTIONAL_BENCHMARKS}; fi
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
        -*) echo "ERROR: task list '${TASK_LIST}' starts with '-': expected comma-separated task names."
            exit 1 ;;
    esac
done
# The length passes (long_eval_plan.py), each merging its window and YaRN into the YAML's overrides; after the main
# line (EVAL_MAIN=1) they leave out its tasks and its NIAH tasks at its lengths.
LONG_EVAL_PASSES=()
if [ -n "${EVAL_LENGTHS}" ]; then
    LONG_EVAL_SKIP=""
    if [ "${EVAL_MAIN}" = "1" ]; then
        LONG_EVAL_SKIP=$("${PYTHON_BIN}" -m "${planner}" measured "${BENCHMARKS}" "${NIAH_BENCHMARKS}" "${EVAL_MAX_SEQ_LENGTHS}") || exit 1
    fi
    LONG_EVAL_PLAN=$("${PYTHON_BIN}" -m "${planner}" "${EVAL_LENGTHS}" "${EVAL_SUITES:-}" "${YARN_MODE:-per_length}" \
        "${NIAH_ONLY:-0}" "${CONFIG_OVERRIDES}" "${EVAL_OUTPUT_DIR}" "${LONG_EVAL_SKIP}") || exit 1
    while IFS= read -r pass; do LONG_EVAL_PASSES+=("$pass"); done <<< "${LONG_EVAL_PLAN}"
    echo "Long-context passes (YARN_MODE=${YARN_MODE:-per_length}, EVAL_SUITES=${EVAL_SUITES:-all}, main line first: ${EVAL_MAIN:-0}):"
    printf '  %s\n' "${LONG_EVAL_PASSES[@]}"
fi
if [[ "${print_only}" == 1 ]]; then
    if [[ -z "${EVAL_LENGTHS}" || "${EVAL_MAIN}" == 1 ]]; then
        echo "Main: ${BENCHMARKS}"
        echo "NIAH: ${NIAH_BENCHMARKS} at ${EVAL_MAX_SEQ_LENGTHS}"
    fi
    echo "Checkpoint: ${SAVE_DIR}"
    echo "Results: ${EVAL_OUTPUT_DIR}"
    exit 0
fi
if [[ ! -s "${SAVE_DIR}/model.safetensors" && ! -s "${SAVE_DIR}/model.safetensors.index.json" ]]; then
    echo "Missing model weights in ${SAVE_DIR}" >&2
    exit 1
fi
for file in config.json tokenizer.json tokenizer_config.json; do
    if [[ ! -s "${SAVE_DIR}/${file}" ]]; then
        echo "Missing ${SAVE_DIR}/${file}" >&2
        exit 1
    fi
done
if [[ ${METHOD} == nfini-attention ]]; then
    "${PYTHON_BIN}" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); sys.exit(0 if "infini_segment_length" in json.loads((p/"config.json").read_text()) and (p/"training.json").is_file() else "Expected an exported Infini checkpoint: "+str(p))' "${SAVE_DIR}"
fi
mkdir -p "${EVAL_OUTPUT_DIR}"
# Detect GPUs visible to this job (including CUDA_VISIBLE_DEVICES); managed_cluster
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


# The main stage, then the NIAH stage, with the current task lists, lengths, output dir and PASS_ARGS.
run_stages() {
    echo "Main benchmarks: ${BENCHMARKS}"
    echo "NIAH benchmarks: ${NIAH_BENCHMARKS}"
    # RULER / NIAH context lengths: the YAML's eval_max_seq_lengths (default 1K-32K).
    META=$("${PYTHON_BIN}" -c 'import json,sys; print(json.dumps(dict(pretrained=sys.argv[1], max_seq_lengths=json.loads(sys.argv[2]))))' "${SAVE_DIR}" "${EVAL_MAX_SEQ_LENGTHS}")
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
            --module lla_bench.eval \
            --checkpoint_dir "${SAVE_DIR}" \
            --benchmark "${BENCHMARKS}" \
            --output_path "${EVAL_OUTPUT_DIR}" \
            "${MAIN_METADATA_ARGS[@]}" \
            "${PASS_ARGS[@]}" \
            --tokenizer_dir "${SAVE_DIR}"

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
            --module lla_bench.eval \
            --checkpoint_dir "${SAVE_DIR}" \
            --benchmark "${NIAH_BENCHMARKS}" \
            --metadata "${META}" \
            --output_path "${EVAL_OUTPUT_DIR}" \
            "${PASS_ARGS[@]}" \
            --tokenizer_dir "${SAVE_DIR}"

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


echo "${METHOD} evaluation finished."
