#!/usr/bin/env bash
# Budget sweep in the style of KVCache-Factory's eval_methods.sh; extra args are forwarded.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:?Usage: bash scripts/sweep_budget.sh configs/qwen3_1.7b_32k_pyramidkv.yaml --suite niah [--limit 4] ...}
shift
for budget in ${BUDGETS:-128 256 512 1024 2048 4096}; do
    bash scripts/evaluate.sh "$CONFIG" --max-capacity-prompt "$budget" \
        --output "outputs/pyramidkv-b${budget}" "$@"
done
