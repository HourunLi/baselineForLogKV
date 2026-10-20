#!/usr/bin/env bash
# Run once per node. Default: prepare -> adaptation -> main -> NIAH, every phase in SemanticLogKV's torch218.
set -euo pipefail
cd "$(dirname "$0")"
source scripts/hybrid_csa_runtime.sh
# The bootstrap only needs PyYAML; every phase uses the interpreter hybrid_csa.pipeline selects (torch218).
TORCH218_PYTHON=/home/ma-user/anaconda3/envs/torch218/bin/python
[[ -x "$TORCH218_PYTHON" ]] || TORCH218_PYTHON=python
exec "${BASELINE_PYTHON:-${HYBRID_CSA_PYTHON:-${BASELINE_EVAL_PYTHON:-$TORCH218_PYTHON}}}" -m hybrid_csa.pipeline "$@"
