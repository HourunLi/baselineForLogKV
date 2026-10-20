#!/usr/bin/env bash
# Run once per node. Default: prepare -> adaptation -> main -> NIAH.
set -euo pipefail
cd "$(dirname "$0")"
source scripts/hybrid_csa_runtime.sh
# The bootstrap only needs PyYAML; each phase uses its explicitly selected Python.
exec "${BASELINE_PYTHON:-${HYBRID_CSA_TRAIN_PYTHON:-${HYBRID_CSA_PYTHON:-python}}}" -m hybrid_csa.pipeline "$@"
