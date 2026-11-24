#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
if [[ ${1:-} == --pipeline || ${1:-} == --check ]]; then
    set +u  # managed_cluster conda activation can read unset variables.
    source scripts/hattention_env.sh
    exec "${PYTHON_BIN}" -m nfini_attention.pipeline "$@"
fi
if [[ ${1:-} == --train ]]; then
    shift
    exec bash scripts/train_nfini.sh "$@"
fi
exec bash eval.sh "$@"
