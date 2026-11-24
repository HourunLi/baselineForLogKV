#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
if [[ ${1:-} == --train ]]; then
    shift
    exec bash scripts/train_hattention.sh "$@"
fi
exec bash eval.sh "$@"
