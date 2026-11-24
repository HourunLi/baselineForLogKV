#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
if [[ ${1:-} == --train ]]; then
    shift
    exec bash scripts/train_nfini.sh "$@"
fi
exec bash eval.sh "$@"
