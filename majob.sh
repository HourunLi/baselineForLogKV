#!/usr/bin/env bash
set -e
cd "$(dirname "$0")" || exit 1
case ${1:-} in
    --pipeline)
        [[ $# == 1 ]] || { echo "Usage: bash majob.sh --pipeline" >&2; exit 2; }
        exec bash scripts/train_hattention.sh pipeline ;;
    --train)
        shift
        exec bash scripts/train_hattention.sh "$@" ;;
esac
exec bash eval.sh "$@"
