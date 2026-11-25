#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/mamba2"
case "${1:-}" in
    --prepare) shift; exec bash diag.sh prepare.py "$@" ;;
    --train) shift; exec bash majob.sh "${1:-exp/cpt.yaml}" ;;
    --plan) shift; exec python long_eval_plan.py "${1:-32k,64k,128k,256k,512k,1m}" "${EVAL_SUITES:-}" per_length "${NIAH_ONLY:-0}" '{}' evaluate ;;
    *) exec bash eval.sh "${1:-exp/eval_full.yaml}" "${@:2}" ;;
esac
