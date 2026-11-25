#!/usr/bin/env bash
set -e
cd "$(dirname "$0")" || exit 1
case ${1:-} in
    --eval-student)
        [[ $# == 1 ]] || exit 2
        export LLA_PIPELINE_EVAL_ONLY=1
        exec bash scripts/train_hattention.sh pipeline ;;
    --pipeline)
        if [[ $# == 3 && $2 == --cpt-steps && $3 =~ ^[1-9][0-9]*$ ]]; then
            export LLA_PIPELINE_CPT_STEPS=$3
        elif [[ $# != 1 ]]; then
            echo "Usage: bash majob.sh --pipeline [--cpt-steps N]" >&2
            exit 2
        fi
        exec bash scripts/train_hattention.sh pipeline ;;
    --train)
        shift
        exec bash scripts/train_hattention.sh "$@" ;;
esac
exec bash eval.sh "$@"
