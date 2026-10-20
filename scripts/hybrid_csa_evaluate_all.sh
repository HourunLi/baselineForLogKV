#!/usr/bin/env bash
# Evaluation only, including when no adapter is provided for dense/logkv.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash majob.sh --mode eval "$@"
