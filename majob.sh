#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
exec bash eval.sh "$@"
