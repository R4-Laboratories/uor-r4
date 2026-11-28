#!/usr/bin/env bash
set -euo pipefail
umask 077

if [ "$#" -ne 6 ]; then
    echo "usage: run.sh WICK_RUNS COLD_4333 COLD_4332 COLD_4330 D26_ARTIFACT NEW_OUTPUT" >&2
    exit 2
fi
out=$(realpath -m "$6")
src=$(cd "$(dirname "$0")" && pwd)
mkdir "$out"
bash "$src/capture.sh" "$1" "$2" "$4" "$5" "$out/client"
bash "$src/grade.sh" "$2" "$3" "$4" "$out"
