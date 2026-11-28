#!/usr/bin/env bash
# Never add these score-only tapes to the TRAIN capsule.
set -euo pipefail
umask 077
if [[ $# != 5 ]]; then
    echo 'usage: stage-eval.sh FIT FIT_SHA ROSTER_ROOT ROSTER_SEAL_SHA NEW_OUT' >&2
    exit 2
fi
here=$(cd "$(dirname "$0")" && pwd -P)
# shellcheck source=../repair/custody.sh
source "$here/../repair/custody.sh"
fit=$(realpath "$1")
roster=$(realpath "$3")
out=$(realpath -m "$5")
pin "$2" "$fit"
pin "$4" "$roster/ROSTER.sha256"
(cd "$roster" && sha256 --quiet --strict -c ROSTER.sha256)
(cd "$roster/blind" && sha256 --quiet --strict -c BLIND.sha256)
jq -e '.coverage == "pass"' "$roster/coverage.json" >/dev/null
mkdir "$out"
cd "$out"
for panel in validation protected; do
    for arm in geometry count; do
        "$fit" prepare-eval "$roster/blind/$panel/features.jsonl" "$arm" "$panel.$arm.eval.tape"
    done
done
printf '%s  ROSTER.sha256\n' "$4" > roster-pin.txt
printf '%s  interval-fit\n' "$2" > fit-pin.txt
sha256 ./*.tape roster-pin.txt fit-pin.txt > EVAL.sha256
