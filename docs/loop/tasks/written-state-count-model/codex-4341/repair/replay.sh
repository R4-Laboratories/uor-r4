#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 4 ]] || {
    echo 'usage: replay.sh {select|materialize|qualify} FIRST_OUT FRESH_REPLAY_OUT ORIGINAL_ARGUMENTS_EXCEPT_OUT...' >&2
    exit 2
}
stage=$1
expected=$2
out=$3
shift 3
here=$(cd "$(dirname "$0")" && pwd -P)
case "$stage" in
    select) seal=SELECTION.sha256 ;;
    materialize) seal=EXTRACTION.sha256 ;;
    qualify) seal=SEALED.sha256 ;;
    *) printf 'Unknown stage %s\n' "$stage" >&2; exit 2 ;;
esac
test -s "$expected/$seal"
bash "$here/$stage.sh" "$@" "$out"
test -s "$out/$seal"
# Omit file diffs because these private outputs include reviewer text.
diff -rq "$expected" "$out"
