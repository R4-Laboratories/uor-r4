#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
panel=$(realpath -e "${1:?panel root}")
raw=$(realpath -e "${2:?physical cohort folders}")
states=$(realpath -e "${3:?state manifest}")
out=${4:?new result directory}
shift 4
[[ $# -gt 0 ]]
mkdir "$out"
out=$(realpath -e "$out")
cut -f1 "$states" | xargs -r -P 4 -I STATE bash "$here/grade.sh" "$panel" "$raw" STATE "$out/STATE" "$@"
