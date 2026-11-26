#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
panel=$(realpath -e "$1"); raw=$(realpath -e "$2"); probes=$(realpath -e "$3")
factors=$4; out=$5
mkdir "$out"; out=$(realpath -e "$out")
tail -n +2 "$factors" | cut -f1,5 | xargs -r -n2 -P4 bash -c '
    set -eu
    bash "$1/state.sh" "$2" "$3" "$4" "$6" "$7" b "$5/$6" train
' _ "$here" "$panel" "$raw" "$probes" "$out"
