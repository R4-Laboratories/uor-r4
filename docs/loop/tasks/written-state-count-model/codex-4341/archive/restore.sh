#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# != 3 ]]; then
    echo 'usage: restore.sh PRIVATE_PACKET PINNED_RECEIPT FRESH_RESTORE' >&2
    exit 2
fi
here=$(cd "$(dirname "$0")" && pwd -P)
packet=$(realpath "$1")
receipt=$(realpath "$2")
dest=$(realpath -m "$3")
jq -e '.schema=="uor-r4-natural-reciprocal-repair-archive/1" and (.kind|IN("repair","result"))' "$receipt" >/dev/null
bash "$here/../../codex-4338/archive/restore.sh" "$packet" "$receipt" "$dest"
(
    cd "$dest"
    if [[ $(jq -er .kind "$receipt") == result ]]; then
        jq -e '.trial=="finished"' result/FINAL-RUN.json >/dev/null
    else
        test ! -e result
    fi
    test -z "$(find . ! -type f ! -type d -print -quit)"
    test -z "$(find . -type f -name '*.d11' -print -quit)"
)
