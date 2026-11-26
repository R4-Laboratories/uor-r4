#!/usr/bin/env bash
# Reuse the original audited compact archive and restore path.
set -euo pipefail
umask 077
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-far-transport-margin"
old="$runs/20261126-codex-governor-noun-links/methods/gate-v4/preserve-wick.sh"
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$study") == zfs ]]
printf '%s  %s\n' 0040fbea3450ba6cf976b31e3c897e6f590f0f6f945b4aa9c5170fdb8f9f8c48 "$old" | sha256sum --strict --quiet -c -
adapter=$(mktemp "$study/private-storage-adapter.XXXXXXXX")
sed -e 's/20261126-codex-governor-noun-links/20261126-codex-far-transport-margin/g' \
    -e 's@results/codex/governor-noun-links-20261126-@results/codex/far-transport-margin-20261126-@' \
    -e 's/Preserve the addressed governor-link measurements/Preserve the far transport margin measurements/' "$old" > "$adapter"
bash "$adapter" "$@"
