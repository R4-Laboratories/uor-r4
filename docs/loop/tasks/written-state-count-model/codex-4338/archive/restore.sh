#!/usr/bin/env bash
set -euo pipefail
umask 077
packet=$(realpath "${1:?downloaded private packet}")
receipt=$(realpath "${2:?published JSON receipt}")
dest=$(realpath -m "${3:?fresh restore root}")
[[ $(stat -c %s "$packet") == "$(jq -er .bytes "$receipt")" ]]
printf '%s  %s\n' "$(jq -er .sha256 "$receipt")" "$packet" | sha256sum --strict --quiet -c -
mkdir "$dest"
tar --no-same-owner --no-same-permissions --zstd -xf "$packet" -C "$dest"
(
    cd "$dest"
    printf '%s  MEMBERS.sha256\n' "$(jq -er .members_sha256 "$receipt")" | sha256sum --strict --quiet -c -
    sha256sum --strict --quiet -c MEMBERS.sha256
    count=$(find . -type f ! -path ./MEMBERS.sha256 | wc -l)
    test "$count" -eq "$(wc -l < MEMBERS.sha256)"
    test "$count" -eq "$(jq -er .member_count "$receipt")"
)
