#!/usr/bin/env bash
# Preserve the independent panel and actual picked addresses. Any client rows
# and summaries passed here come from the registered run at those addresses.
set -euo pipefail
umask 077
[[ $(hostname -s) == wick ]]
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-governor-noun-links
[[ $(findmnt -no FSTYPE --target "$study") == zfs ]]
sha() { sha256sum "$1" | cut -d' ' -f1; }
case ${1:?pack or publish} in
pack)
    shift
    out=${1:?new packet path}
    shift
    [[ $out == "$study"/packet-v* ]]
    mkdir "$out" "$out/upload"
    for path in "$@"; do
        file=$(realpath "$path")
        [[ $file == "$study"/* && -f $file ]]
        printf '%s\n' "${file#"$study"/}"
    done | LC_ALL=C sort -u > "$out/members.txt"
    test -s "$out/members.txt"
    (cd "$study" &&
        while IFS= read -r path; do sha256sum "$path"; done < "$out/members.txt") > "$out/upload/MEMBERS.sha256"
    tar -C "$study" --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
        -cf - -T "$out/members.txt" | zstd -q -15 -T1 -o "$out/upload/measurement.tar.zst"
    (cd "$out/upload" && sha256sum MEMBERS.sha256 measurement.tar.zst > SHA256SUMS)
    ;;
publish)
    packet=$(realpath "${2:?sealed packet}/upload")
    out=${3:?new restore receipt directory}
    [[ $packet == "$study"/packet-v*/upload && $out == "$study"/restore-v* ]]
    (cd "$packet" && sha256sum --strict --quiet -c SHA256SUMS)
    mkdir "$out" "$out/extracted"
    hf="$HOME/.local/bin/hf"
    repo=caseyallard/uor-r4-store
    pack_sha=$(sha "$packet/measurement.tar.zst")
    prefix="results/codex/governor-noun-links-20261126-${pack_sha:0:12}"
    "$hf" datasets info "$repo" --json | jq -e 'select(.private==true)|{private}' > "$out/privacy.json"
    "$hf" upload "$repo" "$packet" "$prefix" --repo-type dataset \
        --commit-message '[codex] Preserve the addressed governor-link measurements' --format json > "$out/upload.json"
    rev=$("$hf" datasets info "$repo" --json | jq -er 'select(.private)|.sha')
    [[ $rev =~ ^[0-9a-f]{40}$ ]]
    "$hf" download "$repo" --repo-type dataset --revision "$rev" --include "$prefix/*" \
        --local-dir "$out/download" --quiet > "$out/download.log"
    restored="$out/download/$prefix"
    for name in MEMBERS.sha256 measurement.tar.zst SHA256SUMS; do cmp "$packet/$name" "$restored/$name"; done
    tar --zstd -xf "$restored/measurement.tar.zst" -C "$out/extracted"
    (cd "$out/extracted" && sha256sum --quiet --strict -c "$restored/MEMBERS.sha256")
    jq -n --arg repo "$repo" --arg path "$prefix" --arg rev "$rev" --arg archive "$pack_sha" \
        --argjson count "$(wc -l < "$restored/MEMBERS.sha256")" \
        --argjson bytes "$(du -bc "$packet"/* | awk 'END{print $1}')" \
        '{repo:$repo,path:$path,revision:$rev,archive_sha256:$archive,raw_files_verified:$count,upload_bytes:$bytes}' \
        > "$out/receipt.json"
    ;;
*) exit 64 ;;
esac
