#!/usr/bin/env bash
set -euo pipefail
lib=$(dirname "$(realpath "$0")")/archive-lib.sh
source "$lib"
root=${1:?new Wick fixture directory}
[[ $# == 1 ]]
mkdir "$root" "$root/source" "$root/modified"
printf '{"votes":[-7,0,3],"status":"NOT_RUN"}\n' > "$root/source/receipt.json"
printf 'UORSTT01\000\001\377' > "$root/source/sample.state"
printf '%s\n' receipt.json sample.state > "$root/members.txt"
write_packet "$root/source" "$root/members.txt" "$root/packet"
write_packet "$root/source" "$root/members.txt" "$root/repeated"
for file in MEMBERS.sha256 SHA256SUMS measurement.tar.zst; do
    cmp "$root/packet/$file" "$root/repeated/$file"
    cp "$root/packet/$file" "$root/modified/$file"
done
extract_packet "$root/packet" "$root/extracted"
for file in receipt.json sample.state; do cmp "$root/source/$file" "$root/extracted/$file"; done
chmod u+w "$root/modified/measurement.tar.zst"
printf x >> "$root/modified/measurement.tar.zst"
if bash -c 'source "$1"; extract_packet "$2" "$3"' _ \
    "$lib" "$root/modified" "$root/should-refuse" > "$root/modified.log" 2>&1; then
    echo 'Altered packet was accepted' >&2
    exit 1
fi
test ! -e "$root/should-refuse"
grep -Fq 'computed checksum did NOT match' "$root/modified.log"
jq -n --arg packet "$(digest "$root/packet/measurement.tar.zst")" '
    {pass:true,archive_sha256:$packet,deterministic_recompression:true,
     byte_identical_extraction:true,rejected_modified_packet:true}' > "$root/receipt.json"
(cd "$root" && find . -type f ! -name .seal.pending -print0 | sort -z | xargs -0 -r sha256sum > .seal.pending && \
    mv .seal.pending SHA256SUMS)
cat "$root/receipt.json"
