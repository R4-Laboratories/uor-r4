#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# != 2 ]]; then
    echo 'usage: publish.sh FINAL_PACKET_DIR FRESH_VERIFICATION_DIR' >&2
    exit 2
fi
here=$(cd "$(dirname "$0")" && pwd -P)
pack=$(realpath "$1")
verification=$(realpath -m "$2")
repository=caseyallard/uor-r4-store
path=results/codex/natural-reciprocal-repair-20261127/matched-transfer-result.tar.zst
packet=$pack/matched-transfer-result.tar.zst
base_receipt=$pack/packet.json
jq -e '.schema=="uor-r4-natural-reciprocal-repair-archive/1" and .kind=="result"' "$base_receipt" >/dev/null
(cd "$pack" && sha256sum --strict --quiet -c PACKET.sha256)
mkdir "$verification"
bash "$here/restore.sh" "$packet" "$base_receipt" "$verification/before-upload"
hf datasets info "$repository" --expand private,sha,siblings --json > "$verification/private-before.json"
jq -e --arg path "$path" '.private == true and (.siblings|type)=="array" and all(.siblings[];.rfilename!=$path)' \
    "$verification/private-before.json" >/dev/null
hf upload "$repository" "$packet" "$path" --repo-type dataset \
    --commit-message '[codex] Preserve #4341 matched reciprocal transfer result' > "$verification/upload.txt"
revision=$(sed -n 's@.*commit/\([a-f0-9]\{40\}\).*@\1@p' "$verification/upload.txt")
[[ $revision =~ ^[a-f0-9]{40}$ ]]
hf datasets info "$repository" --revision "$revision" --expand private,sha --json \
    > "$verification/private-after.json"
jq -e --arg revision "$revision" '.private==true and .sha==$revision' "$verification/private-after.json" >/dev/null
hf download "$repository" "$path" --repo-type dataset --revision "$revision" --force-download \
    --local-dir "$verification/download" > "$verification/download.txt"
cmp "$packet" "$verification/download/$path"
bash "$here/restore.sh" "$verification/download/$path" "$base_receipt" "$verification/restored"
cp "$packet" "$verification/corrupt.tar.zst"
printf '\000' | dd of="$verification/corrupt.tar.zst" bs=1 count=1 conv=notrunc status=none
if bash "$here/restore.sh" "$verification/corrupt.tar.zst" "$base_receipt" "$verification/forged" \
    > "$verification/corruption-test.log" 2>&1; then
    echo 'Restore accepted a corrupted packet' >&2
    exit 1
fi
test ! -e "$verification/forged"
jq --arg repository "$repository" --arg path "$path" --arg revision "$revision" \
   --slurpfile run "$verification/restored/result/FINAL-RUN.json" \
   '. + {$repository,$path,$revision,
        private_repository_verified:true,forced_download_byte_identical:true,
        restored_members_verified:true,corrupted_packet_refused_before_extraction:true,
        outcome:($run[0]|{trial,source_status,writer_status,locked_evaluation})}' \
   "$base_receipt" > "$verification/archive.json"
