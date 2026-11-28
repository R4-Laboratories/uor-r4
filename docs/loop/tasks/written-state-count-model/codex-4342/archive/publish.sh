#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $# == 2 ]] || { echo 'usage: publish.sh PACKED_WORK FRESH_VERIFICATION' >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd -P)
packed=$(realpath "$1")
verification=$(realpath -m "$2")
mkdir "$verification"
repo=caseyallard/uor-r4-store
path=results/codex/directed-relation-20261128/complete-words.tar.zst
restore="$here/../../codex-4338/archive/restore.sh"
receipt="$packed/receipt.json"
jq -e '.schema=="uor-r4-directed-relation-screen/1"' "$receipt" >/dev/null
bash "$restore" "$packed/screen.tar.zst" "$receipt" "$verification/preflight"
hf datasets info "$repo" --expand private,sha,siblings --json > "$verification/before.json"
jq -e --arg path "$path" '.private==true and (.siblings|type)=="array" and all(.siblings[]; .rfilename!=$path)' "$verification/before.json" >/dev/null
hf upload "$repo" "$packed/screen.tar.zst" "$path" --repo-type dataset --commit-message '[codex] Preserve #4342 source qualification and native interval screen' > "$verification/upload.txt"
revision=$(sed -n 's@.*commit/\([a-f0-9]\{40\}\).*@\1@p' "$verification/upload.txt")
[[ "$revision" =~ ^[0-9a-f]{40}$ ]]
hf datasets info "$repo" --revision "$revision" --expand private,sha --json |
    jq -e --arg revision "$revision" '.private==true and .sha==$revision' >/dev/null
hf download "$repo" "$path" --repo-type dataset --revision "$revision" --force-download --local-dir "$verification/download" > "$verification/download.txt"
cmp "$packed/screen.tar.zst" "$verification/download/$path"
bash "$restore" "$verification/download/$path" "$receipt" "$verification/restored"
jq --arg repository "$repo" --arg path "$path" --arg revision "$revision" \
    '. + {$repository,$path,$revision,private_repository_verified:true,forced_download_byte_identical:true,restored_members_verified:true}' "$receipt" > "$verification/archive.json"
