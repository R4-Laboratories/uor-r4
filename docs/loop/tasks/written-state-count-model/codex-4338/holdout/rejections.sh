#!/usr/bin/env bash
set -euo pipefail
umask 077
official=$(realpath "${1:?official DEV JSON}")
roster=$(realpath "${2:?sealed roster JSONL}")
out=$(realpath -m "${3:?fresh results}")
here=$(cd "$(dirname "$0")" && pwd)
mkdir "$out"

reject() {
    local name=$1 edit=$2 reason=$3
    jq -c "if input_line_number == 1 then $edit else . end" "$roster" > "$out/$name.jsonl"
    if jq -e --slurpfile roster "$out/$name.jsonl" -f "$here/check.jq" "$official" > "$out/$name.stdout" 2> "$out/$name.stderr"; then
        printf 'Accepted forged roster: %s\n' "$name" >&2
        return 1
    fi
    grep -Fq "$reason" "$out/$name.stderr"
    jq -nc --arg case "$name" --arg reason "$reason" '{case:$case, result:"REFUSED", reason:$reason}' >> "$out/results.jsonl"
}

reject changed_question '.positive.question += " altered"' 'official question drift'
reject changed_window '.shared_passage += " altered"' 'window bytes drift'
reject changed_annotation '.positive.original_annotation_bytes[0] += 1' 'window annotation byte origin drift'
reject changed_shared_id '.negative.prefix[.negative.user_bounds[0]] += 1' 'shared token IDs diverge'
reject changed_copy_address '.positive.label.first_position = .positive.user_bounds[1]' 'positive original addresses drift'
reject changed_split '.split = "forged"' 'article crosses split'
