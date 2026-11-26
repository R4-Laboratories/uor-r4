#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $(hostname -s) == wick ]]
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-governor-noun-links
methods=$(cd "$(dirname "$0")" && pwd)
capture=$(realpath "${1:?sealed ordinary capture}")
out=${2:?new audit directory}
[[ $capture == "$study"/client/capture-v* && $out == "$study"/reply-audit-v* ]]
(cd "$capture" && sha256sum --quiet --strict -c SHA256SUMS)
mkdir "$out"
reference="$study/panel/panel-v1/fresh/reference.json"
jq -c '.record.turns[1] as $t | {id,ids:$t.reply_ids,stop:$t.stop,text:$t.reply_text}' \
    "$capture/stock/fresh/rows.jsonl" > "$out/stock.jsonl"
for arm in native governor_only nouns_only correct_link exchanged_link; do
    jq -nc --arg arm "$arm" --slurpfile stock "$out/stock.jsonl" --slurpfile ref "$reference" \
        -f "$methods/read-replies.jq" "$capture/$arm/fresh/rows.jsonl" > "$out/$arm.jsonl"
    # Both methods independently reconstruct and compare every original token pair.
    jq -sc 'sort_by(.id)|map([.id,.first.wrong,.returned.wrong,.returned.exact,.returned.cap_prefix,.retry])' \
        "$out/$arm.jsonl" > "$out/$arm.own"
    jq -sc 'def err: if .==null then null else {position,expected:.expected_id,actual:.actual_id} end;
        sort_by(.id)|map([.id,(.first.first_wrong|err),(.returned.first_wrong|err),
                            .exact_terminal,.returned.correct_prefix_at_cap,.retry])' \
        "$capture/grade/$arm.jsonl" | cmp - "$out/$arm.own"
done
jq -se -f "$methods/summarize-replies.jq" \
    "$out/"{native,governor_only,nouns_only,correct_link,exchanged_link}.jsonl > "$out/summary.json"
sha256sum "$methods/"{audit-replies-wick.sh,read-replies.jq,summarize-replies.jq} "$reference" "$capture/SHA256SUMS" > "$out/inputs.sha256"
(cd "$out" && find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS &&
    sha256sum --strict --quiet -c SHA256SUMS)
