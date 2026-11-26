#!/usr/bin/env bash
set -euo pipefail
s=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-all-outranker-interval
here=$(cd "$(dirname "$0")" && pwd)
[[ $(hostname -s) == wick && $here == "$s"/replay-summary-methods-v* ]]
m=$(realpath "$1"); a=$(realpath "$2"); h=$(realpath "$3"); v=${4:?new version}
[[ $m == "$s"/measurement-v* && $a == "$s"/audit/train-v* && $h == "$s"/audit/held-v* && $v =~ ^v[1-9][0-9]*$ ]]
for d in "$m" "$a" "$h"; do (cd "$d" && sha256sum --strict --quiet -c SHA256SUMS); done
out="$s/replay-summary-$v"; mkdir "$out"
jq -ne --slurpfile endpoint "$m/train/all/round-1/device/base.edges.jsonl" \
    --slurpfile ordered "$m/train/ordered-all/round-1/device/base.edges.jsonl" -f "$here/starting-costs.jq" > "$out/starting-costs.json"
jq -n --slurpfile train "$a/readback.json" --slurpfile held "$h/readback.json" \
    --arg train_sha "$(sha256sum "$a/SHA256SUMS"|cut -d' ' -f1)" --arg held_sha "$(sha256sum "$h/SHA256SUMS"|cut -d' ' -f1)" '
    $train[0] as $t|$held[0] as $h|[$t.arms[].rounds[].independent] as $rounds|
    {train_manifest:$train_sha,held_manifest:$held_sha,train_pass:$t.pass,held_pass:$h.pass,train_batches:$t.batches,
     native_states:([$rounds[].state_sha256|length]|add),joint_fields:([$rounds[].joint_fields]|add),
     signed_fields:([$rounds[].signed_fields]|add),held_raw_fields:([$h.grades[].arms[].raw_fields]|add),
     writer_released:$h.writer_released}' > "$out/independent.json"
sha256sum "$m/SHA256SUMS" "$a/SHA256SUMS" "$h/SHA256SUMS" "$here/start"*.jq "$here/summarize-replay-wick.sh" > "$out/inputs.sha256"
(cd "$out" && sha256sum *.json inputs.sha256 > SHA256SUMS)
