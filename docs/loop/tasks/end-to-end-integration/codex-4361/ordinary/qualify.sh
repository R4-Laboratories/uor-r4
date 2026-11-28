#!/usr/bin/env bash
set -euo pipefail
umask 077
root=$(cd "${1:?expected Wick run root}" && pwd)
cd "$root"
(cd received/attempt02 && sha256sum --strict --status -c members.sha256)
(cd received/attempt03 && sha256sum --strict --status -c members.sha256)
checkpoint_receipt=received/attempt02/checkpoints/COMPLETE.json
jq -e '.verdict=="COMPLETE" and .updates==64 and .total_targets_per_arm==2151 and
  .initial_checkpoint_sha256=="20262de550c2a96a9cc87a7fecfd178567a11cf58dbd6ad2cb1ffc3e5bddb589" and
  .full_sha256!=.reset_sha256' "$checkpoint_receipt" >/dev/null
sha() { sha256sum "$1" | cut -d' ' -f1; }
for arm in frozen reset full; do
    model=inputs/base.uorbfy03
    manifest=ordinary/baseline.manifest.json
    expected=$(jq -er .initial_checkpoint_sha256 "$checkpoint_receipt")
    if [[ $arm != frozen ]]; then
        model=received/attempt02/checkpoints/$arm.uorbfy03
        expected=$(jq -er --arg key "${arm}_sha256" '.[$key]' "$checkpoint_receipt")
        manifest=ordinary/arm-$arm.manifest.json
        (set -C; jq --arg sha "$expected" '.checkpoint_sha256=$sha' ordinary/baseline.manifest.json >"$manifest")
    fi
    test "$(sha "$model")" = "$expected"
    rows=received/attempt03/$arm.rows.jsonl
    jq -e --arg expected "$expected" '.model==$expected and .rows==3 and .parity_articles==3 and
        .parity_transitions>0 and .tape=="cc747479b693456e2d4e6b200fd0ac6b79dce16218050ee70165760c9a2fdb12"' \
        "received/attempt03/$arm.receipt.json" >/dev/null
    out=ordinary/result-$arm
    ordinary/replay-held.sh "$manifest" "$(sha "$manifest")" "$model" "$rows" "$(sha "$rows")" 32 "$out"
    bin/check_client_bytes inputs/tokenizer.json "$out"
    jq -s -e '([.[].id] == [1,2,3]) and all(.[]; .prefix_equal and .prefix_hash_equal and .emitted_equal and .stop_equal)' \
        "$out/compare.jsonl" >/dev/null
    jq -sn --arg role "$arm" --arg sha "$expected" --argjson checks "$(jq -s . "$out/compare.jsonl")" \
        --arg manifest "$(sha "$manifest")" --arg raw "$(sha "$rows")" \
        '{role:$role,checkpoint_sha256:$sha,manifest_sha256:$manifest,capture_sha256:$raw,checks:$checks,stdout_verified:3}' \
        >"$out/qualified.json"
done
(set -C; jq -s '{schema:"grounded-ordinary/1",articles:[1,2,3],verified_cold_responses:9,arms:.}' \
    ordinary/result-{frozen,reset,full}/qualified.json >ordinary-gate.json)
