#!/usr/bin/env bash
# Preserve the original partial attempt and run the integer-only count control once.
set -euo pipefail
umask 077
cd /root/c4361
out=/workspace/attempt05
prior=/workspace/attempt04
mkdir "$out"
exec 2>"$out/errors.log"
sha256sum --strict -c correction.sha256 >"$out/intake.txt"
test ! -e "$prior/COMPLETE"
test -e "$prior/counts.receipt.json" && test ! -s "$prior/counts.receipt.json"
test ! -e "$prior/counts.rows.jsonl"
test ! -e bin/counts.sm89.cubin
jq -e '.verified_cold_responses==9' ordinary-gate.json > /dev/null
train_sha=f96cc65ed0fd76c6cb9f75b5434ef6a0d7edc65d42ec01a4145781e2d2a94f4a
held_sha=6ffd2228957ca42705c17fa995071a8360c5f265b719674f8d0c1990e984e4ee
kernel_sha=1c89893d84fc4af303c254146c2cb72498ff810e0406b194bdc3d561c5225368
for role in frozen reset full; do
    model=$(jq -er --arg role "$role" '.arms[]|select(.role==$role)|.checkpoint_sha256' ordinary-gate.json)
    jq -e --arg sha "$model" --arg tape "$held_sha" \
      '.model==$sha and .tape==$tape and .rows==222 and .parity_articles==3 and .matched_pairs==0' \
      "$prior/$role.receipt.json" >/dev/null
    jq -es --arg sha "$model" --arg tape "$held_sha" \
      'length==222 and ([.[].id]==[range(1;223)]) and all(.[];.model_sha256==$sha and .tape_sha256==$tape)' \
      "$prior/$role.rows.jsonl" >/dev/null
done
(cd "$prior"; sha256sum *.json *.jsonl *.txt errors.log > "$out/attempt04-members.sha256")
export CUDA_DISABLE_PTX_JIT=1
count-correction/evaluate-count counts count-correction/counts-int.sm89.cubin "$kernel_sha" \
  inputs/train.jsonl "$train_sha" inputs/held.jsonl "$held_sha" "$out/counts.rows.jsonl" >"$out/counts.receipt.json"
jq -e '.training_targets==2151 and .rows==222 and .histogram_words_checked==(4097*4096) and .elected_rows_checked==4097' \
    "$out/counts.receipt.json" > /dev/null
(cd "$out" && sha256sum *.json *.jsonl *.txt errors.log attempt04-members.sha256 >members.sha256)
touch "$out/COMPLETE"
