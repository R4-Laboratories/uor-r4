#!/usr/bin/env bash
# Inspect the persisted banks directly; changing only training nouns must leave lexical bytes fixed.
set -euo pipefail
umask 077
root=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-text-governor-links
test "$(hostname -s)" = wick
dir=$(realpath "${1:?measured train directory}")
out=${2:?new summary directory}
[[ $dir == "$root"/measurement-v*/train && $out == "$root"/state-audit-v* ]]
(cd "$(dirname "$dir")" && sha256sum --strict --quiet -c SHA256SUMS)
mkdir "$out"
for arm in learned corrupt_noun flipped_sign; do
  path="$dir/$arm.state"
  test "$(wc -c < "$path")" -eq 13336
  test "$(head -c8 "$path")" = UORSTT01
  for sign in 0 1; do
    for bank in lexical transport; do
      offset=$((8 + 6664 * sign))
      if [[ $bank == lexical ]]; then size=6148; else offset=$((offset + 6148)); size=516; fi
      count=$(od -An --endian=little -j "$offset" -N4 -tu4 "$path" | tr -d ' ')
      digest=$(dd if="$path" iflag=skip_bytes,count_bytes skip="$offset" count="$size" status=none | sha256sum | cut -d' ' -f1)
      jq -nc --arg arm "$arm" --argjson sign "$sign" --arg bank "$bank" --argjson count "$count" --arg sha "$digest" \
        '{arm:$arm,sign:$sign,bank:$bank,count:$count,sha256:$sha}' >> "$out/banks.jsonl"
    done
  done
done
jq -se '
  def bank($a;$s;$b): map(select(.arm==$a and .sign==$s and .bank==$b))[0];
  . as $banks | {bytes:13336,banks:$banks,
  corrupted_only_transport:all(range(2); . as $s | $banks |
    bank("learned";$s;"lexical").sha256==bank("corrupt_noun";$s;"lexical").sha256 and
    bank("learned";$s;"transport").sha256!=bank("corrupt_noun";$s;"transport").sha256),
  signed_banks_exchanged:all(range(2); . as $s | $banks |
    all(["lexical","transport"][]; . as $b | $banks |
      bank("learned";$s;$b).sha256==bank("flipped_sign";(1-$s);$b).sha256))} |
  if .corrupted_only_transport and .signed_banks_exchanged then . else error("training controls failed to isolate their registered banks") end' \
  "$out/banks.jsonl" > "$out/summary.json"
(cd "$out" && sha256sum banks.jsonl summary.json > SHA256SUMS)
