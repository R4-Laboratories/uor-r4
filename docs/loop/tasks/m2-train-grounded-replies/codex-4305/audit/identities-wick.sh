#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
[[ $# == 4 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "${4%/*}") == zfs ]]
panel=$(realpath -e "$1")
physical=$(realpath -e "$2")
archived=$(realpath -e "$3")
out=$4
here=$(cd "$(dirname "$0")" && pwd -P)
for split in fresh near regression; do
    (cd "$panel/$split" && sha256sum --strict --quiet -c SHA256SUMS)
done
(cd "$physical" && sha256sum --strict --quiet -c raw.members)
(cd "$archived" && sha256sum --strict --quiet -c SHA256SUMS)
mkdir "$out"
for split in regression near fresh; do
    for arm in baseline G N GN wrong-G wrong-N GN-wrong-G GN-wrong-N GN-wrong-both; do
        jq -nc --arg arm "$arm" --arg split "$split" \
            --slurpfile requests "$panel/$split/requests.jsonl" \
            --slurpfile native "$physical/raw/$split/$arm.jsonl" \
            --slurpfile archived "$archived/near.json" -f "$here/identities.jq"
    done > "$out/$split.jsonl"
done
jq -es '.[0] | (.signed_pairs|length) == 4 and (.signed_pairs == .original_ids)' "$out/regression.jsonl" > /dev/null
jq -es '.[0] | (.full_pairs|length) == 24 and (.full_pairs == .original_ids)' "$out/near.jsonl" > /dev/null
jq -n --slurpfile fresh "$out/fresh.jsonl" --slurpfile reg "$out/regression.jsonl" --slurpfile near "$out/near.jsonl" '
   def change($base; $current): {gains:($current-$base),losses:($base-$current)};
   [$fresh[] as $r | $r.arm as $name |
      ($reg[] | select(.arm == $name)) as $s |
      ($near[] | select(.arm == $name)) as $t |
      {arm:$name, fresh_signed_pairs:($r.signed_pairs|length),
       fresh_full_pairs:($r.full_pairs|length),
       regression_signed_pairs:($s.signed_pairs|length),
       near_full_pairs:($t.full_pairs|length), selector_outcomes:$r.selector_outcomes,
       signed_rows:$r.signed_rows, full_rows:$r.full_rows, tied_rows:$r.tied_rows, absent_rows:$r.absent_rows,
       fresh_signed_change:change($fresh[0].signed_pairs;$r.signed_pairs),
       fresh_full_change:change($fresh[0].full_pairs;$r.full_pairs),
       regression_signed_change:change($reg[0].signed_pairs;$s.signed_pairs),
       near_full_change:change($near[0].full_pairs;$t.full_pairs)}]
' > "$out/summary.json"
sha256sum "$panel/PANELS.sha256" "$physical/raw.members" "$archived/SHA256SUMS" \
    "$here/identities.jq" "$(realpath -e "$0")" > "$out/input.sha256"
(cd "$out" && sha256sum *.json *.jsonl input.sha256 > SHA256SUMS)
