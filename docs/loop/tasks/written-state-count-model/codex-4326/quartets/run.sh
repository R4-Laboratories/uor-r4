#!/usr/bin/env bash
set -euo pipefail
code=$(cd "$(dirname "$0")" && pwd)
measured=${1:?device output root}
panel=${2:?original joint panel root}
out=${3:?new ledger directory}
mkdir "$out"
for split in train held; do
 for component in zero period post-b full; do
   case "$component" in
     zero) trace=zero-period/first-period;;
     period) trace=zero-period/joint;;
     post-b) trace=post-full/first-period;;
     full) trace=post-full/joint;;
   esac
   jq -cn --slurpfile rows "$measured/raw/$split/$trace.jsonl" --slurpfile keys "$panel/$split/keys.jsonl" -f "$code/scores.jq" > "$out/$split-$component.json"
 done
 jq -s '
   to_entries | map(. as $entry | .value as $q |
      {component:(["zero","period","post-b","full"][$entry.key]), contexts:($q|length),
       strict_all:([$q[]|select(.weakest_signed_margin>0)]|length),
       interaction_zero:([$q[]|select(.copy_minus_stop==0)]|length),
       interaction_negative:([$q[]|select(.copy_minus_stop<0)]|length),
       interaction_positive:([$q[]|select(.copy_minus_stop>0)]|length),
       signed_interactions:([$q[].copy_minus_stop]|unique)})
 ' "$out/$split-zero.json" "$out/$split-period.json" "$out/$split-post-b.json" "$out/$split-full.json" > "$out/$split-summary.json"
done
