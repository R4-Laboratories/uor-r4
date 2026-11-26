#!/usr/bin/env bash
# Summarize sealed measurements without changing election or release criteria.
set -euo pipefail
umask 077
s=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-mixed-transport
m=$(realpath "${1:?measurement}"); g=$(realpath "${2:?grade}"); out=${3:?new receipt directory}
[[ $(hostname -s) == wick && $m == "$s"/measurement-v* && $g == "$s"/grade-v* && $out == "$s"/report-v* ]]
for d in "$m" "$g"; do (cd "$d" && sha256sum --strict --quiet -c SHA256SUMS); done
mkdir "$out"
for split in mixed far; do
 d="$m/train/$split"
 jq -ne --slurpfile e "$d/election.json" --slurpfile q "$d/first/quality.jsonl" --slurpfile c "$d/coordinate-checks.json" \
    --slurpfile v "$d/first/votes.jsonl" --slurpfile signed "$d/old.checks.json" '
  def direction: if .>0 then 1 elif .<0 then -1 else 0 end;
  ($v|map(select(.near*.far<0))) as $opposed | ($v|map([.sign,.slot,.axis])|unique|length) as $active |
  {elected:$e[0].selections,parent_near_full_pairs:($e[0].original_near|length),parent_common_far_full_pairs:($e[0].original_common_far|length),
   checkpoints:[$q[] as $x|$e[0].trajectory[$x.step] as $t| {step:$x.step,hinge:$x.hinge,strict_wins:$x.strict_wins,
      near:$x.near,far:$x.far,train_exact_pairs:$t.exact_pairs,train_full_pairs:$t.full_pairs,
      protected_near_lost:($t.lost_original_near|length),near_full:$t.near_full,common_far_full:$t.common_far_full,
      common_far_gained:($t.gained_common_far|length),common_far_lost:($t.lost_common_far|length),feasible:$t.feasible}],
   votes:{active_axes:$active,coordinate_steps:($v|length),opposed_coordinate_steps:($opposed|length),
      distinct_opposed_axes:($opposed|map([.sign,.slot,.axis])|unique|length),
      exact_cancellations:($opposed|map(select(.near+.far==0))|length),
      per_sweep:$c[0].vote_steps,
      examples:($opposed|sort_by([.step,.sign,.slot,.axis])|group_by(.step)|map(.[0]|.+{result_direction:((.near+.far)|direction)}))},
   audit:{checkpoints:$c[0].checkpoint_bytes_verified,individual_votes:$c[0].individual_coordinate_records_verified,
         differences:$c[0].disagreements,edge_replay:$signed[0]}}' > "$out/$split.json"
done
jq -n --slurpfile p "$m/premise.json" --slurpfile mixed "$out/mixed.json" --slurpfile far "$out/far.json" \
 --arg retained "$(sha256sum "$m/states/retained.state"|cut -d' ' -f1)" --arg free "$(sha256sum "$m/states/mixed_free.state"|cut -d' ' -f1)" --arg far_sha "$(sha256sum "$m/states/far_only.state"|cut -d' ' -f1)" \
 --rawfile wall "$m-lease/resources.txt" --arg measurement "$(sha256sum "$m/SHA256SUMS"|cut -d' ' -f1)" --arg grade "$(sha256sum "$g/SHA256SUMS"|cut -d' ' -f1)" \
 '{premise:($p[0]|.original_full_near|=length|.common_far_losing_pairs|=length),
  states:{retained:$retained,mixed_free:$free,far_only:$far_sha},mixed:$mixed[0],far_only:$far[0],resources:($wall|rtrimstr("\n")),measurement_sha256:$measurement,grade_sha256:$grade}' > "$out/training.json"
jq -ne --slurpfile h "$g/held.scores.json" --slurpfile b "$g/bridge.scores.json" --slurpfile br "$g/bridge-retention.json" \
 --slurpfile release "$g/summary.json" '
 def numbers: with_entries(.value|={exact,full,prompts,governor_prompts,neutral_prompts});
 def score: {all:(.all|numbers),distance:(.distance|map_values(numbers)),family_distance:(.family_distance|
  map_values(with_entries(select(.key|IN("baseline","far_only","mixed_free","retained","last_far","correct_link")))|map_values(.exact)))};
 {held:($h[0]|score),bridge:($b[0]|score),historical_retention:($br[0]|.arms|=map_values(del(.missing_far,.missing_near))),
  gate:($release[0]|{pass,gate,reference_sha256,endpoints_sha256,far_gain_retention_informative,
  original_near:(.original_near|length),lost_near:(.lost_near|length),far_only_gains:(.far_only_gains|length),far_only_gains_lost:(.far_only_gains_lost|length)})}' > "$out/transfer.json"
(cd "$out" && sha256sum {mixed,far,training,transfer}.json > SHA256SUMS)
