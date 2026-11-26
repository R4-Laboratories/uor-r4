#!/usr/bin/env bash
here=$(cd "$(dirname "$0")" && pwd)
source "$here/common-wick.sh"
mv=${1:?measurement}; gv=${2:?grade}; av=${3:?audit}; cv=${4:?cross-check}; v=${5:?unused report}
for ver in "$mv" "$gv" "$av" "$cv" "$v"; do version "$ver"; done
m="$study/measurement-$mv"; g="$study/grade-$gv"; a="$study/audit/captured-$av"; c="$study/audit/compare-$cv"; out="$study/report-$v"
for dir in "$m" "$m-lease" "$g" "$a" "$c"; do verify "$dir"; done
sha256sum --quiet --strict -c "$g/inputs.sha256" "$a/inputs.sha256"
grep -Fxq "$(sha256sum "$m/SHA256SUMS")" "$a/inputs.sha256"
jq -e '.pass and .comparisons==144 and .production_and_independent_rank_tie_address_disagreements==0' "$c/comparison.json" >/dev/null
mkdir "$out"
jq -r '
 def loc: if .==null then "none" else "\(.word) [\(.bpe[0]),\(.bpe[1]))" end;
 def joint($s): if .joint_selected==null then "unresolved" else (.joint_selected[$s]|loc) end;
 (["pair","instruction","failed_noun_role","original_governor","gold_noun","gold_cost",
   "native_choice","native_gold_rank","winning_occurrence_omitted","next_joint_choice","gold_rank_after_omission",
   "next_strongest_signed_rivals_and_costs","matched_nonwinner_omitted","control_joint_choice","control_gold_rank",
   "full_native","full_winner_omitted","full_control"]|@tsv),
 (.per_prompt[] | .failed_sign as $s | [.pair_id,.admission,["affirmed","excluded"][$s],(.governor|loc),(.gold|loc),
   .gold.cost,(.neutral|joint($s)),.neutral.gold_rank,(.old_winner|loc),(.winner|joint($s)),.winner.gold_rank,
   ([.winner.strongest_remaining_wrong[] | ((loc)+" cost="+(.cost|tostring))]|join(" / ")),
   (.matched_control|loc),(.control|joint($s)),.control.gold_rank,.neutral.full,.winner.full,.control.full]|@tsv)' \
   "$g/analysis.json" > "$out/prompt-traces.tsv"
jq '{pass,coverage,enrolled_rows,enrolled_pairs,unmasked_rows,controls_missing,historical_neutral_and_234_fields}' \
   "$a/readback.json" > "$out/verification.json"
jq -n --arg m "$(digest "$m/SHA256SUMS")" --arg grade "$(digest "$g/SHA256SUMS")" --arg audit "$(digest "$a/SHA256SUMS")" \
   --argjson bytes "$(du -sb "$m"|cut -f1)" --rawfile device "$m-lease/device.txt" \
   --rawfile start "$m-lease/started-at.txt" --rawfile end "$m-lease/finished-at.txt" --rawfile time "$m-lease/resources.txt" \
   '{measurement_sha256:$m,grade_sha256:$grade,audit_sha256:$audit,device:($device|rtrimstr("\n")),
     start_utc:($start|rtrimstr("\n")),end_utc:($end|rtrimstr("\n")),lease_resources:($time|rtrimstr("\n")),
     capture_tree_apparent_bytes:$bytes}' > "$out/measurement.json"
sha256sum "$g/SHA256SUMS" "$a/SHA256SUMS" "$c/SHA256SUMS" "$m/SHA256SUMS" > "$out/inputs.sha256"; seal "$out"
