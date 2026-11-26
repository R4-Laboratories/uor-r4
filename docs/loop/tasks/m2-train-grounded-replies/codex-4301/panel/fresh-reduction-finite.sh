#!/usr/bin/env bash
set -euo pipefail
umask 077
here=$(cd "$(dirname "$0")" && pwd -P)
out=${1:?unused finite output directory}
mkdir "$out"

jq -n '
  def range_at($p): {start:$p,end:($p+1)};
  def a($p): {bytes:range_at($p),tokens:range_at($p)};
  def audit($p): {bytes:[$p,$p+1],tokens:[$p,$p+1]};
  def native($i;$p;$t): {i:$i,bytes:range_at($p),bpe:range_at($p),text:$t};
  def third($cost;$words;$ties): {cost:$cost,eligible_sites:1,sites:[{site:native(2;6;"x"),
      words:$words,ties:$ties,first_tuple:2}]};
  def replay($cost;$words;$ties): {best:$cost,sites:[{site:2,address:audit(6),
      governors:$words,governor_addresses:[$words[]|audit(if .==0 then 0 else 6 end)],
      ties:$ties,tuple_ordinals:[2,3]}]};
  {physical:{row_index:0,pair_id:"test",case_id:"one",selector:"first",family:"fixture",text:"g t t x y h",
      coordinate_map:{words:[a(0),a(6)],sites:[a(2),a(4),a(6),a(8),a(10)],
      tuples:[[0,3,4],[1,3,4],[2,3,4],[2,0,4],[3,2,4]]}},
   grade:{row:0,pair_id:"test",family:"fixture",selector:"first",sign:1,order:"before",
      requested:native(0;2;"t"),rival:native(1;4;"t"),governor:native(0;0;"g"),
      local:{requested:[0,9],rival:[0,10],relation:"win"},
      joint:{requested:[0,14,4],rival:[0,16,4],relation:"win"},requested_free:[0,14,4],
      third_fixed:third([0,15,4];[0];2),third_free:third([0,12,4];[0,1];4),
      gold_vs_fixed_third:"win",gold_vs_third:"loss",
      source:false,full:false,native:{ties:1,choice:"third site"}},
   raw:{row:0,sign:1,word:0,requested:0,rival:1,sites:[{i:0,local:[0,9]},
      {i:1,local:[0,10]},{i:2,local:[0,8],fixed:{best:[0,15,4]}},
      {i:3,local:[0,1],fixed:{best:null}}]},
   observed:{rows:[{row:0,case_id:"one",pair_id:"test",sign:1,selector:"first",family:"fixture",
      follows:false,full:false,native_ties:1,requested_address:audit(2),rival_address:audit(4),
      fixed_third:replay([0,15,4];[0];2), free_third:replay([0,12,4];[0,1];4)}]}}
' > "$out/input.json"
for field in physical grade raw; do
    jq -ec --arg field "$field" '.[$field]' "$out/input.json" > "$out/$field.jsonl"
done
jq -ec '.observed' "$out/input.json" > "$out/observed.json"
jq -ecn --argjson expected 1 --slurpfile physical "$out/physical.jsonl" \
    --slurpfile grade "$out/grade.jsonl" --slurpfile raw "$out/raw.jsonl" \
    --slurpfile observed "$out/observed.json" -f "$here/fresh-state.jq" > "$out/witness.json"
jq -e '.case_id=="one" and .local_third_relation=="loss" and .local_third.eligible_sites==1 and
    .local_third.sites[0].text=="x" and .free_target_vs_third=="loss" and
    .deciding_keys.fixed_target_vs_free_third=={key:"transport",requested:14,rival:12,requested_minus_rival:2} and
    .third_free.all_minimizing_tuple_ordinals and (.third_free.sites[0]|.ties==4 and
        (.governors|map(.text))==["g","x"] and (.tuples|map(.ordinal))==[2,3] and
        .tuples[1].sites[1].text=="t")' "$out/witness.json" >/dev/null
jq -ecn --argjson expected 1 --slurpfile physical "$out/physical.jsonl" \
    --slurpfile grade "$out/grade.jsonl" --slurpfile raw "$out/raw.jsonl" \
    --argjson observed '[]' -f "$here/fresh-state.jq" | jq -ec '
        select((.third_free.all_minimizing_tuple_ordinals|not) and
          (.third_free.sites[0].tuples|map(.ordinal)) == [2])' > "$out/provisional.json"
jq -ec '.rows[0].free_third.sites[0].tuple_ordinals[1]=0' "$out/observed.json" > "$out/bad.json"
if jq -ecn --argjson expected 1 --slurpfile physical "$out/physical.jsonl" \
    --slurpfile grade "$out/grade.jsonl" --slurpfile raw "$out/raw.jsonl" \
    --slurpfile observed "$out/bad.json" -f "$here/fresh-state.jq" > /dev/null 2> "$out/rejection.txt"; then
    exit 1
fi

jq -cn --slurpfile w "$out/witness.json" '
    def pair($i;$source;$masked):
      $w[0] as $a | [($a + {row:($i*2),case_id:("a"+($i|tostring)),pair_id:("p"+($i|tostring)),
            family:("f"+($i|tostring)),sign:(if $i==0 then 1 else -1 end),
            source:$source,full:$source,gap:(if $i==0 then [25,18] else [15,27] end),
            native:{ties:1,choice:(if $masked then "third site" elif $source then "requested" else "same-spelling rival" end)},
            local:{relation:(if $source or $masked then "win" else "loss" end)},
            joint:{relation:(if $source or $masked then "win" else "loss" end)}}),
        null] | .[1]= (.[0] | .selector="later" | .row+=1 | .case_id=("b"+($i|tostring)) |
            .requested=$a.rival | .rival=$a.requested);
    [{id:"original",rows:(pair(0;true;false)+pair(1;false;false))},
     {id:"median",rows:(pair(0;false;true)+pair(1;false;false))},
     {id:"fit",rows:(pair(0;false;true)+pair(1;true;false))}]
' > "$out/lanes.json"
jq -ec --argjson election '[{"selected":"fit","selection":"admissible","best_initial":"original","best_learned":"fit"}]' \
    -f "$here/fresh-comparison.jq" "$out/lanes.json" > "$out/comparison.json"
jq -e '.selected=="fit" and (.comparisons[0] | .all.source.gain_ids==["p1"] and
    .all.source.loss_ids==["p0"] and .historic_gaps.source.loss==1 and
    .by_family[1].source.gain==1 and .by_sign[0].source.gain==1 and
    .historic_gap_examples.first_paired_source_loss.to.sides[1].requested.bpe.start==4) and
    .initial_vs_learned.all.source.loss==1' "$out/comparison.json" >/dev/null
