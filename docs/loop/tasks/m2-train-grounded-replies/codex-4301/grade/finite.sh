#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
out=${1:?unused output}
mkdir "$out"
jq -nc '
  def red($id;$num;$reg;$near;$train;$miss):
    {id:$id,kind:(if ($id|startswith("fit")) then "learned" else "initial" end),ordinal:$num,
     regression:{pairs:4,source:$reg,source_ids:[range($reg)]}, near:{pairs:24,correct:$near,wins:[range($near)]},
     train:{pairs:32,source:$train,local:2,missing_margins:$miss,worst_deficit:5,sum_deficit:6,full:1,strict_third:1}};
  [red("original";0;4;16;20;0),red("median";1;0;24;31;0),red("fit-partial";2;3;24;32;0),
   red("fit-missing";3;4;24;10;2),red("fit-observed";4;4;24;10;0)]
' > "$out/records.json"
jq -e -f "$here/elect.jq" "$out/records.json" > "$out/election.json"
jq -e '.selection=="admissible" and .selected=="fit-observed" and
  .admissible_ids==["fit-observed","fit-missing"] and .best_initial=="median"' "$out/election.json" > /dev/null
jq -c '.[0:3]' "$out/records.json" | jq -ec -f "$here/elect.jq" | jq -ec '
  select(.selection=="diagnostic" and .selected=="fit-partial" and (.admissible_ids|length)==0)
' > "$out/diagnostic.json"
jq -c '.[2].near.wins[1]=.[2].near.wins[0]' "$out/records.json" > "$out/broken.json"
if jq -e -f "$here/elect.jq" "$out/broken.json" > /dev/null 2> "$out/expected-error.txt"; then exit 1; fi
jq -nc '
  def row($action;$req;$rival;$win):
    {pair_id:"p",selector:$action,governor:{i:2},sign:1,requested:{i:$req},rival:{i:$rival},family:"f",
     source:$win,full:false,local:{relation:"win",margin:3},joint:{relation:"tie"},
     gold_vs_third:"unopposed",third_free:{cost:null},native:{choice:"requested"}};
  row("first";0;1;true),row("later";1;0;false)
' > "$out/sides.jsonl"
jq -ecs -f "$here/pairs.jq" "$out/sides.jsonl" |
 jq -ec 'select(.source==0 and .first_rows==1 and .later_rows==0 and .local==1 and .joint==0 and .strict_third==0 and .third_present==0)' > "$out/paired.json"
