#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); out=${1:?unused fixture path}
[[ $(hostname -s) == wick && $out =~ /20261126-codex-protected-transport/grade-fixture-v[1-9][0-9]*$ ]]
mkdir "$out"
jq -n '[{id:0,kind:"center"},{id:1,kind:"coordinate"},{id:2,kind:"coordinate"},{id:3,kind:"center"}][]' > "$out/menu.jsonl"
jq -nc '
 {id:null,hinge:10,strict_wins:100,changed:0,near_full:["n0","n1"],far_full:["f0"],lost_protected:[],far_gained:[],far_lost:[]} as $b |
 $b,( $b + {id:0,hinge:2,strict_wins:110,changed:16,near_full:["n1","n2"],far_full:["f0","f1"],lost_protected:["n0"],far_gained:["f1"]} ),
 ( $b + {id:1,hinge:5,changed:1} ),( $b + {id:2,hinge:5,changed:1,far_full:["f0","f1"],far_gained:["f1"]} ),
 ( $b + {id:3,hinge:5,changed:4,far_full:["f0","f1"],far_gained:["f1"]} )' > "$out/candidates.jsonl"
elect(){ jq -es --slurpfile menu "$out/menu.jsonl" -f "$here/elect.jq" "$1"; }
elect "$out/candidates.jsonl" > "$out/elected.json"
jq -e '.free.id==0 and .protected.id==2 and .first_far_improving_near_rejection.id==0 and .counts.protected_and_far_improving==3' "$out/elected.json" >/dev/null
jq -c 'if .id!=null then .hinge=10 else . end' "$out/candidates.jsonl" > "$out/no-gain.jsonl"
elect "$out/no-gain.jsonl" | jq -e '.free.id==null and .protected.id==null' >/dev/null
tail -n +2 "$out/candidates.jsonl" > "$out/missing-base.jsonl"
if elect "$out/missing-base.jsonl" >/dev/null 2>"$out/missing-base.stderr"; then exit 1; fi
jq -n '{parent:{near_full:[range(24)],hinge:10},protected:{lost_protected:[],far_gained:["f1"],hinge:5}}' > "$out/training.json"
jq -n '
 def arm($wins):{exact:$wins,full:$wins,earlier:$wins,prompts:(($wins|length)*2)};
 [range(9)] as $n | [range(100;110)] as $f | [range(128)] as $all |
 {native:arm([]),unfitted:arm([]),baseline:arm([0]),protected:arm($n),far_only:arm([0]),correct_link:arm($all),exchanged_link:arm([])} as $near |
 ($near + {baseline:arm([]),protected:arm($f),far_only:arm($f[:2])}) as $far |
 {distance:{near:$near,far:$far},family:{a:$near,b:$far},all:($near+{protected:arm($n+$f)})}' > "$out/held.json"
gate(){ jq -e --slurpfile train "$out/training.json" --arg ref_sha fixture --arg endpoint_sha fixture -f "$here/gate.jq" "$1"; }
gate "$out/held.json" | jq -e .pass >/dev/null
jq '.distance.near.protected.exact[0]=99' "$out/held.json" > "$out/near-swap.json"
gate "$out/near-swap.json" | jq -e '(.pass|not) and (.gate.retains_held_near|not)' >/dev/null
jq '.distance.far.protected.exact[0]=99' "$out/held.json" > "$out/far-swap.json"
gate "$out/far-swap.json" | jq -e '(.pass|not) and (.gate.retains_matched_far_gains|not)' >/dev/null
jq -n '{pass:true,free_winner:0,protected_winner:2,counts_preserving_near_swap_rejected:true,
 matched_far_swap_rejected:true,unchanged_without_gain:true,positive_gate:true,missing_base_rejected:true}' > "$out/checks.json"
(cd "$out"; find . -type f ! -name .pending -print0|LC_ALL=C sort -z|xargs -0 sha256sum >.pending; mv .pending SHA256SUMS)
