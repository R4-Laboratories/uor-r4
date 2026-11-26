#!/usr/bin/env bash
set -euo pipefail
umask 077
s=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-protected-transport
methods=$(realpath "${1:?methods}")
version=${2:?receipt version}
[[ $(hostname -s) == wick && $version =~ ^v[1-9][0-9]*$ ]]
out="$s/fitness/controls-$version"
mkdir "$out"
cd "$out"
jq -n '
  [range(128) as $i |
    {pair:("p"+($i|tostring)), ids:[["take", $i],["defer", $i]],
      distance:(if $i<64 then "far" else "near" end),
      block:(if $i<64 then "common" else "replacement" end),
      full:($i<40 or ($i>=64 and $i<88))}]
' > parent.json
jq -n '{id:0,hinge:19,strict_wins:39,strict_losses:1,eligible:256,changed:1}' > scores.json
jq '[.[] | if .pair=="p64" then .full=false elif .pair=="p88" then .full=true else . end]'     parent.json > count-substitution.json
jq -e --slurpfile score scores.json --slurpfile initial parent.json     -f "$methods/grade/candidate.jq" count-substitution.json > substituted.json
jq -e '(.near_full|length)==24 and .lost_protected==["p64"]' substituted.json > checked-substitution.json

# Four metrics and the final ID each have an eligible loser that would win if skipped.
jq -n --slurpfile substitution substituted.json '
  $substitution[0] as $bad |
  ($bad + {id:null,hinge:50,strict_wins:4,changed:0,lost_protected:[]}) as $base |
  ($base + {hinge:20,strict_wins:6,changed:2,far_full:["a","b"],far_gained:["a"]}) as $ok |
  [$base, $bad,
   ($ok + {id:1,hinge:21,far_full:["a","b","c"],strict_wins:20,changed:1}),
   ($ok + {id:2,far_full:["a"],strict_wins:20,changed:1}),
   ($ok + {id:3,strict_wins:5,changed:1}), ($ok + {id:4,changed:3}),
   ($ok + {id:5}), ($ok + {id:6}),
   ($ok + {id:7,hinge:50})]
' > candidates.json
jq -c '.[]|select(.id!=null)|{id,kind:"coordinate"}' candidates.json > menu.jsonl
jq -e --slurpfile menu menu.jsonl -f "$methods/grade/elect.jq" candidates.json > election.json
jq -e '.protected.id==5 and .free.id==0 and .first_near_rejection.id==0 and
    .first_far_improving_near_rejection.id==0 and .counts.proposals==8' election.json >/dev/null
jq '.[0].hinge as $h | [.[0], (.[1:][] | .hinge=$h)]' candidates.json > no-move.json
jq -e --slurpfile menu menu.jsonl -f "$methods/grade/elect.jq" no-move.json > no-move-election.json
jq -e '.base as $base | all(.protected,.free;
    .gained_from_round_base==[] and .lost_from_round_base==[] and
    del(.gained_from_round_base,.lost_from_round_base)==$base) and
    .counts.far_improving==0' no-move-election.json >/dev/null
jq 'del(.[2])' candidates.json > incomplete.json
if jq -e --slurpfile menu menu.jsonl -f "$methods/grade/elect.jq" incomplete.json > incomplete-output.txt 2> incomplete-error.txt; then
    exit 1
fi
grep -Fq 'candidate budget, base or order differs' incomplete-error.txt

jq -n '
   def all_ids: [range(128)|tostring];
   def arms($b;$p;$f):
     {baseline:{exact:$b},protected:{exact:$p,full:["link"]},far_only:{exact:$f},
      native:{exact:[]},unfitted:{exact:[]},
      correct_link:{exact:all_ids,earlier:all_ids},exchanged_link:{exact:[],prompts:0}};
   {distance:{near:arms(["n"];["n"];[]),
      far:arms(["p"];["p","f","x","y","z","u","v","w","q"];["f"])},
    all:arms(["n","p"];["n","p","f","x","y","z","u","v","w","q"];["f"]),
    family:{keeps:{protected:{exact:["f","n"]},native:{exact:[]}}}}
' > held.json
jq -n '{parent:{near_full:[range(24)],hinge:100},
    protected:{lost_protected:[],far_gained:["one"],hinge:90}}' > train.json
check_gate() {
    jq -e --arg ref_sha fixture --arg endpoint_sha fixture --slurpfile train train.json         -f "$methods/grade/gate.jq" "$1" > "$2"
}
check_gate held.json accepted.json
jq -e '.pass and .matched_far_gain_informative' accepted.json >/dev/null
jq '.distance.near.protected.exact=["different-near-id"]' held.json > held-near-swap.json
check_gate held-near-swap.json rejected-near.json
jq -e '.pass==false and .gate.retains_held_near==false' rejected-near.json >/dev/null
jq '.distance.far.protected.exact[1]="different-far-id"' held.json > held-far-swap.json
check_gate held-far-swap.json rejected-far.json
jq -e '.pass==false and .gate.retains_matched_far_gains==false' rejected-far.json >/dev/null
jq '.all.exchanged_link.prompts=1' held.json > held-exchanged.json
check_gate held-exchanged.json rejected-exchanged.json
jq -e '.pass==false and .gate.supplied_controls==false' rejected-exchanged.json >/dev/null
jq '.distance.far.far_only.exact=.distance.far.baseline.exact' held.json > held-empty-increment.json
check_gate held-empty-increment.json admitted-empty-increment.json
jq -e '.pass and .matched_far_gain_informative==false' admitted-empty-increment.json >/dev/null

sha256sum "$methods/grade/"{candidate,elect,gate}.jq "$0" > inputs.sha256
jq -n '{pass:true,protected_identity:true,four_tiebreaks_and_id:true,positive_move:true,
    strict_improvement:true,bad_budget_rejected:true,writer_controls:true,empty_gain_denominator:true}' > checks.json
find . -type f ! -name .pending -print0 | sort -z | xargs -0 sha256sum > .pending
mv .pending SHA256SUMS
