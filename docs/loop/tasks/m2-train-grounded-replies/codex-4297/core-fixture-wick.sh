#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
v=${1:?new fixture version}; version "$v"; out="$s/finite-core-$v"; mkdir "$out"
rustc --edition=2021 -C opt-level=2 "$here/bundle.rs" -o "$out/bundle"
for geom in endpoint ordered; do
    "$out/bundle" pack "$geom" "$state" "$out/$geom.bridge"
    test "$(stat -c %s "$out/$geom.bridge")" = 13352
    test "$("$out/bundle" unpack "$out/$geom.bridge" "$out/$geom.state")" = "$geom"
    cmp "$state" "$out/$geom.state"
done
cmp -l "$out/endpoint.bridge" "$out/ordered.bridge" > "$out/tag-differences.txt" || test "$?" = 1
test "$(wc -l < "$out/tag-differences.txt")" -eq 1
for invalid in tag reserved truncate; do
    cp "$out/ordered.bridge" "$out/$invalid.bridge"
    case $invalid in
        tag) printf '\177' | dd of="$out/$invalid.bridge" bs=1 seek=8 conv=notrunc status=none ;;
        reserved) printf '\001' | dd of="$out/$invalid.bridge" bs=1 seek=10 conv=notrunc status=none ;;
        truncate) truncate -s 13351 "$out/$invalid.bridge" ;;
    esac
    if "$out/bundle" unpack "$out/$invalid.bridge" "$out/bad.state" 2>"$out/$invalid.error"; then exit 1; fi
    test ! -e "$out/bad.state"
done
jq -nc 'range(68) as $id|{id:$id,group:(($id/4)|floor),step:([1,2,4,8][$id%4])}' > "$out/menu.jsonl"
jq -nc '{id:null,changed:0,near_full:["a","b"],far_full:["f"],lost_protected:[],strongest_loss:12,all_loss:60} as $b|
    $b, (range(68) as $id|$b+{id:$id,changed:1}+
        (if $id==0 then {strongest_loss:1,all_loss:2,near_full:["a","c"],lost_protected:["b"]}
         elif $id==1 then {strongest_loss:4,all_loss:41}
         elif $id==2 then {strongest_loss:6,all_loss:20,far_full:["f","g"]} else {} end))' > "$out/choices.jsonl"
for arm in strongest all ordered-all; do
    jq -es --arg arm "$arm" --slurpfile menu "$out/menu.jsonl" -f "$here/grade/elect.jq" "$out/choices.jsonl" > "$out/$arm.json"
done
jq -e '.qualified and .chosen.id==1' "$out/strongest.json" >/dev/null
for arm in all ordered-all; do jq -e '.qualified and .chosen.id==2' "$out/$arm.json" >/dev/null; done
jq -c '.lost_protected=["b"]' "$out/choices.jsonl" | jq -es --arg arm all --slurpfile menu "$out/menu.jsonl" \
    -f "$here/grade/elect.jq" | jq -e '(.qualified|not) and .chosen.id==null' >/dev/null
jq -n '{qualified:true,objective_before:10,objective_after:5,final:{lost_protected:[],far_gained:["gain"]}}' > "$out/training.json"
jq -n 'def win($v):{exact:$v,full:$v,earlier:$v,prompts:(($v|length)*2)};
    [range(9)] as $n|[range(100;110)] as $f|[range(128)] as $ids|
    {native:win([]),unfitted:win([]),baseline:win([0]),all:win($n),far_only:win([0]),
     correct_link:win($ids),exchanged_link:win([])} as $near|
    ($near+{baseline:win([]),all:win($f),far_only:win($f[:2])}) as $far|
    {distance:{near:$near,far:$far},family:{x:$near,y:$far},all:($near+{all:win($n+$f)})}' > "$out/held.json"
gate(){ jq -e --arg arm all --arg ref_sha fixture --arg ep_sha fixture --slurpfile training "$out/training.json" -f "$here/grade/gate.jq" "$1"; }
gate "$out/held.json" | jq -e .pass >/dev/null
jq '.distance.near.all.exact[0]=99' "$out/held.json" > "$out/near-swap.json"
gate "$out/near-swap.json" | jq -e '(.pass|not) and (.gate.retains_held_near|not)' >/dev/null
jq '.all.exchanged_link.prompts=1' "$out/held.json" > "$out/control-leak.json"
gate "$out/control-leak.json" | jq -e '(.pass|not) and (.gate.supplied_controls|not)' >/dev/null
join_rows held "$old/panel/panel-v1/held" learned "$old/transfer-v1/held/protected.raw.jsonl" > "$out/historical.rows.jsonl"
cmp "$out/historical.rows.jsonl" "$old/transfer-v1/held/protected.rows.jsonl"
pair_rows held < "$out/historical.rows.jsonl" > "$out/historical.pairs.json"
cmp "$out/historical.pairs.json" "$old/transfer-v1/held/protected.pairs.json"
jq -nc 'range(64) as $i | (if $i%2==0 then 1 else 3 end) as $at |
    ([{start:$at,end:($at+1)},{start:7,end:8},{start:9,end:10}]) as $sites |
    {pair_id:(($i/2|floor)|tostring),family:"x",distance:"far",block:"evaluation",admission:"take",
     case_id:($i|tostring),prompt_sha256:($i|tostring),selector:(if $i%2==0 then "first" else "later" end),
     gold:$sites,gold_tokens:$sites,arm:{full:true,exact:true,earlier:true,both_governors:true,
     neutral:false,selection:$sites,tokens:$sites}}' > "$out/contrast.rows.jsonl"
pair_rows contrast < "$out/contrast.rows.jsonl" | jq -e 'length==32 and all(.[];.full)' >/dev/null
jq -es -f "$here/grade/contrast.jq" "$out/contrast.rows.jsonl" | jq -e '.correct_source_flips==32 and .full==32' >/dev/null
jq -c '.arm.selection[0]={start:1,end:2}|.arm.tokens[0]={start:1,end:2}' "$out/contrast.rows.jsonl" \
    | jq -es -f "$here/grade/contrast.jq" | jq -e '.correct_source_flips==0 and .distinct_reads==0' >/dev/null
sha256sum "$here/SHA256SUMS" "$state" > "$out/inputs.sha256"
jq -n '{pass:true,header_bytes:16,state_bytes:13336,recovered_payload_bytes:13336,
 strongest_election:1,all_rivals_election:2,near_identity_substitution_rejected:true,
 no_feasible_election_rejected:true,positive_writer_gate:true,exchanged_control_leak_rejected:true,
 historical_primary_rows_rejoined:256,matched_control_flips:32,fixed_address_control_flips:0}' > "$out/checks.json"
seal "$out"
