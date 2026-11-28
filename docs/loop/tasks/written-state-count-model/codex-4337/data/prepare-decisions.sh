#!/usr/bin/env bash
# Rebuild the grader's expected decisions from sealed cold GPU-parity evidence.
set -euo pipefail
umask 077

if [[ $# != 6 ]]; then
    echo 'usage: prepare-decisions.sh COLD GPU PREPARATION GATE_PANEL HELD_PANEL FRESH_OUTPUT' >&2
    exit 2
fi
cold=$(realpath "$1")
gpu=$(realpath "$2")
prep=$(realpath "$3")
gate=$(realpath "$4")
held=$(realpath "$5")
out=$(realpath -m "$6")

bind() {
    local directory=$1 manifest=$2 digest=$3
    printf '%s  %s\n' "$digest" "$directory/$manifest" | sha256sum --strict --quiet -c -
    (cd "$directory" && sha256sum --strict --quiet -c "$manifest")
}
hash() { sha256sum "$1" | cut -d ' ' -f1; }

bind "$cold" OUTPUTS.sha256 3d7aa04a3ac1221badbeceb6b355e86595ca9836d1968f63dec9ef5a9b1c6926
bind "$gpu" OUTPUTS.sha256 0075b9935d630ba9867d97b51064279ffaaa0751c51868f5543eab3398385953
bind "$prep" OUTPUTS.sha256 65f6ffe5a2c281d30ca063ce016d95928379bc33431ac736b3fb2809cdd10b7e
bind "$gate" SEALED.sha256 da8667c7f7659ac0553bf0ff84ab9d2f836f1e22a103a19a399993b847e839b7
bind "$held" SEALED.sha256 39a13bb1af9e681b2c1d281165cd49a0944a7fde135ba4292f810b65da07cdf9

mkdir "$out"
cd "$out"
for panel in gate protected held; do
    case "$panel" in
        gate) prefix="$gate/train" ;;
        protected) prefix="$gate/protected" ;;
        held) prefix="$held/held" ;;
    esac
    jq -nce --arg panel "$panel" --arg gold_sha "$(hash "$prefix-gold.jsonl")" \
        --arg truth_sha "$(hash "$prefix-truth.jsonl")" --arg blind_sha "$(hash "$prefix-input.jsonl")" \
        --slurpfile gold "$prefix-gold.jsonl" --slurpfile truth "$prefix-truth.jsonl" \
        --slurpfile blind "$prefix-input.jsonl" '
        ($blind|map(.case_id)) as $order |
        if ($order|length)==0 or ($order|length)!=($order|unique|length) or
           $order!=($gold|map(.case_id)) or $order!=($truth|map(.case_id))
        then error("sealed panel identities differ: "+$panel)
        else {key:$panel,value:{gold_sha256:$gold_sha,truth_sha256:$truth_sha,
                                blind_sha256:$blind_sha,roster:$order}} end' >> panels.jsonl
done

for arm in geometry count; do
    for control in actual permuted zero; do
        model="$arm.$control"
        pack="$cold/$model.pack.json"
        words="$gpu/$model.words"
        if [[ $control == zero ]]; then words="$prep/device/zero.words"; fi
        for panel in held protected; do
            capture="$cold/$model.$panel.cold.jsonl"
            parity="$cold/$model.$panel.parity.json"
            jq -cse --arg arm "$arm" --arg model "$model" --arg panel "$panel" \
                --arg state_sha "$(hash "$words")" --slurpfile panels panels.jsonl \
                --slurpfile pack "$pack" --slurpfile parity "$parity" '
                def sha: type=="string" and test("^[a-f0-9]{64}$");
                def signed_word_sum: type=="number" and .==floor and .>=-524288 and .<=524272;
                def authority($row):
                    $row as $r |
                    ($r.scores.candidates | max) as $best |
                    $r.scores.selection as $read |
                    if ($r.candidates|length)!=($r.scores.candidates|length) or
                       any($r.scores.candidates[]; signed_word_sum|not) or
                       ($read.no_read_score|signed_word_sum|not) or
                       $read.best_candidate_score!=$best or
                       $read.span!=(if $best!=null and $best>$read.no_read_score
                           then ($r.candidates[($r.scores.candidates|index($best))]|{start,end})
                           else null end)
                    then error("inconsistent frozen candidate scores for "+$r.case_id)
                    else {case_id:$r.case_id,user:$r.user,selection:$read} end;
                ($panels|from_entries|.[$panel].roster) as $roster |
                (map(select(.arm==$arm))) as $rows |
                if ($pack|length)!=1 or ($parity|length)!=1 or
                   $pack[0].schema!="uor-r4.lut-discrete-native-interval/1" or
                   ($pack[0].artifact_sha256|sha|not) or $pack[0].interval.arm!=$arm or
                   $pack[0].interval.state_sha256!=$state_sha or $pack[0].interval.state_bytes!=8192 or
                   ($parity[0].payload_bytes!=8192 or $parity[0].banks!=16) or
                   ($rows|map(.case_id))!=$roster or
                   any(.[]; (.arm|IN("geometry","count")|not) or ((.scores!=null)!=(.arm==$arm))) or
                   (length!=2*($roster|length)) or
                   $parity[0].cases!=($roster|length) or
                   $parity[0].exact_scores!=($rows|map((.candidates|length)+1)|add)
                then error("invalid reference evidence for "+$model+" "+$panel)
                else {model:$model,panel:$panel,artifact_sha256:$pack[0].artifact_sha256,
                      identity:$pack[0].interval,decisions:($rows|map(authority(.)))} end' \
                "$capture" >> models.jsonl
        done
    done
done
jq -S -nce --slurpfile panels panels.jsonl --slurpfile models models.jsonl '
    {schema:"uor-r4.ordinary-interval-decision-reference/1",
     balanced:{artifact_sha256:"5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa",
               model_schema:"uor-r4.lut-discrete-native-source-choice/1"},
     panels:($panels|from_entries),
     models:($models|group_by(.model)|map({key:.[0].model,
       value:{artifact_sha256:.[0].artifact_sha256,identity:.[0].identity,
              panels:(map({key:.panel,value:.decisions})|from_entries)}})|from_entries)}
' > decision-reference.json
sha256sum decision-reference.json panels.jsonl models.jsonl > OUTPUTS.sha256
