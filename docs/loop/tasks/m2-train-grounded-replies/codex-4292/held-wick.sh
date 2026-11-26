#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
mode=${1:?run or inner}; mv=${2:?TRAIN version}; tv=${3:?fresh version}
[[ $mv =~ ^v[1-9][0-9]*$ && $tv =~ ^v[1-9][0-9]*$ ]]
m="$s/measurement-$mv"; transfer="$s/transfer-$tv"
permit=$(realpath "${4:?replayed TRAIN authorization}"); digest=${5:?authorized SHA}
[[ $permit == "$s"/train-verified-v*/inputs.sha256 ]]; pin "$digest" "$permit"; sha256sum --strict --quiet -c "$permit"
for file in "$m/"{SHA256SUMS,selection.sha256,selection.json} "$here/"{SHA256SUMS,authorize-held-wick.sh} "$(dirname "$permit")/verified.json"; do
 grep -Fxq "$(sha256sum "$file")" "$permit"
done
jq -e '.pass and .compared_unique_batches>0 and .compared_unique_batches==(.batches|length)' "$(dirname "$permit")/verified.json" >/dev/null
verify "$m"; sha256sum --quiet --strict -c "$m/selection.sha256"; jq -e '.pass' "$m/premise.json" >/dev/null
manifest=$(cut -d' ' -f3- "$m/inputs.sha256"); check_release "$manifest" "$(cut -d' ' -f1 "$m/inputs.sha256")"
panel=$(cat "$(dirname "$manifest")/panel-path.txt"); p="$panel/held"
if [[ $mode == run ]]; then exec bash "$lock" "$transfer-lease" bash "$0" inner "$mv" "$tv" "$permit" "$digest"; fi
[[ $mode == inner && -e /proc/$$/fd/9 ]]
verify "$panel"; mkdir "$transfer" "$transfer/held" "$transfer/controls"
d="$transfer/held"
read_arm(){
 local arm=$1 kind=$2; shift 2
 "$reader" "$kind" "$banks" "$p/blind/text.bin" "$@" "$d/$arm.raw.jsonl"
 case $kind in correct|exchanged) kind="${kind}_link";; esac
 rows held "$p" "$kind" "$d/$arm.raw.jsonl" > "$d/$arm.rows.jsonl"
 jq -es -f "$prev/grade/pairs.jq" "$d/$arm.rows.jsonl" > "$d/$arm.pairs.json"
}
read_arm native native; read_arm unfitted unfitted; read_arm gap governor_gap "$parent"
for arm in baseline free protected far_only; do
 case $arm in baseline) state="$parent";; far_only) state="$matched";; *) state="$m/states/$arm.state";; esac
 read_arm "$arm" learned "$state"
 "$joint" "$p/blind/text.bin" "$banks" "$state" "$d/$arm.raw.jsonl" "$d/gap.raw.jsonl" > "$d/$arm.replay.json"
 jq -e '.pass and .learned.rows==256 and .learned.disagreements==0 and .governor_gap.disagreements==0' "$d/$arm.replay.json" >/dev/null
done
read_arm correct_link correct "$p/aid.bin"; read_arm exchanged_link exchanged "$p/aid.bin"
struct="$p/structural-links"; cd "$transfer/controls"
for arm in native governor_only nouns_only correct_link exchanged_link; do
 "$legacy/governor-noun-links-sm89" "$banks" "$struct/geometry.bin" "$struct/$arm.bin" "$PWD/$arm.rank"
done
"$legacy/target/release/governor-noun-links" grade "$struct" "$model" "$banks" "$PWD"
for arm in native correct_link exchanged_link; do
 jq -ne --arg a "$arm" --slurpfile rows "$d/$arm.rows.jsonl" --slurpfile other endpoints.jsonl '
 def sig:[.selection,.tokens,.bank_cost,.ties,.eligible]; ($other|INDEX(.case_id)) as $idx |
 if ($rows|length)==256 and ($other|length)==256 and all($rows[];. as $r|$idx[.case_id] as $o|
   $o.prompt_sha256==$r.prompt_sha256 and ($r.arm|sig)==($o.arms[$a]|sig)) then {decisions:256,disagreements:0}
 else error("independent source-address control failed") end' > "$arm.checks.json"
done
sha256sum "$permit" "$m/SHA256SUMS" "$here/SHA256SUMS" "$panel/SHA256SUMS" > "$transfer/inputs.sha256"
seal "$transfer"
