#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
mode=${1:?run or inner}; v=${2:?unused measurement version}; [[ $v =~ ^v[1-9][0-9]*$ ]]
manifest=$(realpath "${3:?release manifest}"); digest=${4:?approved SHA}
check_release "$manifest" "$digest"
release=$(dirname "$manifest"); gpu=$(cat "$release/gpu-path.txt"); panel=$(cat "$release/panel-path.txt")
[[ $gpu == "$s"/fit/build-*/transport-proposals-sm89 && $panel == "$s"/panel/panel-v1 ]]
for file in "$gpu" "$panel/SHA256SUMS" "$release/"{gpu-path.txt,panel-path.txt,FIT.md,DECISION.md} "$reader" "$joint" "$lock"; do
 grep -Fxq "$(sha256sum "$file")" "$manifest"
done
out="$s/measurement-$v"
if [[ $mode == run ]]; then exec bash "$lock" "$out-lease" bash "$0" inner "$v" "$manifest" "$digest"; fi
[[ $mode == inner && -e /proc/$$/fd/9 ]]
mkdir "$out" "$out/train" "$out/train/initial"
start="$out/train/initial"
old_fit="$r/20261126-codex-far-transport-margin/fit/build-v3/transport-margin-sm89"
"$old_fit" diagnose "$train_panel/blind/text.bin" "$train_panel/links.bin" "$parent" "$start/edges.jsonl"
"$reader" learned "$banks" "$train_panel/blind/text.bin" "$parent" "$start/reader.jsonl"
cmp "$start/reader.jsonl" "$original/step-0.raw.jsonl"
cmp "$start/edges.jsonl" "$original/initial.jsonl"
rows mixed "$train_panel" learned "$start/reader.jsonl" | jq -es -f "$prev/grade/pairs.jq" > "$start/pairs.json"
cmp "$start/pairs.json" "$original/step-0.pairs.json"
jq -ne --slurpfile p "$start/pairs.json" --slurpfile e "$start/edges.jsonl" '
 {protected:[$p[0][]|select(.distance=="near" and .full)|.pair],
  far:[$p[0][]|select(.block=="common" and .full)|.pair],
  eligible:([$e[]|select(.eligible_for_fit)]|length),
  far_eligible:([$e[]|select(.row<128 and .eligible_for_fit)]|length),
  far_hinge:([$e[]|select(.row<128)|.before.loss]|add),
  far_strict_wins:([$e[]|select(.row<128 and .before.margin>0)]|length),
  far_strict_losses:([$e[]|select(.row<128 and .before.margin<0)]|length)} |
 .pass=((.protected|length)==24 and (.far|length)==40 and .eligible==512 and .far_eligible==256 and .far_strict_losses==104)' > "$out/premise.json"
sha256sum "$manifest" > "$out/inputs.sha256"
if ! jq -e .pass "$out/premise.json" >/dev/null; then seal "$out"; exit 0; fi
batch(){
 local label=$1 state=$2 dest
 dest="$out/train/$label"; mkdir "$dest"
 "$gpu" "$banks" "$train_panel/blind/text.bin" "$train_panel/links.bin" "$parent" "$full_step" "$state" "$dest/device"
 cmp "$state" "$dest/device/base.state"
 if [[ $label == root ]]; then
   cmp "$dest/device/base.reader.jsonl" "$start/reader.jsonl"
 else
   cmp "$out/train/root/device/menu.jsonl" "$dest/device/menu.jsonl"
 fi
 seal "$dest/device"
 bash "$here/grade-batch-wick.sh" grade "$dest"
}
winner_state(){
 local group=$1 arm=$2 id
 id=$(jq -r --arg arm "$arm" '.[$arm].id' "$out/train/$group/grade/election.json")
 if [[ $id == null ]]; then echo "$out/train/$group/device/base.state"; else echo "$out/train/$group/device/$id.state"; fi
}
batch root "$parent"
master="$out/train/root/grade/election.json"
jq -ne --slurpfile p "$out/premise.json" --slurpfile e "$master" '
 $e[0].base as $b | $p[0] as $p | $b.near_full==$p.protected and $b.far_full==$p.far and
 $b.hinge==$p.far_hinge and $b.strict_wins==$p.far_strict_wins and $b.strict_losses==$p.far_strict_losses' >/dev/null
mkdir "$out/states"
for arm in protected free; do
 first=$(winner_state root "$arm"); cp "$first" "$out/states/$arm-first.state"
 if cmp -s "$first" "$parent"; then group=root
 elif [[ $arm == free ]] && cmp -s "$first" "$out/states/protected-first.state"; then group=$(cat "$out/train/protected.batch")
 else group="$arm"; batch "$group" "$first"; fi
 echo "$group" > "$out/train/$arm.batch"
 cp "$(winner_state "$group" "$arm")" "$out/states/$arm.state"
done
jq -ne --slurpfile root "$master" --slurpfile p "$out/train/$(cat "$out/train/protected.batch")/grade/election.json" \
 --slurpfile f "$out/train/$(cat "$out/train/free.batch")/grade/election.json" \
 --arg ps "$(sha "$out/states/protected.state")" --arg fs "$(sha "$out/states/free.state")" '
 def chain($end;$start): $end.hinge<=$start.hinge and ($start|{hinge,strict_wins,near_full,far_full})
   == ($end|{hinge,strict_wins,near_full,far_full}) ;
 if chain($p[0].base;$root[0].protected) and chain($f[0].base;$root[0].free) and
   ($p[0].protected.lost_protected|length)==0 and ($p[0].protected.hinge<=$p[0].base.hinge) and
   ($f[0].free.hinge<=$f[0].base.hinge) and $p[0].counts.proposals==$root[0].counts.proposals and
   $f[0].counts.proposals==$root[0].counts.proposals then
 {parent:$root[0].base,protected_first:$root[0].protected,free_first:$root[0].free,
  protected:$p[0].protected,free:$f[0].free,first_counts:$root[0].counts,
  second_counts:{protected:$p[0].counts,free:$f[0].counts},state_sha256:{protected:$ps,free:$fs}}
 else error("round paths or equal proposal budgets do not match") end' > "$out/selection.json"
seal "$out/train"
sha256sum "$out/train/SHA256SUMS" "$out/selection.json" "$out/states/"*.state "$manifest" > "$out/selection.sha256"
seal "$out"
