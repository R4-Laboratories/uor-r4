#!/usr/bin/env bash
# Released physical update and blind collection. A zero exit with no strict premise records a curriculum failure.
set -euo pipefail
umask 077
export LC_ALL=C CUDA_DISABLE_PTX_JIT=1 CUDA_FORCE_PTX_JIT=0
runs=/mnt/plexpool/uor-r4-corpus/runs
s="$runs/20261126-codex-far-transport-margin"
prior="$runs/20261126-codex-text-governor-links"
history="$runs/20261125-codex-governing-source-pairs"
old="$runs/20261126-codex-governor-noun-links/device/build-v2"
fixed="$prior/measurement-v1/train/learned.state"
reader="$prior/learner/build-v2/textlinks-sm89"
joint_oracle="$prior/parity/build-v2/release/textlinks-trace-audit"
banks="$history/fit-v1/primary.words"
model="$history/models/fold-8/ordered-all.d11"
lock="$runs/20261125-codex-within-prompt-scope-margin/device-build-v1/source/codex-4281/fit/device-wick.sh"
panel="$s/panel/panel-v1"
here=$(cd "$(dirname "$0")" && pwd)
test "$(hostname -s)" = wick
test "$(findmnt -no FSTYPE --target "$s")" = zfs
[[ $here == "$s"/methods-v* ]]
mode=${1:?run, inner, or grade}
version=${2:?new measurement version}; [[ $version =~ ^v[1-9][0-9]*$ ]]
manifest=$(realpath "${3:?release manifest}"); digest=${4:?release sha}
[[ $manifest == "$s"/release-v*/inputs.sha256 && $digest =~ ^[0-9a-f]{64}$ ]]
sha(){ sha256sum "$1"|cut -d' ' -f1; }
seal(){ (cd "$1" && test ! -e SHA256SUMS && find . -type f ! -name .pending -print0 | sort -z | xargs -0 sha256sum >.pending && mv .pending SHA256SUMS); }
verify(){ (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS); }
test "$(sha "$manifest")" = "$digest"
sha256sum --strict --quiet -c "$manifest"
reldir=$(dirname "$manifest")
trainer=$(cat "$reldir/trainer-path.txt"); oracle=$(cat "$reldir/oracle-path.txt")
[[ $trainer == "$s/fit/build-"*/transport-margin-sm89 && $oracle == "$s/audit/build-"*/target/release/transport-margin-audit ]]
for f in "$reldir/"{trainer-path.txt,oracle-path.txt,FIT.md,DECISION.md} "$trainer" "$oracle" "$reader" "$joint_oracle" "$fixed" "$banks" "$model" "$lock" \
  "$panel/SHA256SUMS" "$here/"{measure-wick.sh,PRE-RUN.md,SHA256SUMS}; do
  grep -Fxq "$(sha256sum "$f")" "$manifest"
done
verify "$here"
out="$s/measurement-$version"
if [[ $mode == run ]]; then
  bash "$lock" "$out-lease" bash "$here/measure-wick.sh" inner "$version" "$manifest" "$digest"
elif [[ $mode == inner ]]; then
  test -e /proc/$$/fd/9
  verify "$panel"; verify "$old"
  mkdir "$out" "$out/train" "$out/held" "$out/independent"
  d="$out/train"; p="$panel/train"
  "$trainer" diagnose "$p/blind/text.bin" "$p/links.bin" "$fixed" "$d/initial.jsonl"
  "$oracle" diagnose "$p/blind/text.bin" "$p/links.bin" "$fixed" "$fixed" "$fixed" "$d/initial.oracle.jsonl" "$d/initial.reconstruction.json"
  jq -es --slurpfile physical "$d/initial.jsonl" -f "$here/audit/compare.jq" "$d/initial.oracle.jsonl" > "$d/initial.checks.json"
  jq -es 'if length!=512 or (map([.row,.sign])|unique|length)!=512 then error("signed enrollment changed") else
   {all:length,eligible:map(select(.eligible_for_fit))|length,strict_losses:map(select(.eligible_for_fit and .before.margin<0))|length,
    ties:map(select(.eligible_for_fit and .before.margin==0))|length,strict_wins:map(select(.eligible_for_fit and .before.margin>0))|length} end' \
   "$d/initial.jsonl" > "$d/premise.json"
  if ! jq -e '.strict_losses>0' "$d/premise.json" >/dev/null; then
    echo 'no original strict TRAIN loss' > "$out/premise-rejected.txt"
    seal "$out"; exit 0
  fi
  for arm in contrastive positive; do
    for trial in first repeat; do
      "$trainer" train "$arm" "$p/blind/text.bin" "$p/links.bin" "$fixed" "$d/$arm.$trial.state" "$d/$arm.$trial.trace" "$d/$arm.$trial.final.state"
    done
    for suffix in state trace final.state; do cmp "$d/$arm.first.$suffix" "$d/$arm.repeat.$suffix"; done
    "$oracle" "$arm" "$p/blind/text.bin" "$p/links.bin" "$fixed" "$d/$arm.first.state" "$d/$arm.first.final.state" "$d/$arm.oracle.jsonl" "$d/$arm.updates.json"
    jq -es --slurpfile physical "$d/$arm.first.trace" -f "$here/audit/compare.jq" "$d/$arm.oracle.jsonl" > "$d/$arm.checks.json"
  done
  # Commit both selected states and their complete TRAIN evidence before opening the blind held packet.
  sha256sum "$fixed" "$d/"{initial.jsonl,initial.checks.json,premise.json,{positive,contrastive}.first.{state,final.state,trace},{positive,contrastive}.checks.json} > "$out/train-selection.sha256"
  for split in train held; do
    p="$panel/$split"; d="$out/$split"
    sha256sum --strict --quiet -c "$out/train-selection.sha256"
    for mode in native unfitted; do "$reader" "$mode" "$banks" "$p/blind/text.bin" "$d/$mode.jsonl"; done
    "$reader" governor_gap "$banks" "$p/blind/text.bin" "$fixed" "$d/gap.jsonl"
    for arm in baseline positive learned; do
      case $arm in baseline) state=$fixed ;; positive) state="$out/train/positive.first.state" ;; learned) state="$out/train/contrastive.first.state" ;; esac
      "$reader" learned "$banks" "$p/blind/text.bin" "$state" "$d/$arm.jsonl"
      "$joint_oracle" "$p/blind/text.bin" "$banks" "$state" "$d/$arm.jsonl" "$d/gap.jsonl" > "$d/$arm.replay.json"
    done
    "$reader" learned "$banks" "$p/blind/text.bin" "$out/train/contrastive.repeat.state" "$d/learned.repeat.jsonl"
    cmp "$d/learned.jsonl" "$d/learned.repeat.jsonl"
    "$reader" learned "$banks" "$p/blind/text.bin" "$out/train/positive.repeat.state" "$d/positive.repeat.jsonl"
    cmp "$d/positive.jsonl" "$d/positive.repeat.jsonl"
    # The trained source measurements have no labels or aid in their argv.
    for arm in correct exchanged; do "$reader" "$arm" "$banks" "$p/blind/text.bin" "$p/aid.bin" "$d/$arm.jsonl"; done
  done
  # Independent instrument recomputes supplied and native sources from the actual original intervals.
  p="$panel/held/structural-links"
  for arm in native governor_only nouns_only correct_link exchanged_link; do
    "$old/governor-noun-links-sm89" "$banks" "$p/geometry.bin" "$p/$arm.bin" "$out/independent/$arm.rank"
  done
  "$old/target/release/governor-noun-links" grade "$p" "$model" "$banks" "$out/independent"
  sha256sum "$manifest" > "$out/inputs.sha256"
  seal "$out"
elif [[ $mode == grade ]]; then
  verify "$out"
  g="$s/grade-${5:?new grade version}"
  mkdir "$g"
  for arm in contrastive positive; do
    jq -nce --slurpfile reference "$panel/train/reference.json" --slurpfile coords "$panel/train/blind/coordinates.jsonl" \
      -f "$here/grade/loss-cases.jq" "$out/train/$arm.first.trace" > "$g/$arm.edges.jsonl"
    jq -es -f "$here/grade/loss-summary.jq" "$g/$arm.edges.jsonl" > "$g/$arm.loss.json"
  done
  for split in train held; do
    d="$out/$split"; p="$panel/$split"
    jq -nce --slurpfile reference "$p/reference.json" --slurpfile coords "$p/blind/coordinates.jsonl" \
      --slurpfile native "$d/native.jsonl" --slurpfile unfitted "$d/unfitted.jsonl" --slurpfile learned "$d/learned.jsonl" \
      --slurpfile gap "$d/gap.jsonl" --slurpfile correct "$d/correct.jsonl" --slurpfile exchange "$d/exchanged.jsonl" \
      --slurpfile base "$d/baseline.jsonl" --slurpfile pos "$d/positive.jsonl" -f "$here/grade/rows.jq" > "$g/$split.jsonl"
    jq -es --arg split "$split" -f "$here/grade/source-scores.jq" "$g/$split.jsonl" > "$g/$split.scores.json"
  done
  jq -nce --slurpfile r "$g/held.jsonl" --slurpfile other "$out/independent/endpoints.jsonl" '
    ($other|INDEX(.case_id)) as $index |
    [$r[] as $p| $index[$p.case_id] as $o|["native","correct_link","exchanged_link"][] as $a|
      select($p.prompt_sha256!=$o.prompt_sha256 or
        ($p.arms[$a]|[.selection,.tokens,.bank_cost,.ties,.eligible])!=($o.arms[$a]|[.selection,.tokens,.bank_cost,.ties,.eligible]))|
      [$p.case_id,$a]] as $bad |
    if ($r|length)==256 and ($other|length)==256 and $bad==[] then {original_address_comparisons:768,disagreements:0}
    else error("unresolved comparison with independent controls") end' > "$g/control-replay.json"
  jq -c '.arms |= with_entries(select(.key|IN("native","baseline","learned","correct_link","exchanged_link")))' "$g/held.jsonl" > "$g/endpoints.jsonl"
  jq -e --slurpfile training "$g/contrastive.loss.json" --arg reference_sha "$(sha "$panel/held/reference.json")" \
      --arg writer_sha "$(sha "$g/endpoints.jsonl")" --arg source_sha "$(sha "$g/held.jsonl")" -f "$here/grade/release.jq" "$g/held.scores.json" > "$g/summary.json"
  jq -es --slurpfile reference "$panel/held/reference.json" -f "$here/grade/trace.jq" "$g/held.jsonl" > "$g/held-errors.json"
  sha256sum "$manifest" "$out/SHA256SUMS" "$here/SHA256SUMS" > "$g/inputs.sha256"
  seal "$g"
else exit 64; fi
