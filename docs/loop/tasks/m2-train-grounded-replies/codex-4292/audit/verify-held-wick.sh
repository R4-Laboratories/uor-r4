#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
s="$runs/20261126-codex-protected-transport"
prior="$runs/20261126-codex-mixed-transport/measurement-v1"
panel="$s/panel/panel-v1"
words="$runs/20261125-codex-governing-source-pairs/fit-v1/primary.words"
here=$(cd "$(dirname "$0")" && pwd)
measurement=${1:?measurement version}
transfer=${2:?transfer version}
grade=${3:?grade version}
build=${4:?audit build version}
version=${5:?new held audit version}
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$s") == zfs ]]
for id in "$measurement" "$transfer" "$grade" "$build" "$version"; do [[ $id =~ ^v[1-9][0-9]*$ ]]; done
m="$s/measurement-$measurement"
t="$s/transfer-$transfer"
g="$s/grade-$grade"
d="$s/audit/transfer-$version"
v="$s/audit/build-$build"
verify() (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS)
for dependency in "$m" "$t" "$g" "$v" "$panel"; do verify "$dependency"; done
sha256sum --strict --quiet -c "$t/inputs.sha256" "$g/inputs.sha256"
mkdir "$d"
for arm in baseline free protected far_only; do
    case "$arm" in
        baseline) state="$prior/train/mixed/first/step-0.state" ;;
        far_only) state="$prior/states/far_only.state" ;;
        *) state="$m/states/$arm.state" ;;
    esac
    "$v/protected-transport-audit" held "$words" "$panel/held/blind/text.bin" "$state" \
        "$t/held/$arm.raw.jsonl" "$panel/held/reference.json" "$d/$arm.readback.json"
    jq -e --arg a "$arm" --slurpfile grade "$g/ids.json" '
      . as $p | $grade[0].distance as $d |
      $p.pass and $p.rows==256 and $p.near_exact==$d.near[$a].exact and
      $p.far_exact==$d.far[$a].exact and $p.near_full==$d.near[$a].full and
      $p.far_full==$d.far[$a].full
    ' "$d/$arm.readback.json" > "$d/$arm.counts-check.json"
done
for control in native correct_link exchanged_link; do
    jq -e '.decisions==256 and .disagreements==0' "$t/controls/$control.checks.json" > "$d/$control.check.json"
done
jq -ne --slurpfile baseline "$d/baseline.readback.json" --slurpfile free "$d/free.readback.json" \
    --slurpfile protected "$d/protected.readback.json" --slurpfile far_only "$d/far_only.readback.json" \
    --slurpfile gate "$g/summary.json" '
  ($baseline[0].near_exact-$protected[0].near_exact) as $near_lost |
  ($far_only[0].far_exact-$baseline[0].far_exact) as $matched_gain |
  ($matched_gain-$protected[0].far_exact) as $far_lost |
  if $near_lost==$gate[0].lost_near and $matched_gain==$gate[0].matched_far_gains and
      $far_lost==$gate[0].matched_far_gains_lost then
     def scores: {near_exact:(.near_exact|length),far_exact:(.far_exact|length),
                  near_full:(.near_full|length),far_full:(.far_full|length)};
     {pass:true, source_gate_released:$gate[0].pass, original_near_lost:$near_lost,
      matched_far_gains_lost:$far_lost,
      baseline:($baseline[0]|scores),free:($free[0]|scores),protected:($protected[0]|scores),
      far_only:($far_only[0]|scores)}
  else error("fresh moved pair identities contradict the source gate") end
' > "$d/verified.json"
sha256sum "$m/SHA256SUMS" "$t/SHA256SUMS" "$g/SHA256SUMS" "$v/SHA256SUMS" "$panel/SHA256SUMS" \
    "$words" "$here/verify-held-wick.sh" > "$d/inputs.sha256"
(cd "$d" && sha256sum -- *.json inputs.sha256 > SHA256SUMS)
