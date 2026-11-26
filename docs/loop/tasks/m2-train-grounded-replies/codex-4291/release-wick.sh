#!/usr/bin/env bash
# Bind the approved sheet and both independent fixtures before measuring the registered prompts.
set -euo pipefail
umask 077
runs=/mnt/plexpool/uor-r4-corpus/runs
s="$runs/20261126-codex-mixed-transport"; prev="$runs/20261126-codex-far-transport-margin"
prior="$runs/20261126-codex-text-governor-links"; history="$runs/20261125-codex-governing-source-pairs"
old="$runs/20261126-codex-governor-noun-links"; donor="$runs/20261125-codex-bound-instruction-state"
m=$(realpath "${1:?sealed methods}"); panel=$(realpath "${2:?panel}"); build=$(realpath "${3:?physical trainer build}")
finite=$(realpath "${4:?qualified physical fixture}"); ab=$(realpath "${5:?audit build}"); af=$(realpath "${6:?canonical audit fixture}")
out=$(realpath "${7:?new release directory containing nonauthor FIT and lab decision}"); fit_sha=${8:?approved FIT SHA}
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$s") == zfs && $m == "$s"/methods-v* && $panel == "$s"/panel/panel-v* && $build == "$s"/fit/build-v* &&
 $finite == "$s"/fit/finite-v* && $ab == "$s"/audit/build-v* && $af == "$s"/audit/finite-v* && $out == "$s"/release-v* ]]
test ! -e "$out/inputs.sha256"
printf '%s  %s\n' "$fit_sha" "$out/FIT.md" | sha256sum --strict --quiet -c -
for sealed in "$m" "$panel" "$build" "$finite" "$finite-lease" "$ab" "$af" "$s/grade-fixture-v1" "$s/client/"{methods-v1,fixture-v1}; do
 (cd "$sealed" && sha256sum --strict --quiet -c SHA256SUMS)
done
test "$(cat "$finite-lease/exit-code.txt")" = 0
printf '%s\n' "$build/transport-mixed-sm89" > "$out/trainer-path.txt"
printf '%s\n' "$ab/target/release/transport-margin-audit" > "$out/oracle-path.txt"
printf '%s\n' "$panel" > "$out/panel-path.txt"
sha256sum "$out/"{trainer-path.txt,oracle-path.txt,panel-path.txt,DECISION.md,FIT.md} "$m/"{common-wick.sh,measure-wick.sh,grade-wick.sh,PRE-RUN.md,ORDER.md,SHA256SUMS} \
 "$build/"{transport-mixed-sm89,SHA256SUMS} "$finite/"{checks.json,SHA256SUMS} "$finite-lease/SHA256SUMS" "$ab/"{target/release/transport-margin-audit,SHA256SUMS} \
 "$af/SHA256SUMS" "$s/audit/finite-wick.sh" "$s/grade-fixture-v1/SHA256SUMS" "$s/client/"{methods-v1/SHA256SUMS,fixture-v1/SHA256SUMS,run-wick.sh} "$panel/"{SHA256SUMS,authority.sha256} \
 "$panel/"{mixed,far,held}/{reference.json,blind/text.bin,blind/coordinates.jsonl,aid.bin,links.bin,distance.txt} \
 "$prev/fit/build-v3/transport-margin-sm89" "$prev/audit/build-v2/target/release/transport-margin-audit" "$prev/methods-v1/audit/compare.jq" \
 "$prior/learner/build-v2/textlinks-sm89" "$prior/parity/build-v2/release/textlinks-trace-audit" "$prior/measurement-v1/train/learned.state" \
 "$prev/measurement-v2/train/contrastive.first.state" "$prev/panel/panel-v1/SHA256SUMS" "$prev/measurement-v2/held/"{baseline,learned}.jsonl \
 "$old/device/build-v2/SHA256SUMS" "$history/fit-v1/primary.words" "$history/models/fold-8/ordered-all.d11" "$history/reserve/history.json" \
 "$donor/clients/v4/uor-chat" "$donor/custody-v1/signed/tokenizer.json" "$runs/20261125-codex-within-prompt-scope-margin/device-build-v1/source/codex-4281/fit/device-wick.sh" "$0" > "$out/inputs.sha256"
sha256sum --strict --quiet -c "$out/inputs.sha256"; sha256sum "$out/inputs.sha256"
