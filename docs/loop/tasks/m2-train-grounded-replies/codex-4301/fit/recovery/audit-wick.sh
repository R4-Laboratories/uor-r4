#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-paired-occurrence-margin"
root="$study/fit/recovery"
prior="$runs/20261126-codex-fixed-assignment-median"
here=$(cd "$(dirname "$0")" && pwd -P)
[[ $# == 2 && $1 =~ ^v[1-9][0-9]*$ && $(hostname -s) == wick ]]
fit=$(realpath -e "$2")
[[ $(findmnt -n -o FSTYPE --target "$root") == zfs && $here == "$root"/methods-v* ]]
[[ $fit == "$study"/rented-train-v*/fit-out || $fit == "$study"/rented-finite-v*/fit-out/first || $fit == "$study"/rented-finite-v*/fit-finite-v*/first ]]
version=$1
out="$root/lineage-$version"
mkdir "$out" "$out/source"
cp -- "$here/"{native.rs,lineage.rs,schedule.jq,audit-wick.sh} "$out/source/"
original="$prior/methods-v2/inputs/protected.state"
median="$prior/train-v1/states/ordered.state"
packet="$study/panel/train/ordered/text.bin"
(
    cd "$study/panel"
    sha256sum --quiet --strict -c training.sha256
)
if [[ $fit == "$study"/rented-finite-v* ]]; then
    original="$(dirname "$fit")/original.state"
    median="$(dirname "$fit")/median.state"
    packet="$(dirname "$fit")/text.bin"
else
    printf '%s  %s\n' a2f8d37a5304e557996d2a364ac064e96f9aeeed034357738a3c2989a81b8d8a "$original" \
        0edd2cb3ee9251b9fa40cd829b52e3aff2fc477bf05d515c787cea9435af01ea "$median" | sha256sum --quiet --strict -c -
fi
date -u +%FT%TZ > "$out/audit-utc.txt"
rustc --version > "$out/rustc.txt"
sha256sum "$original" "$median" "$packet" "$fit/"{complete.json,budget-17.json,candidates.tsv,objectives.jsonl,schedule.jsonl} \
    "$fit/"*.state "$fit/"*.native.jsonl > "$out/inputs.sha256"
rustc --edition 2021 -D warnings --cfg recovery_audit "$out/source/lineage.rs" -o "$out/lineage"
jq -es -f "$out/source/schedule.jq" "$fit/schedule.jsonl" > "$out/strict-acceptance.json"
jq -er '
    if .phase == "initial" then
        [.phase, .initial_index, .name, .candidate, .evaluated, .after.movement, .after.changed]
    else
        [.phase,.path,.round,.priority,.parent,.proposal,.accepted,(.candidate // "-"),.evaluated,
         .before.movement,.before.changed,.after.movement,.after.changed]
    end | @tsv' "$fit/schedule.jsonl" > "$out/schedule.tsv"
jq -er '[.candidate,.pair,.selector,.row,.sign,.word,.requested,.rival] +
    .local_center + [.local_debt,.pair_local_worst,.joint_debt,.pair_joint_worst] +
    .joint[0].opposite + .joint[1].opposite | @tsv' "$fit/objectives.jsonl" > "$out/objectives.tsv"
jq -e '(.mode == "fit" and .initial == 57 and .exported <= 105 and .state_bytes == 13336 and
    .proposal_evaluations <= 2649 and .snapshot_replays == .exported)' "$fit/complete.json" > "$out/bounded.json"
jq -e '(.proposals == 17 and (.best_proposal >= 0 and .best_proposal < 17) and
    (has("candidate") or has("state") | not))' "$fit/budget-17.json" > "$out/shadow-objective-only.json"
budget=$(jq -er .proposal_evaluations "$fit/complete.json")
"$out/lineage" "$original" "$median" "$packet" "$fit" "$out" "$budget" > "$out/checks.json"
jq -en --slurpfile checks "$out/checks.json" --slurpfile complete "$fit/complete.json" '
    $checks[0].pass and $checks[0].immutable_states == $complete[0].exported and
    $checks[0].packet_rows == $complete[0].rows' > "$out/complete-agreement.json"
sha256sum --quiet --strict -c "$out/inputs.sha256"
(cd "$out" && find . -type f -print0 | sort -z | xargs -0 sha256sum > "$root/lineage-$version.pending")
mv -- "$root/lineage-$version.pending" "$out/SHA256SUMS"
cat "$out/checks.json"
sha256sum "$out/SHA256SUMS"
