#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $# == 2 ]] || { echo 'usage: stage-view.sh restored|replay NEW_VIEW_DIRECTORY' >&2; exit 2; }
side=$1
out=$2
ops=$(cd "$(dirname "$0")/.." && pwd)
[[ $side == restored || $side == replay ]]
mkdir -m700 "$out"
cd "$out"
base=/mnt/plexpool/uor-r4-corpus/runs
replay=$base/20261128-codex-grounded-opposed-replication
files=(base.uorbfy03 train.jsonl role-and-source.jsonl tokenizer.json historical.json serving.sm89.cubin original-train.sm89.cubin)
if [[ $side == restored ]]; then
    source=forced-original/packet
    old=forced-original/replay/attempt02
    finite=$source/received/attempt01
    readout=$source/received/attempt04/readout
    input_files=()
    for file in "${files[@]}"; do input_files+=("forced-original/replay/inputs/$file"); done
    archive_sha=9e5f42b6759ae3709d32060cfe74553bd700ab3af037ad915d68ba9fb58a8e20
    ssh wick "set -eu; cd '$replay'; printf '%s  %s\n' '$archive_sha' forced-original/original.tar.zst | sha256sum --strict --status -c -; (cd '$source'; sha256sum --strict --status -c PACKET.sha256); (cd '$old'; sha256sum --strict --status -c members.sha256); tar -chf - forced-original/verified.log '$source/PACKET.sha256' '$source/audit/training-sources.sha256' '$source/evaluators.sha256' '$source/bin/checker' '$source/bin/readout-v2' '$source/cold' '$old/checkpoints' '$old/device.csv' '$finite/finite.json' '$finite/serving-finite.json' '$finite/device.csv' '$readout' ${input_files[*]}" | tar -xf -
    for path in "${input_files[@]}"; do ln -s "$path" "${path##*/}"; done
    ln -s "$source/bin/checker" checker
    ln -s "$source/bin/readout-v2" readout-v2
    ln -s "$source/cold" cold
    ln -s "$old/device.csv" authentic.csv
    ln -s "$finite/device.csv" calibration.csv
    ln -s "$old/checkpoints" checkpoints
    ssh wick "cd '$replay'; tar -chf - audit/uor-chat" | tar -xf -
else
    source=.
    finite=received/calibration
    readout=received/readout/readout
    input_files=()
    for file in "${files[@]}"; do input_files+=("intake/inputs/$file"); done
    ssh wick "set -eu; cd '$replay'; (cd intake; sha256sum --strict --status -c intake.sha256); (cd audit; sha256sum --strict --status -c wick-only.sha256); for s in calibration authentic readout; do (cd received/\$s; test -f COMPLETE; sha256sum --strict --status -c members.sha256); done; test -s cold/report.json; tar -chf - intake/intake.sha256 audit/wick-only.sha256 audit/uor-chat intake/bin/checker intake/bin/readout-v2 received/calibration received/authentic received/readout cold ${input_files[*]}" | tar -xf -
    for path in "${input_files[@]}"; do ln -s "$path" "${path##*/}"; done
    mkdir -m700 operations
    cp "$ops/"{pod.id,lease.json,up.log,down.log,calibration.pins,training.sha256} operations/
    pod=$(cat operations/pod.id)
    jq -c --arg pod "$pod" '
      select(.pod==$pod and .lab=="codex" and .session=="grounded-opposed-replication")
    ' "$HOME/.local/share/uor-r4/compute/ledger.jsonl" > operations/compute.jsonl
    for stage in calibration authentic readout; do cp "$ops/$stage."{job,retrieval}.log operations/; done
    ln -s intake/bin/checker checker
    ln -s intake/bin/readout-v2 readout-v2
    ln -s received/authentic/device.csv authentic.csv
    ln -s "$finite/device.csv" calibration.csv
    ln -s received/readout/device.csv readout.csv
    ln -s received/authentic/checkpoints checkpoints
fi
ln -s audit/uor-chat uor-chat
ln -s checkpoints/report.json training.json
for file in sham.uorbfy03 active.uorbfy03; do ln -s "checkpoints/$file" "$file"; done
for file in finite.json serving-finite.json; do ln -s "$finite/$file" "$file"; done
for file in rows.jsonl calibration.json; do ln -s "$readout/$file" "$file"; done
ln -s "$readout/report.json" readout.json
