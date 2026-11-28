#!/usr/bin/env bash
set -euo pipefail
umask 077
restore=$(realpath "${1:?authenticated extracted packet}")
replay=$(realpath -m "${2:?fresh replay root}")
repo=$restore/source
unit=$repo/docs/loop/tasks/written-state-count-model/codex-4338
sealed=$restore/roster/sealed
inputs=$restore/roster/source-inputs
official=$inputs/20261127-codex-whole-turn-census/supply-inputs/squad-dev-v2.0.json
mkdir "$replay"
(cd "$sealed" && sha256sum --strict -c SEAL.sha256 ROSTER.sha256)
CARGO_NET_OFFLINE=true bash "$unit/holdout/freeze.sh" "$inputs" "$restore/workspace" "$unit" "$replay/build" "$replay/roster"
(cd "$replay/roster" && sha256sum --strict -c "$sealed/ROSTER.sha256")
jq -e --slurpfile roster "$sealed/roster.jsonl" -f "$unit/holdout/check.jq" \
    "$official" > "$replay/official-identity.json"
cmp "$restore/roster/official-identity.json" "$replay/official-identity.json"
bash "$unit/holdout/rejections.sh" "$official" "$sealed/roster.jsonl" "$replay/rejections"
cmp "$restore/roster/rejections.jsonl" "$replay/rejections/results.jsonl"
