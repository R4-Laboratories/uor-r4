#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
if [[ $# != 5 ]]; then
    echo 'usage: replay-repair.sh RESTORED_4341 RESTORED_4340 RESTORED_4338 BALANCED_D11 FRESH_OUT' >&2
    exit 2
fi
packet=$(realpath "$1")
prior=$(realpath "$2")
supply=$(realpath "$3")
base=$(realpath "$4")
out=$(realpath -m "$5")
here=$(cd "$(dirname "$0")" && pwd -P)
source "$here/../repair/custody.sh"
tasks=$packet/sources/docs/loop/tasks/written-state-count-model
unit=$tasks/codex-4341
repair=$unit/repair
old=$tasks/codex-4340/validation
src=$packet/repair
(cd "$packet" && sha256 --strict --quiet -c MEMBERS.sha256)
pin 265fe39846ff9cc7654d5a3284f1e6d2029bdd823abe72867464c6da287725c5 "$prior/MEMBERS.sha256"
(cd "$prior" && sha256 --strict --quiet -c MEMBERS.sha256)
pin 3b2d3d587beb6e557bd6bb95992687d34b809e27aa0e3e66c0919e69db19cc51 "$supply/MEMBERS.sha256"
(cd "$supply" && sha256 --strict --quiet -c MEMBERS.sha256)
pin 5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa "$base"
official=$supply/roster/source-inputs/20261127-codex-whole-turn-census/supply-inputs/squad-dev-v2.0.json
tokenizer=$supply/roster/source-inputs/20261127-codex-visible-answer/panel/inputs/tokenizer.json
mkdir "$out"
install -m 700 "$supply/runtime/uor-chat" "$out/uor-chat"
install -m 700 "$supply/runtime/native-opening" "$out/native-opening"
install -m 700 "$supply/runtime/native-fit" "$out/native-fit"

bash "$repair/replay.sh" select "$src/selection-v1" "$out/selection" "$prior/validation-selection-v1"
bash "$repair/replay.sh" materialize "$src/extraction-v4" "$out/extraction" \
    "$prior/validation-selection-v1" "$out/selection" "$unit/REGISTRATION.json" "$supply/roster/sealed" \
    "$official" "$old"
bash "$repair/replay.sh" qualify "$src/consensus-v1" "$out/qualification" "$out/extraction" "$prior" \
    "$src/semantic-a/decisions.jsonl" "$src/semantic-b/decisions.jsonl" "$src/repair-pins.json" \
    "$old" "$out/uor-chat" "$out/native-opening" "$base" "$tokenizer"

score=$unit/score
(cd "$score" && sha256 --strict --quiet -c SOURCE.sha256)
CARGO_TARGET_DIR="$out/cargo-target" cargo build --locked --offline --manifest-path "$score/Cargo.toml"
tool=$out/cargo-target/debug/validation-transfer-score
digest() { sha256 < "$1" | cut -d' ' -f1; }
bash "$score/assemble-roster.sh" "$tool" 4341 "$out/extraction" "$(digest "$src/extraction-v4/EXTRACTION.sha256")" \
    "$out/qualification" "$(digest "$src/consensus-v1/SEALED.sha256")" "$packet/controls" \
    "$(digest "$packet/controls/CONTROLS.sha256")" "$out/roster"
diff -rq "$packet/roster" "$out/roster"
bash "$score/stage-eval.sh" "$out/native-fit" 7cd8f52ad299622eef75579364ec5c31d9d98313051248df85565c74a2cc6eb5 \
    "$out/roster" "$(digest "$packet/roster/ROSTER.sha256")" "$out/eval"
diff -rq "$packet/eval" "$out/eval"
(cd "$unit/client" && sha256 --strict --quiet -c SOURCE.sha256)
bash "$unit/client/stage.sh" "$out/roster/roster.json" "$(digest "$packet/roster/roster.json")" \
    "$out/extraction/selected-semantic/source.jsonl" "$(digest "$src/extraction-v4/selected-semantic/source.jsonl")" \
    "$out/writer-stage"
diff -rq "$packet/writer-stage" "$out/writer-stage"
