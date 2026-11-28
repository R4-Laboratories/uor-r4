#!/usr/bin/env bash
set -euo pipefail
umask 077
packet=$(realpath "${1:?restored #4340 packet}")
parent=$(realpath "${2:?restored #4338 packet}")
base=$(realpath "${3:?pinned balanced artifact}")
out=$(realpath -m "${4:?fresh replay directory}")
source=$packet/source/docs/loop/tasks/written-state-count-model
unit=$source/codex-4340
fixtures=$parent/fixtures
supply=$fixtures/20261127-codex-grounded-interval/next-supply-v1
official=$parent/roster/source-inputs/20261127-codex-whole-turn-census/supply-inputs
tokenizer=$parent/roster/source-inputs/20261127-codex-visible-answer/panel/inputs/tokenizer.json
prior=$packet/validation-selection-v1
pins=$packet/panel-pins-v1.json
review_a=$packet/semantic-a/decisions.jsonl
review_b=$packet/semantic-b/decisions.jsonl

pin() {
    printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -
}
(
    cd "$packet"
    sha256sum --strict --quiet -c MEMBERS.sha256
)
pin 3b2d3d587beb6e557bd6bb95992687d34b809e27aa0e3e66c0919e69db19cc51 "$parent/MEMBERS.sha256"
(
    cd "$parent"
    sha256sum --strict --quiet -c MEMBERS.sha256
)
jq -en --slurpfile registered "$unit/evidence/input-pins.json" --slurpfile actual "$pins" '
    $registered[0] == ($actual[0] | del(.extraction,.review_a,.review_b))
' >/dev/null
pin "$(jq -er .extraction "$pins")" "$prior/EXTRACTION.sha256"
pin "$(jq -er .semantic_source_rows_sha256 "$pins")" "$prior/selected-semantic/rows.jsonl"
pin "$(jq -er .review_a "$pins")" "$review_a"
pin "$(jq -er .review_b "$pins")" "$review_b"
pin "$(jq -er .training_capsule_sha256 "$pins")" "$packet/fit-input-v1/INPUTS.sha256"
(
    cd "$prior"
    sha256sum --strict --quiet -c EXTRACTION.sha256
    (cd selection && sha256sum --strict --quiet -c SELECTION.sha256)
    (cd selected-semantic && sha256sum --strict --quiet -c SEMANTIC.sha256)
)

mkdir "$out"
for executable in uor-chat native-opening native-fit; do
    install -m 700 "$parent/runtime/$executable" "$out/$executable"
done
extract=$out/extraction
bash "$unit/validation/extract.sh" "$supply" "$parent/roster/sealed" "$official" "$extract"

# Original extraction manifests retain absolute paths. Match the ordered input
# digests and every relative output before deriving a relocation-only pin.
cmp <(cut -d ' ' -f 1 "$prior/INPUTS.sha256") <(cut -d ' ' -f 1 "$extract/INPUTS.sha256")
for directory in selection selected-semantic; do
    diff -rq "$prior/$directory" "$extract/$directory"
done
extraction=$(sha256sum "$extract/EXTRACTION.sha256" | cut -d ' ' -f 1)
jq --arg extraction "$extraction" '.extraction=$extraction' "$pins" > "$out/relocated-pins.json"

qualify() {
    bash "$unit/validation/qualify-panel.sh" "$extract" "$@" "$out/relocated-pins.json" \
        "$out/uor-chat" "$out/native-opening" "$base" "$tokenizer" "$qualification"
}
qualification=$out/qualification
qualify "$review_a" "$review_b"
diff -rq "$packet/qualification-v1" "$qualification"
bash "$unit/fit/stage.sh" "$parent/unit/matched-v1" "$out/native-fit" "$out/train-input"
diff -rq "$packet/fit-input-v1" "$out/train-input"

cp "$review_a" "$out/altered-review.jsonl"
printf '\n' >> "$out/altered-review.jsonl"
qualification=$out/rejected-review
if qualify "$out/altered-review.jsonl" "$review_b" > "$out/rejected-review.log" 2>&1; then
    printf 'Qualification accepted a modified review ledger\n' >&2
    exit 1
fi
test -d "$qualification"
test -z "$(find "$qualification" -mindepth 1 -print -quit)"
