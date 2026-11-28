#!/usr/bin/env bash
set -euo pipefail
umask 077
restore=$(realpath "${1:?authenticated extracted packet}")
base=$(realpath "${2:?balanced #4322 artifact}")
replay=$(realpath -m "${3:?fresh replay root}")
client=$restore/runtime/uor-chat
prior_fit=$restore/runtime/prior-interval-fit
tasks=$restore/source/docs/loop/tasks/written-state-count-model
native=$restore/unit/native-v2
parent_receipt=$tasks/codex-4337/evidence/archive.json
tokenizer=$restore/roster/source-inputs/20261127-codex-visible-answer/panel/inputs/tokenizer.json
drivers=$tasks/codex-4338/data
mkdir "$replay"

assert_pin() {
    printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -
}
for pin in client fit balanced tokenizer; do
    case "$pin" in
        client) input=$client ;;
        fit) input=$prior_fit ;;
        balanced) input=$base ;;
        tokenizer) input=$tokenizer ;;
    esac
    assert_pin "$(jq -er .inputs."${pin}_sha256" "$parent_receipt")" "$input"
done
assert_pin "$(awk 'NR == 3 {print $1}' "$native/SOURCES.sha256")" "$restore/runtime/native-fit"
assert_pin "$(awk 'NR == 4 {print $1}' "$native/SOURCES.sha256")" "$restore/runtime/native-opening"

# The parent archive extracts files at mode 0600. Use authenticated private
# executable copies in this replay root regardless of the input file modes.
install -m 700 "$client" "$replay/uor-chat"
install -m 700 "$prior_fit" "$replay/prior-interval-fit"
(cd "$native" && sha256sum --strict -c SEALED.sha256)
bash "$drivers/native-build.sh" "$restore/workspace" "$tasks" "$replay/build"
bash "$drivers/native.sh" "$replay/uor-chat" "$replay/prior-interval-fit" \
    "$replay/build/fit/target/release/interval-fit" \
    "$replay/build/opening/target/release/whole-turn-supervision" "$base" "$tokenizer" \
    "$restore/unit/qualified-v1" "$replay/native"
# Relocating source and binaries changes the path-bearing source manifest.
# Compare each capture, opening, coverage report and device tape byte for byte.
(
    cd "$replay/native"
    awk '$2 != "SOURCES.sha256"' "$native/SEALED.sha256" | sha256sum --strict -c -
)
rejections=$restore/unit/native-rejections-v1
bash "$drivers/native-rejections.sh" "$replay/prior-interval-fit" "$restore/unit/qualified-v1" \
    "$native" "$replay/rejections"
(
    cd "$replay/rejections"
    awk '$2 != "SOURCES.sha256"' "$rejections/SEALED.sha256" | sha256sum --strict -c -
)
match=$restore/unit/matched-v1
bash "$drivers/match.sh" "$restore/fixtures/20261127-codex-grounded-interval" "$native" \
    "$replay/build/fit/target/release/interval-fit" "$replay/matched"
cmp "$match/SEALED.sha256" "$replay/matched/SEALED.sha256"
(cd "$replay/matched" && sha256sum --strict -c "$match/SEALED.sha256")
