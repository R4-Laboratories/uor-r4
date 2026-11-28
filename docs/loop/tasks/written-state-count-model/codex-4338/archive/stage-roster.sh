#!/usr/bin/env bash
set -euo pipefail
umask 077
runs=$(realpath "${1:?original Wick runs root}")
run=$(realpath "${2:?unit Wick root}")
source=$(realpath "${3:?source tree with unit holdout files}")
workspace=$(realpath "${4:?original Wick tokenizer workspace}")
stage=$(realpath -m "${5:?fresh custody root}")
unit=docs/loop/tasks/written-state-count-model/codex-4338
supply=20261127-codex-whole-turn-census/supply-inputs
mkdir "$stage"
install -d -m 700 "$stage/roster" "$stage/source" "$stage/workspace/crates"
cp -r "$run/holdout-sealed-v1" "$stage/roster/sealed"
install -m 600 "$run/holdout-verification-v1/official-identity.json" "$stage/roster/official-identity.json"
install -m 600 "$run/holdout-rejections-v1/results.jsonl" "$stage/roster/rejections.jsonl"
cp -r "$workspace/crates/uor-r4-tokenizer" "$stage/workspace/crates/"
(
    cd "$source"
    files=(
      "$unit/holdout/freeze.sh" "$unit/holdout/roster.rs" "$unit/holdout/check.jq"
      "$unit/holdout/rejections.sh" "$unit/holdout/README.md" "$unit/archive/ATTRIBUTION.md"
      "docs/loop/tasks/written-state-count-model/codex-4329/supply/Cargo.toml"
      "docs/loop/tasks/written-state-count-model/codex-4329/supply/Cargo.lock"
      "docs/loop/tasks/written-state-count-model/codex-4329/supply/src/main.rs"
      "docs/loop/tasks/written-state-count-model/codex-4337/data/held.rs"
    )
    cp --parents "${files[@]}" "$stage/source/"
)
jq -r --arg supply "$supply" '
  (.history_files | to_entries[] | [.value,.key]),
  (.pins[] | .[0] |= if contains("/") then . else $supply + "/" + . end | reverse)
  | @tsv' "$stage/roster/sealed/summary.json" |
  while IFS=$'\t' read -r expected path; do
    [[ $expected =~ ^[0-9a-f]{64}$ && -n $path && $path != /* && $path != *..* ]]
    printf '%s  %s\n' "$expected" "$runs/$path" | sha256sum --strict --quiet -c -
    install -D -m 600 "$runs/$path" "$stage/roster/source-inputs/$path"
  done
rustc -vV > "$stage/roster/build-rust.txt"
