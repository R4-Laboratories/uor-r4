#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
[[ $# == 2 ]]
root=$(realpath -e "$1")
out=$2
mkdir "$out"
authority="$root/inputs/panel-authority"
seed="$root/inputs/seed"
panel="$root/panel"
recipe="$root/methods/panel"
old="$authority/20261126-codex-third-margin-retain"
paired="$authority/20261126-codex-paired-occurrence-margin/panel"
interval="$authority/20261126-codex-all-outranker-interval/panel"
median="$authority/20261126-codex-fixed-assignment-median/panel/panel-v1"
switches="$authority/20261126-codex-switch-source-competition/panel/panel-v2"
prior="$old/storage/cold-v1/packet"
generator="$old/panel/generator-build-v1/target/release/governor-links-panel"
exporter="$interval/build-v3/target/release/source-link-text"
contrast="$paired/generator-build-v3/target/release/source-link-text"
history="$panel/fresh/history.json"

(cd "$authority" && sha256sum --strict --quiet -c SHA256SUMS)
(cd "$panel" && sha256sum --strict --quiet -c PANELS.sha256 guards.sha256)
for part in near regression fresh; do
    (cd "$panel/$part" && sha256sum --strict --quiet -c SHA256SUMS)
done
for f in fresh-design.json grammar.json; do cmp "$panel/fresh/$f" "$recipe/$f"; done
"$generator" --contrast "$seed/tokenizer.json" "$seed/ordered-all.d11" "$history" "$recipe/fresh-design.json" \
    "$out/generated" > "$out/generator.log"
cmp "$out/generated/reference.json" "$panel/fresh/reference.json"
jq -ce 'map(.turns[1]) | select(all(.[]; type == "string"))' "$out/generated/reference.json" > "$out/user-only.json"
cmp "$out/user-only.json" "$panel/fresh/user-only.json"
"$exporter" pack "$out/user-only.json" "$history" "$seed/ordered-all.d11" "$seed/tokenizer.json" "$out/endpoint"
"$exporter" pack-ordered "$out/user-only.json" "$history" "$seed/ordered-all.d11" "$seed/tokenizer.json" "$out/ordered"
for space in endpoint ordered; do
    for name in text.bin coordinates.jsonl; do cmp "$out/$space/$name" "$panel/fresh/$space/$name"; done
done
"$contrast" compare-contrast "$out/generated/reference.json" "$out/endpoint/text.bin" "$out/ordered/text.bin" > "$out/matched.json"
cmp "$out/matched.json" "$panel/fresh/matched.json"
jq -s 'add' "$interval/panel-v1/"{held,contrast}/reference.json \
    "$median/"{held,contrast,mixed}/reference.json "$switches/"{historical,train-derived}/reference.json \
    "$paired/"{regression,fresh}/reference.json "$prior/panel/"{train,fresh}/reference.json > "$out/earlier.json"
jq -ce --argjson expected 1672 --slurpfile prior "$out/earlier.json" \
    -f "$prior/methods/panel/provenance.jq" "$panel/fresh/reference.json" > "$out/provenance.json"
cmp "$out/provenance.json" "$panel/fresh/provenance.json"

for part in near regression fresh; do
    cohort="$panel/$part"
    mkdir "$out/$part"
    jq -ce --arg cohort "$part" --slurpfile refs "$cohort/reference.json" \
        --slurpfile guard <(jq -s . "$panel/near/gold.jsonl") \
        --slurpfile grammar "$recipe/grammar.json" --slurpfile design "$recipe/fresh-design.json" \
        -f "$recipe/mask-evidence.jq" "$cohort/rows.jsonl" > "$out/$part/mask-evidence.jsonl"
    cmp "$out/$part/mask-evidence.jsonl" "$cohort/mask-evidence.jsonl"
    jq -er '[.row_index,.sign,.governor.index,.wrong_governor.index,(.good|length),.good[].index,
        (.wrong|length),.wrong[].index]|map(tostring)|join(" ")' "$out/$part/mask-evidence.jsonl" > "$out/$part/masks.tsv"
    cmp "$out/$part/masks.tsv" "$cohort/masks.tsv"
    if [[ $part == near ]]; then continue; fi
    for lane in gold swapped; do
        refs="$cohort/reference.json"
        if [[ $lane == swapped ]]; then refs="$cohort/swapped-reference.json"; fi
        "$exporter" supervise "$refs" "$cohort/ordered/coordinates.jsonl" \
            "$out/$part/$lane-links.bin" "$out/$part/$lane-aid.bin"
        cmp "$out/$part/$lane-links.bin" "$cohort/$lane-links.bin"
        cmp "$out/$part/$lane-aid.bin" "$cohort/$lane-aid.bin"
    done
done
printf 'Rebuilt fresh text and exact coordinates, membership, all masks and both complete-address controls.\n'
