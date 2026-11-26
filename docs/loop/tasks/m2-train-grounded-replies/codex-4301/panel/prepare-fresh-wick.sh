#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-paired-occurrence-margin/panel"
precedent="$runs/20261126-codex-all-outranker-interval/panel"
median="$runs/20261126-codex-fixed-assignment-median/panel/panel-v1"
switches="$runs/20261126-codex-switch-source-competition/panel/panel-v2"
codec="$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
d11="$runs/20261125-codex-governing-source-pairs/models/fold-8/ordered-all.d11"
history="$root/history.json"
here=$(cd "$(dirname "$0")" && pwd)
build_version=${1:?existing positive generator build version}
attempt=${2:?unused positive fresh panel version}
[[ $# == 2 && $build_version =~ ^[1-9][0-9]*$ && $attempt =~ ^[1-9][0-9]*$ ]]
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$root") == zfs && $here == "$root"/methods-v* ]]
stage="$root/fresh-v$attempt.partial"
build="$root/generator-build-v$build_version"
generator="$build/target/release/governor-links-panel"
compare="$build/target/release/source-link-text"
exporter="$precedent/build-v3/target/release/source-link-text"
[[ ! -e $root/fresh && ! -e $stage ]]

pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
pin 5c843187d34c3a6ca94560a161c8d82c3655d16dd65e283c36753d1e9f55e7b8 "$d11"
pin a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$codec"
pin 9cf2c96fcb088bbcf6b7132f8a8845c1a04c05150cfb06fc160c5f0ee57b6dc7 "$exporter"
pin 92704b9d3ffb8201245a26fe05c5648e29360d0a383dd0db9ccb1389f845eb4b "$precedent/panel-v1/SHA256SUMS"
pin 80bbfaa53532c8e113000445285721c1e52d6f060b51a880a7c17681980c4a93 "$median/SHA256SUMS"
pin 8f46a07da62c7243cf2d5965c460ce2ba3f6484c1fa7266b77ecb7a4f0f09d92 "$switches/SHA256SUMS"
for directory in "$precedent/panel-v1" "$median" "$switches"; do
    (cd "$directory" && sha256sum --strict --quiet -c SHA256SUMS)
done
(cd "$root" && sha256sum --strict --quiet -c training.sha256 training-authority.sha256)
(cd "$build" && sha256sum --strict --quiet -c SHA256SUMS authority.sha256)

mkdir "$stage"
cp "$here/fresh-design.json" "$stage/design.json"
cp "$history" "$stage/history.json"
"$generator" --contrast "$codec" "$d11" "$stage/history.json" "$stage/design.json" \
    "$stage/generated" > "$stage/generator.txt"
cp "$stage/generated/reference.json" "$stage/reference.json"
jq -ce 'map(.turns[1]) | select(all(.[]; type == "string"))' \
    "$stage/reference.json" > "$stage/user-only.json"
"$exporter" pack "$stage/user-only.json" "$stage/history.json" "$d11" "$codec" "$stage/endpoint"
"$exporter" pack-ordered "$stage/user-only.json" "$stage/history.json" "$d11" "$codec" "$stage/ordered"
cmp "$stage/endpoint/coordinates.jsonl" "$stage/ordered/coordinates.jsonl"
"$compare" compare-contrast "$stage/reference.json" "$stage/endpoint/text.bin" \
    "$stage/ordered/text.bin" > "$stage/matched.json"
jq -e '.pairs == 128 and .prompt_rows == 256 and
    .matched_edges_with_full_endpoint_alias_and_ordered_change == 256' "$stage/matched.json" >/dev/null
jq -ce -f "$here/fresh-census.jq" "$stage/reference.json" > "$stage/census.json"

earlier=(
    "$precedent/panel-v1/"{held,contrast}/reference.json
    "$median/"{held,contrast,mixed}/reference.json
    "$switches/"{historical,train-derived}/reference.json
    "$root/regression/reference.json"
)
jq -ce --slurpfile prior <(jq -s 'add' "${earlier[@]}") \
    -f "$here/provenance.jq" "$stage/reference.json" > "$stage/provenance.json"

printf '[]\n' > "$stage/no-historical-losses.json"
jq -ce --arg cohort fresh --slurpfile coordinates "$stage/ordered/coordinates.jsonl" \
    --slurpfile lost "$stage/no-historical-losses.json" -f "$here/enroll.jq" \
    "$stage/reference.json" > "$stage/rows.jsonl"
jq -ce '.requests[]' "$stage/rows.jsonl" > "$stage/requests.jsonl"
jq -er '[.row_index,.sign,.governor_word_index,.requested_site_index,.rival_site_index] | @tsv' \
    "$stage/requests.jsonl" > "$stage/requests.tsv"
jq -ser --argjson pairs 128 -f "$root/methods-v2/fit.jq" "$stage/requests.jsonl" \
    > "$stage/pair-index.tsv"
jq -se 'group_by(.pair_id) | length == 128 and all(.[];
    length == 2 and .[0].coordinate_map == .[1].coordinate_map)' "$stage/rows.jsonl" >/dev/null
jq -ce -f "$here/fresh-gold.jq" "$stage/rows.jsonl" > "$stage/gold.jsonl"

jq -ce -f "$here/fresh-controls.jq" "$stage/reference.json" > "$stage/swapped-reference.json"
for lane in gold swapped; do
    reference="$stage/reference.json"
    if [[ $lane == swapped ]]; then reference="$stage/swapped-reference.json"; fi
    "$exporter" supervise "$reference" "$stage/ordered/coordinates.jsonl" \
        "$stage/$lane-links.bin" "$stage/$lane-aid.bin"
    [[ $(wc -c < "$stage/$lane-links.bin") == 4108 && $(head -c8 "$stage/$lane-links.bin") == UORLNK01 ]]
    jq -se --arg lane "$lane" --rawfile bytes <(od -An -v -tu4 -j8 -w4 "$stage/$lane-links.bin") '
        ([length] + [ .[] | .governors as [$a,$e] |
            (if $lane == "gold" then .sites else .swapped_sites end) as [$as,$es,$h] |
            $a,$as,$e,$es ]) == [$bytes | scan("[0-9]+") | tonumber]' "$stage/gold.jsonl" >/dev/null
done
if cmp -s "$stage/"{gold,swapped}-links.bin; then
    echo "gold and same-spelling swap links are identical" >&2
    exit 1
fi

sha256sum "$here/"{prepare-fresh-wick.sh,fresh-design.json,enroll.jq,fresh-census.jq,fresh-controls.jq,fresh-gold.jq,provenance.jq} \
    "$root/methods-v2/fit.jq" "$build/"{SHA256SUMS,authority.sha256} "$generator" "$exporter" "$compare" \
    "$codec" "$d11" "$history" "${earlier[@]}" > "$stage/authority.sha256"
(
    cd "$stage"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum --strict --quiet -c SHA256SUMS authority.sha256
)
chmod -R a-w "$stage"
mv -T --no-clobber "$stage" "$root/fresh"
test ! -e "$stage"
(cd "$root/fresh" && sha256sum --strict --quiet -c SHA256SUMS authority.sha256)
sha256sum "$root/fresh/"{SHA256SUMS,authority.sha256}
