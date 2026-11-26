#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-switch-source-competition"
root="$study/matched-train"
primary="$study/panel/panel-v1"
base_methods="$study/panel/methods-v5"
parent_build="$runs/20261126-codex-all-outranker-interval/panel/build-v3"
ordered="$parent_build/target/release/source-link-text"
selector="$parent_build/target/release/governor-links-panel"
parser="$runs/20261126-codex-fixed-assignment-median/audit/build-v6/src/packet.rs"
codec="$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
model="$runs/20261125-codex-governing-source-pairs/models/fold-8/ordered-all.d11"
here=$(cd "$(dirname "$0")" && pwd -P)
version=${1:?unused positive panel version}
[[ $# == 1 && $version =~ ^[1-9][0-9]*$ ]]
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$root") == zfs && $here == "$root"/methods-v* ]]
final="$root/panel-v$version"
work="$root/prepare-v$version"
[[ ! -e $final && ! -e $work ]]
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --quiet --strict -c -; }

pin 8f46a07da62c7243cf2d5965c460ce2ba3f6484c1fa7266b77ecb7a4f0f09d92 "$primary/SHA256SUMS"
(cd "$primary" && sha256sum --quiet --strict -c SHA256SUMS authority.sha256 train-derived.sha256)
(cd "$parent_build" && sha256sum --quiet --strict -c SHA256SUMS authority.sha256)
pin b125d17182c8751dcaa06bf1d3b268544b6b0757800c546b806fa69f08ff51a9 "$parser"
mkdir "$work"
stage="$work/panel"
mkdir "$stage"
cp "$primary/history.json" "$stage/"
cp "$parser" "$work/packet.rs"
cp "$here/slice.rs" "$work/"
rustc -D warnings --edition 2021 "$work/slice.rs" -o "$work/slice"
rustc --version > "$stage/rustc.txt"

jq -ce '
    (.wordings[] | select(.family=="planning-review").contrast.after) |=
        if (split(" put in the red file,")|length)==2 and
           (split(", when the small board meets,")|length)==2 then
            sub(" put in the red file,"; " in the file,") |
            sub(", when the small board meets,"; ", when the board meets,")
        else error("TRAIN frame has changed") end
' "$primary/train-derived/design.json" > "$stage/design.json"
"$selector" --contrast "$codec" "$model" "$stage/history.json" "$stage/design.json" \
    "$work/generated" > "$stage/generator.txt"
jq -ce 'map(.turns[1])' "$work/generated/reference.json" > "$work/users.json"
"$ordered" pack "$work/users.json" "$stage/history.json" "$model" "$codec" "$work/endpoint"
"$ordered" pack-ordered "$work/users.json" "$stage/history.json" "$model" "$codec" "$work/ordered"
cmp "$work/endpoint/coordinates.jsonl" "$work/ordered/coordinates.jsonl"
"$ordered" compare-contrast "$work/generated/reference.json" "$work/endpoint/text.bin" \
    "$work/ordered/text.bin" > "$work/matched.json"

jq -ce '
    [.records[] | select(.family=="planning-review" and .governor_order=="after")] |
    select(length==4 and all(.[]; [.edges[].gap_bpe]==[25,18])) |
    map(.pair_id = ("matched-train/" + .pair_id)) |
    {pairs:length, prompt_rows:(length*2), matched_edges_with_full_endpoint_alias_and_ordered_change:(length*2),
     records:.}
' "$work/matched.json" > "$stage/matched.json"
jq -re 'to_entries[] | select(.value.family=="planning-review" and .value.governor_order=="after") | .key' \
    "$work/generated/reference.json" > "$stage/source-indices.txt"
jq -ce '
    map(select(.family=="planning-review" and .governor_order=="after") |
        (.pair_id,.admission_pair,.distance_pair,.fact_horizon_pair,.neutral_pair,.role_pair,.order_pair) |=
            ("matched-train/" + .)) |
    select(length == 8 and (map(.admission)|unique)==["defer","take"] and (map(.fact)|unique)==["road","school"])
' "$work/generated/reference.json" > "$stage/reference.json"
jq -ce --slurpfile primary "$primary/train-derived/reference.json" -f "$here/deletion.jq" \
    "$stage/reference.json" > "$stage/deletion.json"
jq -ce 'map(.turns[1])' "$stage/reference.json" > "$stage/user-only.json"
for representation in endpoint ordered; do
    command=pack
    if [[ $representation == ordered ]]; then command=pack-ordered; fi
    "$ordered" "$command" "$stage/user-only.json" "$stage/history.json" "$model" "$codec" "$stage/$representation"
    "$work/slice" "$work/$representation/text.bin" "$stage/$representation/text.bin" \
        "$stage/source-indices.txt" > "$stage/$representation-subset.json"
done
cmp "$stage/endpoint/coordinates.jsonl" "$stage/ordered/coordinates.jsonl"
jq -ce --arg cohort train-derived --slurpfile coordinates "$stage/ordered/coordinates.jsonl" \
    --slurpfile lost "$primary/no-historical-losses.json" -f "$base_methods/enroll.jq" \
    "$stage/reference.json" > "$stage/rows.jsonl"
jq -ce '.requests[]' "$stage/rows.jsonl" > "$stage/requests.jsonl"
jq -er '[.row_index,.sign,.governor_word_index,.requested_site_index,.rival_site_index] | @tsv' \
    "$stage/requests.jsonl" > "$stage/requests.tsv"
jq -se --arg cohort matched-train --argjson expected 8 -f "$base_methods/census.jq" \
    "$stage/rows.jsonl" > "$stage/census.json"
jq -sce 'group_by(.pair_id) | map(. as [$a,$b] |
    select(length==2 and $a.selector=="first" and $b.selector=="later" and $a.sign==$b.sign and
        $a.governor==$b.governor and $a.opposite_governor==$b.opposite_governor and
        $a.requested==$b.rival and $a.rival==$b.requested and $a.opposite==$b.opposite and $a.heading==$b.heading) |
    {pair_id:$a.pair_id,sign:$a.sign,governor:$a.governor,first:$a.requested,later:$b.requested,
     opposite:$a.opposite,heading:$a.heading}) | select(length==4)' \
    "$stage/requests.jsonl" > "$stage/addresses.json"
sha256sum "$here/"{prepare-wick.sh,slice.rs,deletion.jq} "$base_methods/"{enroll.jq,census.jq} \
    "$primary/"{SHA256SUMS,train-derived.sha256} "$parent_build/"{SHA256SUMS,authority.sha256} \
    "$ordered" "$selector" "$parser" "$codec" "$model" > "$stage/authority.sha256"
(
    cd "$stage"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum --quiet --strict -c SHA256SUMS authority.sha256
)
mv -T --no-clobber "$stage" "$final"
test ! -e "$stage"
chmod -R a-w "$final"
sha256sum "$final/"{SHA256SUMS,authority.sha256,matched.json,ordered/text.bin,requests.tsv}
