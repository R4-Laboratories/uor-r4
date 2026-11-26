#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-switch-source-competition/panel"
prior="$runs/20261126-codex-fixed-assignment-median"
prior_panel="$prior/panel/panel-v1"
build="$runs/20261126-codex-all-outranker-interval/panel/build-v3"
mixed_source="$runs/20261126-codex-mixed-transport/panel/private/a-v2.json"
codec="$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
model="$runs/20261125-codex-governing-source-pairs/models/fold-8/ordered-all.d11"
selector="$build/target/release/governor-links-panel"
exporter="$build/target/release/source-link-text"
here=$(cd "$(dirname "$0")" && pwd)
version=${1:?unused positive panel version}
[[ $# == 1 && $version =~ ^[1-9][0-9]*$ ]]
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$root") == zfs && $here == "$root"/methods-v* ]]
final="$root/panel-v$version"
stage="$final.partial"
[[ ! -e $final && ! -e $stage ]]

pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --quiet --strict -c -; }

pin 80bbfaa53532c8e113000445285721c1e52d6f060b51a880a7c17681980c4a93 "$prior_panel/SHA256SUMS"
pin 838d9da02091ccb336496b55cb69f8aeee303ed7d92d59b0a09c9c04b95c4ab1 "$mixed_source"
pin a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$codec"
pin 5c843187d34c3a6ca94560a161c8d82c3655d16dd65e283c36753d1e9f55e7b8 "$model"
(cd "$prior_panel" && sha256sum --quiet --strict -c SHA256SUMS blind.sha256 labels.sha256)
(cd "$build" && sha256sum --quiet --strict -c SHA256SUMS authority.sha256)
mkdir "$stage"
mkdir "$stage/"{historical,train-original,train-derived}
cp "$prior_panel/history.json" "$stage/"
copy_cohort() {
    local from=$1 to=$2
    cp -r "$from/"{reference.json,user-only.json,ordered} "$to/"
    cmp "$from/ordered/text.bin" "$to/ordered/text.bin"
    cmp "$from/ordered/coordinates.jsonl" "$to/ordered/coordinates.jsonl"
    cmp "$from/reference.json" "$to/reference.json"
}
copy_cohort "$prior_panel/mixed" "$stage/train-original"

# Bind the intervention grammar and packets without opening held rows or old reads.
derived="$stage/train-derived"
jq -ce --slurpfile mixed "$stage/train-original/reference.json" --slurpfile codec "$codec" \
    --slurpfile frames "$here/train-frames.json" -f "$here/train-design.jq" \
    "$mixed_source" > "$derived/design.json"
"$selector" --contrast "$codec" "$model" "$stage/history.json" "$derived/design.json" \
    "$derived/generated" > "$derived/generator.txt"
cp "$derived/generated/reference.json" "$derived/reference.json"
jq -ce 'map(.turns[1])' "$derived/reference.json" > "$derived/user-only.json"
"$exporter" pack "$derived/user-only.json" "$stage/history.json" "$model" "$codec" "$derived/endpoint"
"$exporter" pack-ordered "$derived/user-only.json" "$stage/history.json" "$model" "$codec" "$derived/ordered"
cmp "$derived/endpoint/coordinates.jsonl" "$derived/ordered/coordinates.jsonl"
"$exporter" compare-contrast "$derived/reference.json" "$derived/endpoint/text.bin" \
    "$derived/ordered/text.bin" > "$derived/matched.json"
jq -e '.pairs == 32 and .prompt_rows == 64 and
    .matched_edges_with_full_endpoint_alias_and_ordered_change == 64' \
    "$derived/matched.json" >/dev/null
(
    cd "$stage"
    sha256sum history.json train-derived/{design.json,user-only.json,reference.json,matched.json} \
        train-derived/{endpoint,ordered}/{text.bin,coordinates.jsonl} > train-derived.sha256
    sha256sum --quiet --strict -c train-derived.sha256
)

# Access the spent held reference only after the TRAIN intervention is fixed.
copy_cohort "$prior_panel/contrast" "$stage/historical"
cp "$prior_panel/contrast/matched.json" "$stage/historical/"
prior_report="$prior/analysis/held-v3/report.json"
pin 0503ccb26b6741f2d0646d55ed9d2c699e23bd412363b19a1c129b41e25358ca "$prior_report"
jq -ce '[.domains[] | select(.domain=="contrast") | .arms[] | select(.geometry=="ordered") |
    .comparisons[] | select(.from=="frozen" and .to=="full") | .signed_selectors[] |
    .role as $role | .lost[] | {role:$role,pair_id:.}] |
    select(length == 4)' "$prior_report" > "$stage/historical/lost.json"
jq -ne --slurpfile derived "$derived/reference.json" \
    --slurpfile original "$stage/train-original/reference.json" \
    --slurpfile historical "$stage/historical/reference.json" '
    def nouns: [.[] | .gold_binding as $gold | .turns[1] as $text |
        $gold | (.affirmed,.excluded,.referenced) | $text[.start:.end]] | unique;
    ($derived[0]|nouns) as $derived_nouns | ($original[0]|nouns) as $original_nouns |
    ($historical[0]|nouns) as $held_nouns |
    def overlap($a;$b): [$a[] | . as $value | select($b|index($value)!=null)];
    {
        derived_nouns:$derived_nouns, original_nouns:$original_nouns,
        shared_held_nouns:overlap($derived_nouns;$held_nouns),
        shared_held_fact_ids:overlap(($derived[0]|map(.fact)|unique);($historical[0]|map(.fact)|unique)),
        shared_held_frames:overlap(($derived[0]|map(.scope_clauses[])|unique);
            ($historical[0]|map(.scope_clauses[])|unique)),
        shared_original_prompt_ids:overlap(($derived[0]|map(.id));($original[0]|map(.id))),
        shared_held_prompt_ids:overlap(($derived[0]|map(.id));($historical[0]|map(.id)))
    } | select((.derived_nouns - .original_nouns | length) == 0 and
        ([.shared_held_nouns,.shared_held_fact_ids,.shared_held_frames,
         .shared_original_prompt_ids,.shared_held_prompt_ids] | all(.[]; length==0)))' \
    > "$derived/provenance.json"

printf '[]\n' > "$stage/no-historical-losses.json"
for cohort in train-original train-derived historical; do
    output="$stage/$cohort"
    expected=64
    lost="$stage/historical/lost.json"
    if [[ $cohort != historical ]]; then
        lost="$stage/no-historical-losses.json"
    fi
    if [[ $cohort == train-original ]]; then
        expected=256
    fi
    jq -ce --arg cohort "$cohort" --slurpfile coordinates "$output/ordered/coordinates.jsonl" \
        --slurpfile lost "$lost" -f "$here/enroll.jq" "$output/reference.json" > "$output/rows.jsonl"
    jq -ce '.requests[]' "$output/rows.jsonl" > "$output/requests.jsonl"
    jq -er '[.row_index,.sign,.governor_word_index,.requested_site_index,.rival_site_index] |
        @tsv' "$output/requests.jsonl" > "$output/requests.tsv"
    jq -se --arg cohort "$cohort" --argjson expected "$expected" -f "$here/census.jq" \
        "$output/rows.jsonl" > "$output/census.json"
done
jq -se --slurpfile lost "$stage/historical/lost.json" '
    [ .[] | select(.historical_lost) ] as $known |
    ($known|length)==8 and
    all($lost[0][]; . as $pair |
        [$known[] | select(.pair_id==$pair.pair_id and .role==$pair.role) | .selector] |
        sort == ["first","later"])' "$stage/historical/requests.jsonl" >/dev/null

sha256sum "$here/"{prepare-wick.sh,enroll.jq,census.jq,train-design.jq,train-frames.json} \
    "$prior_panel/"{SHA256SUMS,blind.sha256,labels.sha256} "$prior_report" "$codec" "$model" \
    "$mixed_source" "$build/"{SHA256SUMS,authority.sha256} "$selector" "$exporter" > "$stage/authority.sha256"
(
    cd "$stage"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum --quiet --strict -c SHA256SUMS authority.sha256 train-derived.sha256
)
chmod -R a-w "$stage"
mv -T --no-clobber "$stage" "$final"
test ! -e "$stage"
sha256sum "$final/"{SHA256SUMS,authority.sha256,train-derived.sha256}
