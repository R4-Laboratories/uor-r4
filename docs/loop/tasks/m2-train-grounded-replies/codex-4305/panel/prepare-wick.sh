#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-governor-argument-sites/panel"
predecessor="$runs/20261126-codex-third-margin-retain"
cold="$predecessor/storage/cold-v1/packet"
paired="$runs/20261126-codex-paired-occurrence-margin/panel"
interval="$runs/20261126-codex-all-outranker-interval/panel"
median="$runs/20261126-codex-fixed-assignment-median/panel/panel-v1"
switches="$runs/20261126-codex-switch-source-competition/panel/panel-v2"
d11="$cold/inputs/ordered-all.d11"
codec="$cold/inputs/tokenizer.json"
history="$cold/panel/history.json"
generator="$predecessor/panel/generator-build-v1/target/release/governor-links-panel"
exporter="$interval/build-v3/target/release/source-link-text"
compare="$paired/generator-build-v3/target/release/source-link-text"
helpers="$paired/methods-v8"
indexer="$paired/methods-v2/fit.jq"
prior_methods="$cold/methods/panel"
here=$(cd "$(dirname "$0")" && pwd -P)
attempt=${1:?unused positive preparation attempt}
[[ $# == 1 && $attempt =~ ^[1-9][0-9]*$ && $(hostname -s) == wick &&
    $(findmnt -no FSTYPE --target "$root") == zfs && $here == "$root"/methods-v* ]]
stage="$root/attempt-v$attempt.partial"
targets=(fresh regression near guards.sha256 authority.sha256 PANELS.sha256)
for item in "$stage" "${targets[@]/#/$root/}"; do [[ ! -e $item ]]; done

pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
pin 048d01e22cc4719d0d3eaeaafe19eb4d216761f5ac887bce804ef125066db47d "$cold/MEMBERS.sha256"
pin 9426c9251237fec1a856e02ca187d0bde392f7038a2b2d81a1b4c152af526916 "$cold/panel/guards.sha256"
pin 45731c74f2141da030adf0fb3920a601dd9edf2c0516cb2962b538fe194eb389 "$history"
pin 5c843187d34c3a6ca94560a161c8d82c3655d16dd65e283c36753d1e9f55e7b8 "$d11"
pin a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$codec"
pin 2e3596907881eabd5d894c398c047ff63009146bea68a9b414993e6110edd9d1 "$generator"
pin 9cf2c96fcb088bbcf6b7132f8a8845c1a04c05150cfb06fc160c5f0ee57b6dc7 "$exporter"
pin 9f57bbc2d47bbc2949ed0a5c16d8fae1e810f5719c9cb14ac729ccf56b858586 "$compare"
(cd "$cold/panel" && sha256sum --strict --quiet -c guards.sha256)
(cd "$cold" && awk '$2 ~ /^\.?(\/)?panel\/(train|fresh)\/reference\.json$/' MEMBERS.sha256 | sha256sum --strict --quiet -c -)

mkdir "$stage"
cp -R "$cold/panel/"{near,regression} "$stage/"
cp "$cold/panel/guards.sha256" "$stage/"
(cd "$stage" && sha256sum --strict --quiet -c guards.sha256)
for guard in near regression; do chmod u+w "$stage/$guard"; done

fmt='[.row_index,.sign,.governor.index,.wrong_governor.index,(.good|length),.good[].index,(.wrong|length),.wrong[].index]|map(tostring)|join(" ")'
mask() {
    local cohort=$1 expected=$2
    local packet="$stage/$cohort"
    jq -ce --arg cohort "$cohort" --slurpfile refs "$packet/reference.json" \
        --slurpfile guard <(jq -s . "$stage/near/gold.jsonl") \
        --slurpfile grammar "$here/grammar.json" --slurpfile design "$here/fresh-design.json" \
        -f "$here/mask-evidence.jq" "$packet/rows.jsonl" > "$packet/mask-evidence.jsonl"
    jq -er "$fmt" "$packet/mask-evidence.jsonl" > "$packet/masks.tsv"
    jq -se --argjson expected "$expected" --slurpfile guard "$stage/near/gold.jsonl" 'length == $expected and
        map(.row_index) == [range($expected)] and
        all(.[]; . as $row | .guard == (.cohort == "regression" or
            (.cohort == "near" and any($guard[]; .row_index == $row.row_index))))' \
        "$packet/mask-evidence.jsonl" >/dev/null
    if [[ $cohort != near ]]; then
        jq -se --argjson pairs "$((expected / 2))" 'group_by(.pair_id) |
            length == $pairs and all(.[]; length == 2 and
                (.[0].selector == "first" and .[1].selector == "later") and
                (.[0] | {sign,governor,wrong_governor,good,wrong}) ==
                (.[1] | {sign,governor,wrong_governor,good,wrong}))' \
            "$packet/mask-evidence.jsonl" >/dev/null
    fi
    jq -sc 'group_by(.family) | map({
        family:.[0].family, rows:length, guard_rows:(map(select(.guard))|length),
        good_sizes:(map(.good|length)|unique),
        legal_site_count:(map([.good[]|select(.legal_with_governor > 0)]|length)|[min,max]),
        independent_alternate_tuple_counts:
            ([.[].good[]|select(.role == "independent_alternative")|.legal_with_governor]|[min,max]),
        independent_alternates_with_original_opposite_and_alternate_heading:
            ([.[].good[]|select(.role == "independent_alternative" and
                .completes_with_original_opposite_and_alternate_heading)]|length)
    })' "$packet/mask-evidence.jsonl" > "$packet/mask-census.json"
}

# Check in each enrolled original near row that its registered contested sign
# selects the same exact physical governor and source as archived full labels.
jq -se --slurpfile guard "$stage/near/gold.jsonl" '
    length == 256 and ($guard|length) == 48 and
    ([$guard[].row_index] | unique | length) == 48 and
    ([$guard[].pair_id] | group_by(.) | length == 24 and all(.[]; length == 2))' \
    "$stage/near/rows.jsonl" >/dev/null
jq -se --slurpfile guard "$stage/near/gold.jsonl" '. as $rows |
    all($guard[]; . as $g | $rows[$g.row_index] as $row | $row.requests[0] as $r |
        (if $r.sign == 1 then 0 else 1 end) as $slot |
        $row.pair_id == $g.pair_id and $g.governors[$slot] == $r.governor.index and
        $g.sites[$slot] == $r.requested.index and $g.governors[1-$slot] == $r.opposite_governor.index and
        $g.sites[1-$slot] == $r.opposite.index and $g.sites[2] == $r.heading.index)' \
    "$stage/near/rows.jsonl" >/dev/null
cmp "$stage/near/guard.tsv" <(jq -er '[.row_index]+.governors+.sites|@tsv' "$stage/near/gold.jsonl")

fresh="$stage/fresh"
mkdir "$fresh"
cp "$here/"{fresh-design.json,grammar.json} "$fresh/"
cp "$history" "$fresh/history.json"
"$generator" --contrast "$codec" "$d11" "$history" "$fresh/fresh-design.json" "$stage/generated" \
    > "$fresh/generator.txt"
mv "$stage/generated/reference.json" "$fresh/reference.json"
jq -ce 'map(.turns[1]) | select(all(.[]; type == "string"))' "$fresh/reference.json" > "$fresh/user-only.json"
"$exporter" pack "$fresh/user-only.json" "$history" "$d11" "$codec" "$fresh/endpoint"
"$exporter" pack-ordered "$fresh/user-only.json" "$history" "$d11" "$codec" "$fresh/ordered"
cmp "$fresh/endpoint/coordinates.jsonl" "$fresh/ordered/coordinates.jsonl"
"$compare" compare-contrast "$fresh/reference.json" "$fresh/endpoint/text.bin" "$fresh/ordered/text.bin" \
    > "$fresh/matched.json"
jq -e '.pairs == 128 and .prompt_rows == 256 and
    .matched_edges_with_full_endpoint_alias_and_ordered_change == 256' "$fresh/matched.json" >/dev/null

old_refs=(
    "$interval/panel-v1/"{held,contrast}/reference.json
    "$median/"{held,contrast,mixed}/reference.json
    "$switches/"{historical,train-derived}/reference.json
    "$paired/"{regression,fresh}/reference.json
    "$cold/panel/"{train,fresh}/reference.json
)
jq -s 'add' "${old_refs[@]}" > "$stage/earlier.json"
jq -ce --argjson expected 1672 --slurpfile prior "$stage/earlier.json" -f "$prior_methods/provenance.jq" \
    "$fresh/reference.json" > "$fresh/provenance.json"
jq -ce --arg cohort fresh --slurpfile coordinates "$fresh/ordered/coordinates.jsonl" \
    --slurpfile lost <(printf '[]\n') -f "$helpers/enroll.jq" "$fresh/reference.json" > "$fresh/rows.jsonl"
jq -ce '.requests[]' "$fresh/rows.jsonl" > "$fresh/requests.jsonl"
jq -er '[.row_index,.sign,.governor_word_index,.requested_site_index,.rival_site_index] | @tsv' \
    "$fresh/requests.jsonl" > "$fresh/requests.tsv"
jq -ser --argjson pairs 128 -f "$indexer" "$fresh/requests.jsonl" > "$fresh/pair-index.tsv"
jq -se 'group_by(.family,.fact,.governor_order,.admission) | length == 128 and
    all(.[]; length == 2 and .[0].coordinate_map == .[1].coordinate_map)' "$fresh/rows.jsonl" >/dev/null
jq -e 'group_by(.family) | length == 4 and all(.[]; length == 64 and
    (map(.fact)|unique|length) == 8 and
    (group_by(.requested_role,.requested_horizon) | length == 4 and all(.[]; length == 16)))' \
    "$fresh/reference.json" >/dev/null

for cohort in regression fresh; do
    packet="$stage/$cohort"
    jq -ce -f "$helpers/fresh-gold.jq" "$packet/rows.jsonl" > "$packet/gold.jsonl"
    jq -ce -f "$helpers/fresh-controls.jq" "$packet/reference.json" > "$packet/swapped-reference.json"
    expected=$(wc -l < "$packet/gold.jsonl")
    verify() {
        jq -se --arg lane "$1" \
            --rawfile links <(od -An -v -tu4 -j8 -w4 "$packet/$2-links.bin") \
            --rawfile aid <(od -An -v -tu4 -j8 -w4 "$packet/$2-aid.bin") \
            -f "$prior_methods/verify-aid.jq" "$packet/gold.jsonl" >/dev/null
    }
    for lane in gold swapped; do
        reference="$packet/reference.json"
        if [[ $lane == swapped ]]; then reference="$packet/swapped-reference.json"; fi
        "$exporter" supervise "$reference" "$packet/ordered/coordinates.jsonl" \
            "$packet/$lane-links.bin" "$packet/$lane-aid.bin"
        [[ $(head -c8 "$packet/$lane-links.bin") == UORLNK01 &&
            $(head -c8 "$packet/$lane-aid.bin") == UORAID01 &&
            $(wc -c < "$packet/$lane-links.bin") == $((12+16*expected)) &&
            $(wc -c < "$packet/$lane-aid.bin") == $((12+24*expected)) ]]
        verify "$lane" "$lane"
    done
    if verify gold swapped 2> "$packet/swapped-as-gold.rejection.txt"; then
        echo "swapped address control incorrectly passed the gold oracle" >&2
        exit 1
    fi
    rg -F 'original word/site sequence disagrees with the registered lane' \
        "$packet/swapped-as-gold.rejection.txt" >/dev/null
    mask "$cohort" "$expected"
done
mask near 256
jq -ce --slurpfile prior "$stage/earlier.json" --slurpfile annotations "$fresh/mask-evidence.jsonl" '
    [ $prior[0][] | .turns[1] as $text | .gold_binding |
        (.affirmed,.excluded,.referenced) | $text[.start:.end] ] | unique as $old |
    ([ $annotations[].good[] | select(.role == "independent_alternative") | .text ] | unique) as $new |
    {fresh_explicit_alternative_surfaces:$new,
     shared_earlier_enrolled_surfaces:[$new[] as $surface | select($old|index($surface) != null) | $surface]} |
    select((.fresh_explicit_alternative_surfaces|length) == 4 and
        (.shared_earlier_enrolled_surfaces|length) == 0)' "$fresh/provenance.json" > "$fresh/alternative-provenance.json"

sha256sum "$here/"{fresh-design.json,grammar.json,mask-evidence.jq,prepare-wick.sh} \
    "$cold/MEMBERS.sha256" "$d11" "$codec" "$history" "$generator" "$exporter" "$compare" \
    "$prior_methods/"{provenance.jq,verify-aid.jq} "$helpers/"{enroll.jq,fresh-controls.jq,fresh-gold.jq} \
    "$indexer" "${old_refs[@]}" > "$stage/authority.sha256"
(
    cd "$stage"
    sha256sum --strict --quiet -c guards.sha256 authority.sha256
    for cohort in fresh regression near; do
        (cd "$cohort" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$cohort.sha256.partial"
        mv "$cohort.sha256.partial" "$cohort/SHA256SUMS"
        (cd "$cohort" && sha256sum --strict --quiet -c SHA256SUMS)
    done
    sha256sum fresh/SHA256SUMS regression/SHA256SUMS near/SHA256SUMS guards.sha256 authority.sha256 > PANELS.sha256
)
for item in "${targets[@]}"; do
    mv -T --no-clobber "$stage/$item" "$root/$item"
    [[ ! -e $stage/$item ]]
    chmod -R a-w "$root/$item"
done
(cd "$root" && sha256sum --strict --quiet -c PANELS.sha256 guards.sha256 authority.sha256)
for cohort in fresh regression near; do (cd "$root/$cohort" && sha256sum --strict --quiet -c SHA256SUMS); done
sha256sum "$root/"{PANELS.sha256,fresh/SHA256SUMS,regression/SHA256SUMS,near/SHA256SUMS}
