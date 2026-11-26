#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-fixed-assignment-median
root="$study/analysis"
here=$(cd "$(dirname "$0")" && pwd -P)
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$root") == zfs && $here == "$root/"* ]]
mode=${1:?train or held}; shift
train=$(realpath "${1:?sealed TRAIN capture}"); shift
if [[ $mode == held ]]; then held=$(realpath "${1:?sealed held capture}"); shift; fi
panel=$(realpath "${1:?released panel}")
methods=$(realpath "${2:?released methods}")
out="$root/${3:?new output directory basename}"
[[ $# == 3 && ${out##*/} =~ ^(train|held)-v[1-9][0-9]*$ && ${out##*/} == "$mode-"* ]]
[[ $train == "$study"/train-v* && $panel == "$study"/panel/panel-v* && $methods == "$study"/methods-v* ]]
if [[ $mode == held ]]; then [[ $held == "$study"/held-v* ]]; fi
mkdir "$out"

bind_file() (
    cd "$1"
    local file=$2
    grep -Fxq "$(sha256sum -- "./$file")" SHA256SUMS
    sha256sum -- "$1/$file"
)
bind() { bind_file "$1" "$2" >> "$out/inputs.sha256"; }
read -r release_hash release_manifest < "$train/inputs.sha256"
[[ $release_manifest == "$study"/release-v*/inputs.sha256 ]]
printf '%s  %s\n' "$release_hash" "$release_manifest" | sha256sum --strict --quiet -c -
grep -Fxq "$(sha256sum "$panel/SHA256SUMS")" "$release_manifest"
grep -Fxq "$(sha256sum "$methods/SHA256SUMS")" "$release_manifest"
sha256sum "$train/SHA256SUMS" "$panel/SHA256SUMS" "$methods/SHA256SUMS" "$release_manifest" \
    "$here/sources.jq" "$here/run-wick.sh" > "$out/inputs.sha256"
bind "$train" selection.json
bind "$train" inputs.sha256
bind "$methods" inputs/premise.json

case "$mode" in
train) splits=(mixed); bind "$methods" original-join.jq ;;
held)
    grep -Fxq "$(sha256sum "$train/SHA256SUMS")" "$held/inputs.sha256"
    grep -Fxq "$(sha256sum "$panel/SHA256SUMS")" "$held/inputs.sha256"
    sha256sum "$held/SHA256SUMS" >> "$out/inputs.sha256"
    bind "$held" inputs.sha256
    splits=(held contrast)
    ;;
*) exit 64 ;;
esac

compare() {
    local domain=$1 geometry=$2 from=$3 to=$4 raw_a=$5 raw_b=$6 joined_a=$7 joined_b=$8
    jq -ne -f "$here/sources.jq" --arg domain "$domain" --arg geometry "$geometry" --arg from "$from" --arg to "$to" \
        --slurpfile reference "$panel/$domain/reference.json" --slurpfile protected "$methods/inputs/premise.json" \
        --slurpfile before_raw "$raw_a" --slurpfile after_raw "$raw_b" \
        --slurpfile before "$joined_a" --slurpfile after "$joined_b"
}

verify_protected_bundle() {
    local lane=$1 state="states/$1.state" bridge="states/$1.bridge" unpacked="$out/$1.fallback.state"
    bind "$methods" inputs/protected.state
    bind "$methods" inputs/bundle
    bind "$train" "$state"
    bind "$train" "$bridge"
    bind "$train" "device/$lane/base.state"
    cmp "$methods/inputs/protected.state" "$train/device/$lane/base.state"
    cmp "$methods/inputs/protected.state" "$train/$state"
    [[ $("$methods/inputs/bundle" unpack "$train/$bridge" "$unpacked") == endpoint ]]
    cmp "$methods/inputs/protected.state" "$unpacked"
}

verify_held_lane() {
    local lane=$1 geometry=$2
    bind "$held" "unpacked/$lane.mode"
    bind "$held" "unpacked/$lane.state"
    bind "$train" "states/$lane.state"
    [[ $(cat "$held/unpacked/$lane.mode") == "$geometry" ]]
    cmp "$train/states/$lane.state" "$held/unpacked/$lane.state"
}

for geometry in endpoint ordered; do
    election=$(jq -er --arg arm "$geometry" '
        .[$arm] | select(.arm==$arm and (.qualified|type)=="boolean" and (.qualified or .chosen=="base")) |
        [.qualified,(.chosen | select(.=="base" or .=="full" or test("^slot-(0[0-9]|1[0-5])$")))] |
        select(length==2) | @tsv' "$train/selection.json")
    read -r qualified elected <<< "$election"
    printf '%s\n' "$election" > "$out/$geometry.election.tsv"
    lane_geometry=$geometry
    if [[ $qualified == false ]]; then
        verify_protected_bundle "$geometry"
        lane_geometry=endpoint
    fi
    if [[ $mode == held ]]; then
        verify_held_lane "$geometry" "$lane_geometry"
        verify_held_lane "${geometry}_full" "$geometry"
    fi
done

for domain in "${splits[@]}"; do
    bind "$panel" "$domain/reference.json"
    dest="$out/$domain"; mkdir "$dest"
    report_inputs=()
    for geometry in endpoint ordered; do
        read -r qualified elected < "$out/$geometry.election.tsv"
        if [[ $mode == train ]]; then
            bind "$panel" "$domain/blind/coordinates.jsonl"
            lanes=(frozen:base full:full)
            if [[ $qualified == true ]]; then lanes+=("elected:$elected"); fi
            for pair in "${lanes[@]}"; do
                name=${pair%%:*}; id=${pair##*:}
                bind "$train" "$geometry/$id.raw.jsonl"
                raw="$train/$geometry/$id.raw.jsonl"
                printf '%s\n' "$raw" > "$dest/$geometry-$name.raw-path"
                jq -nce --arg split mixed --arg kind learned --argjson row_count 256 \
                    --slurpfile reference "$panel/$domain/reference.json" \
                    --slurpfile coords "$panel/$domain/blind/coordinates.jsonl" --slurpfile physical "$raw" \
                    -f "$methods/original-join.jq" > "$dest/$geometry-$name.rows.jsonl"
            done
        else
            frozen=baseline; if [[ $geometry == ordered ]]; then frozen=ordered_base; fi
            lanes=("frozen:$frozen" "full:${geometry}_full")
            if [[ $qualified == true ]]; then lanes+=("elected:$geometry"); fi
            for pair in "${lanes[@]}"; do
                name=${pair%%:*}; lane=${pair##*:}
                bind "$held" "$domain/$lane.raw.jsonl"; bind "$held" "$domain/$lane.rows.jsonl"
                printf '%s\n' "$held/$domain/$lane.raw.jsonl" > "$dest/$geometry-$name.raw-path"
                printf '%s\n' "$held/$domain/$lane.rows.jsonl" > "$dest/$geometry-$name.rows-path"
            done
        fi
        raw_frozen=$(cat "$dest/$geometry-frozen.raw-path"); raw_full=$(cat "$dest/$geometry-full.raw-path")
        joined_frozen="$dest/$geometry-frozen.rows.jsonl"
        joined_full="$dest/$geometry-full.rows.jsonl"
        if [[ $mode == held ]]; then
            joined_frozen=$(cat "$dest/$geometry-frozen.rows-path"); joined_full=$(cat "$dest/$geometry-full.rows-path")
        fi
        compare "$domain" "$geometry" frozen full "$raw_frozen" "$raw_full" "$joined_frozen" "$joined_full" \
            > "$dest/$geometry-frozen-full.json"
        report_inputs+=("$dest/$geometry-frozen-full.json")
        if [[ $qualified == false ]]; then
            if [[ $mode == held ]]; then
                for format in raw rows; do
                    bind "$held" "$domain/$geometry.$format.jsonl"
                    bind "$held" "$domain/baseline.$format.jsonl"
                    cmp "$held/$domain/$geometry.$format.jsonl" "$held/$domain/baseline.$format.jsonl"
                done
            fi
            jq -n --arg mode "$mode" '{elected_candidate:null,training_diagnostic_candidate:"base",
                deployment_fallback:{geometry:"endpoint",state:"protected",training_bundle_verified:true} +
                    (if $mode=="held" then {held_reads_byte_identical_to_endpoint_baseline:true} else {} end)}' \
                > "$dest/$geometry-election.json"
            continue
        fi
        raw_elected=$(cat "$dest/$geometry-elected.raw-path")
        joined_elected="$dest/$geometry-elected.rows.jsonl"
        if [[ $mode == held ]]; then joined_elected=$(cat "$dest/$geometry-elected.rows-path"); fi
        if [[ $elected == full ]]; then
            cmp "$raw_full" "$raw_elected"; cmp "$joined_full" "$joined_elected"
            jq -n '{elected_candidate:"full",elected_reads_byte_identical_to_full:true}' > "$dest/$geometry-election.json"
        else
            jq -n --arg elected "$elected" '{elected_candidate:$elected}' > "$dest/$geometry-election.json"
            compare "$domain" "$geometry" frozen elected "$raw_frozen" "$raw_elected" "$joined_frozen" "$joined_elected" \
                > "$dest/$geometry-frozen-elected.json"
            compare "$domain" "$geometry" full elected "$raw_full" "$raw_elected" "$joined_full" "$joined_elected" \
                > "$dest/$geometry-full-elected.json"
            report_inputs+=("$dest/$geometry-frozen-elected.json" "$dest/$geometry-full-elected.json")
        fi
    done
    jq -es --arg domain "$domain" '{domain:$domain,arms:(group_by(.geometry)|map({geometry:.[0].geometry,comparisons:.}))}' \
        "${report_inputs[@]}" > "$dest/report.json"
    jq -e --slurpfile endpoint "$dest/endpoint-election.json" --slurpfile ordered "$dest/ordered-election.json" \
        '.arms |= map(. + (if .geometry=="endpoint" then $endpoint[0] else $ordered[0] end))' \
        "$dest/report.json" > "$dest/attribution.json"
done
jq -cs '{schema:"uor-r4.codex-4299.source-attribution/1",domains:.}' "$out/"*/attribution.json > "$out/report.json"
sort -u "$out/inputs.sha256" > "$out/unique-inputs.sha256"
(
    cd "$out"
    find . -type f ! -name .seal.pending -print0 | sort -z | xargs -0 -r sha256sum > .seal.pending
    mv .seal.pending SHA256SUMS
)
sha256sum "$out/SHA256SUMS" "$out/report.json"
