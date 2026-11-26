#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-third-margin-retain/panel"
previous="$runs/20261126-codex-paired-occurrence-margin/panel"
precedent="$runs/20261126-codex-all-outranker-interval/panel"
median="$runs/20261126-codex-fixed-assignment-median/panel/panel-v1"
switches="$runs/20261126-codex-switch-source-competition/panel/panel-v2"
codec="$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
d11="$runs/20261125-codex-governing-source-pairs/models/fold-8/ordered-all.d11"
here=$(cd "$(dirname "$0")" && pwd)
build_version=${1:?existing positive generator build version}
attempt=${2:?unused positive preparation attempt}
[[ $# == 2 && $build_version =~ ^[1-9][0-9]*$ && $attempt =~ ^[1-9][0-9]*$ ]]
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$root") == zfs && $here == "$root"/methods-v* ]]
stage="$root/panel-v$attempt.partial"
build="$root/generator-build-v$build_version"
generator="$build/target/release/governor-links-panel"
compare="$previous/generator-build-v3/target/release/source-link-text"
exporter="$precedent/build-v3/target/release/source-link-text"
helpers="$previous/methods-v8"
indexer="$previous/methods-v2/fit.jq"
history="$previous/history.json"
targets=(train fresh near regression history.json guards.sha256 training.sha256)
for item in "$stage" "${targets[@]/#/$root/}"; do [[ ! -e $item ]]; done

pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
pin 5c843187d34c3a6ca94560a161c8d82c3655d16dd65e283c36753d1e9f55e7b8 "$d11"
pin a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$codec"
pin 9cf2c96fcb088bbcf6b7132f8a8845c1a04c05150cfb06fc160c5f0ee57b6dc7 "$exporter"
pin 92704b9d3ffb8201245a26fe05c5648e29360d0a383dd0db9ccb1389f845eb4b "$precedent/panel-v1/SHA256SUMS"
pin 80bbfaa53532c8e113000445285721c1e52d6f060b51a880a7c17681980c4a93 "$median/SHA256SUMS"
pin 8f46a07da62c7243cf2d5965c460ce2ba3f6484c1fa7266b77ecb7a4f0f09d92 "$switches/SHA256SUMS"
pin be6a9c7d086036b5eb5a885302aa7537d584e83b8c3b063ce60dd8c9a4edf02e "$previous/training.sha256"
pin 34699010b8cf175a8b3544e97088d35f0b844dc226e4ecda0693fb8c10382e30 "$previous/fresh/SHA256SUMS"
for directory in "$precedent/panel-v1" "$median" "$switches"; do
    (cd "$directory" && sha256sum --strict --quiet -c SHA256SUMS)
done
(cd "$previous" && sha256sum --strict --quiet -c training.sha256 training-authority.sha256)
(cd "$previous/fresh" && sha256sum --strict --quiet -c SHA256SUMS authority.sha256)
(cd "$previous/generator-build-v3" && sha256sum --strict --quiet -c SHA256SUMS authority.sha256)
(cd "$build" && sha256sum --strict --quiet -c SHA256SUMS authority.sha256)

mkdir "$stage"
cp "$history" "$stage/"
cp -R "$previous/"{near,regression} "$stage/"
for guard in near regression; do
    diff -qr "$previous/$guard" "$stage/$guard"
done
[[ $(wc -l < "$stage/near/guard.tsv") == 48 ]]
[[ $(wc -l < "$stage/regression/regression-index.tsv") == 8 ]]
jq -se '(map(.pair_id)|unique|length) == 24' "$stage/near/gold.jsonl" >/dev/null
jq -se '(map(.pair_id)|unique|length) == 4' "$stage/regression/requests.jsonl" >/dev/null
(
    cd "$stage"
    find near regression -type f -print0 | sort -z | xargs -0 sha256sum > guards.sha256
    sha256sum --strict --quiet -c guards.sha256
)
(cd "$previous" && sha256sum --strict --quiet -c "$stage/guards.sha256")

earlier=(
    "$precedent/panel-v1/"{held,contrast}/reference.json
    "$median/"{held,contrast,mixed}/reference.json
    "$switches/"{historical,train-derived}/reference.json
    "$previous/"{regression,fresh}/reference.json
)

for split in train fresh; do
    packet="$stage/$split"
    mkdir "$packet"
    if [[ $split == train ]]; then
        facts=4
        compare_rows=1288
        provenance=("${earlier[@]}")
    else
        facts=8
        compare_rows=1416
        provenance=("${earlier[@]}" "$stage/train/reference.json")
    fi
    pairs=$((facts * 16))
    rows=$((pairs * 2))
    cp "$here/$split-design.json" "$packet/design.json"
    cp "$history" "$packet/history.json"
    "$generator" --contrast "$codec" "$d11" "$packet/history.json" "$packet/design.json" \
        "$packet/generated" > "$packet/generator.txt"
    jq -c --arg split "$split" 'map(.cohort = $split)' "$packet/generated/reference.json" > "$packet/reference.json"
    jq -ce 'map(.turns[1]) | select(all(.[]; type == "string"))' \
        "$packet/reference.json" > "$packet/user-only.json"
    "$exporter" pack "$packet/user-only.json" "$packet/history.json" "$d11" "$codec" "$packet/endpoint"
    "$exporter" pack-ordered "$packet/user-only.json" "$packet/history.json" "$d11" "$codec" "$packet/ordered"
    cmp "$packet/endpoint/coordinates.jsonl" "$packet/ordered/coordinates.jsonl"
    "$compare" compare-contrast "$packet/reference.json" "$packet/endpoint/text.bin" \
        "$packet/ordered/text.bin" > "$packet/matched.json"
    jq -e --argjson pairs "$pairs" --argjson rows "$rows" '.pairs == $pairs and .prompt_rows == $rows and
        .matched_edges_with_full_endpoint_alias_and_ordered_change == $rows' "$packet/matched.json" >/dev/null
    jq -ce --argjson expected "$compare_rows" --slurpfile prior <(jq -s 'add' "${provenance[@]}") \
        -f "$here/provenance.jq" "$packet/reference.json" > "$packet/provenance.json"

    printf '[]\n' > "$packet/no-historical-losses.json"
    jq -ce --arg cohort "$split" --slurpfile coordinates "$packet/ordered/coordinates.jsonl" \
        --slurpfile lost "$packet/no-historical-losses.json" -f "$helpers/enroll.jq" \
        "$packet/reference.json" > "$packet/rows.jsonl"
    jq -ce '.requests[]' "$packet/rows.jsonl" > "$packet/requests.jsonl"
    jq -er '[.row_index,.sign,.governor_word_index,.requested_site_index,.rival_site_index] | @tsv' \
        "$packet/requests.jsonl" > "$packet/requests.tsv"
    jq -ser --argjson pairs "$pairs" -f "$indexer" "$packet/requests.jsonl" > "$packet/pair-index.tsv"
    if [[ $split == train ]]; then cp "$packet/pair-index.tsv" "$packet/fit.tsv"; fi
    jq -se --argjson pairs "$pairs" 'group_by(.pair_id) | length == $pairs and all(.[]; length == 2
        and .[0].coordinate_map == .[1].coordinate_map)' "$packet/rows.jsonl" >/dev/null
    jq -ce -f "$helpers/fresh-gold.jq" "$packet/rows.jsonl" > "$packet/gold.jsonl"
    jq -sce --slurpfile refs "$packet/reference.json" --slurpfile rows "$packet/rows.jsonl" \
        -f "$here/eligibility.jq" "$packet/gold.jsonl" > "$packet/eligibility.jsonl"
    jq -sce --argjson facts "$facts" -f "$here/census.jq" \
        "$packet/eligibility.jsonl" > "$packet/census.json"
    if [[ $split == train ]]; then
        jq -er 'select(.affirmative.text == "retain") | [.row_index,.affirmative.i] | @tsv' \
            "$packet/eligibility.jsonl" > "$packet/retain.tsv"
        jq -er 'select(.sign == 1 and .affirmative.text == "retain") | [.row_index,.affirmative.i] | @tsv' \
            "$packet/eligibility.jsonl" > "$packet/retain-request.tsv"
        jq -er 'select(.affirmative.text == "retain") | [.row_index,.affirmative.i,-1] | @tsv' \
            "$packet/eligibility.jsonl" > "$packet/retain-wrong-sign.tsv"
        jq -er 'select(.affirmative.text == "retain") | [.row_index,.excluded.i] | @tsv' \
            "$packet/eligibility.jsonl" > "$packet/retain-wrong-word.tsv"
        [[ $(wc -l < "$packet/retain.tsv") == 64 && $(wc -l < "$packet/retain-request.tsv") == 32 ]]
        cmp <(cut -f1,2 "$packet/retain-wrong-sign.tsv") "$packet/retain.tsv"
        if cmp -s "$packet/retain.tsv" "$packet/retain-wrong-word.tsv"; then
            echo "wrong-governor control duplicates the original positive ordinals" >&2
            exit 1
        fi
    fi

    jq -ce -f "$helpers/fresh-controls.jq" "$packet/reference.json" > "$packet/swapped-reference.json"
    verify() {
        jq -se --arg lane "$1" \
            --rawfile links <(od -An -v -tu4 -j8 -w4 "$packet/$2-links.bin") \
            --rawfile aid <(od -An -v -tu4 -j8 -w4 "$packet/$2-aid.bin") \
            -f "$here/verify-aid.jq" "$packet/gold.jsonl" >/dev/null
    }
    for lane in gold swapped; do
        reference="$packet/reference.json"
        if [[ $lane == swapped ]]; then reference="$packet/swapped-reference.json"; fi
        "$exporter" supervise "$reference" "$packet/ordered/coordinates.jsonl" \
            "$packet/$lane-links.bin" "$packet/$lane-aid.bin"
        [[ $(head -c8 "$packet/$lane-links.bin") == UORLNK01 ]]
        [[ $(head -c8 "$packet/$lane-aid.bin") == UORAID01 ]]
        [[ $(wc -c < "$packet/$lane-links.bin") == $((12+16*rows)) ]]
        [[ $(wc -c < "$packet/$lane-aid.bin") == $((12+24*rows)) ]]
        verify "$lane" "$lane"
    done
    if verify gold swapped 2> "$packet/swapped-as-gold.rejection.txt"; then
        echo "canonical same-spelling corrupted source passed the gold oracle" >&2
        exit 1
    fi
    rg -F 'original word/site sequence disagrees with the registered lane' \
        "$packet/swapped-as-gold.rejection.txt" >/dev/null

    # Only prior, frozen inputs use absolute authority paths here. The separate
    # fresh provenance binds the current TRAIN by a path valid after publication.
    sha256sum "$here/"{prepare-wick.sh,eligibility.jq,census.jq,verify-aid.jq,provenance.jq} \
        "$here/$split-design.json" "$helpers/"{enroll.jq,fresh-gold.jq,fresh-controls.jq} \
        "$indexer" "$build/"{SHA256SUMS,authority.sha256} "$generator" "$exporter" "$compare" \
        "$codec" "$d11" "$history" "${earlier[@]}" > "$packet/authority.sha256"
    if [[ $split == fresh ]]; then
        (cd "$packet" && sha256sum ../train/reference.json) >> "$packet/authority.sha256"
    fi
    (
        cd "$packet"
        find . -type f -print0 | sort -z | xargs -0 sha256sum > "$stage/$split.sha256.partial"
        mv -T "$stage/$split.sha256.partial" SHA256SUMS
        sha256sum --strict --quiet -c SHA256SUMS authority.sha256
    )
done

(
    cd "$stage"
    find train near regression -type f -print0 | sort -z | xargs -0 sha256sum > training.sha256
    sha256sum history.json guards.sha256 >> training.sha256
    sha256sum --strict --quiet -c training.sha256 guards.sha256
)
for item in "${targets[@]}"; do
    if [[ -d $stage/$item ]]; then chmod u+w "$stage/$item"; fi
    mv -T --no-clobber "$stage/$item" "$root/$item"
    [[ ! -e $stage/$item ]]
    chmod -R a-w "$root/$item"
done
(cd "$root" && sha256sum --strict --quiet -c training.sha256 guards.sha256)
for split in train fresh; do
    (cd "$root/$split" && sha256sum --strict --quiet -c SHA256SUMS authority.sha256)
done
sha256sum "$root/"{training.sha256,guards.sha256,train/SHA256SUMS,fresh/SHA256SUMS}
