#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
mode=${1:?release, capture or inner}; v=${2:?new version}; version "$v"
case $mode in
release)
    panel=$(realpath -e "${3:?panel root}"); binary=$(realpath -e "${4:?SM89 binary}")
    finite=$(realpath -e "${5:?finite fixture dir}"); review=$(realpath -e "${6:?FIT dir}")
    [[ $# == 6 && $panel == "$study"/panel/panel-v* && $binary == "$study"/device/* &&
       $finite == "$study"/device/* && $review == "$study"/audit/* ]]
    for dir in "$here" "$panel" "${binary%/*}" "$finite" "$review"; do check_dir "$dir"; done
    physical_inputs
    grep -Fq 'TEST FITNESS: FIT' "$review/FIT.md"
    for file in "$here/SHA256SUMS" "$here/PRE-RUN.md" "$panel/SHA256SUMS" "$binary" "$finite/SHA256SUMS"; do
        grep -Fq "$(sha "$file")" "$review/FIT.md"
    done
    r="$study/release-$v"; mkdir "$r"
    cp "$here/"{ORDER,PRE-RUN,DECISION}.md "$review/FIT.md" "$r/"
    [[ $("$bundle" unpack "$prior/train-v1/states/ordered.bridge" "$r/ordered.unpacked") == ordered ]]
    cmp "$refit" "$r/ordered.unpacked"
    jq -n --arg panel "$panel" --arg binary "$binary" --arg finite "$finite" --arg review "$review" \
        '{panel:$panel,binary:$binary,finite:$finite,review:$review}' > "$r/paths.json"
    sha256sum "$r/"{ORDER.md,PRE-RUN.md,DECISION.md,FIT.md,paths.json,ordered.unpacked} "$here/SHA256SUMS" \
        "$panel/SHA256SUMS" "$binary" "${binary%/*}/SHA256SUMS" "$finite/SHA256SUMS" "$review/SHA256SUMS" \
        "$bank" "$original" "$refit" "$bundle" "$lease" "$prior/train-v1/SHA256SUMS" "$prior/held-v1/SHA256SUMS" \
        "$prior/panel/panel-v1/SHA256SUMS" > "$r/inputs.sha256"
    seal "$r"; sha256sum "$r/SHA256SUMS"
    ;;
capture|inner)
    r=$(realpath -e "${3:?sealed release}"); expected=${4:?release SHA}
    [[ $# == 4 && $r == "$study"/release-v* ]]
    pin "$expected" "$r/SHA256SUMS"; check_dir "$r"; sha256sum --strict --quiet -c "$r/inputs.sha256"
    grep -Fxq "$(sha256sum "$here/SHA256SUMS")" "$r/inputs.sha256"
    binary=$(jq -er .binary "$r/paths.json"); panel=$(jq -er .panel "$r/paths.json")
    check_dir "$panel"; check_dir "$here"
    m="$study/capture-$v"
    if [[ $mode == capture ]]; then exec bash "$lease" "$m-lease" bash "$0" inner "$v" "$r" "$expected"; fi
    [[ -e /proc/$$/fd/9 ]]; mkdir "$m"
    for cohort in historical train-original train-derived; do
        d="$m/$cohort"; mkdir "$d"; source_dir="$panel/$cohort"
        for state in before after; do
            payload=$original; if [[ $state == after ]]; then payload=$refit; fi
            /usr/bin/time -f 'wall_s=%e user_s=%U system_s=%S peak_rss_kib=%M' -o "$d/$state.time" \
                "$binary" "$bank" "$source_dir/ordered/text.bin" "$payload" "$source_dir/requests.tsv" "$d/$state"
            test "$(wc -l < "$d/$state/requests.jsonl")" = "$(wc -l < "$source_dir/requests.tsv")"
            test "$(wc -l < "$d/$state/native.jsonl")" = "$(wc -l < "$source_dir/rows.jsonl")"
        done
    done
    for arm in before after; do
        previously=ordered_base; id=base
        if [[ $arm == after ]]; then previously=ordered_full; id=full; fi
        cmp "$m/historical/$arm/native.jsonl" "$prior/held-v1/contrast/$previously.raw.jsonl"
        cmp "$m/train-original/$arm/native.jsonl" "$prior/train-v1/ordered/$id.raw.jsonl"
    done
    sha256sum "$r/SHA256SUMS" "$here/SHA256SUMS" "$panel/SHA256SUMS" "$binary" "$bank" "$original" "$refit" \
        "$prior/held-v1/contrast/"{ordered_base,ordered_full}.raw.jsonl "$prior/train-v1/ordered/"{base,full}.raw.jsonl > "$m/inputs.sha256"
    seal "$m"; sha256sum "$m/SHA256SUMS"
    ;;
*) exit 64 ;;
esac
