#!/usr/bin/env bash
# Build and finite tests have their own artifacts. This entry point measures real rows only on a signed release.
here=$(cd "$(dirname "$0")" && pwd)
source "$here/common-wick.sh"
[[ $here == "$study"/methods-v* ]]
mode=${1:?release, run or inner}
case $mode in
release)
    shift
    v=${1:?release version}; fit_sha=${2:?FIT SHA}; gpu=$(realpath "${3:?device executable}")
    panel=$(realpath "${4:?sealed panel}"); shift 4
    version "$v"; release="$study/release-$v"
    [[ $gpu == "$study"/device/*/omission-sm89 && $panel == "$study"/panel/panel-v* && $# -ge 2 ]]
    test ! -e "$release/inputs.sha256"
    pin "$fit_sha" "$release/FIT.md"; test -s "$release/DECISION.md"
    historical; verify "$old_run"; verify "$mixed/panel/panel-v1"; verify "$here"; verify "$panel"
    verify "$(dirname "$gpu")"
    receipts=()
    for f in "$@"; do
        f=$(realpath "$f"); [[ $f == "$study/"* && -f $f ]]; receipts+=("$f")
    done
    printf '%s\n' "$gpu" > "$release/gpu.txt"
    printf '%s\n' "$panel" > "$release/panel.txt"
    sha256sum "$here/"{SHA256SUMS,run-wick.sh,common-wick.sh,PRE-RUN.md,ORDER.md} \
        "$release/"{FIT.md,DECISION.md,gpu.txt,panel.txt} "$gpu" "$(dirname "$gpu")/SHA256SUMS" \
        "$panel/"{SHA256SUMS,masks.tsv,rows.jsonl} "$old_run/"{SHA256SUMS,selection.sha256} \
        "$state" "$old_raw" "$old_edges" "$old/far-residuals-v1/summary.json" \
        "$mixed/panel/panel-v1/SHA256SUMS" "$text" "$gold_labels" "$original/reference.json" \
        "$original/blind/coordinates.jsonl" "$frozen" "$join" "$lock" "${receipts[@]}" > "$release/inputs.sha256"
    sha256sum --quiet --strict -c "$release/inputs.sha256"; sha256sum "$release/inputs.sha256"
    ;;
run|inner)
    v=${2:?unused measurement version}; version "$v"
    release=$(realpath "${3:?release directory}"); key=${4:?approved release SHA}
    [[ $release == "$study"/release-v* ]]; pin "$key" "$release/inputs.sha256"
    sha256sum --quiet --strict -c "$release/inputs.sha256"; verify "$here"; historical
    gpu=$(cat "$release/gpu.txt"); panel=$(cat "$release/panel.txt")
    for f in "$gpu" "$here/"{SHA256SUMS,run-wick.sh} "$panel/"{SHA256SUMS,masks.tsv,rows.jsonl} \
             "$release/"{gpu.txt,panel.txt,FIT.md,DECISION.md} "$state" "$old_raw" "$lock"; do
        grep -Fxq "$(sha256sum "$f")" "$release/inputs.sha256"
    done
    out="$study/measurement-$v"
    if [[ $mode == run ]]; then exec bash "$lock" "$out-lease" bash "$0" inner "$v" "$release" "$key"; fi
    test -e /proc/$$/fd/9; mkdir "$out"
    sha256sum "$release/inputs.sha256" > "$out/inputs.sha256"
    "$gpu" "$frozen" "$text" "$state" "$gold_labels" "$panel/masks.tsv" "$out/raw"
    cmp "$out/raw/neutral.jsonl" "$old_raw"
    for arm in neutral winner control; do
        test "$(wc -l < "$out/raw/$arm.jsonl")" -eq 256
        test "$(wc -l < "$out/raw/$arm.ranks.jsonl")" -eq 48
    done
    sha256sum "$state" "$text" "$frozen" "$gold_labels" > "$out/unchanged.sha256"
    seal "$out"
    ;;
*) exit 64 ;;
esac
