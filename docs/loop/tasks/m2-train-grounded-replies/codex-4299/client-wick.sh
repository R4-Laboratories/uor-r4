#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
mode=${1:?stage, finite or capture}; v=${2:?adapter version}; version "$v"; adapter="$s/client/methods-$v"
case $mode in
stage)
    earlier="$parent/client/methods-v4"; pin 86d86a0e52f03de50f17a5807ba303c955fd38bb80f7f92f659bfcaf07e61e60 "$earlier/SHA256SUMS"
    verify "$earlier"; mkdir -p "$s/client"; mkdir "$adapter"
    cp "$earlier/"{plan,grade,summary,gate,jobs,tokens-summary,audit-tokens}.jq "$earlier/lanes-wick.sh" "$adapter/"
    for f in lanes.jq common.jq; do
        grep -Fq '"strongest","all","ordered-all"' "$earlier/$f"
        sed 's/"strongest","all","ordered-all"/"endpoint","ordered"/' "$earlier/$f" > "$adapter/$f"
    done
    sed 's/20261126-codex-all-outranker-interval/20261126-codex-fixed-assignment-median/g' \
        "$earlier/run-wick.sh" > "$adapter/run-wick.sh"
    grep -Fq 'study="$runs/20261126-codex-fixed-assignment-median"' "$adapter/run-wick.sh"
    cp "$here/client-fixture-wick.sh" "$adapter/"
    sha256sum "$earlier/SHA256SUMS" "$here/SHA256SUMS" > "$adapter/inputs.sha256"
    seal "$adapter"; sha256sum "$adapter/SHA256SUMS"
    ;;
finite)
    fv=${3:?fixture version}; version "$fv"; verify "$adapter"
    bash "$adapter/client-fixture-wick.sh" "$s/client/fixture-$fv"
    verify "$s/client/fixture-$fv"
    ;;
capture)
    gv=${3:?grade version}; cv=${4:?result version}; pv=${5:?panel version}
    for version_id in "$gv" "$cv" "$pv"; do version "$version_id"; done
    grade="$s/grade-$gv"; gate="$grade/summary.json"; panel="$s/panel/panel-$pv"
    verify "$grade"; verify "$adapter"; verify "$panel"; listed "$grade/inputs.sha256" "$panel/SHA256SUMS"
    grep -Fxq "$(sha "$gate")  ./summary.json" "$grade/SHA256SUMS"
    selection=$(jq -ce -L "$adapter" -f "$adapter/gate.jq" "$gate")
    jq -e --arg ref "$(sha "$panel/held/reference.json")" --arg ep "$(sha "$grade/endpoints.jsonl")" \
        '.pairs==128 and .reference_sha256==$ref and .endpoints_sha256==$ep' <<< "$selection" >/dev/null
    if jq -e '.pass==false' <<< "$selection" >/dev/null; then
        out="$s/client/refusal-$cv"; mkdir "$out"
        jq -e 'select(.passing==[]) | {status:"NOT_RUN",reason:"fresh source gate",passing,
            arms:(.arms|map_values([.gate|to_entries[]|select(.value==false)|.key]))}' "$gate" > "$out/receipt.json"
        sha256sum "$gate" "$grade/SHA256SUMS" "$adapter/SHA256SUMS" > "$out/inputs.sha256"; seal "$out"
    else
        export UOR_BRIDGE_HELD_PANEL="$panel/held"
        capture="$s/client/capture-$cv"
        bash "$adapter/run-wick.sh" capture "$gate" "$(sha "$gate")" "$grade/endpoints.jsonl" "capture-$cv"
        verify "$capture"; bash "$adapter/run-wick.sh" writer "$capture" "writer-$cv"; verify "$s/client/writer-$cv"
        out="$s/client/tokens-$cv"; mkdir "$out"
        here="$adapter"; source "$here/lanes-wick.sh"; grade_tokens "$capture" "$out"
        sha256sum "$gate" "$grade/SHA256SUMS" "$adapter/SHA256SUMS" "$capture/"{selection.json,SHA256SUMS} "$s/client/writer-$cv/SHA256SUMS" > "$out/inputs.sha256"
        seal "$out"
    fi
    ;;
*) exit 64 ;;
esac

