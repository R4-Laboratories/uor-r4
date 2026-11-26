#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
c="$s/client"; previous="$c/methods-v1"; tokens="$text_source/client/adapter-v1/audit-tokens.jq"
mode=${1:?stage, finite, or capture}; v=${2:?version}; version "$v"; adapter="$c/methods-$v"
case $mode in
stage)
    [[ $# == 2 && $v != v1 ]]
    pin 5ca0675b6a4863d961da3a039245e3e5aad8ca3afef6f485205d278faf19bfab "$previous/SHA256SUMS"
    verify "$previous"
    mkdir "$adapter"
    cp "$previous/"{run-wick.sh,common.jq,plan.jq,grade.jq,summary.jq} "$adapter"
    cp "$here/client/"{adapter.patch,lanes-wick.sh,lanes.jq,gate.jq,jobs.jq,tokens-summary.jq,fixture-wick.sh} "$adapter"
    cp "$tokens" "$adapter/audit-tokens.jq"
    (cd "$adapter" && patch --fuzz=0 -p0 < adapter.patch)
    bash -n "$adapter/run-wick.sh"
    sha256sum "$previous/SHA256SUMS" "$tokens" \
        "$here/client/"{adapter.patch,lanes-wick.sh,lanes.jq,gate.jq,jobs.jq,tokens-summary.jq,fixture-wick.sh} > "$adapter/inputs.sha256"
    seal "$adapter"
    ;;
finite)
    f=${3:?new fixture version}; version "$f"; verify "$adapter"
    [[ $# == 3 ]]
    bash "$adapter/fixture-wick.sh" "fixture-$f"
    verify "$c/fixture-$f"
    jq -e '.pass and .schema=="uor-r4.codex-4297.client-selection-fixture/1"' "$c/fixture-$f/receipt.json" >/dev/null
    ;;
capture)
    gv=${3:?grade version}; cv=${4:?new capture or refusal version}; version "$gv"; version "$cv"
    grade="$s/grade-$gv"; gate="$grade/summary.json"; verify "$grade"; verify "$adapter"
    grep -Fxq "$(sha "$gate")  ./summary.json" "$grade/SHA256SUMS"
    selection=$(jq -ce -L "$adapter" -f "$adapter/gate.jq" "$gate")
    jq -e '.pairs==128' <<< "$selection" >/dev/null
    if jq -e '.pass==false' <<< "$selection" >/dev/null; then
        [[ $# == 4 || $# == 5 ]]
        out="$c/refusal-$cv"; mkdir "$out"
        jq -e '{status:"NOT_RUN",reason:"fresh source gate",passing,arms:(.arms|map_values([.gate|to_entries[]|select(.value==false)|.key]))}' "$gate" > "$out/receipt.json"
        sha256sum "$gate" "$grade/SHA256SUMS" "$adapter/SHA256SUMS" > "$out/inputs.sha256"; seal "$out"; exit 0
    fi
    pv=${5:?pinned panel version}; version "$pv"; [[ $# == 5 ]]; p="$s/panel/panel-$pv"
    listed "$grade/inputs.sha256" "$p/SHA256SUMS"; verify "$p"; export UOR_BRIDGE_HELD_PANEL="$p/held"
    capture="$c/capture-$cv"
    bash "$adapter/run-wick.sh" capture "$gate" "$(sha "$gate")" "$grade/endpoints.jsonl" "capture-$cv"
    verify "$capture"; bash "$adapter/run-wick.sh" writer "$capture" "writer-$cv"; verify "$c/writer-$cv"
    out="$c/tokens-$cv"; mkdir "$out"
    here="$adapter"
    source "$here/lanes-wick.sh"
    grade_tokens "$capture" "$out"
    sha256sum "$here/audit-tokens.jq" "$here/SHA256SUMS" "$gate" "$capture/"{selection.json,SHA256SUMS} \
        "$c/writer-$cv/SHA256SUMS" > "$out/inputs.sha256"; seal "$out"
    ;;
*) exit 64 ;;
esac
