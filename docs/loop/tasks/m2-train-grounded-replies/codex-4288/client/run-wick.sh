#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

here=$(cd "$(dirname "$0")" && pwd)
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-text-governor-links"
client="$study/client"
previous="$runs/20261126-codex-governor-noun-links/client"
legacy="$previous/methods-v6"
reference="$study/panel/panel-v1/fresh/reference.json"
reference_sha=fcc8031daf05d66a47fda878cbe3015ead90b1b9990dd67a268f316e716edce3

[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$client") == zfs && $here == "$client/"* ]]
mode=${1:?stage, fixture, or capture}
version=${2:?methods version}
[[ $version =~ ^v[1-9][0-9]*$ ]]
methods="$client/methods-$version"

sha() { sha256sum "$1" | cut -d' ' -f1; }
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
verify() ( cd "$1" && sha256sum --strict --quiet -c SHA256SUMS; )
seal() (
    cd "$1"
    test ! -e SHA256SUMS
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum --strict --quiet -c SHA256SUMS
    chmod -R a-w .
)
check_gate() {
    local grade=$1 approved_sha=$2 endpoints_sha=$3 panel_sha=$4
    pin "$approved_sha" "$grade" &&
        jq -e --arg endpoints "$endpoints_sha" --arg reference "$panel_sha" '
          .pass == true and .pairs == 128 and .endpoints_sha256 == $endpoints and
          .reference_sha256 == $reference' "$grade" >/dev/null
}
audit() { jq -ce -f "$here/audit-tokens.jq" "$@"; }

case "$mode" in
stage)
    pin 6ab042d912e0629b33175c54be47c7f54dcdc40e3228a49161714f3665d1cade "$legacy/SHA256SUMS"
    verify "$legacy"
    mkdir "$methods"
    for file in common.jq fixture-wick.sh run-wick.sh; do
        sed -e 's/20261126-codex-governor-noun-links/20261126-codex-text-governor-links/g' \
            -e 's/governor_only/governor_gap/g' -e 's/nouns_only/learned/g' \
            -e 's@/gate-v@/grade-v@g' "$legacy/$file" > "$methods/$file"
    done
    cp "$legacy/"{grade,plan,summary}.jq "$methods/"
    bash -n "$methods/run-wick.sh"
    bash -n "$methods/fixture-wick.sh"
    sha256sum "$here/run-wick.sh" "$here/audit-tokens.jq" "$legacy/SHA256SUMS" > "$methods/adaptation.sha256"
    seal "$methods"
    ;;
fixture)
    verify "$methods"
    sha256sum --strict --quiet -c "$methods/adaptation.sha256"
    fixture="$client/fixture-$version"
    out="$client/adapter-fixture-$version"
    [[ ! -e $fixture && ! -e $out ]]
    mkdir "$out"
    bash "$methods/fixture-wick.sh" "fixture-$version" > "$out/predecessor.stdout"
    verify "$fixture"
    jq -es 'length == 10 and
      (map(.arm) | unique) == ["correct_link","exchanged_link","governor_gap","learned","native"] and
      all(.[] | select(.arm == "learned"); (.wire.affirmed.end > .wire.affirmed.start) and
        (.wire | has("learned_field") | not))' "$fixture/plan.jsonl" > "$out/arm-mapping.ok"

    historical="$previous/capture-v1/grade/exchanged_link.jsonl"
    pin b8dbd478dbe049841a0bba6cb2b044888e604fc230e231314b429fe5bb386fbf "$historical"
    audit "$fixture/"{take,defer}-grade.json "$historical" > "$out/token-audit.jsonl"
    jq -es 'length == 258 and .[0].returned.exact_through_eos and
      .[1].first.correct_prefix_at_cap and .[1].returned.correct_prefix_at_cap and
      (.[1].returned.exact_through_eos | not) and
      .[2].first.first_differing_id == {position:1,expected_id:2997,actual_id:617} and
      .[2].returned.first_differing_id == .[2].first.first_differing_id' \
      "$out/token-audit.jsonl" > "$out/token-audit.ok"
    if head -n 1 "$historical" | jq -c '.returned.first_wrong.actual_id += 1' |
         audit > "$out/bad-divergence.stdout" 2> "$out/bad-divergence.stderr"; then
        echo 'Altered actual divergent ID passed the independent audit' >&2
        exit 1
    fi
    grep -q 'first differing position or ID' "$out/bad-divergence.stderr"
    jq -nc --arg sha "$reference_sha" '
      {pass:true,pairs:128,endpoints_sha256:("8" * 64),reference_sha256:$sha}' > "$out/accepted-grade.json"
    check_gate "$out/accepted-grade.json" "$(sha "$out/accepted-grade.json")" "$(printf '%064d' 0 | tr 0 8)" "$reference_sha"
    jq -c '.pass = false' "$out/accepted-grade.json" > "$out/rejected-grade.json"
    if check_gate "$out/rejected-grade.json" "$(sha "$out/rejected-grade.json")" "$(printf '%064d' 0 | tr 0 8)" "$reference_sha"; then
        echo 'A failed source gate admitted the writer' >&2
        exit 1
    fi
    sha256sum "$methods/SHA256SUMS" "$fixture/SHA256SUMS" "$historical" > "$out/inputs.sha256"
    seal "$out"
    sha256sum "$out/SHA256SUMS"
    ;;
capture)
    gate=$(realpath "${3:?reviewed passing grade summary}")
    approved_sha=${4:?approved grade SHA-256}
    label=${5:?unused capture-vN}
    [[ $gate =~ ^"$study"/grade-v[1-9][0-9]*/summary\.json$ && $approved_sha =~ ^[a-f0-9]{64}$ ]]
    [[ $label =~ ^capture-v[1-9][0-9]*$ ]]
    endpoints="$(dirname "$gate")/endpoints.jsonl"
    capture="$client/$label"
    out="$client/token-${label#capture-}"
    [[ ! -e $capture && ! -e $out ]]
    verify "$methods"
    sha256sum --strict --quiet -c "$methods/adaptation.sha256"
    verify "$client/adapter-fixture-$version"
    verify "$(dirname "$gate")"
    pin "$reference_sha" "$reference"
    check_gate "$gate" "$approved_sha" "$(sha "$endpoints")" "$reference_sha"
    bash "$methods/run-wick.sh" capture "$gate" "$approved_sha" "$endpoints" "$label"
    verify "$capture"
    mkdir "$out"
    audit "$capture/grade/"*.jsonl > "$out/endpoints.jsonl"
    jq -es 'if length == 1280 then group_by(.arm) | map({
        arm:.[0].arm, endpoints:length,
        first_correct_prefix_at_cap:([.[] | select(.first.correct_prefix_at_cap)] | length),
        returned_correct_prefix_at_cap:([.[] | select(.returned.correct_prefix_at_cap)] | length),
        returned_exact_eos:([.[] | select(.returned.exact_through_eos)] | length)
      }) else error("ordinary token audit did not cover each arm and endpoint") end' \
      "$out/endpoints.jsonl" > "$out/summary.json"
    sha256sum "$methods/SHA256SUMS" "$client/adapter-fixture-$version/SHA256SUMS" \
        "$capture/SHA256SUMS" "$capture/grade/summary.json" "$gate" "$endpoints" "$reference" > "$out/inputs.sha256"
    seal "$out"
    sha256sum "$capture/SHA256SUMS" "$out/SHA256SUMS"
    ;;
*) exit 64 ;;
esac
