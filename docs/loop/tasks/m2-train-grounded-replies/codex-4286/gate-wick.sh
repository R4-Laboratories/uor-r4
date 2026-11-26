#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $(hostname -s) == wick ]]
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-governor-noun-links
methods=$(cd "$(dirname "$0")" && pwd)
sha() { sha256sum "$1" | cut -d' ' -f1; }
score() {
    jq -ne --slurpfile reference "$1" --slurpfile rows "$2" \
        --arg reference_sha256 "$(sha "$1")" --arg endpoints_sha256 "$(sha "$2")" \
        -f "$methods/gate.jq"
}
case ${1:?finite, enroll or score} in
finite)
    out=${2:?new output directory}
    [[ $out == "$study"/gate-finite-v* ]]
    mkdir "$out"
    for part in reference pass fail damaged; do
        jq -nc --arg part "$part" -f "$methods/gate-fixture.jq" > "$out/$part.json"
    done
    for part in pass fail; do score "$out/reference.json" "$out/$part.json" > "$out/$part-result.json"; done
    jq -e '.pass and .gate.net_over_native==128 and .arms.correct_link.exact_pairs==128' "$out/pass-result.json" >/dev/null
    jq -e '(.pass|not) and .gate.net_over_native==0' "$out/fail-result.json" >/dev/null
    if score "$out/reference.json" "$out/damaged.json" > "$out/rejected.json" 2> "$out/error.txt"; then exit 1; fi
    grep -Fq 'byte/token disagreement' "$out/error.txt"
    ;;
enroll)
    reference=$(realpath "${2:?frozen panel reference}")
    out=${3:?new enrollment output directory}
    [[ $reference == "$study"/panel/* && $out == "$study"/enrollment-v* ]]
    test "$(sha "$reference")" = "${4:?frozen reference digest}"
    previous=/mnt/plexpool/uor-r4-corpus/runs/20261125-codex-governing-source-pairs
    mkdir "$out"
    jq -ne --slurpfile ref "$reference" --slurpfile train "$previous/train-v1/teacher.jsonl" \
       --slurpfile held "$previous/held/primary/panel-v1/fresh/reference.json" \
       --slurpfile reserve "$previous/reserve/panel-v1/fresh/reference.json" '
       ($ref[0]|map(.turns[1])) as $texts |
       (($train|map(.user))+($held[0]|map(.turns[1]))+($reserve[0]|map(.turns[1]))) as $past |
       ($past|INDEX(.)) as $index |
       ($ref[0]|group_by([.family,.distance])|map({
         family:.[0].family,distance:.[0].distance,rows:length,pairs:(map(.pair_id)|unique|length),
         gap_bpe:(map(.governors.affirmative.gap,.governors.excluded.gap)|{min:min,max:max}),
         governing_verb_orders:(map(.governor_order)|unique),
         same_noun_orders:(map(.placement_name)|unique),fact_orders:(map(.requested_role)|unique)
       })) as $cells |
       {fresh_distinct:($texts|unique|length),past_training:($train|length),
        past_ordinary:($held[0]|length),past_reserve:($reserve[0]|length),
        overlap:([$texts[]|select($index[.]!=null)]|length),cells:$cells} |
       if .fresh_distinct==256 and .overlap==0 and (.cells|length)==8 and
         all(.cells[];.pairs==16 and .rows==32 and (.governing_verb_orders|length)==2
            and (.same_noun_orders|length)==2 and (.fact_orders|length)==2)
       then . else error("freshness or crossed panel enrollment failed") end' > "$out/summary.json"
    sha256sum "$reference" "$previous/train-v1/teacher.jsonl" \
        "$previous/"{held/primary/panel-v1,reserve/panel-v1}/fresh/reference.json > "$out/inputs.sha256"
    ;;
score)
    reference=$(realpath "${2:?frozen panel reference}")
    rows=$(realpath "${3:?GPU endpoints}")
    out=${4:?new gate output directory}
    [[ $reference == "$study"/panel/* && $rows == "$study"/device/* && $out == "$study"/gate-v* ]]
    test "$(sha "$reference")" = "${5:?reference digest}"
    test "$(sha "$rows")" = "${6:?verified physical output digest}"
    mkdir "$out"
    score "$reference" "$rows" > "$out/summary.json"
    jq -e '.pass|type=="boolean"' "$out/summary.json" >/dev/null
    sha256sum "$methods/"{gate-wick.sh,gate.jq} "$reference" "$rows" > "$out/inputs.sha256"
    ;;
*) exit 64 ;;
esac
(cd "$out" && find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS &&
    sha256sum --strict --quiet -c SHA256SUMS)
