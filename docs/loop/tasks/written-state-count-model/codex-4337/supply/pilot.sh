#!/usr/bin/env bash
set -euo pipefail
umask 077

mode=${1:?freeze or extract}
supply=$(realpath "${2:?sealed next-supply-v1}")
official=$(realpath "${3:?official archived SQuAD 2.0 TRAIN JSON}")
pilot=$(realpath -m "${4:?pilot directory}")
salt=uor-r4/codex-4337/next-supply-pilot-v1

hash() {
    printf '%s\n%s' "$salt/$1" "$2" | sha256sum | cut -d' ' -f1
}

freeze() {
    mkdir "$pilot"
    (cd "$supply" && sha256sum -c OUTPUTS.sha256 >&2)
    sha256sum "$supply/pairs.json" "$supply/questions.jsonl" "$official" "${BASH_SOURCE[0]}" > "$pilot/INPUTS.sha256"

    # Freeze article membership from names alone. A JSON string in each TSV
    # preserves the exact article or case identity for jq on replay.
    jq -c 'map(.article)|unique[]' "$supply/pairs.json" |
        while IFS= read -r article; do
            title=$(jq -r . <<< "$article")
            printf '%s\t%s\n' "$(hash article "$title")" "$article"
        done | LC_ALL=C sort > "$pilot/articles-ranked.tsv"
    test "$(wc -l < "$pilot/articles-ranked.tsv" | tr -d ' ')" -eq 38
    awk -F '\t' 'BEGIN{OFS="\t"} {group=(NR<=24?"train":(NR<=31?"validation":"locked")); print group,$1,$2}' \
        "$pilot/articles-ranked.tsv" > "$pilot/split.tsv"
    (cd "$pilot" && sha256sum articles-ranked.tsv split.tsv > SPLIT.sha256)

    # Within TRAIN rank each paragraph by its passage ID, then take the first
    # paragraph of each new article and the first eight such articles.
    awk -F '\t' '$1=="train"{print $3}' "$pilot/split.tsv" | jq -s . > "$pilot/train-articles.json"
    jq -r --slurpfile train "$pilot/train-articles.json" '
      map(select(.article as $article | $train[0]|index($article))) |
      group_by(.passage)[] | group_by(.article)[] |
      [.[0].passage,(.[0].article|tojson)] | @tsv' "$supply/pairs.json" |
        while IFS=$'\t' read -r passage article; do
            printf '%s\t%s\t%s\n' "$(hash passage "$passage")" "$passage" "$article"
        done | LC_ALL=C sort > "$pilot/train-passages-ranked.tsv"
    awk -F '\t' '!seen[$3]++ && ++picked<=8{print $2}' "$pilot/train-passages-ranked.tsv" |
        jq -R -s 'split("\n")|map(select(length>0))' > "$pilot/picked-passages.json"
    jq -c --slurpfile selected "$pilot/picked-passages.json" '
      [$selected[0][] as $passage | map(select(.passage==$passage)) |
       sort_by(.left,.right) | first] | .[]' "$supply/pairs.json" > "$pilot/picked-pairs.jsonl"
    test "$(wc -l < "$pilot/picked-pairs.jsonl" | tr -d ' ')" -eq 8
    (cd "$pilot" && sha256sum train-articles.json train-passages-ranked.tsv picked-passages.json picked-pairs.jsonl > PICKS.sha256)
}

case "$mode" in
    freeze) freeze ;;
    *) printf 'Unknown mode: %s\n' "$mode" >&2; exit 2 ;;
esac
