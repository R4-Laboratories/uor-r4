#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# != 4 ]]; then
    echo 'usage: extract.sh NEXT_SUPPLY SEALED_ANSWERABILITY OFFICIAL_INPUTS FRESH_OUTPUT' >&2
    exit 2
fi
supply=$(cd "$1" && pwd -P)
sealed=$(cd "$2" && pwd -P)
official=$(cd "$3" && pwd -P)
driver=$(cd "$(dirname "$0")" && pwd -P)
out=$4

pin() {
    printf '%s  %s\n' "$1" "$2" | sha256sum --strict -c --quiet -
}

pin f6eea0707c6880a2858638f494b7c8887d5b8609285cb41b1865a8899e4ed131 "$supply/pairs.json"
pin 36ef265ab8346bb4cc51ede1ea83f210622edc9472297410e5ad8c859373ff28 "$supply/questions.jsonl"
pin 1899b7c8e52bc56e08d084e1286855e167d4a5665c13a72538e75e9f1c6c984f "$sealed/SEAL.sha256"
(cd "$sealed" && sha256sum --strict -c --quiet SEAL.sha256 ROSTER.sha256)
pin 8502372b2e063cbcd05b7619b24d3b6f9ebe5e33efec9b8f36cfdb029a86439a "$sealed/prior-positive-split.tsv"
pin 68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002 "$official/squad-train-v2.0.json"
pin 80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8 "$official/squad-dev-v2.0.json"

mkdir "$out"
cd "$out"
mkdir selection selected-semantic
sha256sum "$supply/pairs.json" "$supply/questions.jsonl" "$sealed/SEAL.sha256" \
    "$sealed/ROSTER.sha256" "$sealed/prior-positive-split.tsv" \
    "$official/squad-train-v2.0.json" "$official/squad-dev-v2.0.json" \
    "$driver"/{Cargo.toml,Cargo.lock,src/main.rs,REGISTRATION.md,SELECTION.md,extract.sh,extract-selected.jq} \
    > INPUTS.sha256

awk -F '\t' '$1=="validation"{print $3}' "$sealed/prior-positive-split.tsv" |
    jq -s . > selection/two-answer-titles.json
jq -c --slurpfile titles selection/two-answer-titles.json '
  .[] | select(.article as $title | $titles[0] | index($title) != null) |
  {article,passage_sha256:.passage,left,right}
' "$supply/pairs.json" > selection/two-answer.metadata.jsonl
jq -c '
  select(.split == "validation") |
  {article,passage_sha256,left:.positive.published_id,right:.negative.published_id}
' "$sealed/roster.jsonl" > selection/answerability.metadata.jsonl
jq -se --slurpfile articles "$sealed/articles.jsonl" '
  (map(.article) | unique) == ($articles | map(select(.split=="validation") | .article) | unique)
' selection/answerability.metadata.jsonl > /dev/null

CARGO_TARGET_DIR="$PWD/target" cargo run --quiet --locked --offline --manifest-path "$driver/Cargo.toml" -- \
    selection/two-answer.metadata.jsonl selection/answerability.metadata.jsonl > selection/selection.json
jq -e '([.two_answer[].passages] | sort) == [1,2,7,9,10,18,30] and
       ([.two_answer[].selected] | add) == 23 and
       ([.answerability[].selected] | add) == 44' selection/selection.json > /dev/null
jq -c '
  to_entries[] | select(.key == "two_answer" or .key == "answerability") |
  .key as $family | .value[].candidates[] | select(.disposition == "review") |
  .pair + {family:$family,rank,pair_id:($family + "-" + .pair.passage_sha256)}
' selection/selection.json > selection/review-identities.jsonl
(cd selection && sha256sum ./*.json ./*.jsonl > SELECTION.sha256)

# Project the exact selected cases before handing their fields to the extractor.
jq -cs --slurpfile picks selection/review-identities.jsonl '
  ($picks | map(select(.family=="two_answer") | .left,.right) | INDEX(.)) as $want |
  .[] | select($want[.case_id] != null)
' "$supply/questions.jsonl" > selected-semantic/two-answer-source.jsonl
jq -c --slurpfile picks selection/review-identities.jsonl '
  ($picks | map(select(.family=="answerability")) | INDEX(.passage_sha256)) as $want |
  select(.split == "validation" and $want[.passage_sha256] != null)
' "$sealed/roster.jsonl" > selected-semantic/answerability-source.jsonl

for family in two_answer answerability; do
    if [[ $family == two_answer ]]; then
        dataset="$official/squad-train-v2.0.json"
        source=selected-semantic/two-answer-source.jsonl
    else
        dataset="$official/squad-dev-v2.0.json"
        source=selected-semantic/answerability-source.jsonl
    fi
    jq -c --arg family "$family" --slurpfile picks selection/review-identities.jsonl \
        --slurpfile source "$source" -f "$driver/extract-selected.jq" "$dataset"
done > selected-semantic/source.jsonl

# The grader sees exact client-visible text and candidate source evidence only.
jq -c 'del(.left.prefix,.right.prefix,.left.user_bounds,.right.user_bounds)' \
    selected-semantic/source.jsonl > selected-semantic/rows.jsonl
jq -se 'length == 67 and
       ((map(.pair_id) | unique | length) == 67) and
       ((map(.left.case_id,.right.case_id) | unique | length) == 134)' \
    selected-semantic/rows.jsonl > /dev/null
(cd selected-semantic && sha256sum ./*.jsonl > SEMANTIC.sha256)
sha256sum INPUTS.sha256 selection/SELECTION.sha256 selected-semantic/SEMANTIC.sha256 > EXTRACTION.sha256
