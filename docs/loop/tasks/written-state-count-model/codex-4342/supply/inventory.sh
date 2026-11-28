#!/usr/bin/env bash
set -euo pipefail
umask 077

[[ $# == 2 ]] || { echo 'usage: inventory.sh WICK_RUNS FRESH_DESTINATION' >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd -P)
runs=$(realpath "$1")
out=$(realpath -m "$2")
registration=$here/REGISTRATION.json
export LC_ALL=C.UTF-8

source_file() {
    printf '%s/%s\n' "$runs" "$(jq -er --arg key "$1" '.[$key].member' "$registration")"
}

rank() {
    printf '%s\n%s\n' "$1" "$2" | sha256sum | cut -d ' ' -f 1
}

# Authenticate only the enumerated files. The parent split is used for its
# TRAIN metadata projection. No development or locked source is an input.
mkdir "$out"
for key in official parent_split parent_train parent_qualification_truth parent_natural_fit_ids parent_original_fit_ids; do
    printf '%s  %s\n' "$(jq -er --arg key "$key" '.[$key].sha256' "$registration")" "$(source_file "$key")"
done > "$out/SOURCE.sha256"
sha256sum --quiet --strict -c "$out/SOURCE.sha256"
cp "$registration" "$out/registration.json"

awk -F '\t' '$1=="train" {print $3}' "$(source_file parent_split)" |
    jq -se --slurpfile allowed "$(source_file parent_train)" '
      if length == 24 and (unique | length) == 24 and . == $allowed[0] then .
      else error("parent TRAIN article projection mismatch") end' > "$out/train-articles.json"
article_domain=$(jq -er .article_domain "$registration")
screen_count=$(jq -er .article_counts.screen "$registration")
jq -c '.[]' "$out/train-articles.json" |
    while IFS= read -r title; do
        printf '%s\t%s\n' "$(rank "$article_domain" "$(jq -r . <<< "$title")")" "$title"
    done | LC_ALL=C sort > "$out/articles-ranked.tsv"
jq -Rn --argjson screen "$screen_count" '[inputs|split("\t")]|to_entries|map({
  rank:.value[0], article:(.value[1]|fromjson), split:(if .key < $screen then "screen" else "construction" end)
})' "$out/articles-ranked.tsv" > "$out/articles.json"

official=$(source_file official)
jq -ce --slurpfile articles "$out/articles.json" --slurpfile registered "$registration" '
  ($articles[0]|INDEX(.article)) as $allowlist |
  if .version!=$registered[0].official.dataset_version then error("wrong official dataset version") else . end |
  [.data[] | .title as $title | select($allowlist[$title] != null)] as $train |
  if ($train|map(.title)|sort)!=($allowlist|keys) then error("official TRAIN allowlist projection mismatch") else . end |
  $train[] as $article | $allowlist[$article.title] as $scope |
  $article.paragraphs | to_entries[] |
  {article:$scope.article,split:$scope.split,paragraph:.key,context:.value.context,qas:.value.qas}
' "$official" > "$out/train-source.jsonl"

passage_domain=$(jq -er .passage_domain "$registration")
while IFS= read -r row; do
    digest=$(jq -j .context <<< "$row" | sha256sum | cut -d ' ' -f 1)
    printf '%s\t%s\n' "$digest" "$(rank "$passage_domain" "$digest")"
done < "$out/train-source.jsonl" > "$out/passages.tsv"

natural_ids=$(source_file parent_natural_fit_ids)
historical_ids=$(source_file parent_original_fit_ids)
jq -c --slurpfile articles "$out/articles.json" --rawfile natural "$natural_ids" --rawfile historical "$historical_ids" '
  . as $prior | select(any($articles[0][]; .article==$prior.article)) |
  {case_id, source_id, article, paragraph, passage_sha256,
   natural:($natural|split("\n")|index($prior.case_id)!=null),
   historical:($historical|split("\n")|index($prior.case_id)!=null)}
' "$(source_file parent_qualification_truth)" > "$out/prior-train-ids.jsonl"

jq -ne --slurpfile registered "$registration" --slurpfile articles "$out/articles.json" \
  --rawfile hashes "$out/passages.tsv" --slurpfile prior "$out/prior-train-ids.jsonl" \
  -f "$here/inventory.jq" "$out/train-source.jsonl" > "$out/inventory.json"
jq -c '.selected[]' "$out/inventory.json" > "$out/review.jsonl"
jq '.summary' "$out/inventory.json" > "$out/counts.json"

(cd "$out" && sha256sum SOURCE.sha256 registration.json train-articles.json articles-ranked.tsv articles.json \
    train-source.jsonl passages.tsv prior-train-ids.jsonl inventory.json review.jsonl counts.json > INVENTORY.sha256)
