#!/usr/bin/env bash
set -euo pipefail
umask 077
parent=$(realpath "${1:?4337 private run}")
official=$(realpath "${2:?pinned official train JSON}")
extract=$(realpath "${3:?4337 pilot-extract.jq}")
out=$(realpath -m "${4:?fresh output}")
pilot="$parent/next-supply-pilot-v1"
supply="$parent/next-supply-v1"
sha256sum --quiet --strict -c - <<PINS
22e9d64fc7b423364f254447769d14e42819ccd4030d418643de930ac77d9cbb  $supply/OUTPUTS.sha256
ff88d9040c5f700c0bce30edd227c9e9933eae3229298ec6af8f4f6c3c1c51de  $pilot/FOLLOW-THROUGH.sha256
68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002  $official
bfa110b89d20fec53b1695ed526947811af56b653b697fbe3cf8d892c4280db7  $extract
PINS
mkdir "$out"
(cd "$pilot" && sha256sum --quiet --strict -c FOLLOW-THROUGH.sha256)
(cd "$supply" && sha256sum --quiet --strict -c OUTPUTS.sha256)
jq -c --rawfile ranking "$pilot/train-passages-ranked.tsv" '
  (group_by(.passage)|map(sort_by(.left,.right)|first)|INDEX(.passage)) as $by_passage |
  ($ranking|split("\n")|map(select(length>0)|split("\t"))) as $rows |
  if ($rows|length)!=149 then error("TRAIN roster size") else . end |
  $rows[]|. as $r|$by_passage[.[1]]|select(.article==($r[2]|fromjson))
' "$supply/pairs.json" > "$out/picks.jsonl"
jq -c --slurpfile picked "$out/picks.jsonl" --slurpfile questions "$supply/questions.jsonl" \
    -f "$extract" "$official" > "$out/rows.jsonl"
jq -es 'length==149 and (map(.passage_sha256)|unique|length)==149 and (map(.article)|unique|length)==24' "$out/rows.jsonl" >/dev/null
while IFS= read -r row; do
    digest=$(jq -j .context <<< "$row" | sha256sum | cut -d" " -f1)
    test "$digest" = "$(jq -r .passage_sha256 <<< "$row")"
done < "$out/rows.jsonl"
(cd "$out" && sha256sum picks.jsonl rows.jsonl > SEALED.sha256)
