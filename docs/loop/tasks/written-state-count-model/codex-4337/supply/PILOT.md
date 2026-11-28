# Prospective natural question pilot

The frozen TRAIN sample yielded 3 of 8 reciprocal pairs with two independently
licensed spans. The other 5 pairs have at least one ambiguous or overreaching
question. Ten of 16 individual answers are licensed. Questions that list
several competing answers, add a qualifier, ask about an unstated actor, or turn
a present-tense description into a claim about origins can pass the mechanical
screen and still fail passage adjudication.

None of the eight sampled TRAIN passages has an official impossible label. The
separate census of all 149 TRAIN positive-pair passages found four such labels
on one passage. The first fixed pick is ambiguous. Review of all four labels
found one genuinely unsupported question and two answerable ones. The same
passage has two supported positive questions. One asks the affirmative version
of the unsupported question, giving one matched supported/unsupported pair.
Pairing both positives with that same unsupported question would reuse the
single negative. The pilot and the targeted scan use one adjudicator. The
counts describe source supply only. Validation and locked questions, passages
and answers remained outside adjudication.

## Rebuild selection and authenticate source on Wick

Run from this directory. Use a fresh destination inside the unit root. The
private manifests retain source text and the individual reasons.

```sh
runs=/mnt/plexpool/uor-r4-corpus/runs
unit=$runs/20261127-codex-grounded-interval
supply=$unit/next-supply-v1
official=$runs/20261127-codex-whole-turn-census/supply-inputs/squad-train-v2.0.json
sealed=$unit/next-supply-pilot-v1
replay=$unit/next-supply-pilot-replay-v1
umask 077
bash pilot.sh freeze "$supply" "$official" "$replay"
cmp "$sealed/split.tsv" "$replay/split.tsv"
cmp "$sealed/picked-pairs.jsonl" "$replay/picked-pairs.jsonl"
jq -c --slurpfile picked "$replay/picked-pairs.jsonl" \
  --slurpfile questions "$supply/questions.jsonl" \
  -f pilot-extract.jq "$official" > "$replay/selected-source.jsonl"
cmp "$sealed/selected-source.jsonl" "$replay/selected-source.jsonl"
jq -c --rawfile train_passages "$replay/train-passages-ranked.tsv" \
  --slurpfile pairs "$supply/pairs.json" \
  --slurpfile questions "$supply/questions.jsonl" \
  -f pilot-negative-inventory.jq "$official" > "$replay/train-negative-inventory.jsonl"
cmp "$sealed/train-negative-inventory.jsonl" "$replay/train-negative-inventory.jsonl"
jq -sc 'map(select(.impossible_candidates>0))[:4][]' \
  "$replay/train-negative-inventory.jsonl" > "$replay/train-negative-selected.jsonl"
cmp "$sealed/train-negative-selected.jsonl" "$replay/train-negative-selected.jsonl"
jq -c --slurpfile sampled "$replay/train-negative-selected.jsonl" '
  $sampled[] as $row |
  map(select(.article==$row.article and .passage==$row.passage_sha256)) |
  sort_by(.left,.right)[]' "$supply/pairs.json" > "$replay/train-negative-all-pairs.jsonl"
jq -c --slurpfile picked "$replay/train-negative-all-pairs.jsonl" \
  --slurpfile questions "$supply/questions.jsonl" \
  -f pilot-extract.jq "$official" > "$replay/train-negative-all-positive-pairs.jsonl"
jq -c --slurpfile sampled "$replay/train-negative-selected.jsonl" '
  $sampled[] as $row | .data[] | select(.title==$row.article) |
  .paragraphs[$row.paragraph].qas | map(select(.is_impossible==true)) |
  sort_by(.id)[]' "$official" > "$replay/train-negative-all-impossible.jsonl"
cmp "$sealed/train-negative-all-positive-pairs.jsonl" "$replay/train-negative-all-positive-pairs.jsonl"
cmp "$sealed/train-negative-all-impossible.jsonl" "$replay/train-negative-all-impossible.jsonl"
(cd "$sealed" && sha256sum -c INPUTS.sha256 SPLIT.sha256 PICKS.sha256 SOURCE.sha256 \
  TRAIN-NEGATIVES.sha256 MANIFEST.sha256 FOLLOW-THROUGH.sha256)
```

The extraction authenticates the official question, original annotation, exact
character span and reconstructed prompts. The original capture also checked the
UTF-8 paragraph SHA-256. Compare the sourced case identities against the private
adjudication ledgers when re-reviewing the judgments.
