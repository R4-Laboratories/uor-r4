# Reproduce the validation sample inside custody

The selector receives exactly two validation-only JSONL files. Each row contains `article`, lowercase `passage_sha256`, `left`, and `right`. The selector accepts no prompt, score, labels, gold addresses, artifacts or split-changing flag. It was tested only with invented passages and IDs.

## Custodian projection

Keep original files under their existing sealed manifests and permissions. Use a private fresh output directory (`umask 077`, `mkdir`) and isolate the unsealing process from graders and model processes. For the two-answer pool, `SPLIT` is #4337's authenticated `split.tsv` (pin in `supply/PILOT.json` or the byte-identical `prior-positive-split.tsv` in the #4338 seal) and `PAIRS` is #4337's pinned `next-supply-v1/pairs.json`. For answerability, `SEALED` is the restored #4338 `roster/sealed` directory. Check its `SEAL.sha256` digest and verify both `SEAL.sha256` and `ROSTER.sha256` in that directory. `DRAFT` identifies this selector directory and `VALIDATION` the private new output directory. Run this step only after the registration is approved.

In the administrative custody process, filter split membership before projecting row metadata. From the mixed originals emit only the selected type of validation metadata. Never display or publish either original. Do not give the selector, model or grader access to the originals.

```bash
set -euo pipefail
umask 077
mkdir "$VALIDATION"
awk -F '\t' '$1=="validation"{print $3}' "$SPLIT" | jq -s . > "$VALIDATION/two-answer-titles.json"
jq -c --slurpfile titles "$VALIDATION/two-answer-titles.json" '
  .[] | select(.article as $title | $titles[0] | index($title) != null) |
  {article,passage_sha256:.passage,left,right}
' "$PAIRS" > "$VALIDATION/two-answer.metadata.jsonl"
jq -c '
  select(.split == "validation") |
  {article,passage_sha256,left:.positive.published_id,right:.negative.published_id}
' "$SEALED/roster.jsonl" > "$VALIDATION/answerability.metadata.jsonl"
jq -s -e --slurpfile articles "$SEALED/articles.jsonl" '
  (map(.article) | unique) == ($articles | map(select(.split=="validation") | .article) | unique)
' "$VALIDATION/answerability.metadata.jsonl" > /dev/null

cargo run --locked --offline --manifest-path "$DRAFT/Cargo.toml" -- \
  "$VALIDATION/two-answer.metadata.jsonl" "$VALIDATION/answerability.metadata.jsonl" \
  > "$VALIDATION/selection.json"
jq -e '([.two_answer[].passages] | sort) == [1,2,7,9,10,18,30] and
       ([.two_answer[].selected] | add) == 23 and
       ([.answerability[].selected] | add) == 44' "$VALIDATION/selection.json" > /dev/null
jq -c '
  to_entries[] | select(.key == "two_answer" or .key == "answerability") |
  .key as $family | .value[].candidates[] | select(.disposition == "review") |
  .pair + {family:$family,rank}
' "$VALIDATION/selection.json" > "$VALIDATION/review-identities.jsonl"
(cd "$VALIDATION" && sha256sum two-answer-titles.json two-answer.metadata.jsonl \
  answerability.metadata.jsonl selection.json review-identities.jsonl > SELECTION.sha256)
```

The one- and two-passage titles retain their quota shortfalls in the manifest. Preserve the manifest if any coverage check aborts. The custodian can regenerate metadata and the selection manifest from the pinned parents and compare the full output byte for byte. Supply source review with a fresh bundle containing only `review-identities.jsonl` joined one-to-one to the authenticated selected source rows, carrying exact source coordinates, client prompt bytes, candidate annotation and original question IDs. For the two-answer family, reuse #4337 `supply/pilot-extract.jq`'s strict original-question and span rejoin, omit its unrelated impossible candidates from the review copy, and retain exact client-visible prompts from the screened question rows. For answerability use the exact #4338 validation window, prompts and annotations, never the larger full passage as evidence available to the client. Full-source identities are for custody authentication. Do not expose locked source or show validation alternatives that the registration did not take.

## Fixed rank and ledger

For each family, use exact UTF-8 official title and original whole passage SHA-256. Rank passages in a title by ascending lowercase SHA-256 of the bytes formed by joining `uor-r4/codex/natural-reciprocal-transfer-20261127/validation-passage/v1`, family name (`two_answer` or `answerability`), exact title and passage SHA-256 with LF, with no trailing LF. Break digest ties by ascending original passage SHA-256. Two-answer pairs are internally ordered by case ID then ranked lexicographically by the two IDs. The first ordered pair represents a passage. Answerability preserves the roster's positive/impossible side order and its single pair per passage. Use up to the first four passage representatives per title for review. The two-answer title capacities of one and two are part of the fixed sample, giving 23 plus 44 review pairs. Source decisions and outcomes cannot change these representatives.

Every mechanical candidate gets a passage rank and a disposition. `review` enters source adjudication, `alternate_within_passage` is a nonrepresentative, and `outside_article_quota` is a passage representative beyond rank four. Use the full manifest to reconcile the per-title totals, then append one source disposition and one instrument disposition per `review` pair, including which side, signed reviewer reason and isolated instrument cause. Keep the complete excluded set. Every gate-ready result accounts for all 187 two-answer and 256 answerability mechanical pairs, all selected pairs, and every strict result from the final common qualified panel. No resampling, even when adjudication rejects most of a title.

Verify the implementation offline with `cargo test --locked --offline --manifest-path "$DRAFT/Cargo.toml"`. The tests exercise deterministic ranking, selected count, shortage accounting, distinct passage selection, and duplicate rejection with synthetic IDs.
