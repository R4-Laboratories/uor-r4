# Replay and verify from Wick

All private work is inside `/mnt/plexpool/uor-r4-corpus/runs/20261128-codex-grounded-release-supply`. The [custody manifest](results/WICK.sha256) hashes to `a27c1895a6917f3e29082ce1691ae1a4eb2f448dde5ee42d1e951e0cc8d9b7ed`. It authenticates the donor runs, both private source reviews, the review grade, and the three human census manifests. The existing donor and extractive corpus data remain in their original custody.

## Donor and human counts

Set `UNIT` to this checkout's absolute task directory on Wick. Use a fresh private output directory for each replay. The Rust drivers use the bound tokenizer, Rust 1.97.1 and the pinned #4329 host dependencies. The two unchanged human eligibility searches use the [SQuAD recipe](../codex-4336/supply/continuations/README.md) and the [DROP recipe](../codex-4336/supply/drop/README.md). The [human-v1 scope](results/human-v1/SCOPE.md) describes the original custody readback. The stricter supplement reuses the pinned DROP intake and rejects writer delimiters and repeated original word surfaces even when tokenized differently.

```sh
RUNS=/mnt/plexpool/uor-r4-corpus/runs
STORE=$RUNS/20261128-codex-grounded-release-supply
bash "$UNIT/replay-d87.sh" "$RUNS" "$STORE/d87-replay-v1"
cmp "$STORE/d87-v5/SHA256SUMS" "$STORE/d87-replay-v1/SHA256SUMS"
cmp "$UNIT/results/d87.json" "$STORE/d87-replay-v1/d87.json"

bash "$UNIT/human/replay-supplement.sh" "$RUNS" "$STORE/human-replay-v1"
cmp "$STORE/human-v3/SHA256SUMS" "$STORE/human-replay-v1/SHA256SUMS"
cmp "$UNIT/results/human.json" "$STORE/human-replay-v1/strict-supplement.json"
```

The retained donor review packet SHA-256 is `175d447dfe8ed8ac41a0420ba64f1aa81a15cb0c072d36765715380549ff95f0`. Packet rows contain exact visible requests, answers and source addresses. Each source reviewer independently judged whether the request directly asked for the exact supporting occurrence and the terminal reply fulfilled that request. The four TSV columns are record ID, zero-based turn, decision and a private reason. The decisions are `s` for supported, `r` for incidental or unsupported, and `a` for ambiguous. The grade requires one decision per sealed stop candidate in packet order and counts the decisions and disagreements separately.

```sh
(
  cd "$STORE"
  sha256sum --strict -c "$UNIT/results/WICK.sha256"
  for run in d87-v2 d87-v3 d87-v5 d87-v6 review-a-v1 review-grade-v1 human-v3 human-v4; do
    (cd "$run" && sha256sum --strict -c SHA256SUMS)
  done
  (cd review-b-v1 && sha256sum --strict -c RETURN.sha256)
  (cd human-v1 && sha256sum --strict -c VERIFIED.sha256)

  jq -Ss --rawfile a review-a-v1/judgments.tsv --rawfile b review-b-v1/judgments.tsv \
    -f "$UNIT/grade-reviews.jq" d87-v2/private-review.jsonl | cmp - "$UNIT/results/source-review.json"
)
```

The published JSON files contain counts. The source text and judgments remain on Wick. The releases and #4336 client controls were already measured and are linked in the [report](REPORT.md).
