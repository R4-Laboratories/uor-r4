# Independent observation grading and D11 replay

The scorer joins returned device rows to reference rows and separately retained
wording keys. It uses [the #4321 provenance audit](../../codex-4321/audit.jq)
before computing new metrics. Run with the audit copied beside these files.

On Wick, run once for each returned configuration and split. Process baseline
first to establish the pair-ID reference for each comparison. Run the three
splits of a configuration serially so that they reuse the same packed
checkpoint. Run held replay only after phase 2 returns.

```sh
bash replay.sh baseline train
bash replay.sh baseline dev
bash replay.sh baseline protect
# Then run the same three splits for balanced, weak and both.
bash selection.sh
```

`analysis-v1/CONFIG/SPLIT/` records the frozen audit, the per-pair and
stratified scores, differences from baseline, and original protected pair
retention. A directed contrast requires a negative short score and positive
long score. Signed margins are the negated short score and the raw long
score. Both must reach eight to count toward the registered margin check.
Exact requires terminal EOS and the full expected original-ID sequence.

`cold-v1/CONFIG/` contains the assigned weight state packaged with its exact
fit receipt digest. Each split has all-arm device parity and a separate
ordinary serving trace, with only the tokenized prompt and four positions
in the input. The comparison covers all ordinary trace fields except
`first_mismatch`, which the reference-free client must leave null.
