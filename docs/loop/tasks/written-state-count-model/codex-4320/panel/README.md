# Frozen stop and continue panel

The same two facts, queried person, question and two authorized original token addresses occur on both sides of a pair. Each person has a separately named opening word and follow-up word. The final directive asks for the opening word and period on both sides. The continue side also asks for one space and the queried person's follow-up word. The original follow-up token is one leading-space piece, written exactly as in the source. The two reference sequences use EOS ID 1 and period ID 16.

| Side | Raw reference |
|---|---|
| 0, stop | `[target, 16, 1]` |
| 1, continue | `[target, 16, next_id, 1]` |

The held tape contains 200 pairs across four wording families of 50 pairs each. In each held family there are 25 pairs per directive variant, 25 per queried fact position, 25 per source order, and 25 per directive-final punctuation condition. Both sides share the same punctuation condition. Each of the ten values occurs five times in each lexical role within each family. The two collision IDs 1808 and 3984 are intentional prior-value exceptions. The other held values and all finite values are screened against earlier tracked target/rival and donor-opening identities. The finite tape has 16 pairs, with separate facts, people, words, and final wording.

The Rust preparation tool uses the ordinary protocol-2 dialogue encoder with the pinned 4,096-token vocabulary. It checks the raw user round trip without fallbacks, the exact fact and question boundaries and assistant marker, and the complete reference response with its emission mask. Each of the four factual value IDs occurs exactly once in the raw prefix. Every value position binds the original token bytes within its authored source statement. Both held variants preserve this binding. The paired suffix check allows differences only after the question mark. The screen rejects repeated prefixes and full text from the prior evaluation and donor-prefix records, and repeated facts, questions, people, and templates from the immediately preceding sealed #4318/#4319 panels, including #4319 TRAIN and finite. The broad corpus screens are exact digest and lexical-identity checks. Words can occur as ordinary tokens in the broad corpus.

## Prepare and verify

The private tapes are on Wick at `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-postperiod/panel-v1`. Each split has its own rows, keys, census, and digest manifest. Their reproduction generated identical files in a separate directory. The [root manifest](SEALED.sha256) SHA-256 is `56091ad40d7e5da6f3d6742e02f04e37cdb01171fc3fddb4acba9bcf4f11b876`.

```sh
cargo build --locked --release
target/release/postperiod-panel prepare "$RUNS" "$NEW_PANEL"
POSTPERIOD_RUNS="$RUNS" cargo test --locked --release -- --include-ignored
target/release/postperiod-panel score "$TOKENIZER" "$SEALED_PANEL" held "$OBSERVATIONS" "$NEW_ANALYSIS"
```

The scorer joins every observation to its complete sealed row and the actual opposite side. It requires both source addresses, the authenticated copy, the supplied period, provenance for the intervention, and equal subsequent events for equal committed prefixes. An exact reply requires the whole sequence through observed EOS. The scorer counts an EOS selected at step three and a native EOS separately. A non-EOS at the fourth and last allowed emission counts as an open reply at the cap. The private analysis contains per-row decoded output, fixed input-selected examples and separately labeled outcome-selected examples. The aggregate records directions of the paired third-token choice, first wrong IDs, and full exact replies and pairs by wording family, directive punctuation, and collision status. The Wick instrument test plants both natural and supplied EOS and a final wrong token at the cap, then checks that a false terminal flag is refused.
