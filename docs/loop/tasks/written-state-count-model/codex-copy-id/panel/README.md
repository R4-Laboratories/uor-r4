# Authenticated-copy opening panel

The panel holds 200 independently screened prompt groups. Each group crosses two questions with two assignments of two literal labels. Under a question swap the factual prefix is identical and the correct and rival occurrences reverse. Under an assignment swap only the two label IDs change in the full causal prefix, at their registered addresses. Questions use four new families with 50 groups each. Every family balances fact order, label placement before or after the subject, and whether the requested response includes a period.

| Tape | Groups | Rows | Purpose |
|---|---:|---:|---|
| `held/rows.jsonl` | 200 | 800 | Primary four-way comparison |
| `absent/rows.jsonl` | 200 | 600 | 400 omitted-value rows and 200 unbound-subject rows |
| `finite/rows.jsonl` | 16 | 16 | Eight identical-ID alternate occurrences and eight malformed inputs |

There are 40 primary groups with both original IDs 1808 (` lake`) and 3984 (` anchor`). They reuse labels from the previous held panel in independently screened questions, subject pairs and factual text. The [frozen writer census](../../codex-4315/evidence/writer-identity.json) records root code 47768, cell 139 and native EOS under the earlier supplied-source diagnostic for each ID. The other 160 groups draw from 20 distinct whole-word IDs, with zero overlaps against the #4315 controlled training and held-or-human answer or rival IDs, the #4314 fallback training contrast and the 4,345 canonical donor opening IDs and first words. All 800 target occurrences have an ID somewhere in the canonical donor tape. There are 272 of 640 distinct-word targets and 160 of 160 collision targets in the set of IDs seen at any canonical donor colon. The crossed assignment gives the same exposure for the rival. The input audit also screens earlier panel text and pairs, the D87 prompt digest bank and canonical donor prefixes. [Registered census and input pins](RECEIPT.json)

The exact protocol-2 dialogue encoder supplies the full prefix through the ordinary assistant marker, both answer pieces and their positions. All primary answers open with the leading-space whole-word ID. A registered response to the period request contains that ID, period ID 16 and EOS ID 1. A response to the no-punctuation request contains the ID followed immediately by EOS ID 1. The first two tokens therefore grade 400 responses with an EOS successor and 400 with a period successor. The row-level key retains literal user and response text, the token IDs and exact decoded pieces. `expected_terminal_after_first` is false on all answerable rows, and `expected_terminal_after_second` is true where the written response reaches EOS at step two. An unsupported or malformed item has no supervised target or successor. Its `target` and `rival` fields identify the matched diagnostic words.

A duplicate row has an authorizing factual sentence and an unrelated example note, each with a separate occurrence of the same literal target. The key records both sentence byte ranges, sentence text, absolute token positions and original ID. Testing a claim for the unrelated note against the genuine fact read exercises address authentication even though both positions decode to the same text. Four rows put the note before the facts and four after. Malformed inputs are four single-space fact/question separators and four broken assistant markers. Every finite row is outside the primary denominator.

The sealed private Wick directory is `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-copy-id/panel-v1`. Its manifest SHA-256 is `a6b8962b01e0a9bbb866496ad4f1d95a00770c5e419ca98025039c43ed83d8b0`. The panel builder only uses registered corpus and tokenizer evidence. [Sealed files](SEALED.sha256)

## Rebuild and tabulate

Build the Rust package here. `RUNS` points to the private directory holding the four prior run roots listed in the receipt. `TOKENIZER` selects the exact tokenizer pinned in the receipt. `OBSERVATIONS` is the returned GPU JSONL with each sealed row and the eight measured arms, keyed by pair ID and raw prefix. Use fresh output paths.

```sh
cargo build --locked --release
target/release/copy-opening-panel build "$RUNS" "$NEW_PANEL"
target/release/copy-opening-panel verify "$RUNS" "$SEALED_PANEL"
target/release/copy-opening-panel report "$TOKENIZER" "$SEALED_PANEL" "$OBSERVATIONS" "$NEW_ANALYSIS"
```

The report binds each GPU observation to the sealed prompt and its text-derived grading by pair ID and full prefix. It verifies recorded address authorization, rejection, identical learned/native reading, exact EOS termination and native successor consistency after identical committed IDs. The CSV gives an arm-level decoded trajectory and identifies the exact ID sent into the second step. A separate CSV gives the matched first-token and both-token comparisons per row, assignment and quartet. The summary gives these match counts, paired discordance and conditional second-step counts. It includes three preselected comparisons of learned Copy against ordinary Generate, supplied correct against its same-read native proposal, and supplied correct against a valid rival. The collision examples are the four cells of the lexicographically first collision group, selected independently of output.
