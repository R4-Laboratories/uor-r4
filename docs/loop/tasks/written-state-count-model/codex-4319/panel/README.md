# Word ending panel

Each pair uses the same two facts, question, word and authenticated source address. The instructions following the question mark ask either for the word alone or for the word followed by a period. The expected raw replies are `[word, 1]` and `[word, 16, 1]`, respectively.

| Split | Pairs | Rows | Literal ending instructions |
|---|---:|---:|---:|
| Train | 192 | 384 | 24 across six families |
| Held | 200 | 400 | 16 across four families |
| Finite | 16 | 32 | 4 across two families |

The four held families have 50 separate fact and question pairs each. Each family uses two paired instruction variants, 25 fact and question pairs per variant.

| Held family | What the wording asks the reply to do |
|---|---|
| Contrastive correction | Follow the phrase contrasted against the alternative after “not.” The same words occur in both instructions within each variant, with their order changed. |
| Final character | Constrain the last character to a letter in the copied word or to a period after the word. |
| Referenced rule | Apply the rule stated in the preceding sentence. “Use that allowance” is an imperative to include the period authorized by that sentence. |
| Scope exception | Follow a copy-only instruction with a period exception that applies to this reply or to other replies. |

Every held family has 25 targets in the first fact, 25 in the second, and an equal split of the target before and after the person's name. Each family uses five pairs for each of ten target IDs, including ten combined `lake` (1808) or `anchor` (3984) cases. The other 160 held pairs use eight distinct target IDs. All three splits have disjoint word IDs, people, pair IDs, prefixes, fact and question text, and question and ending templates. The #4318 screen covers its sealed exact contexts, bare questions, full prompts, prefixes and pair IDs, as well as its question and ending templates. The earlier #4318 corpus, donor token and historical prompt screens run using the pinned history loader. Across held instructions, 49 of 104 distinct tokenizer IDs also occur in train instructions.

The tokenizer's protocol-2 encoder constructs the open and complete dialogues. The preparation tool checks the exact fact delimiter `[201, 201]`, exact question token `33`, and assistant marker `[201, 35, 560, 652, 714, 28]`. The answer token must occur exactly once in the prefix, within the designated fact. The complete assistant tail must match the expected IDs, have response masks on all emitted positions, and decode to the authored reply before its EOS. Private keys retain the original rendered messages, reference answers and source facts.

## Reproduce and score

The two preparation passes produced identical manifests. The private Wick panel is at `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-word-ending/panel-v1`. The [file digest manifest](SEALED.sha256) binds each tape and the full private receipt. Its SHA-256 is `7df4af3f38ef7775caaf02326cee8dd8fa14de9ba043f23cee038cdd61f25fc0`.

```sh
cargo build --locked --release
cargo test --locked --release
target/release/word-ending-panel prepare "$RUNS" "$NEW_PANEL"
target/release/word-ending-panel score "$TOKENIZER" "$SEALED_PANEL" held "$OBSERVATIONS" "$NEW_ANALYSIS"
```

Use a fresh private output directory on Wick. The scorer joins every observation to its entire sealed row, verifies that the swapped state came from that row's opposite instruction, and requires the same authenticated first commit in every arm. At step two it verifies the one-use ending choice and its native fallback. When step three runs, it verifies that the ordinary writer emitted the observed token and repeats the same transition across arms that committed the same actual prefix. The scorer counts an exact pair only if both arms' corresponding replies match every target through observed EOS. It reports first mismatches, token-two choices, and stop and period counts by family, including gains, losses and retention against native exact replies on the same cases. The private CSV includes actual and expected token IDs, decoded replies and token-two pieces, terminal flags and first mismatches. Private examples select up to two faults per family, class and arm in ascending pair-ID order.

The grading test uses planted events. It accepts `[1808,16,1]`, marks `[1808,16,73]` as wrong at token three and still open at the cap, and rejects a claim of termination without EOS. Planted events are separate from the generated finite and held measurements.
