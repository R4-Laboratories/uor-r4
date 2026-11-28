# Natural whole-turn source and NoRead candidates

SQuAD 2.0 supplies human-authored extractive questions and adversarial questions marked impossible. The original address of the published positive annotation was verified mechanically. The dataset answerability labels have known semantic errors in all three splits; see the later adjudication note linked below. Every NoRead label comes from an official impossible question with an empty gold list and an exact annotated plausible distractor inside the retained passage. The model-facing user message preserves the full source paragraph and question, joined by one space, in the normal protocol-2 user role. The original text and IDs are in the Wick packet. [Published source and license](https://rajpurkar.github.io/SQuAD-explorer/)

| Split | Paired paragraphs | Source articles | Same first original position | Same full annotated span | Immediate positive opening | After one space ID 223 | Complete word at immediate opening |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fit from official TRAIN | 130 | 25 | 96 | 92 | 114 | 16 | 7 |
| Article reserve from official TRAIN | 200 | 29 | 176 | 173 | 170 | 30 | 6 |
| Published DEV | 25 | 3 | 18 | 17 | 23 | 2 | 2 |

Each pair includes one supported and one impossible row. The pools before pairing contained 1,227 supported and 387 impossible fit rows, 1,586 supported and 643 impossible reserve rows, and 53 supported and 50 impossible published DEV rows. Identical-span pairs with one word-level question edit numbered 44, 74 and 3, respectively. Paired positives with immediate lexical openings and the same first distractor position numbered 81, 149 and 16. These are question pairs nested within articles. The published DEV cohort covers three articles and remains descriptive. [Census](../evidence/supply-census.json)

The 192-ID product window reserves 12 IDs through EOS and admits prefixes up to 180 IDs. The selected prefixes span 84–180 fit IDs, 72–180 reserve IDs, and 76–173 DEV IDs. The context limit excluded 51,205 fit rows, 41,814 reserve rows and 7,999 DEV rows after earlier filters. Long paragraphs were discarded whole. Every selected positive encodes to the same lexical IDs in the source span and canonical answer. A first source ID is sufficient for the complete lexical span on 16 of 355 positives, including one after a leading standalone space in the reserve. The other 339 require one to ten later IDs before EOS. A current one-shot copy at the assistant colon can directly target only the immediate rows. An after-space diagnostic appends one 223 without moving the original source address. [Opening and continuation inventory](../evidence/supply-openings.json)

## Label and split checks

- The verifier checks the version and SHA-256 of the author-published files, the production tokenizer, the prior #4315 human keys, CoQA DEV and the D87 prompt digest bank. It verifies each source annotation by its original character offset and UTF-8 bytes, round-trips the unmodified role content through the pinned encoder, and locates the original ID at the annotated byte range in the single user role. Ambiguous occurrences across the user turn are excluded. Published DEV positives require a strict majority of at least two exact annotations.
- The initial census used all published TRAIN to establish supply. The candidate split then used a fixed salted SHA-256 threshold on the normalized article title independently of individual labels. All 199 reserve articles are withheld from the fit pool. Prompt, normalized question, title, passage and 16-word passage overlap checks protect the reserve and published DEV from fit. The CoQA DEV article names where available and passages, the prior #4315 selected prompt and question text, and D87 bank digests screen historical overlap. Those checks cover identified inputs and exact normalized overlap. The earlier capacity dry run mechanically enumerated reserve candidates, and broader language-model corpus exposure was unmeasured.
- The original IDs, labels and prompt bytes rejoined across all 710 rows in a second reader on Wick. The independently rendered role bounds, gold or distractor token, span, pair membership and SQuAD label agreed. A second build invocation produced byte-identical files before sealing. Human supervision here concerns extractive reading from a supplied passage. Explicitly denied or merely queried facts inside the passage, prior-assistant leakage, and asymmetric speaker roles require the separate controlled and raw conversation inputs. One-word edits in the question exercise some wrong argument or negation conditions, with softer question matches retained as their own analysis cohort. [Rejoined source counts](../evidence/supply-readback.jsonl)

## Run and custody

The Wick packet is `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-whole-turn-census/supply-v1`. Its manifest SHA-256 is `b8cc4c6ff0e65f731cf909df48c72e1497d72217d431d1d75c23ae50cdd5201e`. The source-extraction readback selection is `supply-check-v1.jq` beside the packet. [Sealed files](../evidence/supply-manifest.sha256)

Use the existing Wick input root and Rust in this directory. Run the tokenizer-only preparation on Wick, with a new private output path.

```sh
cargo build --locked --release
RUNS=/mnt/plexpool/uor-r4-corpus/runs
SUPPLY="$RUNS/20261127-codex-whole-turn-census"
umask 077
target/release/whole-turn-supervision "$RUNS" "$SUPPLY/supply-inputs" "$SUPPLY/supply-rebuild"
```

Each `*-prompts.jsonl` row holds `case_id` and raw `prompt`. The matching private `*-truth.jsonl` row holds the official question ID, source article and paragraph, reciprocal pair ID, whole-user token bounds, exact prefix and original char/byte ranges. `gold_source` is null for impossible cases. `plausible_only` stores the exact distractor, including the diagnostic reply encoding, for impossible cases. `response_lexical_offset` measures distinct leading space IDs. `response_ids` follows canonical protocol-2 answer encoding before terminal EOS.

A later reader inspected the passage and question meaning and found erroneous published labels. Use the [adjudication note](../evidence/label-review.md) with both the archived tapes and the revision-pinned preparation recipe.
