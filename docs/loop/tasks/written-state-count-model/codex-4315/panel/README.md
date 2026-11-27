# Source-occurrence panel for #4315

The panel tests source selection with the frozen protocol-2 tokenizer and original token IDs. Each positive controlled group crosses two ordinary questions with two assignments of the same two labels. Under a fact swap, only the two label IDs change in the causal prefix for either question. Under a question swap, the fact region stays identical and the gold and rival occurrences reverse. Both answers are whole words and each has exactly one original-ID occurrence in the complete prompt. Each split uses four distinct wording families with 50 groups per family. Every split has 50 groups in each joint cell of fact order and whether the value precedes its subject. The source key records the topic, actual subject and value assignments, and matching information for the controls.

| Private tape | Groups | Rows | Contents |
|---|---:|---:|---|
| `train/rows.jsonl` | 200 | 800 | Museum badges, radio pseudonyms, bakery counters, repair workshop signs |
| `train-absent/rows.jsonl` | 200 | 200 | One third-subject question per TRAIN group with both original facts preserved |
| `held/rows.jsonl` | 200 | 800 | Greenhouse inventory, archive shelves, regatta pennants, courtyard inscriptions |
| `absent/rows.jsonl` | 200 | 600 | 400 omitted-value prompts and 200 third-subject prompts matched to HELD |
| `human/rows.jsonl` | 29 | 29 | Unseen CoQA dev v1.0 excerpts and original human questions |

The omitted-value control asks the original question after replacing the requested subject's value-bearing fact with a neutral statement about that subject. The requested token ID is absent from the entire prefix and the rival fact remains. The third-subject control keeps both tempting facts and asks about a different subject. Its two diagnostic IDs remain present under a query about an unasserted subject. Both controls have a null `gold_pos` and a null private `reply`. `target` and `raw_target` on these rows trace to the matched controlled positive. Any selection of a source counts as unsupported. The supplied-missing intervention is the separate runtime control applied to ordinary present-source prompts.

## Answer novelty and prior exposure

All 76 controlled HELD words and their original target/rival IDs are disjoint from new contrast TRAIN, the #4314 fitted fallback contrast, and the genuine canonical D87 donor opening set. The donor check also excluded IDs that were the first non-whitespace token of a genuine opening and exact first words of genuine donor replies, by decoding the pinned canonical answer spans. The canonical exposure scan used all 4,345 genuine openings and all 118,915 colon rows. All 800 HELD target occurrences have IDs already present somewhere in the canonical donor tape. Of those 800 held rows, 362 have a target ID that occurred at some donor colon. Another exposure count in the sealed receipt, `ids_anywhere_in_positive_training_prefixes`, counts held rows. Its value of 12 corresponds to one distinct held target ID, 2057, occurring incidentally in positive TRAIN prompt text and serving as the target on 12 held rows. The same numbers for rival occurrences follow from the crossed assignments. Novelty concerns supervised answer openings and contrasts. The raw donor tape contains ordinary vocabulary exposure.

Exact full prefixes and normalized prompt text are unique across all five tapes. The verifier excludes 2,844 historical exact prefixes, 928 historical normalized question types, 590 unordered historical subject pairs, exact normalized prompts or questions in the D87 digest bank, and every exact canonical donor colon prefix. The positive controlled split retains the four different wording families on each side. [Frozen audit and input pins](RECEIPT.json)

## Human passages and evidence

The text screen excludes the 210 source IDs used by #4312–#4314 and their duplicate full normalized story text. It offers bounded original excerpts that contain the literal evidence, with an exact literal answer match inside that evidence and agreement among at least three original annotations. The panel reviewer inspected the wording, evidence and excerpt in private and selected one self-contained question per distinct passage before fitting. All selected answers have four agreeing annotations under normalization. The user message contains only the original excerpt and original CoQA question. The original CoQA excerpts span CNN, Gutenberg, MCTest, RACE and Wikipedia. [Text-only selection record](coqa-review.tsv) · [CoQA data attribution](https://stanfordnlp.github.io/coqa/)

A human source is **unique** when the first non-whitespace reply ID occurs exactly once in the factual span and that occurrence begins the original answer within its annotated evidence. An **ambiguous** source has multiple occurrences of that ID in the facts. **Absent** means the lexical text is grounded but that exact token ID cannot be assigned to a supported occurrence, for example because case or the serialization boundary changes the token. Both non-unique statuses have a null `gold_pos`. Keys preserve the passage filename, source conversation and turn, original annotation span in characters, excerpt/evidence/answer byte ranges relative to the original UTF-8 story, exact rationale text and all occurrences of the target ID in the excerpt. The human rival is a separate unique occurrence representing a complete word outside the evidence. It is a positional distractor. This cohort has 22 unique, three ambiguous and four absent exact source addresses. All 29 first lexical IDs are absent from the question portion of their own prompt.

Two natural gold continuations begin with a standalone space ID 223, including one row with an otherwise unique lexical address. `raw_target` is the actual first token even on those rows. `target` remains the first non-whitespace ID and `lexical_offset` locates that step. Fourteen human `raw_target` IDs and 12 human `target` IDs appear in the canonical donor's immediate-opening set. One human target appears in the old fallback contrast and none appears in the new contrast targets/rivals. This transfer cohort was selected for grounded text and original questions with observed answer-ID exposure reported alongside the controlled novelty test.

## Causal rows and custody

`prefix` is the full ordinary prompt through the assistant colon, bounded to 192 original IDs. It uses one user message containing the factual text, the literal two-newline separator, and the question. On every row, `context_start` and `context_end` are zero-based half-open absolute offsets into that prefix. The two annotated positions are absolute offsets into the factual region and, when present, index the reported target and rival IDs. The additional metadata supports supervision, grading, matched counterfactual interventions and audit. Learned forward selection consumes causal `prefix` IDs. Private `keys.jsonl` holds raw text, topics, answer IDs and source annotations. The legacy #4314 fields remain directly readable by its tooling.

The sealed private Wick directory is `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-shared-source/panel-v1`. Its manifest SHA-256 is `5a405952795ed13cff8d7737ee2027f021664252bfeed1d7e7f52f4213602dd8`. The preparation verifier reconstructs the 2,429 canonical prefixes, checks the source-key cross and absence semantics, and audits exposure separately. The persisted-rows pass reopens the source dataset to confirm the exact human question, reply, excerpt and evidence. A fresh x86-64 build on Wick reproduced every file that the arm64 preparation build sealed. Preparation used only the CPU corpus and tokenizer instrument. [Sealed file manifest](SEALED.sha256)

Rebuild with the Rust package in this directory. Set `RUNS` to the input root that contains the pinned #4312, #4313 and #4314 directories shown in the receipt. Write to a new private directory. The optional preview prints raw CoQA text and belongs in private custody.

```sh
cargo build --locked --release
target/release/shared-source-panel preview "$RUNS" > "$PRIVATE/coqa-preview.jsonl"
target/release/shared-source-panel build "$RUNS" coqa-review.tsv "$PRIVATE/new-panel"
target/release/shared-source-panel verify "$RUNS" coqa-review.tsv "$PRIVATE/new-panel"
```
