# Which natural openings actually teach original-source admission?

## What did not work

The old five “unique” openings reproduce, but their target tokens start two editing tasks, two proofs and one open completion. None identifies a requested asserted value. Expanding the candidate range recovers hundreds of single-occurrence spelling matches. An exhaustive semantic review of such matches, a separate stratified sample, and a targeted review of one recurring filename family produced 39 verified requested-value openings. Every one follows a separate space ID 223. None teaches an immediate authenticated factual copy at the ordinary assistant marker. The 39 are known witnesses from the reviewed subsets; the semantic census of all duplicate-ID turns remains incomplete. [Source audit](semantic/README.md)

## What changed

The Wick Rust census rebuilt the corrected D87 visible conversations from the original records with the product tokenizer, preserved the actual current user boundary and answer offset, and compared the results with the full donor UORT and mask. It accounted for 4,345 assistant openings across 4,060 groups and 5,731,000 visible IDs with their response masks. It separates the actual first ID, the first non-whitespace ID, the current user's positions, older history, context length, ambiguity and complete-word evidence. No writer or learned weight changed. [Census and replication](census/README.md)

## Measurements

| At the first relevant target | Unique original ID in current user turn | Unique and reachable in the ordinary context | Reachable and a whole word at both locations |
|---|---:|---:|---:|
| Immediate reply ID | 329 | 194 | 74 |
| First non-whitespace ID | 527 | 341 | 80 |
| First non-whitespace after a separate space (subset of preceding row) | 198 | 147 | 6 |

The existing facts-before-blank-line framing parsed 188 genuine openings. In ordinary initial context 2,262 complete prefixes fit and 2,083 exceeded 192 IDs. Of 285 conversations with an earlier assistant reply, 25 fit with the real history. Removing that history changes the experiment. Three prompts at exactly 192 remain reachable for the first content token only by sliding the window after space. Actual marker-ID sequences occur inside the user’s literal text in 303 turns, including 63 context-eligible turns. Use the current message bounds from serialization for training and raw serving. [Count tables](evidence/census.json)

Of the 39 reviewed positive addresses, 38 ask for a field in a filename, and one asks for a stated inventory count. Thirty-eight confirmed filename examples and two genuinely unsupported filenames fit the ordinary window. Two independent readers agreed on a separate enriched 24-row panel, including all five historical matches, and checked the inventory and structured examples against original bytes. A single initial digit can occur eight times yet only one occurrence begins the correct episode number. Conversely an initial digit occurring once in a formatting example can be unrelated to the answer. Keep these distinctions when choosing negative examples. [Semantic review and coverage](semantic/README.md)

**Actual client behavior.** On Wick the balanced artifact replayed five exposed natural first turns under native generation, authenticated copy and occurrence-disabled generation (15 replies). The inventory answer begins with IDs 223, 25, 24 (space and the digits of 76). The authenticated client rejected its framing and emitted 13 as token one. The tagged episode answer begins 223, 22, 24, 1 (space, the digits of 46, EOS). That client instead copied 1255 from prompt position 14 at token one; the requested first digit is at position 52. Disabling the reader on that example produced the actual space and then another space instead of ID 22. The ordinary one-shot copy action is already consumed after the first step. All three arms completed 0/5 references exactly. Three references exceed the registered cap of 12; the two shorter ones also failed. [Fitted input identities and diagnostic scope](replay/FIT.md) · [Actual commits and first errors](evidence/donor-replay.json)

**Usable supervision.** From SQuAD 2.0 we prepared unmodified paragraphs and questions in paired, single-paragraph user messages. Each pair contains a published positive annotation whose source bytes and address reproduce, and a question the dataset marks impossible with a plausible span in the same paragraph. Semantic spot checks found answerable questions among these official NoRead labels. The tapes preserve both original coordinates and full annotated lexical IDs. They are designed for source selection. A legitimate NoRead decision does not specify a uniquely correct conversational reply. [Dataset source and license](https://rajpurkar.github.io/SQuAD-explorer/) · [Build, exclusions and limitations](supply/README.md)

| Published QA pairs | Articles | Positive first ID is immediate | Positive needs space first | Same first original address as annotated plausible source |
|---|---:|---:|---:|---:|
| TRAIN, 130 | 25 | 114 | 16 | 96 |
| Article-disjoint reserve, 200 | 29 | 170 | 30 | 176 |
| Published DEV, 25 | 3 | 23 | 2 | 18 |

All 710 model-facing rows survived an independent metadata and source-offset join. There are 339 multi-lexical-ID positives among the 355 pairs. On TRAIN, seven positives are complete words in a single immediately copyable ID. The article reserve was constructed after a capacity audit and is disjoint from the fit articles. A separate authored tape provides 240 TRAIN rows (144 positive, 96 NoRead) and 120 DEV rows (72 positive, 48 NoRead) crossing fact order, requested person, ownership, duplicate answer tokens, denial and unanswered questions. In the denial and question conditions, changing the person asked about while keeping both factual clauses identical flips the correct action. These examples are controls for learning and testing bindings; the human QA and unconstrained dialogue outcomes remain separate. [Contrast generator](contrasts/README.md) · [Original opening phase counts](evidence/supply-openings.json)

## Checks run at the head

The previous #4315 admission executable reproduced its receipt byte for byte. The new census reproduced all private per-opening rows and its summary on a second run. A changed class-table digest was refused. The served prefix was independently checked for all 15 diagnostic replies. The supplemental data builder checked every source against original text and tokenizer IDs and reproduced the SQuAD tapes. [Census inputs and outputs](evidence/census-wick.sha256) · [Natural tape manifest](evidence/supply-manifest.sha256) · [Restore](RESTORE.md)

## Skills used

The native-space-cycle protocol governed original integer position identity, role boundaries, serving phase and token parity. This was corpus inventory and frozen client serving on Wick. No optimizer update or GPU rental ran in this unit.

## Still open

The immediate fit has annotation-based candidates and controlled authored negatives. Independently adjudicate both sides of the natural pairs and reseal accepted pairs before using them as verified supervision. It requires the same role-bounded candidate operator in fitting and ordinary serving, protection of the four earlier duplicate-address cases, and a separate causal test for after-space sources. [Next unit](PICKUP.md)

## Plain summary

The current donor alone gives misleading opening labels. The authored controls support a bounded comparison. The SQuAD pairs supply annotation-based candidates for passage-level review. The existing writer still fails on the natural examples even before it can use their verified source.

A revision-pinned cold download reconstructed all 13 reproducible outputs byte for byte. The 65 packaged members and the separate manifest of 33 private Wick files were verified. [Restore and archive identity](RESTORE.md)

**Correction after cold restoration.** An independent reader found published-label errors in TRAIN, article reserve and published DEV, including a reserve pair whose questions reverse the applicable entity. The 710-row mechanical verification establishes agreement with the published record. It does not measure question answerability. The sealed packet remains reproducible under its original annotations. [Question IDs and disposition](evidence/label-review.md)
