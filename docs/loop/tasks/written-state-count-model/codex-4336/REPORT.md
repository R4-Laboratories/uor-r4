# The ordinary writer can release a verified span and still miss the next requested token

## What did not work

[codex] The balanced checkpoint completed 0/24 fresh supported replies through EOS. It missed the first content ID on every reply. Supplying the correct span moved the failure to the next decision on all 16 replies that could complete the span, but leaving that decision to the ordinary writer still gave 0/16 exact. On the other eight replies, the requested first span was nine tokens long and the source cursor exhausted its eight-token budget. The two shorter lengths identify a reachable action failure. The longer length identifies a copy limit. These are ordinary-client integration measurements with frozen weights. [Registered study](ORDER.md) · [Fresh panel](evidence/panel.json)

## What changed

The integer library now has an opt-in span-release operator for the protocol-2 single-user client. A complete serialized prompt and a trusted original user span provide the reference. A separately presented claim must agree on start, exclusive end and every exact ID in the original user range. The cursor uses the existing exact-copy writer, retains the prompt safety fence and enforces its eight-token cap. At a verified end, the current decode step may take the ordinary writer token, emit a supplied EOS, or enter a second authenticated span and copy its first token immediately. A second span has its own end and subsequent action. The same ordinary decoder then consumes the actual committed ID. The diagnostic accepts trusted annotations and supplied choices. The balanced D11 artifact and its weights are unchanged. [Span transition](../../../../../crates/uor-r4-integer/src/discrete_d11/native_span_release.rs) · [Ordinary client integration](../../../../../crates/uor-r4-integer/src/bin/uor-chat.rs)

## Measurements

The fresh panel contains twelve matched stop/continue pairs and three unsupported requests. It crosses two request wordings and two fact orders within each of three source families. Each family has one distinct first-span length, so the panel alone cannot estimate a length effect across families. Offsets below are zero-based. Repeated arms share prompts. [Panel construction and grading](panel/README.md)

| Ordinary decode path on 24 supported replies | First wrong offset | Exact replies through EOS | Pairs with both replies exact and correct source addresses |
|---|---|---:|---:|
| Balanced, native authenticated opening | 0 on all 24 | 0/24 | 0/12 |
| Verified original span, native release choice | 7 on eight, 8 on sixteen | 0/24 | 0/12 |
| Verified original spans, supplied correct choices | No error on sixteen, 8 on eight cap exhaustions | 16/24 | 8/12 |
| Verified original spans, swapped first choice | 7 on eight, 8 on sixteen | 0/24 | 0/12 |

For all seven-token spans, the native writer emitted ID 210 at the verified end on both sides of the request swap. Stops required EOS (1) and continuations required the next source ID (269). At eight tokens, the native writer emitted 16 on all four stops and on three of four continuations. It emitted 465 on the remaining continuation. With correct choices all eight shorter stop/continue pairs completed both sides. Swapping the choice made all 16 finish the opposite reply through EOS. On the eight nine-token cases, every action arm lost the ninth source token (1413) at offset 8 before reaching the release decision. [First wrong IDs and paired controls](evidence/panel.json)

One eight-token continuation copied addresses 10 through 17. On the decode step at exclusive end 18, the ordinary proposal was 16. Taking the supplied continuation copied ID 269 from address 48 on that same step. It continued through address 52 and emitted EOS at exclusive end 53. No ninth token was read from the first source. The source for a continuation and the decision to continue are distinct inputs to this arm.

All 36 first-claim perturbations refused at the opening. The two wrong second claims refused at continuation. The two delimiter controls and the three unsupported-claim controls also followed their registered paths. The native fallback still copied an original token for two of the three unsupported requests, the unspecified and denied cases. It made no such copy for the explicitly excluded case. A refused supplied claim therefore protects the diagnostic writer while leaving the checkpoint's own source decision in force. [Controls](evidence/panel.json)

### Answers ending inside a passage

The authored fresh sources all ended before a period. We therefore ran a supplemental comparison on the 18 previously exposed human questions from #4335. The original balanced checkpoint had completed 0/13 supported extractive replies. Supplying actual source bounds recovered the exact words on 12/13. Taking the native token at release still gave 0/13 exact through EOS. Supplying stop at the same end gave 12/13. Of those 12 successes, ten were followed in the original passage by a token outside the legacy delimiter set. The remaining answer required nine source tokens. Both arms refused all five missing supplied sources and the underlying ordinary writer still copied on four. This is a replay of exposed extractive questions, using their original source annotations. [Transfer method](transfer/README.md) · [Transfer counts](evidence/transfer.json)

### Self-fed state behind the release token

A checker built against the unmodified #4335 source independently reconstructed every ordinary proposal and its full writer event on the emitted prefix. This covered 115 traced intervention sessions and 1,680 decode steps. On 690 steps the intervention changed the ordinary ID. There were 658 observable successor steps after those changed commits. Replacing just the last emitted ID with its pre-intervention ordinary proposal changed the next read code on 637/658, the recurrent code on 637/658 and the ordinary token proposal on 571/658. Self-feeding changes native state even on this frozen checkpoint. Whether that state selects the user's requested action is a separate measured failure above. [Independent replay](replay/README.md) · [Counts](evidence/self-feed.json)

### One human question requiring two sources

The TRAIN supply audit found one DROP question whose entire passage, two disjoint original spans and space-separated answer in annotation order fit this client exactly. On that question the ordinary checkpoint first missed at offset 0, emitting 395 instead of 799. With a verified first span it copied addresses 32–35, then emitted 315 instead of the second span's opening ID 301 at offset 4. Supplying continuation copied addresses 46–51 and produced the full ten-ID answer followed by EOS. Supplying stop instead emitted EOS at offset 4. This is one TRAIN feasibility control. DROP's multi-span grader accepts unordered answer sets, and this question has no matched natural stop twin in the eligible supply. [Eligibility, source attribution and replay](supply/drop/README.md) · [Four ordinary-client runs](supply/drop/writer.json)

## Checks run at the head

On Wick, six focused Rust tests and all 17 existing tests in the D11 module's test group passed at the pinned product revision. The panel tool regenerated the same sealed grade from all 238 client records. The historical, current-base and modified authenticated clients agreed on all 27 ordinary prompts for each comparison, and their stock records agreed on the nine stock anchors. Trace-on and trace-off emissions agreed on all nine native anchors and six supplied-choice anchors. The self-feed checker rejected a forged ordinary proposal at offset 1 even with the final emitted sequence intact. The raw fixtures and records, hashes and source-recovery procedure are in private custody. [Restore and replay](RESTORE.md) · [Aggregate comparisons](evidence/panel.json)

A separate API review found that the library constructor also accepted a protocol-2 opening made with a different tokenizer. The pinned CLI already checked its tokenizer against the artifact. The library now checks the same canonical tokenizer identity that defines the integer stop IDs. Seven focused tests passed, including rejection of an alternate tokenizer whose EOS is 3. The guarded binary reproduced both complete panel and transfer manifests, including all 238 scheduled panel records and all 36 transfer runs. The four supplemental DROP records also matched byte for byte. [Guarded release replay](evidence/guard-replay.json)

## Skills used

The native-space-cycle protocol grounded the original-address representation, exact source cursor and integer replay. All client measurements and independent replay ran on Wick. No GPU was rented and no weights were fitted.

## Still open

The diagnostic shows that original spans and a separately authenticated continuation can survive the real decoder. The checkpoint still needs to select the supported assertion, its exact start and end, and the requested disposition from ordinary text. The current integration accepts a single user opening. Useful multi-turn language, prose and reasoning, the equal-total-byte longer-context comparison, and a qualified laptop/browser model remain open. [Next training unit](PICKUP.md)

## Plain summary

The copy path can now finish an eight-token span and start another on the correct decode step. The first failure moves from the opening to the verified boundary when we provide the source. The largest remaining improvement on these prompts comes from learning the action at that boundary and selecting the fact that justifies it. Supplying those decisions demonstrates exactly what training must recover.
