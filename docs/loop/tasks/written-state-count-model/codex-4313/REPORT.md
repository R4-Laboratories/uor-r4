# The question changes native state, but neither checkpoint selects the requested opening token

Codex development evidence under [#4063](https://github.com/UOR-Foundation/uor-r4/issues/4063), following the [two trained checkpoints](../codex-4312/REPORT.md). Neither checkpoint selected or emitted the correct ordinary-protocol first content token on any of the **800 controlled openings**. The target always scored below the single-space token at the gate. The recurrent state nevertheless changed on 385 of 400 question swaps in reasoning-first and 389 in answer-first. The frozen [registration](SHEET.md), [independent FIT](FIT.md), and [decision](GO.md) precede the first held observation.

## Matched swaps

The unit is a question pair with the same facts, same encoded length and identical assistant marker. Each of 200 independent fact groups was presented under two assignments of the values to the subjects. Four wording families contributed 50 groups each. Each question changes only the subject, each reference is a single original vocabulary ID containing a word, and both answer IDs occur exactly once in the shared facts. All openings end in the literal six-ID protocol-2 marker. Its last ID is the colon, 28. Neither an answer token nor a standalone leading space was supplied. The controls and exclusions are in the [panel recipe](panel/README.md).

| Within a controlled pair | Reasoning-first | Answer-first |
|---|---:|---:|
| Pre-gate read code changed | 197/400 | 201/400 |
| Raw recurrent sum changed | 392/400 | 394/400 |
| Recurrent code after snapping changed | 385/400 | 389/400 |
| Both final codes changed | 194/400 | 197/400 |
| At least one of the read or recurrent codes changed | 388/400 | 393/400 |
| Native successor source address or ID changed | 0/400 | 0/400 |
| Gate winner changed | 7/400 | 50/400 |
| Each requested label strictly beat the opposite label in its own question | 7/400 | 2/400 |
| Both emitted IDs were correct | 0/400 | 0/400 |

Neither checkpoint passed the strict two-label comparison under both assignments in any of the 200 groups. The opposite label strictly won on both sides of 3 reasoning-first and 4 answer-first pairs. For a fixed gold ID, changing from the crossed question to its own question increased effective score on 62/800 reasoning-first rows, left it identical on 678 and reduced it on 60. The corresponding answer-first counts are 45, 716 and 39. Rank could move even when that ID's score was unchanged. [Paired results and subgroup counts](evidence/summary.json).

On the 12 reasoning-first and 7 answer-first pairs that shared both final codes, the output-side read vectors, source addresses and IDs, gate winners, and emitted IDs all matched. The recurrent frames still differed on 6 of those 12 and 1 of those 7. Four reasoning-first pairs and one answer-first pair reached the same snapped recurrence despite different raw sums. In six pairs per model even the terminal frames and sums matched. The capture is taken at the opening and does not localize an earlier convergence within the question prefix. [Terminal pair counts](evidence/collisions.json).

## Rank against space and the prior

Ranks start at one. The unigram prior ranks all 4,096 original IDs by count with lower ID breaking ties. The gate considers candidates from the last-token, read and recurrence tables and the native successor. For equal positive gate scores, the higher unigram count wins, followed by the lower original ID. Native gate ranks cover eligible positive scores plus the genuine fallback winner when no mass exists. The full-vocabulary comparison places the unsupported tail after positive eligible candidates, in prior order. The writer admits geometric cells, then ranks their IDs by coordinate distance and lower ID. An excluded ID has no native writer rank.

| Across 800 controlled rows | Reasoning-first | Answer-first |
|---|---:|---:|
| Correct token median rank under the unigram prior | 1,136 | 1,136 |
| Correct token median gate rank, full vocabulary | 663 | 680 |
| Middle half of full-vocabulary target ranks | 399–1,146 | 414–1,136 |
| Correct token with eligible positive mass | 752/800 | 749/800 |
| Median native rank among those supported targets | 630 | 619 |
| Correct token ranked better than its prior rank | 663/800 | 669/800 |
| Space ranked first / second at the gate | 781 / 19 | 616 / 184 |
| Correct token outranked space / entered top eight / was emitted | 0 / 0 / 0 | 0 / 0 / 0 |
| Correct token admitted to the final writer | 44/800 | 10/800 |

Space is the fitted unigram winner in both artifacts. The last-token table contributes mass 17,033 to space every time in reasoning-first and 17,008 in answer-first. Even the **largest total gold score** observed in each arm, after all terms, was only 481 and 234 respectively. The smallest observed space-minus-gold gaps were 16,938 and 16,933. Read-, recurrence- and successor-conditioned terms in aggregate added more mass to space than to the target on 775/800 and 781/800 rows. In a supplementary arithmetic check, removing the last-token contribution from every candidate while holding the measured pool and states fixed left **0/800** targets above space in either arm. The test measures this fitted scoring rule and says nothing about how a refitted rule would score. [Independent count-table reconstruction, reasoning-first](evidence/reasoning-gate-audit.json) and [answer-first](evidence/answer-gate-audit.json).

The gate selected only space (223) or newline (201) on these 800 openings. In reasoning-first, the writer emitted each selected ID. In answer-first, it emitted space on 616 rows. On the other 184 the gate chose newline and the writer emitted lowercase `g` (73). In that checkpoint, those two IDs share cell 172 and are the only tokens with their exact four-block vector. Both have zero writer distance, so the lower ID wins. Reasoning-first's newline sits in cell 89 and is the lowest ID with its vector. This secondary alias changes the visible error on 184 rows. The correct content ID had already lost at the gate on all 800.

## One actual question swap

The first controlled pair by its frozen ID assigns **snow** to Edward and **cat** to Brian. Both questions ask for the label of the specified person's specimen. The following are measured raw integer codes and analytical full-vocabulary ranks. The labels have prior ranks 1,072 for cat and 2,143 for snow.

| Checkpoint and requested person | Read code | Recurrent code | Gold gate rank | Gold mass | Space rank | Space mass | Emitted |
|---|---:|---:|---:|---:|---:|---:|---|
| Reasoning-first, Brian (cat) | 32 | 53,649 | 1,019 | 15 | 1 | 38,193 | space |
| Reasoning-first, Edward (snow) | 32 | 57,745 | 1,566 | 6 | 1 | 38,121 | space |
| Answer-first, Brian (cat) | 5,122 | 42,277 | 699 | 9 | 1 | 17,726 | space |
| Answer-first, Edward (snow) | 5,170 | 13,605 | 1,516 | 2 | 1 | 18,996 | space |

In answer-first the two read codes and the two recurrent codes differ, yet cat receives mass 9 on either wording and snow receives mass 2 on either wording. Both arms also read the same original successor at source address 4, ID 2,888. The restricted spreadsheet includes the actual question, both codes, the same token's rank under the crossed question, its prior rank, its effective mass, the space comparison, and the emitted ID for each of 1,620 checkpoint–prompt observations.

## Human passage transfer

Five unused CoQA passages had two independently answerable questions and clear first tokens under the strict filters. The questions shared an excerpt and had empty prior dialogue. One pair had equal encoded lengths, four differed in length. Read codes changed on 2/5 pairs and recurrent codes on 5/5 in both arms. On the equal-length pair the read codes stayed equal and recurrent codes changed in both. The answer-first successor also changed address and ID. The median full gate rank improved from a common prior of 278 to 65 in reasoning-first and 40 in answer-first. The best observed target ranked 11th in both arms. All ten targets per arm ranked below space and all first emissions missed the reference. These five pairs provide a descriptive check on natural questions.

## Which boundary the counts learned

After freezing the held results, the [supplemental audit](BOUNDARY-NOTE.md) examined the original D87 tapes. There are 118,915 eligible predictions immediately after the colon token in each model. Only 4,345 follow the exact deployed six-token assistant marker, and all of those are genuine assistant-turn openings. The other 114,570 positions have the same immediate-target histogram in both arms.

| Original count-fit events | Reasoning-first | Answer-first |
|---|---:|---:|
| Standalone space after a genuine assistant marker | 27 | 2 |
| Standalone space after any other colon | 17,006 | 17,006 |
| Newline after a genuine assistant marker | 0 | 2 |
| Newline after any other colon | 16,245 | 16,245 |
| Genuine-opening first ID identical to ordinary rendering of the same text | 4,345/4,345 | 0/4,345 |

The gate keys its last-token term on the single colon ID. It consequently pools turn starts with all other eligible colons. The reasoning-first tape encoded the leading separator with the reasoning text. Answer-first moved the original final answer tokens directly behind the visible marker and trained the reasoning in a separate reset sequence. Those answer IDs had been encoded without the serving renderer's prepended space. Applying the ordinary renderer to the same answer text changes the immediate target ID on every one of its 4,345 genuine training openings. A standalone space would be the canonical immediate target on 2,196 of those reformatted answers. This establishes a training/serving **text-boundary mismatch**, alongside the previously verified bit parity on identical ID sequences. [Static donor census](evidence/donor-boundaries.json).

For sensitivity, the independent auditor took the same sealed device states and rescored the first ID from encoding each trimmed reference without the protocol's introductory space. All 810 IDs changed. On the controlled set the alternate median prior rank was 60, and the alternate full gate medians were 196 (reasoning-first) and 230 (answer-first). The alternate target still never beat space and the gate selected it on **0/800 in both arms**. Eight controlled pairs shared an alternate first ID. Among the 392 remaining distinct-ID pairs the correct ID beat its opposite on both questions in 6 reasoning-first and 3 answer-first pairs. Six answer-first outputs happened to match the unprefixed first subtoken `g` because the gate selected newline and ID 73 won its vector tie. None of those reference words is `g`; none yielded a two-way first-token match. The five human passage pairs had median alternate prior rank 198 and alternate gate medians 385 and 277, again with zero gate selections. This supplementary check used the held state traces and original weights without a second device run. [Alternate-boundary ranks](evidence/alternate-ranks.json).

## Next model experiment

Use one ordinary dialogue encoder to produce the visible training and serving boundaries. First verify exact ID equality on the actual donor answers, including the 2,196 whose canonical answer begins with a separate space. With identical original donor records and first-content training budgets, compare learning the visible turn start from the final colon alone with learning it from the complete assistant-boundary identity and question-dependent geometric state. Train both sides of each question contrast against space and the counterfactual answer, and include fresh untemplated human questions and prose as transfer checks. The concrete criterion is a two-way improvement in the **correct content score** on new questions and on both fact assignments. Then test full replies through natural EOS. Both the arithmetic subtraction and the alternate token readback failed to recover a correct gate selection with the existing weights. This unit leaves the writer and milestones fixed.

## Qualification and recovery

The original checkpoint digests matched the signed-off #4312 receipts. The 4090 ran 810 openings per model, replayed all of them in reverse order and reproduced every captured device word. Reopening the model through the ordinary Rust integer client on Wick matched the read, recurrent frames and coordinates, active routes, source address and ID, selected gate ID, admitted cells, top eight and emitted ID on all 1,620 observations. A second independent scorer used the original D11 sparse triples and the captured query, read, recurrence and successor to reconstruct the full vocabulary. It checked 3,317,760 candidate scores and 2,430 target/rival/space probes per checkpoint. The [finite device fixture](evidence/finite.json) exercised successes, failures, ties, zero support, and writer aliases before the held comparison.

Wick's 4090 was occupied. The rented RTX 4090 completed both checkpoints and reverse replays in 13.64 seconds. Provider uptime was 358 seconds at $0.869/h, or approximately $0.086 at the compute rate. The ZFS copies were rehashed before release, and the private-store download reproduced the archived measurements. [Compute receipt](evidence/compute.json) and [pinned original measurements, methods, and private CSV](https://huggingface.co/datasets/caseyallard/uor-r4-store/blob/9988b98ee4c47c4bd4bdba3eb54542ad65a6f2bf/results/codex/question-swap-20261127/measurements.tar.zst).

The separate [pinned static readbacks and reconstruction source](https://huggingface.co/datasets/caseyallard/uor-r4-store/blob/03aa4470583b11a7f0a76bd9b98246d107d45678/results/codex/question-swap-20261127/static-readbacks.tar.zst) retain the full sparse-table audit, donor census and alternate-boundary data. Both archives are available at that revision. [Archive and CSV checksums](evidence/storage.sha256).

Pre-merge review found that the earlier #4312 trainer used the original shared helper signature. Restoring the default disabled tap allows that existing call to compile. Reassembling the question-swap scorer with the corrected header reproduced the exact PTX and device binary used on the rented GPU. [Compile and binary identities](evidence/build-compatibility.json).
