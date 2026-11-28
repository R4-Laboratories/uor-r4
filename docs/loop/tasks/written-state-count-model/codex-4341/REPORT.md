# Natural question pairs did not transfer to strict source selection

## What did not work

[codex] The proposed native geometric reader selected both original answer spans on 0 of 15 development pairs and the correct original source and NoRead on 1 of 43 mixed pairs. It lost all four protected addresses. Both direct controls scored 0/15 on the first comparison. The natural count control also won 1/43 mixed pairs, in a different title distribution. The source gate is **negative** for this training mixture and operator. The separate seven- and twenty-title transfer sets remain sealed. [Registered gate](../codex-4340/validation/REGISTRATION.md) · [Source results](evidence/source.json)

## What changed

Two independent source reviews and native capture qualified 15 two-answer and 43 answerability pairs in eighteen development articles, covering 73 supported prompts and 43 negatives. The latter contain 27 genuinely unanswered questions and 16 contradicted premises. The original 67 selections and their dispositions were preserved. All 46 remaining passages in the three previously deficient titles were selected before new text was opened. They supplied 19 additional qualified pairs. The three title counts rose from 0, 1 and 1 to 3, 8 and 10. This was targeted development coverage. [Coverage](evidence/coverage.json) · [FIT](FIT.md)

The four arms used the same 4,096 signed 16-bit learned words, 32 passes, 264 examples per pass and balanced writer. Historical and natural examples respectively presented 25,141 and 42,776 candidate actions per pass. Each complete artifact occupied 1,202,185,782 bytes. Repeated CUDA fits reproduced the signed words and update counts byte for byte. Across all four comparisons, swapping the TRAIN labels strictly reversed at least one complete training pair, and zero weights earned zero strict paired wins. Reopened artifacts matched all signed device scores, including 19,838 development and 284 protected choices per variant. Article-macro answerability for natural geometry was 1/33 higher than historical geometry and 1/66 lower than natural count. No two-answer title beat both controls. [TRAIN results](evidence/train.json) · [All source controls](evidence/source.json)

The first GPU attempt stopped before compilation when the scanner looked for its own old filename. The second linked the already pinned scanner bytes at that name. A presentation fix strips a validated line ending from elapsed seconds before parsing. The successful device job took 11.83 seconds, including all four fits, their permuted-label repeats, image inspection and TRAIN replay. The blind post-fit device score process ran separately. The single Vast RTX 4090 was deleted after 221 seconds at $0.869/hour, an estimated $0.05335 for compute. Wick holds the reopened artifacts and ordinary-client outputs. [Device record](evidence/device.json)

## Measurements

| TRAIN mixture and address | Strict TRAIN targets / 264 | Both development answers / 15 | Both development source and NoRead / 43 | Supported source / 43 | Strict NoRead / 43 | Protected / 4 |
|---|---:|---:|---:|---:|---:|---:|
| Historical count | 264 | 0 | 0 | 6 | 0 | 1 |
| Historical geometry | 264 | 0 | 0 | 4 | 2 | 1 |
| Natural count | 257 | 0 | 1 | 5 | 17 | 0 |
| Natural geometry | 246 | 0 | 1 | 4 | 15 | 0 |

The natural geometric fit won 69/72 two-answer TRAIN pairs against their own alternate address and 66/72 against all addresses. On development it won 2/15 against the alternate alone and 0/15 against the full field. In 9/15 pairs changing the question shifted the relative scores in the correct direction, but a shift can leave the same site ahead both times. Thirteen pairs already fail when the two intended answers are the only choices. On mixed development pairs, geometry won 5/43 against the two intended actions and 1/43 against all alternatives. The count control attained the same four paired totals on development. [Paired margins](evidence/within-passage.json)

## First wrong token and end of reply

[codex] In ordinary `uor-chat --say` the proposed arm made 0/73 supported replies exact through natural EOS, including 0/6 with a strict winning source. All 73 ran to the test's 16-token cap. In 59 replies the first committed ID was already wrong at offset zero, compared with 61 under historical geometry and 59 under natural count. Counting matched prompts against historical geometry, the first error moved later on six, earlier on seven and stayed at the same offset on sixty. Against natural count there were six later, six earlier and sixty-one unchanged. The proposed arm copied every original address on five positives. [Matched token movement](evidence/first-wrong.json) · [Writer guard](evidence/writer-guard.json)

Supplying the original source raised exact address sequences from 5/73 to 70/73. All 70 then emitted a non-EOS token and ran to the cap. The most frequent first error at release was period ID 16 where EOS ID 1 was expected, on 26/70. Supplying EOS at the verified end made all 70 replies exact. The other three failed before copy at offset zero, where the expected standalone space ID 223 became equals-sign ID 31 twice and the token for “ Let” (1042) once. These are supplied-source and supplied-stop diagnostics. [Observed release decisions](evidence/release.json)

The proposed arm selected a source on 28/43 unsupported prompts and actually committed original-source tokens on 25/43. The frozen visibility recognizer found 12 empty responses, zero visible refusals and 31 requiring semantic review. The 31 are preserved for blinded human grading. The count of factual false claims remains unresolved. Both the source and writer results independently keep the locked articles closed. [Visibility and writer totals](evidence/writer-guard.json)

All 12 ordinary arms finished their 116 prompts, including all permuted-label and zero-table controls. Together with the 146 supplied-source interventions, that gives 1,538 recorded replies. The complete writer matrix took 33 minutes 44 seconds on Wick. Each of the twelve variants completed zero of 73 supported replies through natural EOS. Unsupported source commits were 41 for historical count, 39 for historical geometry and 25 for each natural arm, on the same 43 negative prompts. [Ordinary results by arm](evidence/ordinary-arms.json)

## Checks run at the head

[codex] Two independent source assessments and the ordinary tokenizer established the common panel before fitting. Independent finite tests rejected forged or missing EOS, wrong original addresses, altered self-fed history and mismatching control identities. Source scores, TRAIN repeats, protected cases and partner swaps were verified against reopened artifacts. The descriptive bank audit reconstructed every natural-geometry score and all 380 own-target versus mate margins from the fitted integer table. [Independent FIT](FIT.md) · [Bank audit](analysis/banks.md)

The complete private archive is 46,911,015 bytes with 964 members. Force-downloading its pinned revision reproduced both source and writer grades byte for byte. A same-length corrupted packet failed restoration. [Restore instructions](RESTORE.md) · [Archive receipt](evidence/archive.json)

## Skills used

[codex] The native space cycle fixed the persisted address and signed-word representation, common operators and exact replay requirements. The research architecture review used the source-versus-writer experiment to choose the next test.

## Still open

[codex] M2 useful grounded conversation is still open. The next source experiment tests one directed native relation across paraphrases and true reversals on a new TRAIN-only split. Release is an independent writer problem even when the source is supplied. Neither direct control nor the proposed checkpoint earned promotion. [Next experiment](PICKUP.md)

## Plain summary

[codex] We trained real weights, repeated the fits on a GPU, and reopened them in the ordinary client. Seeing natural question pairs in training helped the reader choose no source on some unsupported questions, but it still could not answer both questions about a new passage. Even perfect source information usually carried the right words without telling the model when to finish.
