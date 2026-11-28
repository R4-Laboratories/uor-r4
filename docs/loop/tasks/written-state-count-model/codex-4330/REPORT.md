# The question change disappears before the source score

## What did not work

Retain the balanced checkpoint. All three artifacts scored zero strict immediate pairs on TRAIN, article reserve and published DEV. A strict pair requires the correct original address for the supported question and NoRead for the other question on the very same passage. The fitted reader also lost all four previously passing same-ID address cases. The writer, tokenizer and base model were held fixed. This unit rules out the tested feature representation and margin configuration as an upgrade.

## What changed

The expanded arms use every eligible original token in the actual current user turn and its final 32 tokens as the query. One keeps the original geometric weights, and one fits them against positive addresses and NoRead. Packing adds the same small reader envelope to the balanced artifact.

## Measurements

| Immediate question pairs | Original framed | Whole turn, frozen | Whole turn, fitted |
|---|---:|---:|---:|
| Source wins on 114 published TRAIN positives | 0 | 0 | 2 |
| Strict wins on 114 published TRAIN pairs | 0 | 0 | 0 |
| Source wins on 170 article-reserve positives | 0 | 2 | 1 |
| Source selections on 170 reserve questions annotated impossible | 0 | 170 | 141 |
| Strict wins on 170 article-reserve pairs | 0 | 0 | 0 |
| Strict wins on 23 published DEV pairs | 0 | 0 | 0 |
| Previously passing original addresses retained, c021–c024 | 4/4 | 2/4 | 0/4 |

Thirty reserve and two published DEV pairs require an actual leading space. Those pairs are retained as a separate phase. The controlled authored DEV has 72 supported and 48 unsupported prompts. Neither expanded reader found any of the 72 original addresses, and each selected a source in all 48 unsupported prompts. All graded arms had zero strict authored support flips out of 48. The fitted reader recovered 44/144 authored TRAIN positives, against zero with frozen whole-turn words; it correctly abstained on only 2/96 authored TRAIN negatives. [All original addresses, pairs and split denominators](evidence/source.json)

## Where the information is lost

All 170 immediate reserve sources were present among the whole-turn candidates. The fitted gold address ranked 22nd at the median, and 155/170 gold scores were at or below NoRead. Only one gold address ranked first. Sixty-seven strongest lexical rivals lay in the query suffix. Lowering the threshold cannot repair an occurrence that another candidate already outranks. In all four protected cases the fitted reader chose position 40, token 759, while the correct positions in order were 19, 36, 36 and 19. [Rank and rival measurements](evidence/rival-summary.json) · [Protected address records](evidence/source.json)

A separate post-run diagnostic holds the original paragraph position fixed and compares its 16 active feature indices under the two different questions. All 16 are identical on 66/114 immediate natural TRAIN pairs, 108/170 reserve pairs and 13/23 published DEV pairs. On the authored assertion/denial and question pairs with known truth they are identical on **96/96 TRAIN and 48/48 DEV pairs**. The score for that occurrence is a sum of the same 16 selected integer weights on both sides. With a shared NoRead threshold, no values for those weights can give that occurrence the required positive margin over the threshold on one side and negative margin on the other. A lexical tie can still trigger abstention, and other candidates may change. This is a restriction on the fitted objective at the relevant address. It says nothing about the capacity of the recurrent trunk or other project geometries. [Counterfactual comparison](run/paired-features.jq) · [Per-cohort counts](evidence/paired-features-v2.json)

## Supervision quality

The natural baseline used published SQuAD annotations and mechanically verified offsets. After the fit, a separate passage review flagged five TRAIN pairs, two reserve pairs and one published DEV pair. On the unchanged fitted checkpoint, excluding both members of every flagged pair leaves strict immediate scores of 0/109 TRAIN, 0/168 reserve and 0/22 published DEV. These are retrospective sensitivity counts on the same checkpoint. The unflagged natural pairs have not all been semantically adjudicated. The independently authored comparisons above provide the cleanest evidence of the representation problem. The original sealed preregistration and inputs retain what the team knew when it trained. [Pair exclusions and scoring](run/label-sensitivity.sh) · [Sensitivity result](evidence/label-sensitivity.json) · [Label review](../codex-4329/evidence/label-review.md)

## Training, parity and cost

Each four-epoch fit used the same 468 original prefixes (258 positive, 210 NoRead). It changed 228/513 signed i16 words. The two fresh fits returned the same bytes, with SHA-256 64465acceb876799cf1ebe06c36029c87f6f31863f37ead21fd07cf41aa439d0. AOT CUDA and the restored ordinary Rust reader agreed on all 3,258 arm/case decisions, all 214,776 candidates and all 3,436,416 feature indices. The finite fixture demonstrated actual post-update selection, abstention on ties, and selection of a later original address sharing a token ID. One Vast RTX 4090 cost $0.869/hour at the quoted rate; about seven minutes from provision request to deletion gives an estimated $0.10 compute charge. The provider invoice was not measured. Training and scientific device scoring ran on the rental. Wick ran build, restore and ordinary integer client work alongside its existing GPU job. [FIT](FIT.md) · [Device fit](evidence/fit.json) · [Parity](evidence/parity.json) · [Cold restoration](RESTORE.md)

## Ordinary replies and the first wrong token

Of 618 held single-turn prompts, 393 have a defined reference answer through EOS. All 393 fit the 16-token allowance. The other 225 are published impossible questions with no uniquely specified reply. Treating an abstention on those rows as an exact answer would inflate the score.

| On the 393 held references | Original framed | Whole turn, frozen | Whole turn, fitted |
|---|---:|---:|---:|
| First output token correct | 37 | 19 | 13 |
| First error at token 1 / 2 / 3 (one-based) | 356 / 30 / 7 | 374 / 13 / 6 | 380 / 10 / 3 |
| Exact reply through natural EOS | 0 | 0 | 0 |

On protected case c021 the original client copied “ green” (ID 1074) from position 19, emitted the period, then emitted “g” (73) instead of EOS (1) at token three. The expanded frozen reader copied “ green” from the wrong original address and made the same first visible error. The fitted reader copied “Wh” (759) from position 40 and failed on token one. All first decisions and fallback commits joined to their CUDA decisions across 1,086 rows per arm. The 141 incorrect fitted admissions on the 170 immediate reserve questions labelled impossible were actual authenticated copies. [Original held token errors](evidence/held-generation.json) · [Commit provenance](evidence/first-commit.json)

The same normal post-load serving path ran across the panel with fresh per-prompt state. Its complete traces matched 309 cold-process calls covering 103 representative inputs in every arm. On eight additional real human requests, every arm hit the 16-token cap eight times. Reading the 24 replies found whitespace, punctuation and short fragments; none fulfilled its request. These eight prompts were a descriptive sample with no human panel grade. [Ordinary comparison and private replay](RESTORE.md)

## Checks run at the head

The GPU fixture, repeated fit, reversed evaluation order, full CUDA–Rust candidate replay and cold D11 reconstruction passed. The product tests compared legacy balanced traces, authenticated initial copy, native fallback, true user-role bounds and two subsequent self-fed steps. The focused occurrence unit tests used the existing stack-test import correction only in a disposable Wick test checkout. The delivered product executable was built from the unmodified production source at its recorded pin.

## Skills used

The native-space-cycle protocol supplied the typed-state, serialization and train/serve replay checks.

## Still open

The test leaves question-to-occurrence interaction, generation after the opening, useful free conversation and browser qualification open. The [next experiment](PICKUP.md) specifies a finite representability test, fresh semantic supervision and a matched fixed-question control.

## Plain summary

The fit changed actual deployed weights but gave the same score to the same passage address under every controlled support reversal. The retained checkpoint is the original balanced one.
