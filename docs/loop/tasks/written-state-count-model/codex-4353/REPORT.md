# Fitted geometric first-word scores transfer on the paired contrast and fail the full election

## What did not work, first

[codex] The preregistered advancement gate failed for both seeds on 200 human follow-ups. In more pairs, both prompts strictly preferred their own target, but both seeds matched only one live first lexical token. Both lost the frozen model's single live match and gained one by emitting the genuine leading space on a different row where P4 already ranked the gold word first. Neither created a new conditional top-1 nor finished an exact reply. The matched ordered control beat both geometry seeds at the same deployed bytes. The result closes this update, address representation and training distribution together. It does not isolate the cause of that failure.

## What changed

[codex] Learned four 64-slot signed i16 banks directly on the device, sharing the P4 prior, original-ID score scale, strongest-rival update, full 4,095-way election, example order and number of training examples between geometric and ordered arms. Geometry selects four relative D11 addresses. Ordered uses four exact suffix hashes. Each native image consists of 520 B of header/words and occupies 584 B bound to its P4 state. All five configurations act at each actual step with the unchanged P4 prior. Real text training supervised one lexical opening per admitted row, with all 3,891 rows per epoch, eight epochs and two seeds. [Registration](SHEET.md) · [Independent FIT](FIT.md) · [GO](GO.md) · [Inputs](../codex-4352/REPORT.md)

[codex] The original finite runner and client compiled historical squashing functions through the old serving feature. Before any model execution, introduced the p4-only-client build and verified absence in its Rust archives and unstripped executables. It retains the ordinary uor-chat executable name and shares its message encoder and P4 commit step. The first cloud dispatch lacked GNU time and stopped before opening the model. Kept that attempt and installed the missing timer before launching to a separate directory.

## Measurements

[codex] The primary source-held panel contains 200 original human questions with genuine earlier assistant answers. Live lexical applies to the actual self-fed opening. Conditional rank supplies a gold initial space only on the eleven prompts that genuinely start that way. The paired preference requires both sides to prefer their own target over their partner's. Two unpaired rows leave 99 comparisons. All reply grades include the natural EOS ID; cap 32 including EOS. [Independent recount](evidence/supplement-recount.json)

| Arm | Live first lexical /200 | Conditional top-1 /200 | Median conditional rank | Strict paired /99 | First wrong at ID 1 /200 | Natural EOS /200 | Exact through EOS /200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen prior with zero weights | 1 | 3 | 39 | 8 | 199 | 1 | 0 |
| Geometry 435300 | 1 | 1 | 1,985 | 25 | 197 | 18 | 0 |
| Ordered 435300 | 8 | 8 | 1,605.5 | 14 | 192 | 76 | 3 |
| Geometry 435301 | 1 | 1 | 1,876.5 | 25 | 199 | 5 | 0 |
| Ordered 435301 | 8 | 8 | 1,576 | 14 | 192 | 50 | 3 |

[codex] On the same individual rows, geometry moved the first error later/unchanged/earlier on 3/196/1 and 1/198/1 versus frozen, and 1/192/7 and 0/193/7 versus the matched ordered arms. Both ordered seeds finished the same three replies exactly. No geometric arm won an opening whose lexical ID was absent from its prompt. One reference exceeds the cap. [Primary metrics](evidence/supplement-summary.json)

[codex] On the nine-row original DEV cohort the five arms in frozen/G00/O00/G01/O01 order matched 0/2/0/0/0 live lexical targets, had 1/0/0/1/2 strict pairs out of four, and zero exact replies each. These rows are reported separately. They did not select a seed. [DEV metrics](evidence/dev-recount.json)

[codex] On the primary conditional election each geometry seed missed 199 rows. One has a same-four-address rival that already outranks the gold. Two were regressions from a frozen top-1. Of the other 196 prior misses, 191 and 187 gold ranks fell, respectively. The four geometric banks expose at most 26 different addresses each for the registered root transitions, and both fits ended with 103 nonzero words. The ordered fits ended with 252 and 251 nonzero words out of 256. This matters for interpretation even though the deployed sizes match. In the final epoch 3,790 and 3,768 of 3,891 geometric rows still triggered updates. Highest-rival TRAIN full aliases were zero in both final epochs. [Fit log](evidence/fit.json)

[codex] After grading, bounded arithmetic on the sealed conditional scores established that **199/200 primary rows and 9/9 DEV rows can each elect their own correct word** under the existing address and value bounds if given a separate assignment of the 256 weights. Set the target's four addressed words to i16 maximum and every other word to i16 minimum. Any rival differing in one or more banks then loses at least 65,535 × 2^32 Q48 of relative score, larger than the maximum observed frozen prior disadvantage on every row. Full-address aliases keep their original order, blocking exactly one primary row. This answers individual representability; whether one shared learned assignment can satisfy these rows is open. The analysis used no new predictions or fits. [Arithmetic and invariants](individual-ceiling.jq) · [Primary](evidence/supplement-ceiling.json) · [DEV](evidence/dev-ceiling.json)

[codex] A second post-hoc check derived 581 distinct original-ID margin inequalities from the stored strongest competitors of zero and both geometry seeds. It found zero opposite-vector pairs with incompatible required margins across distinct rows. The strongest observed rivals cover only a subset of all constraints, so the zero count cannot establish joint feasibility. [Exact test](opposite-constraints.jq) · [Counts](evidence/opposite-constraints.json)

[codex] The model trial took 32.20 s wall clock, including 26.101 s to load the frozen model, 0.685 s capturing TRAIN priors and 1.354 s for the four fits. Peak sampled GPU memory was 4,314 MiB and maximum host RSS was 4,137,744 KiB. The one RTX 4090 rental recorded 570 s at $0.869/h, approximately $0.138 compute before fees, and was deleted after both remote result directories and 45 file checks verified on Wick. The full deployed baseline plus tokenizer and either 584-byte head totals 3,988,206,820 bytes. The held panel is a short-answer test; no laptop/browser performance or open-ended ability claim follows. [Rental receipt](evidence/rental.json)

## Checks run at the head

[codex] Ordinary and fitted-device finite controls, independent Rust and jq regrades, and preserved whole-sequence/rank-pair counters. The independent recount ignored the original grades and summaries and reproduced every arm's first-error, lexical, rank, paired and natural EOS totals. Both private and public receipts distinguish supplied context from generated context. [Recount program](recount.jq) · [Device fixture](runner/src/device.rs)

[codex] Twelve independent fresh ordinary-client invocations on Wick reproduced all 384 committed steps on the two preregistered sentinels, including the baseline without any attachment. All 12 hit the cap. Zero and frozen projected traces agreed in both pools. Cold wall time ranged from 17.10 to 17.31 s and peak RSS from 3,888,760 to 3,889,704 KiB. The verified forced restore from the private HF store contained all 131 files, including device traces, the emitted weights, both approved binaries, independent recount, hardware evidence and cold outputs. The 3,996,137-byte archive SHA-256 is 98a16cc6bde565cb6cbd1001776df1b53dee6b1c7ba883f3bf311d76215a8242. [Cold checks](evidence/cold.json) · [Replay](REPLAY.md) · [Private result packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/blob/d25a9bcaa90bc70059988e6a42e63939d421aba1/results/codex/20261128-lexical-native-fit/packet.tar.zst)

[codex] The registered cold sentinels all capped, so an additional post-hoc check replayed the lowest opaque ID among the three supplemental rows exact under both ordered seeds. On that same untouched role transcript, the ordinary client reproduced the two ordered seven-ID answers through natural EOS and the two geometric 32-ID capped continuations, with every score, address and tie matching all 78 archived steps. This selected witness carries no additional accuracy denominator. The separate 23-file private restore verified in full. [Witness](evidence/cold-eos-witness.json) · [Private witness](https://huggingface.co/datasets/caseyallard/uor-r4-store/blob/4d2c56730887f5c6162b61bac9a6216a3f233130/results/codex/20261128-lexical-native-fit/posthoc-witness.tar.zst)

## Skills used

[codex] Native-space cycle for shared integer replay, original IDs and artifact identity. Independent source, instrument and outcome reviews.

## Still open

[codex] The primary comparison rules out promotion of these fitted weights. A TRAIN-only diagnostic can calculate each row's exact attainable four-bank margin by setting its gold words to i16 maximum and all other words to minimum; differing banks help that gold against every rival simultaneously. Among feasible rows compare immediate corrected-rival rank, immediate all-ID rank and shared end-of-epoch rank to distinguish a newly exposed rival from conflicting updates across examples. Preserve a conversation-grouped internal split before modifying any fitting rule. The supplemental panel is now measured and needs replacement for another fresh transfer claim. Useful open-ended dialogue, coding, reasoning, long-source retrieval and compact local deployment remain separate gates.

## Plain summary

[codex] Moving the paired preference was real. It did not carry through the competition against the other original vocabulary IDs. The first lexical word and natural completion are still the bottlenecks in this configuration.
