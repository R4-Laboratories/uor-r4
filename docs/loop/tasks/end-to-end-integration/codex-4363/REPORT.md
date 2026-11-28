# Where the quotation-mark opening comes from

## What did not work, first

[codex] Both learned checkpoints got zero of 256 TRAIN openings right. Each emitted original ID 1228, the space-prefixed opening quotation mark, on 253 articles. All 506 quote emissions beat their nearest eligible fine rival by more than one Q15 unit. The first wrong token stayed at offset zero in every learned TRAIN reply. Only the opening was run here. The separate 222-conversation held evaluation in [#4361](../codex-4361/REPORT.md) measured zero exact replies through EOS in both arms.

[codex] The proposed strict CoQA comparison has three component-disjoint TRAIN pairs and zero held pairs. Preserving genuine assistant history and requiring extractive evidence only for the final answer has 76 TRAIN pairs and two held pairs. Both denominators count complete pairs whose answers open with different original tokens. The original 1,441-pair preview preceded the source, token, length, overlap and split checks. It cannot size a learning experiment. [Pair census and reproduction](pairs/README.md)

## What changed

[codex] We traced the original 256 third-answer TRAIN openings with the frozen parent and both complete #4361 checkpoints, keeping the writer fixed. For each prompt, the first read used fixed placeholder labels. A second read fetched the native score words for the reference and actually elected cells. A third repeated the first read. The recorded output includes the 512 Q15 hidden words, 64 coarse scores, two 64-score fine groups and transition signatures. No optimizer step was applied in this unit. [Registration](SHEET.md)

## Measurements

| TRAIN openings per checkpoint | Exact first ID | Gold coarse cell admitted | Cell 3 elected | Quote emitted | Quote fine margin, min / median / max |
|---|---:|---:|---:|---:|---:|
| Frozen parent | 1 / 256 | 34 | 14 | 0 | No observations |
| Reset during training | 0 / 256 | 47 | 255 | 253 | 16 / 1,523 / 2,496 |
| Preserved state during training | 0 / 256 | 46 | 254 | 253 | 35 / 1,692 / 2,857 |

[codex] The signed cell-3 margin against the best other cell increased on all 256 articles in each trained arm. On the 58 articles where both frozen and learned captures contained cell 3, the quote switched from losing to winning against the strongest eligible fine competitor on all 58 in each comparison. Even with the gold coarse cell supplied, its fine row would elect the correct token on only 41/256 articles for either trained model. All recorded local score derivatives were inside their saturation limit. The tape contains 117 distinct first reference IDs and zero positive ID 1228 labels across 2,151 supervised positions. [Independent recount](evidence/recount.json)

[codex] ID 1228's tied embedding moved farthest among all 4,096 embeddings in both training arms. Its absolute displacement sums were 69,471 and 65,729 signed-weight units. To explain the actual score gap, hold the learned endpoint's strongest rival fixed and compare the two observed hidden vectors and checkpoint rows. The symmetric integer identity splits the unrounded change into a hidden-vector term and an output-row term. On all 58 observable fine comparisons in each arm, the hidden term favored the quote and the output-row term opposed it. Their respective medians were about +7,332 and −2,338 Q15 score units for reset and +3,798 and −2,862 for preserved. These medians need not sum to the median change. Both checkpoint endpoints reproduced exactly. At the coarse stage, the cell-3 versus rival output-row term favored cell 3 on every article, and the hidden term favored it on 228/256 reset and 227/256 preserved articles. The post-projection hidden includes effects from input embeddings, recurrent layers, normalization and readout projection. This calculation cannot localize an upstream operator or optimizer step. [Score arithmetic](evidence/score-change.json) · [Parameter inventory](evidence/weight-movement.json)

[codex] With preserved state, the coarse score sum favored the correct assignment over swapped gold cells on 17,321/29,470 distinct-label TRAIN comparisons (58.8%). No one of 255 label rotations reached that count. Reset scored 16,828 and frozen scored 14,850. Only 543 preserved comparisons had both individual coarse comparisons strictly right. Within the common gold cell, preserved favored the correct assignment on 1,473/2,636 comparisons, versus 1,441 reset and 1,391 frozen. These comparisons reuse articles and vary both question and evidence. They detect a prompt-dependent signal beneath a common score preference on TRAIN. They cannot establish passage-specific generalization. [Prospective supplementary calculation](ASSIGNMENT.md) · [Rotation controls](evidence/assignment.json)

## Checks run at the head

[codex] The same physical RTX 4090 detected an exact tie, both one-unit directions and a masked leading slot at both hierarchy levels. The historical 2,048-ID fixture reproduced the 128 greedy IDs and Q48 masses and its batched 128-step replay. A fresh audit of the pinned sm_89 serving kernel admitted all 85,232 machine instructions. The private derived packet references the original three checkpoints, TRAIN tape, historical fixture and cubin by hash. [Intake](evidence/intake.sha256) · [Restore and reproduction](REPLAY.md)

[codex] A second implementation reproduced all 147,456 measured head-score words from the original checkpoints and retained hidden words. It checked all 768 real transitions' trace signatures and elected slots, and rejected a one-unit corruption in a copy of a real score row. The endpoint arithmetic independently reconstructed all 4,552 individual endpoint scores in 1,138 comparisons, with zero clamped scores. Wick received and verified the raw results before the rental was released. The ledger records 818 seconds at $0.869/hour, approximately $0.197 of GPU rental at that rate. [Compute record](evidence/compute.json)

## Skills used

[codex] The native-space cycle supplied the shared fixed-point schema, ordinary election, checkpoint identity and cold replay checks.

## Still open

[codex] A separate public-TRAIN QuAC screen locates 138 eligible disjoint pairs before partition. The same article-first and family-first separation produced zero held pairs. An exploratory revision reserves whole pairs and question families and rejects cross-split historical questions, preserving 101 TRAIN and 26 internally held pairs with no shared component or question family. No model has run on them. One family contributes 83 of the 101 training pairs. Only two held pairs contrast original token IDs within the same target coarse cell and zero compare the registered near versus far strata. These counts support an opening trainability pilot. They cannot qualify long-range recall or a general fine-head repair. Official validation stays untouched. [Source gate and limitations](evidence/quac/README.md)

[codex] Keep the ordinary writer and both copy and EOS behavior fixed. The next weight test should compare the current full-answer/EOS objective with the same objective plus an opening margin against the strongest eligible rival. Use matched updates and exposures, both members together, counterbalance the dominant wording family and check the actual scores and updated integer weights on TRAIN before evaluating reserved pairs. At the coarse stage the rival is another cell. In the gold fine cell it is another admissible token. Report both-member wins, actual greedy original IDs, self-fed first divergence and exact through natural EOS against the native control. Open-ended language, the whole-artifact equal-byte longer-context comparison, and laptop/browser serving qualification remain open.

## Plain summary

[codex] Both trained models assign a real advantage to the wrong opening. The change in the hidden vector promotes the quote even as its output-row change works against it. The state still contains some information about which training prompt goes with which answer. The next test is whether training can turn that information into two correct ordinary first tokens when the same human question requires different answers.
