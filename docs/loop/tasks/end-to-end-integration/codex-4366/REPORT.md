# Prospective reproduction of the opposed-opening update

## What did not work, first

[codex] The historical run used a smaller pre-training calibration than it registered. This reproduction retained the original two TRAIN pairs, one update and unchanged writer, and supplied the missing four-row calibration before optimizing. The historical pilot remains exploratory. [Original release correction](../codex-4364/GO-TRAIN.md) · [Prospective order](ORDER.md)

[codex] The additional opening loss won **0/2** pairs on both first tokens. Frozen, ordinary loss and added margin each missed at offset zero on all four prompts, reached the 96-token cap four times and produced **0/4 exact replies through natural EOS**.

## What changed

[codex] The new lease ran the original trainer, deployed writer, repaired receipt reader, and full-shape calibration. It started the two training arms from the same frozen step-3333 parent and fresh optimizer moments. Both arms saw the same 3,938 causal transitions and 167 supervised tokens including four EOS. The only intervention was the extra opening margin in the treatment objective.

## Measurements

[codex] Both new step-3334 checkpoint files matched their restored historical counterparts byte for byte. The ordinary-loss checkpoint has SHA-256 `ee4b806de58f66dba01299744494969f744d734d120bed2c3ce7308bf56eab6f` and the added-margin checkpoint `1f8196ad1e66feaa4a48df69fdb04cf94c1a25a267c85fa330a1e07519f9a7fb`. The training reports matched exactly after checking and removing the two provenance fields for the physical device and its calibration. The registered gain altered 3,075,695 gradient words according to the training instrument. The independent byte comparison found 40,776 different deployed weight words between the two trained arms.

| Original TRAIN pair and side | Expected first token | Both trained arms emitted | Added margin versus ordinary training, coarse gap | Added margin versus ordinary training, within-group gap |
|---|---|---|---:|---:|
| 33 A | ` M` (379) | space (223) | +61 | +5 |
| 33 B | ` Fo` (3712) | ` The` (287) | −2 | +22 |
| 120 A | ` While` (2999) | space (223) | +32 | +27 |
| 120 B | ` V` (3821) | ` The` (287) | +2 | +7 |

[codex] Gaps compare the correct coarse group or reference fine token to its strongest eligible rival in that head, measured in Q15 units. Neither rank changed on any prompt relative to ordinary training. On pair 33, both contexts favored A's group. On pair 120 both favored B's. Subtracting the total score for swapping the two answers from the total for assigning them correctly cancels a constant preference for one target. This two-sided advantage was −134 and +474 under ordinary training and −125 and +473 under the added margin. The treatment changed it by +9 and −1. The frozen parent gave −44 and +480.

[codex] Every captured GPU score, rank, actual token ID, mismatch and stop matched the pilot rows byte for byte. Within the repeat, ordinary training and added margin had identical 96-token trajectories on three prompts. Only pair 33 side B differed between trained arms, first at zero-based offset 62. Both were already wrong at offset zero. The ordinary GPU readout file has SHA-256 `fe463c5e4c4f89eaf895ef956dd55289f96124363a2c11f2ef171eaa761dd8f0`.
## Checks run at the head

[codex] The short physical training fixture and full-shape synthetic fixture matched their earlier numerical receipts after accounting for the new GPU UUID. The serving calibration matched its earlier receipt byte for byte. All eight full-shape derivatives were live with zero clips. An independent reviewer verified the physical job, original token tape, entire input manifest and device identity before authentic training. The short check, long check, authentic release and subsequent full-reply readout are separate recorded events. [Independent FIT](FIT.md) · [Actual run release](GO-AUTHENTIC.md)

[codex] All 12 cold ordinary-client processes reproduced the complete self-fed GPU IDs, output bytes and stop reasons. The comparator matched 64 files byte for byte and six complete JSON records after validating the enumerated device and receipt fields against their own runs. It rejected all four planted corruptions and checked the new lease, three ordered GPU jobs, files and teardown. The run used a different physical RTX 4090. The provider confirmed deletion after 342 ledger seconds at $0.869/hour, approximately $0.083 for the instance. [Independent comparison](evidence/reproduction.json) · [Compute](evidence/compute.json) · [How to repeat the checks](grade/README.md)

[codex] The 60,522-byte private packet has 132 members. A fresh download verified every file and reconstructed the two checkpoints from the independently restored pilot files after the repeat proved byte equality. Each reconstructed authentic file passed the new device manifest. [Restore instructions](RESTORE.md) · [Pinned receipt](evidence/restore.json)

## Skills used

[codex] The native-space cycle identified the common fixed-point representation, shared operators, serialization checks and ordinary-client boundary for the two trained artifacts.

## Still open

[codex] These same-question comparisons include different authentic conversation histories and passages. Isolating passage influence requires a controlled swap. This experiment leaves the other 99 TRAIN pairs and all 26 reserved pairs outside its claims. Individual product-key memory slot choices during training were unavailable from the original ABI. Gradient and optimizer moment evidence is recorded as digests, whereas both checkpoint files can be compared directly.

[codex] Before increasing the training budget, hold the two target rows fixed and trace the two prompt states at the first coarse election. Score both targets on both states and repeat with the states exchanged. Compare with the strongest third group and the ordinary global writer. This separates a shared target preference from information that changes with the conversation and can identify whether improving state formation or output-row competition is the next useful intervention. The project still requires complete, distinct answers on fresh opposite-role pairs as part of M2. [M2 milestone](https://github.com/UOR-Foundation/uor-r4/issues/2030)

## Plain summary

[codex] This training signal changes real weights. It improved the score gap inside the correct group on all four examples but left the wrong group winning on every example. The separate physical repeat reproduced that failure.
