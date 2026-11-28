# One opposed opening update on two real question pairs

## What did not work, first

[codex] **Pre-run qualification was incomplete.** I released the authentic update after an original-versus-zero-gain calibration using two rows, three transitions and an identity token permutation. The registered control required four rows, a 1,017-transition workspace and the actual permutation. The authentic 3,938-transition check compared forward state and could not establish zero-gain backward parity. Treat the original run as exploratory. Its saved-checkpoint outputs and parameter differences are reproducible measurements. [Original FIT](FIT-INTAKE.md) · [Release correction](GO-TRAIN.md) · [Supplemental calibration order](CAPACITY-ORDER.md)

[codex] Against ordinary answer/EOS training, the opening-specific term changed no first-token decisions on these four registered TRAIN conversations. Both arms still lost the coarse election on every opening. All four reference tokens also lost to a rival inside their own fine cell. The output-ID result cannot qualify generalization, retrieval or useful chat.

## What changed

[codex] Both complete UORBFY03 checkpoints started from the same 30,270,020-byte frozen parent, with the same fresh optimizer moments, and consumed the same four human conversations in the same order. The physical original-ABI control took one native update with ordinary full-answer/EOS supervision. The treatment took one update with those same 167 supervised IDs plus relative score gradients at the four first answer positions. Both use the ordinary unchanged writer and full conversation state. One pair came from the dominant training wording family and one from another family. No held example entered this unit. [Registration](SHEET.md) · [Input selection](intake/SELECTOR.md)

## Measurements

| Original TRAIN pair / side | Gold ID | Frozen first ID | Control first ID | Treatment first ID | Treatment minus control, gold coarse gap | Treatment minus control, gold fine gap |
|---|---:|---:|---:|---:|---:|---:|
| 33 / A | 379 | 223 | 223 | 223 | +61 | +5 |
| 33 / B | 3712 | 223 | 287 | 287 | −2 | +22 |
| 120 / A | 2999 | 223 | 223 | 223 | +32 | +27 |
| 120 / B | 3821 | 223 | 287 | 287 | +2 | +7 |

[codex] Each gap is the gold score minus the strongest eligible other score in the same head, in Q15 units. The treatment's four coarse gaps were −1,761, −24,815, −29,314 and −3,889. Its four conditional fine gaps were −2,564, −6,717, −3,224 and −8,009. All four within-cell gaps moved toward their targets relative to control. Three of four coarse gaps moved toward their targets. Two of the ordinary outputs changed relative to the frozen parent under both arms. Those changes are shared with ordinary training.

[codex] The device reported 3,075,695 gradient words differing between arms. All eight opening derivatives were nonzero and their device traces recorded zero clamps. Static comparison of their complete checkpoints found 40,776 differing i16 parameter words, including 9,613 tied embedding coordinates, 29 coarse centroid coordinates and 31,134 elsewhere. The gradient and moment digests come from the training receipt. Raw gradient and moment buffers were unavailable for an independent rehash. [Native update](evidence/native-update.json)

[codex] In every arm pair 33 favored A's target group on both conversations, and pair 120 favored B's on both. Each arm therefore got one side of each direct comparison right, with zero ties and **0/2** flips to the correct target on both sides. The treatment moved the four target-versus-partner gaps by +10, −1, −19 and +18 Q15 units relative to control. Neither coarse nor conditional fine rank changed on any of the four prompts.

[codex] The two-sided comparison also measures a weaker effect. Sum each side's own-target-minus-partner score. Any constant preference for A over B cancels in that sum. For pair 33 this gave −44 frozen, −134 control and −125 treatment. For pair 120 it gave +480, +474 and +473. The treatment therefore changed the correct-versus-swapped assignment advantage by +9 and −1 relative to control. On pair 120 the correctly assigned scores had a higher total even though the same target won on both prompts. Under treatment the one-sided target preferences had magnitudes of 22,975–25,967 Q15 units. These measurements include the different authentic dialogue histories of the two sides.

[codex] All three arms missed at output offset zero on all four replies. In the trained arms ID 223 was a space and ID 287 was ` The`. The four expected pieces were ` M`, ` Fo`, ` While` and ` V`. Every arm reached the 96-token cap four times. Each had **0/4 natural EOS and 0/4 exact replies through EOS**. Comparing the two trained arms directly, emitted IDs first differed at offset 62 on pair 33 side B. The other three full 96-token trajectories agreed. Across 12 fresh ordinary CLI processes on Wick, the actual role-derived prefix, all self-fed IDs, stdout bytes and stopping condition matched the GPU result. The independent grader rejected a planted first-token error after checking the original file hash. [Grading results](evidence/independent.json) · [Ordinary client verification](evidence/cold.json)

## Checks run at the head

[codex] A physical 512-wide finite fixture compared original update, patched zero gain and active gain. At an amplified gain it produced the predicted direction after checkpoint reload. At the registered gain and margin it changed 1,023 coarse and 2,045 fine gradient words. The real run consumed 3,938 causal transitions and serialized both step-3334 checkpoints. [Finite decision](GO-TRAIN.md) · [Physical receipt](evidence/training-calibration.json)

[codex] The first full-reply measurement stopped before decoding. Its validator expected a parity flag from an earlier draft of the training receipt. The actual control was the original legacy GPU kernel. The corrected validator verifies its exact cubin identity, receipt hash, optimizer and gradient digest fields and the registered update budget. The original refusal is preserved. Both trained files and the training receipt were reused byte-for-byte.

[codex] The single RTX 4090 rental was deleted after 455 ledger seconds at $0.869/hour, approximately $0.11 for the instance. A forced download of the 13,595,602-byte private packet verified all 205 members and reconstructed both complete checkpoints byte for byte. [Compute ledger extract](evidence/compute.json) · [Restore receipt](evidence/restore.json) · [Replay instructions](RESTORE.md)

[codex] A subsequent synthetic calibration ran on a second RTX 4090 with the original parent and its real vocabulary permutation. It exercised four distinct rows, 3,938 transitions and 167 scored labels, with 1,017 transitions in the longest row. The original and patched zero-gain kernels agreed on all 15,130,880 entries in each of the gradient, weight and two optimizer-moment buffers and serialized an identical checkpoint. The active condition had all eight opening derivatives live and changed the deployed checkpoint. Both comparisons matched 39,573,504 forward-workspace words. Individual product-key memory slot IDs were inaccessible through this training ABI. The working wrapper initially compared two differently formatted GPU UUID strings after teardown. The normalized IDs matched. The device checker had already exited zero. The instance was deleted after 72 ledger seconds (about $0.017). All 45 supplemental archive members verified after a forced download. [Capacity FIT](CAPACITY-FIT.md) · [Device receipt](evidence/capacity.json) · [Rental](evidence/capacity-compute.json) · [Supplemental restore](evidence/capacity-restore.json)

## Skills used

[codex] The native-space cycle supplied the fixed-point state, shared train/serve checks, physical calibration, cold restore and outcome boundary.

## Still open

[codex] There was no extra two-sided opening win in the saved checkpoints. The operating-shape calibration now supplies retrospective evidence. Prospectively repeat the single update from the frozen parent, with both calibration receipts present before authentic training, to check whether the same weights and outputs reproduce. Keep the remaining 99 TRAIN pairs and 26 held pairs reserved. If the repeat reproduces the weak pair contrasts, trace the contributions of the hidden state and competing output rows before spending on a longer run.

## Plain summary

[codex] The new signal reached the deployed weights. It moved the correct token a little closer to its strongest rival inside its own group on every prompt. The writer still chose a different group and missed the first word in every reply.
