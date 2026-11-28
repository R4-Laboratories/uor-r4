# Where the first lexical winner is lost in shared-weight training

## What did not work, first

[codex] Neither seed cleared the registered TRAIN gate. In the last epoch a *different original ID* still defeated the correct word immediately after the chosen rival had been beaten on 3,516 of 3,891 openings in each seed. A post-run join by original ordinal found 3,294 openings with this failure in both seeds. Only 105 of those had the same replacement ID in both. [Cross-seed overlap](evidence/replacement-overlap.json) The group-disjoint refit and any fresh chat panel stayed closed. The initial synthetic ceiling check also failed when it equated equal worst margins with equal ranks. The repaired expectation passed before real measurements. [Registration](SHEET.md) · [FIT](FIT.md)

## What changed

[codex] The runner observes a complete 4,095-ID election before a native update, checks the old rival, repeats the election immediately, and compares with the shared epoch-end checkpoint. Geometry replays the original #4353 update. The comparison now matches 584 deployed bytes, four 26-slot reachable banks, input rows, epochs and order as well. Each bank in the ordered control maps its full finalized prefix hash onto the same reachable slots. The bound attachment records the operator for ordinary serving. [Native operator](../../../../../crates/uor-r4-integer/src/discrete_d11/native_occurrence/lexical.rs) · [Runner](runner/src/trial.rs)

## Measurements

[codex] Both operators reached exactly 26 words in each bank. At both fitted seeds geometry used 26, 25, 26 and 26 nonzero words, compared with 26 in every bank for matched ordered. The zero control used none and every unreachable word remained zero. Each arm occupied 584 deployment bytes beyond the common P4.

[codex] Both seeds reproduced all eight sets of #4353 update counters and both original bound geometric checkpoints. The epoch-eight counts below refer to 3,891 admissions. The retained-win row counts immediate wins that remained strict. Later gains contribute the balance of the final totals. [Epochs](evidence/epochs.json) · [Independent recount](evidence/recount.json)

| Geometric first lexical token on TRAIN | Seed 435300 | Seed 435301 |
|---|---:|---:|
| Strictly ahead before visiting this row | 101 | 123 |
| Strictly ahead of the original strongest rival after the visit | 3,829 | 3,838 |
| Strictly ahead of every other ID immediately | 310 | 319 |
| Immediate wins surviving to epoch end | 59 | 38 |
| Additional wins picked up later | 36 | 14 |
| Strict wins at the shared checkpoint | 95 | 52 |

[codex] On the 3,886 jointly strictly attainable rows from 3,703 groups, zero had 80 strict wins, matched ordered had 104 and 82, and geometry had 95 and 52. The registered thresholds were a five-percentage-point advantage over both controls and retention of at least 80% of instant wins. Geometric retention reached 19.03% and 11.91%. Geometry minus ordered was −0.23 and −0.77 percentage points, with paired group-resampled 95% intervals of [−0.72, +0.26] and [−1.25, −0.33]. Across all admissions the first-loss classes for seed 435300 were 3 structurally infeasible, 62 clipped against the old rival, 3,516 replacement rivals, and 251 erased immediate wins. For 435301 they were 3, 53, 3,516, and 281. These classes identify the earliest failure during the row's visit and subsequent replay. A later update can restore a row. [Grade and intervals](evidence/epochs.json)

[codex] On the preregistered first TRAIN ordinal under seed 435300, target ID 93 began at rank 321 behind ID 143. Its update beat 143, but left 638 ahead and the target at rank 5. At the shared checkpoint the target was rank 20 behind 166. The other seed moved the same target from rank 112 to 12 immediately and then 166 at epoch end. These ranks are for the same supplied first lexical boundary. [Witness](evidence/sentinel.json)

## Checks run at the head

[codex] The exact integer device tests, six fabricated arithmetic checks and seven host grader checks passed. Five ordinary P4 client cases passed. The independent recount agreed on 124,512 visits, all epoch counts and the failed decision. Cold ordinary serving from the pinned TRAIN prefix agreed on all five first commits across zero and four fitted heads. Winner ID, winning prior/score, addresses, ties, applicable carry and input bindings matched. The engineering client and final release client each passed five fresh invocations. The release keeps the existing ordinary trace family label `ordered`. The checker authenticates the distinct matched operator by its UORLEX03 header, attachment digest and computed addresses. Final release processes took 17.17–17.34 s and used 3,887,904–3,888,320 KiB peak RSS. This was a one-step replay on one conversation. [Release cold evidence](evidence/cold-release.json) · [Engineering cold evidence](evidence/cold.json) · [Replay](REPLAY.md)

[codex] The rented RTX 4090 used 31.58 s of process wall time, including 26.20 s to load the base, 0.65 s for original capture and baselines, and 2.86 s for four fits. Its peak one-second device sample was 4,314 MiB. The provider ledger recorded 319 s of uptime at $0.869/hour, approximately $0.077 at the listed compute rate. The pod was deleted after all 28 output and job files were verified on Wick. Local archive extraction verified 87 members, and a forced download from the immutable private store matched the compressed packet byte for byte. The separate release-client packet verified all 25 members and matched on immutable forced download. [Release restore](evidence/restore-release.json) · [Run](evidence/run.json) · [Restore](evidence/restore.json)

## Skills used

[codex] The native-space cycle governed shared state, matched original-token operators, bound native words and ordinary replay. The D85 contract check selected one rule and passed. Operator byte parity here covers the registered witness. [Native-space specification](../../../../harness/native-space.json)

## Still open

[codex] The registered next action is a full-election correction in weight training. Use the present update and a sparse gold-address-only update as controls. For a candidate maximum word displacement, raising the four target words and lowering participating competing words with i16 clipping simultaneously maximizes the margin against each rival. A monotone search therefore finds the least possible maximum change that wins the row when a strict win is attainable. Measure immediate success, how many words move, and how many correct answers survive the **shared** checkpoint. Scaling compute or opening more held conversations before testing this correction would leave the dominant failure unresolved.

## Plain summary

[codex] The current optimizer usually corrects the opponent it chose, while another token wins the actual first-word election. Later training removes most of the few full-election wins. The ordinary client reproduces the learned decisions; selecting a useful first word and sustaining useful replies through EOS remain open.
