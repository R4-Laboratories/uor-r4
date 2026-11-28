# A shared-address contradiction prevents retention at fixed lexical geometry

## What did not work, first

[codex] Repeatedly restoring integer headroom during training failed the preregistered gate at both seeds. On 3,886 common authentic TRAIN openings, the full geometric update put the correct first content token strictly above all competitors at every visit. Only 21 remained strict at either final checkpoint. Independent reconstruction found 1,334 disjoint pairs of real openings in which the two correct tokens require contradictory settings of these same words. Every fixed head on these addresses must therefore miss at least 1,334 of the 3,886 openings. The 3,109-win full retention requirement exceeds the upper bound of 2,552 by 557. The result holds for arbitrary shared weights under this frozen score form, candidate addresses and P4 prior. The #4356 sparse and group-disjoint gates have separate denominators. Both two-seed fit gates failed, so the registered group refit and new human panel stayed closed. [Independent recount](evidence/recount.json)

## What changed

[codex] The trainer translates each bank’s 26 reachable i16 words by the same integer before each visit. This preserves all candidate contrasts at translation time while redistributing the headroom available for later updates. Sixteen fits crossed two seeds, geometric and ordered addresses, full and target-only updates, and centering on/off. Every arm used the same 3,891 real openings, schedule per seed, eight epochs, writer, priors and 584-byte released attachment. The study used the frozen model’s ordinary prefix, including the optional standalone space when present. The target was the donor’s next original token ID. English answer frames, supplied addresses and human-held labels played no part. [Registered comparison](SHEET.md) · [Independent FIT](FIT.md)

## Measurements

[codex] Counts below require a strict win over all 4,094 rival IDs on the common 3,886 openings. The gate required 195 more correct choices than each control and at least 80% retention of the immediate wins, at both seeds. [Epoch-eight counts, paired whole-conversation bootstrap intervals and age bins](evidence/epoch-eight.json)

| Update | Seed | Centered geometry | Same fit without centering | Centered ordered | Historical geometry | Frozen prior | New corrections retained |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 435300 | 21 | 22 | 23 | 95 | 80 | 16 / 3,859 |
| Full | 435301 | 21 | 36 | 6 | 52 | 80 | 18 / 3,872 |
| Target only | 435300 | 116 | 122 | 96 | 95 | 80 | 0 / 0 |
| Target only | 435301 | 127 | 116 | 104 | 52 | 80 | 0 / 0 |

[codex] The four frozen sparse controls retained exactly the same complete elections after translation. The two geometric heads gained a feasible target-only correction on 3,533 and 3,674 common rows, and newly blocked zero. Actual sparse training nevertheless reached zero final-epoch updates at both seeds. Its apparently perfect final retention applied only to the 116 and 127 already-strict openings. In the full geometric arm, centering reduced final lower-rail occupancy from 93 to 61 words and from 91 to 68. It reduced genuinely clipped requests from 219,606 to 203,122 and from 220,643 to 202,981. Those are distinct from the centered directional rail hits of 203,260 and 203,052. A further 249,132 and 256,022 word writes were pure translations. The youngest all-admission quartile retained 7/972 full centered wins at each seed. Losses already affected recent updates. Across the eight fits in each condition, uncentered training took 7.088 seconds and centered training took 8.819 seconds. [Frozen feasibility](evidence/frozen-sparse.json) · [Timing and run identities](evidence/run.json)

[codex] Each strict token victory imposes an integer lower bound on the difference between its four addressed words and the rival’s four words. In one verified pair the equal addresses cancel in three banks. The first opening needs `w[0] − w[1] ≥ −2`, while the other needs `w[1] − w[0] ≥ 41`. Adding the requirements gives the contradiction `0 ≥ 39`. Matching disjoint openings prevents counting the same necessary error twice. The runner found opposing examples touching 3,686 rows. The independent verifier separately reconstructed the signed inequalities from 2,668 raw GPU re-reads, verified distinct groups within each pair, original token IDs, common membership and absence of reused rows. The 2,552 ceiling follows from those 1,334 pairs alone. The maximum achievable score may be lower. [Census](evidence/conflict-bound.json) · [Independent certificate verifier](runner/examples/recount.rs)

## Checks run at the head

[codex] The independent recount reproduced 498,048 visits, the 128 checkpoint movement summaries and 32 paired comparisons with 4,096 group resamples each. The uncentered arm exactly replayed 64 archived checkpoint word arrays and 249,024 archived visits. The physical finite suite passed under the dedicated 4090 lock. Wick’s normal GPU was occupied; one rented 4090 at $0.869/hour was released after verified transfer to Wick. The managed ledger records 356 seconds of uptime, about $0.086 at the posted rate. The Linux run took 53.03 seconds wall clock. [Pre-run release](FIT.md) · [Reproduction instructions](REPLAY.md)

[codex] All 19 attachments matched in 19 separate ordinary client starts on Wick with CUDA hidden. The checked fields include original winning ID, score, prior mass, ties, source addresses and carry. The private packet restored from its immutable revision with the same SHA-256 as the 157-member verified Wick extraction. Seven host tests passed. [Ordinary parity](evidence/ordinary.json) · [Immutable restore](evidence/restore-training.json)

[codex] On the ordinary-client sentinel, the expected first content token was original ID 93. The frozen prior put it fifth and chose 166. For seed 435300 the full uncentered head ranked 93 at 166 and emitted 166; centering ranked it at 332 and still emitted 166. For seed 435301 the same ranks were 335 and 332, with emitted IDs 686 and 1265. The first error remained at content offset zero. [All 19 target ranks and winning IDs](evidence/first-token.json)

## Skills used

[codex] The native-space cycle supplied the train/serve integer, operator and serialized-state checks. The reviewer followed the ordinary raw-ID entry and separately checked conditional held admission and the fixed-map proof.

## Still open

[codex] A different causal prefix must be able to give certain competing tokens different address contrasts. The first candidate is the midpoint root already transported by D11 before the final snap. Its relative transform depends on the suffix after the midpoint. Protect question swaps whose only meaningful difference occurs earlier in the turn. Preserve the current P4 prior and writer, all four banks, learned bytes and original IDs. Compare the current read with that transported root, an untransported root from the same midpoint and a swapped real prefix carry. First replay the 1,334 protected contradictions. At least 557 must cease to be exact contradictions: if 778 still hold, the full target is already impossible. If the candidate survives this inexpensive test, recheck all rivals on the same population, including newly introduced aliases and opposing pairs. A reduction in the certificate count measures separation. An explicit shared assignment or a trained head reaching the target establishes that the target is attainable. Then require shared training retention, group-disjoint transfer and ordinary self-fed replies through natural EOS before a new chat claim. Current measurements supply no frontier quality, browser, useful-chat or EOS rate. [D11 recurrence](../../../../../crates/uor-r4-integer/src/discrete_d11.rs)

## Plain summary

[codex] More integer headroom really did make individual corrections possible. The model still forgot them. We can now identify pairs of conversations that mathematically cannot both be answered correctly by any setting of the current output weights. The next intervention targets how the prefix selects addresses, with the ordinary writer held fixed.
