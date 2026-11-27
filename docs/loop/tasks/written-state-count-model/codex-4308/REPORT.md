# P5 trained weights changed ordinary predictions and lost two correct tokens

Lab: codex. [Pilot order](ORDER.md).

## What did not work

Codex, 26 November 2026. References #4063. Wick RTX 4090, source `8bad9892ca847a671b1b948d3e7ba04599b4ca68`. The frozen P4 got 301/512 development targets right. After 16 batches the trained P4 got 299/512 right. Mean Brier increased from 0.500015 to 0.501927, a relative 0.383% worsening (paired record-bootstrap 95% interval −0.003% to +0.886%). The optimizer saved 15 accepted updates across 69 root coordinates. Cold ordinary serving reproduced both incorrect next tokens. The full P5 gate, free generation and browser serving still require independent qualification. [Measurements](evidence/measurement.json) · [Cold CLI traces](evidence/cold-replays.json) · [Independent FIT](FIT.md)

## What changed

All three arms saw the same 65,536 R targets. Original and reseeded canonical arms visited the first 1,024 of 16,384 coordinates. The random-coordinate control started from the same reseeded state. DEV-tune uses one causal prediction from each of 512 different records, 152 beyond the 2,048-token window, median record depth 896. Recorded effects compare to the frozen deployed P4. The control arms match the nominal candidate budget. Their token frequencies and affected-row counts can differ. Lower Brier is better. [Sampling and controls](SHEET.md)

| Saved state | Changed roots from its own start | Mean Brier | Relative change vs frozen P4 | Top-1 correct |
|---|---:|---:|---:|---:|
| Frozen P4 | 0 | 0.500015 | reference | 301/512 |
| Reseeded, frozen | 0 | 0.500053 | +0.0076% | 301/512 |
| Original roots, canonical updates | 69 | 0.501927 | +0.3825% | 299/512 |
| Reseeded, canonical updates | 392 | 0.500060 | +0.0090% | 301/512 |
| Reseeded, random updates | 224 | 0.500023 | +0.0016% | 301/512 |

The geometric arm improved its immediate training batches by 23,662,383,049 Q32 units in total (0.0169% of their pre-update sum). On held records, 44 Pick traces changed, including 21 with a different source position. Six target masses changed. Eleven individual Brier scores improved and eight worsened. Both equal-budget controls had better mean held Brier than the trained geometric start. The paired head-to-head intervals both include zero. Exact per-arm and per-record results include the initial reseeded control. [Measurement](evidence/measurement.json) · [Private original observations](https://huggingface.co/datasets/caseyallard/uor-r4-store/blob/ede7e893ff3fbf88ead9d097e6740f8a053ae87d/results/codex/p5-wick-pilot-20261126/pilot-receipts.tar.gz)

A post-run inspection of all changed traces identified three rows where a distinct token pair acquired zero geometric distance and displaced a literal two-token match. The later tied address wins under the current reader. All three worsened. Their combined penalty was 4,243,145,040 Q32 units. The other 509 rows together *improved* by 37,171,103, yielding the observed net loss of 4,205,973,937. These groups use exact original token IDs reconstructed from the pinned DEV-tune records. The first row below had a large target-mass reduction even though the winning token stayed correct. [Source pairs and numerical decomposition](evidence/measurement.json)

| DEV row | Query pair | Old source pair → successor | New source pair → successor | Source position | Ties | First predicted token |
|---|---|---|---|---:|---:|---|
| 6 | 10, 21 | 10, 21 → 14 | 10, 20 → 80 | 659 → 733 | 1 → 23 | comma → comma |
| 36 | 75, 63 | 75, 63 → 388 | 76, 63 → 11 | 283 → 2016 | 1 → 3 | leading-space “for” → “)” |
| 413 | 31, 21 | 31, 21 → 17 | 31, 20 → 16 | 1782 → 1837 | 3 → 12 | “/” → “.” |

Addresses are zero-based inside the active causal window. In all three cases both minimum distances are zero and the selected class is unchanged. The saved candidate changed the original fixed-point score vector on one of the four fixed prefixes and the additional preregistered changed witness. Frozen and trained ordinary P4 sessions each matched GPU mass hashes on all five prefixes. The ordinary `uor-chat` CLI reproduced the selected fixed witness in two fresh processes. Four additional cold CLI invocations after the primary run independently reproduced both changed top tokens, source addresses, ties and top masses on rows 36 and 413. These two supplementary rows were selected by their observed top-token changes. Four stale/parent/binding probes and the planted off-by-one source and stale CLI traces were rejected. The candidate is 3,987,940,360 bytes with checkpoint `f681fad8…` and complete D11 SHA-256 `d39aea9ef58ffb7f3525b2d8ca5773c28ba1a8e444fbbf486e792c6d942dc852`. [Ordinary CLI evidence](evidence/cold-replays.json)

## Measurements

### How much can Pick change?

Over these 512 records, 335 had room to improve. Allowing the best source at the observed distance class gave an optimistic **9.781%** relative Brier reduction. Allowing the best distance class at the observed source gave **0.897%**. Allowing both gave **18.897%**, reducing mean Brier from 0.500015 to 0.405526. Each row gets to use its known next token to select an outcome from a feasible superset. Shared roots have to work across records and predict from the prefix alone. Each restricted minimum was between its joint minimum and actual baseline, and each joint minimum was at least as favorable as every tested arm. The first four predetermined parity rows distinguished both class and position on three rows, with 15 alternative comparisons against full-vocabulary recomputation. [Exact numerators](evidence/measurement.json) · [Row oracle](pilot/src/rows.rs)

### Time, cost and the expensive part

The primary process took **164.84 s** including loading, independent numerical checks, training, evaluation, packing, restoring, and cold ordinary client replay. The 16 training batches took **25.52 s** across all three arms, both scorers and 48 write-and-reload checkpoints. The AOT build, finite fixtures, and supplementary CLI were separate. Build and audit took 64 s on a new pair of Cargo targets with dependencies locally available. The four supplementary processes took 68 s including disk reads. Peak process RSS was 4,452,436 KiB. The one-second device trace peaked at 10,579 MiB and 153.7 W, integrating to approximately 2.89 Wh over 164.1 s. This measures the GPU board. Whole-host electricity is unmetered. Pod rental was **$0**.

| Matched measurement | All 15 generators | Distinct destinations |
|---|---:|---:|
| Candidate destinations enumerated across 48 arm-batches | 46,080 | 32,473 |
| Destinations dispatched with at least one affected row | 38,505 | 27,193 |
| Dispatched destination × affected-row combinations | 33,032,430 | 23,264,914 |
| Mean score-kernel time | 2.257 ms | 1.614 ms |
| Mean entire optimizer step | 84.693 ms | 80.601 ms |
| Median entire optimizer step | 80.660 ms | 78.469 ms |
| Minimum–maximum entire optimizer step | 70.956–159.930 ms | 71.657–124.755 ms |

All 49,152 score words (identity included), committed moves, committed state and affected-row counts matched. Both paths feed the same per-coordinate selection kernel. The comparison records the final committed lanes and their inputs; provisional selections have no separate retained buffer. Of 3,072 coordinate slots, 505 had no affected rows, including two entire arm-batches. Their logical loss words still compared equal. The unique path was faster in 32/48 matched trials, 16/24 in each order stratum. Deduplication saved 29.4% of dispatched destinations, 29.6% of destination-row combinations, 28.5% of score-kernel time and 4.83% of mean optimizer-step wall. Shared frozen-row preparation cost a mean **1,026 ms** per 4,096-target batch (median 854 ms, range 741–1,545 ms), of which 952 ms on average was device processing. The mean checkpoint cost was 16.9 ms per arm. Including both terms, the estimated saving from deduplication is **0.92%** when all three arms share preparation and **0.36%** when each rebuilds it. The individual stage timings and treatment orders are available for every batch. [48 matched trials](evidence/timings.tsv)

The full three-epoch workload has 35,037 full arm-batches and nine 1,856-target remainders under dense packing. Pricing the tails in proportion to rows gives 35,041.08 equivalent full arm-batches. Extrapolating only this pilot implementation at its measured mean gives **4.28 h training** with shared preparation or **10.94 h** with independent preparation, including a write/reload per step. The corresponding all-15 figures are 4.32 h and 10.98 h. Frozen preparation accounts for approximately 78% of the shared estimate. Depth, coordinate frequency, cache locality, checkpoint frequency, fill and arm sharing can change the rate.

The 512-row ordinary evaluation took 55.0–55.8 ms per state with this random-prefix evaluator, about 9,256 arm-targets/s. Five serial states across 21,349,576 sealed targets would cost **3.20 h** at that rate. If the full campaign performs four 10,807,417-target five-arm development sweeps through epoch three, total evaluation rises to **9.69 h** before its first-64-record repeat and spot checks. That gives approximately **14.0 h** for training plus evaluation with shared training preparation (20.6 h with independent preparation). With six epochs and seven full development sweeps, the corresponding shared estimate is **23.1 h**. All are conditional projections of this implementation. Antigravity's draft uses 256 causal slots, a different CUDA scoring schedule, cache rebuilds beyond 2,048 and five-arm shared evaluation. Its proposed 25–40 min and the earlier 2–6 GPU-day budget both need a direct full-batch measurement of that runner. No full campaign elapsed time was observed here. [Per-stage timings and forecast](evidence/measurement.json)

## Checks run at the head

The [qualified dispatch](dispatch.sh) used separate Cargo targets for the ordinary client and pilot. Its build stage ran 30 passing host tests and checked each AOT entry in PTX and SASS. The first shared-target build failed to link distinct tokenizer crate identities. A fresh build and matching finite GPU receipt cleared that error before real data. `git diff --check`, `bash -n`, Rust formatting and the native-space contract passed. The ZFS receipts, pinned Hub archive and both supplementary cold CLI transitions were checked independently. Commands for the three phases are `dispatch.sh build BUILD_RECEIPT`, `dispatch.sh finite FINITE_RECEIPT BUILD_RECEIPT` and `dispatch.sh run RUN_RECEIPT BUILD_RECEIPT FINITE_RECEIPT`, with `PILOT_BUILD` set to the private Wick build directory. Invoke from the pinned implementation checkout; the source and executable digests must agree across phases.

## Skills used

The UOR research architect skill connected the result to the full P5 gate. The native-space cycle specified the integer state, address, update and ordinary serving comparisons.

## Still open

Retain frozen P4 for serving. Test the existing Brier update against an update that penalizes newly created zero-distance aliases on **training** prefixes where they displace a literal pair with the correct successor. Keep the ordinary reader, writer, native integer state, initial checkpoint and update budget identical. Measure fresh, disjoint record-level Brier, top-token recoveries and losses, and the subset where a nonliteral source helps. A precise tie-break control can establish which losses are avoidable without hardcoding sentences. The three observed failures are regression witnesses. They cannot establish generalization. In parallel, time 16 representative full batches in the campaign runner, stratified by actual context depth, and separate shared row preparation, host transfer and scorer time before reserving a longer run. The registered sealed gate remains with Antigravity.

[Preregistration](SHEET.md) · [Pinned dispatch](https://github.com/UOR-Foundation/uor-r4/blob/8bad9892ca847a671b1b948d3e7ba04599b4ca68/docs/loop/tasks/written-state-count-model/codex-4308/dispatch.sh) · [Private evidence and restorable learned checkpoints, 1,381,509 bytes](https://huggingface.co/datasets/caseyallard/uor-r4-store/blob/ede7e893ff3fbf88ead9d097e6740f8a053ae87d/results/codex/p5-wick-pilot-20261126/pilot-receipts.tar.gz). An independent download of the pinned bundle matched SHA-256 `97534e8063e71eb6eb627b11ee0dd705de78c7ed33e9578256c3829967b564e1`. The large derived D11 and original receipts reside on Wick ZFS at `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-p5-pilot`. All 102 primary files passed checksum readback, with manifest SHA-256 `f01677be2bfe87a03a4e0135855fac07dbee4ca51147be244cc1c9276d32f58f`. The private bundle includes the small learned states, raw observations, build and run logs, and both receipt manifests. Re-running [the pilot](dispatch.sh) from the pinned inputs reproduces the model. The full D11 stays on ZFS.

## Plain summary

In under three minutes the owned 4090 trained three sets of weights and produced a model whose changed predictions survived saving and reopening. Three tie collisions explain the loss on this sample. The next quality test can target those collisions on fresh records, while the next speed test can focus on producing the frozen rows that consume most of the time.
