# A training penalty recovered literal reads and lost useful nonliteral ones

Codex, #4063. The [development experiment](SHEET.md) compares two sets of learned integer weights against frozen P4. The [campaign clock](campaign/SHEET.md) measures the current three-arm trainer. Both use one leased Vast RTX 4090 while Gemini (Antigravity) runs its separate qualification on physical Wick.

## What did not work

The penalty failed the condition for scaling. It recovered one held next token over ordinary training, and the paired interval for its mean Brier improvement includes zero. On the 44 fresh records where frozen P4 chose a different token pair with the correct successor, ordinary training retained 43 useful reads and the penalty retained 31. The emitted top token stayed correct on the same 33 of those 44 in every arm. A mixed predictor can hide a damaged read in its final accuracy. Frozen P4 remains the serving baseline. [Per-record measurements](evidence/quality.json)

## What changed

The only training change was to charge one Q32 Brier unit when a proposed weight update made a later, different pair at distance zero displace a literal source whose successor was correct. A different address can be harmful even when its token happens to agree. The scalar training adjustment is absent from the serialized model and ordinary inference. Both arms trained on the same 65,536 next-token targets, with 64 coordinates × 15 candidate moves for each of 16 batches. The ordinary train reproduced the historical checkpoint byte for byte. The learned endpoints differed in 296 of 16,384 integer root coordinates. [Method and finite controls](SHEET.md)

## Measurements

### Quality on new records

One causal prediction per record, 1,024 distinct DEV-tune records, zero record overlap with the previous 512, including 305 positions past the context window. Brier measures the full next-token distribution, lower is better. Bootstrap intervals resample whole paired records. The two post-training models share the same reader, mixture and writer. [Raw observation summary and independent recount](evidence/quality.json)

| Model | Mean Brier | Top token correct | Correct literal sources displaced by a wrong successor, of 306 | Useful nonliteral successors retained, of 44 |
|---|---:|---:|---:|---:|
| Frozen P4 | 0.496362 | 626/1,024 | 0 | 44 |
| Ordinary Brier update | 0.497519 | 625/1,024 | 4 | 43 |
| Brier plus alias penalty | 0.496722 | 626/1,024 | 2 | 31 |

Penalty versus ordinary training improved mean Brier by 0.1601% relative to ordinary, paired 95% interval from 0.4164% better to 0.0184% worse. Penalty versus frozen worsened mean Brier by 0.0725%. Ordinary training versus frozen worsened it by 0.2330%. Two fresh records (31 and 650) account for a reduction of 3,642,753,662 summed Q32 Brier units versus ordinary training. Two other harmful alias collisions survived both updates. Across the remaining 1,020 records the penalty worsened Brier by 138,594,712 units. The first changed fresh top choice was row 31. For the target `0` (ID 18), ordinary training read source 2043 and emitted `7` (ID 25). The penalty restored frozen address 2037 and emitted `0`. Other changed top tokens moved between wrong answers. Source addresses are zero-based positions in the active causal window. [Fresh and historical contrasts](evidence/quality.json)

| Prior row | Target | Source position, frozen → ordinary → penalty | Next token, frozen → ordinary → penalty |
|---:|---|---|---|
| 6 | `,` | 659 → 733 → 659 | `,` → `,` → `,` |
| 36 | leading-space `for` | 283 → 2016 → 283 | ` for` → `)` → ` for` |
| 413 | `/` | 1782 → 1837 → 1782 | `/` → `.` → `/` |

The penalty restored the original GPU score and source on all three older regressions. Row 6 changed the selected successor and probability without changing the emitted token. These three previously identified cases are excluded from the fresh comparison. Only the first two training batches have identical incoming roots in both arms. At that boundary 29 of 1,920 candidate lanes acquired a penalty, spanning 70 lane-row events. Ordinary training committed nine protected displacements across its trajectory and the penalty arm committed zero. Differences after the first divergent update are downstream effects of changing the optimization path. [Training and replay evidence](evidence/quality.json)

The normal client reproduced the frozen and trained GPU traces on five selected prefixes, with four extra independent cold CLI processes covering rows 36 and 413 in both artifacts. The mass digests for the frozen and penalty arms agree on those five packed witnesses. The first new disagreement with ordinary training (index 4) changed the tie count. The fresh `0` recovery at index 31 was observed in GPU evaluation. Its record was outside the preregistered cold-client set. Training took 31.28 s for the two arms, fresh and historical GPU evaluation 0.685 s, loading 26.90 s and repacking/reloading/ordinary replay 213.67 s. Total process wall was 272.69 s. Sampled quality peaks were 4,524 MiB of GPU memory and 100.41 W board power. [Checkpoint, row and resource identities](evidence/quality.json)

### The three-arm campaign clock

Sixteen full production batches ran on 65,536 R targets, with 37,128 strict sliding-window targets. Their summed batch time was 2.0462 s, a median of 131.3 ms per 4,096 targets (range 49.1–162.2 ms), or 32,028 targets/s with all three arms. Preparing the causal rows took 1.9531 s (95.45%). Scoring and reducing the candidate moves, including readback, took 0.0860 s (4.20%). The scorer considered 46,080 moves in total. The process-cold artifact load and CUDA setup took 16.87 s. Restoring the sixteen causal snapshots took a separate 90.94 ms in total. At launch 3330 the direct and consecutive paths gave the same Brier sum over all 4,096 targets and the same digest of eight complete spot rows. Peak sampled GPU memory was 4,620 MiB, host high-water RSS 3.80 GiB. [Sixteen batch times, snapshots and verified kernel digests](evidence/timing.json)

| Training span | Estimated contribution from full launches | 4090 compute at the quoted rate for that contribution |
|---|---:|---:|
| One epoch | 8.0 min | $0.12 |
| Three epochs | 24.0 min | $0.35 |
| Six epochs | 48.0 min | $0.69 |

Estimates weight the 3,602 full launches per epoch by their four depth strata, treating the deliberately included first launch as a single observation of itself. The other 1,722 partial launches contribute 1,193,792 targets. Their wall times and the development and sealed evaluation sweeps are unmeasured. Only sampled launches updated weights. Late joint-update frequency may differ on an uninterrupted trajectory. A complete original Wick run had no source-bound training receipt available at the last passive check at 04:10 UTC. These are measurements and extrapolations for the rented card. [Measured scope and timing sensitivity](evidence/timing.json)

## Checks run at the head

The harness counted 7 code files (+1,733/−3), 7 documentation files and 5 additional evidence or patch files. The two evidence summaries are generated from immutable raw observations. The pre-data ORDER was signed at the quality source pinned in the FIT. The delivery edit adds the harness field label to its existing roadmap link.

Independent pre-run reviews pinned both instruments, inputs, device and stops. The quality build passed 32 tests and the campaign build 19. The quality kernels passed audits for integer instructions, and the campaign’s five original kernels matched the disassembled production SASS byte for byte. The rented-device tests passed the ambiguity controls, the production 4,096-row driver, and the 11-row and 512-word causal restore check. The ordinary baseline re-created the previous checkpoint SHA-256 `f681fad83f59376bf5b6a3598c1af0e89834bc6b98e6645c1b918544c990c0ac`. Independent recounts from the raw rows reproduced the paired intervals, record disjointness, and both earlier standard CLI errors. [Quality and timing FIT](FIT.md) · [Kernel comparison](campaign/kernel-comparison.tsv)

The new D11 is 3,987,940,360 bytes, SHA-256 `8489f746aa2e5f5f7dd73eb3ccfa09d38bcb3f8ad1f2f4002b7c98aa9240ebf4`. All 127 rental files verified on readback in Wick ZFS at `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-p5-alias/rental`, manifest SHA-256 `3bee507ad8415029e67712f4a792a4cef4cbc425720614de37df7b91d66b6703`. A 441,716-byte delta reconstructed the complete candidate against the frozen P4 on Wick. The private archive also includes the native learned weights, full row observations, the upstream source bundle and restore instructions. A pinned Hub download verified all 133 included files and the 2,636,996-byte archive SHA-256 `1738107755a555f1b1b9a01de54ca56ab8999ba1359a5cdb2ce14b27903691dd`. On macOS use an extractor that keeps AppleDouble metadata files verbatim, such as `python3 -m tarfile -e`, to reproduce the complete readback. The pod ledger recorded 1,084 s uptime at $0.869/h, about $0.262 compute before any provider fees. The pod was deleted after ZFS verification. [Pinned private evidence](https://huggingface.co/datasets/caseyallard/uor-r4-store/blob/8d9b5afd3383f4c814ab1c65e7dd8d839e26e9de/results/codex/p5-alias-20261126/p5-alias-receipts.tar.gz)

## Skills used

The native-space cycle governed the persistent integer S1 updates, bound artifact and cold replay. The UOR research architect skill kept the development observations tied to the existing predict, recall and ordinary-serving roadmap.

## Still open

Keep frozen P4 in service and leave the registered full P5 decision to its controlled campaign. On the quality track, the next component identified in the [program review](../../end-to-end-integration/codex-4307/REPORT.md) is the learned S4 memory read. Make its signed, source-addressed contribution reach the token scores before emission in the ordinary client. Compare the trained route with frozen P4, memory disabled and shuffled addresses at matched bytes. Use held conversation and prose, include cases requiring a word absent from the context, and report first changed tokens, reply quality through EOS and regression on ordinary next-token prediction. The present finding supports testing whether a trained gate can select a useful read while preserving other ways to emit words.

On speed, first time the partial launches and multi-arm development evaluator on a representative development slice, then split the 95.45% row-preparation cost between sliding-cache maintenance, count/mix lookups and row construction. Check any incremental or Omega implementation on the same source, row words and learned decisions before measuring its end-to-end gain. These measurements give a path to a total runtime estimate. No current result gives a date for frontier quality or qualifies the candidate in a browser. [Model and backend roadmap](../../end-to-end-integration/codex-4307/REPORT.md)

## Plain summary

A change to the training objective restored both earlier wrong tokens and one new one. It also lost twelve useful nonliteral reads relative to ordinary training. Keep the current model. The measured full training batches already run in fractions of a second on one 4090. Preparing what each batch will score consumes almost all of that time.
