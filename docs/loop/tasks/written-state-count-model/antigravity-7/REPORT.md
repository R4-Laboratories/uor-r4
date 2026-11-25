# REPORT: written-state-count-model, antigravity-7 (Stage B2b `DEV-tune` `P2` Equal-Byte S3 Calibration — `NEGATIVE`)

Lab: antigravity. Item: [`written-state-count-model`](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P2`). Order: [`ORDER.md`](ORDER.md) (`f1b68c42a1b0f27a89fdf39aa88e7ec0676635b5ef2dd08e3e8ccb00e4ea4ce0`). Sheet: [`SHEET.md`](SHEET.md) (`87a758f46e1bb83a3bbfc5f77bd4266b902d7152ff6c310182409e168a83dd4b`). Non-author FIT: [`FIT.md`](FIT.md) (`d3a37172aa07553e09e3bd51379fbdda9616849fac9a9ec347b252a01f5e7699`, `TEST FITNESS: FIT`). Pre-run & GO: [`PRE-RUN.md`](PRE-RUN.md), [`GO.json`](GO.json). Evidence: [`evidence/receipts.sha256`](evidence/receipts.sha256).

## What did not work, first

- **Both pre-registered `P2` equal-byte gates failed (`NEGATIVE`) for `e8` (`E8Product` + Starling 7-neighbour miss + affine-linear Q16 output Pick) on `DEV-tune` (`2,670` records, `10,807,417` targets, `21,928,475` B all; `7,566,174` / `15,995,822` B novel; `9,523,958` / `19,286,759` B response; `7,258,229` / `15,302,949` B novel_response):**
  - **Standalone `P2` (`gate_p2_standalone_1pct = false`):** `e8` (`99,999,994` B) scored `387,756,585,766` Q12 (`4,317,087` micro-BPB all / `4,543,972` novel) versus best standalone equal-byte control `tc_2` (`99,999,998` B) at `191,317,210,741` Q12 (`2,130,029` micro-BPB all / `2,349,539` novel). The paired 95% block bootstrap CI of `e8 - tc_2` on `all` is `[+188,298,283,907, +205,308,195,859]` Q12 (`[+137,224,085,354, +151,006,302,059]` Q12 on `novel`). Even `splitmix` (`386,352,499,173` Q12, `4,301,454` micro-BPB) and `rand` (`273,850,328,566` Q12, `3,048,911` micro-BPB) beat standalone `e8` on `all`.
  - **Combined `P1(A1s) + P2` (`gate_p2_combined_1pct = false`):** Adding frozen `e8` to Stage B1a's frozen `P1(A1s)` baseline (`125,750,699,315` Q12, `1,400,045` micro-BPB all) scored `125,750,080,815` Q12 (`1,400,038` micro-BPB all / `1,684,532` novel) versus best combined equal-byte control `tc_3 + P1` at `125,712,269,808` Q12 (`1,399,617` micro-BPB all / `1,683,420` novel). The 95% paired bootstrap CI of `(e8 + P1) - (tc_3 + P1)` on `all` is `[+31,002,630, +44,813,594]` Q12 (`[+67,040,913, +78,735,931]` Q12 on `novel`), strictly worse than `tc_3`. `e8 + P1` also trailed `tc_2`, `tc_4`, `splitmix`, `tc_1`, `rand`, `tc_6`, and `tc_12` on `all`. While `e8 + P1` improved over `P1(A1s)` alone by `-7` micro-BPB (`ci_combined_e8_minus_p1_all_q12 = [-1,155,500, -195,792]` Q12), it fell far short of the `>= 1%` bar and lost cleanly to simple token-context controls.
  - **`oracle_u_geo_subset` vs `U_count` caveat:** In [`evidence/devtune-p2.json`](evidence/devtune-p2.json), `oracle_u_geo_subset` is `125,749,877,450` Q12 (`1,400,035` micro-BPB), `delta_bits_q12 = -2,864,300,806`, `ratio_milli = 977` relative to `u_count_devtune` (`128,614,178,256` Q12, `1,431,925` micro-BPB from `claude-438`). Almost all (`-31,880` micro-BPB) of that `-31,890` micro-BPB margin over `U_count` comes from `P1(A1s)` (`1,400,045` micro-BPB) already present in the mixture; `e8` itself adds `-10` micro-BPB (oracle) / `-7` micro-BPB (frozen).
- **Wick GPU contention required using a dedicated Vast.ai pod (deleted after archiving):** Wick's RTX 4090 was occupied by another session's `train-in-space` (`PID 693149`), so `GO.json` sealed execution on a dedicated Vast.ai RTX 5090 pod (`vast-62563028`, `09f331eba8c5`, `compute_cap 12.0`, `$0.521/h`, `18:45:01Z`–`19:28:51Z`, `~44` min, est. `~$0.38`). All `55` evidence files plus `tables/e8.uors3t01` and `devtune-pick.rows` were archived to Wick `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-p2-s3-devtune/` (`receipts.sha256` `100%` `OK`), and `vast-62563028` was deleted (`scripts/pod/uor-pod status` shows `0` running pods, `$0/h`).
- **Mechanism negative, not an instrument failure (`DEV-test` untouched):** All five pre-registered verification gates passed with `0` PTX/SASS float ops and `0` divergent replay rows across all 11 arms. Per [`DECISION.md`](../../../../research/reviews/pi-review-2026-11-16/DECISION.md#L279-L299) (`L293-L294`) and [`SHEET.md`](SHEET.md) §4, `e8` `S3` at `100 MB` is closed on `DEV-tune`, `DEV-test` stayed unopened/unscored, and Stage B2 advances to `DECISION.md` Fallback Arm 1 (`VSA SplitMix`).

## What changed

- [`docs/loop/tasks/written-state-count-model/antigravity-7/`](.): Added standalone Rust crate [`p2-s3/`](p2-s3/Cargo.toml) (`#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]` in `src/main.rs` with two `// SAFETY:`-annotated `launch` calls in `src/cuda.rs`, and unconditional `#![forbid(unsafe_code)]` in `build.rs`, `src/builder.rs`, `src/eval.rs`, `src/transport.rs`, plus `src/s3_kernels.cu`), [`ORDER.md`](ORDER.md), [`SHEET.md`](SHEET.md), [`dispatch.sh`](dispatch.sh), non-author [`FIT.md`](FIT.md) (`5` review rounds -> `TEST FITNESS: FIT`, `0` post-FIT source changes across all `13` `GO.json`-pinned files), [`PRE-RUN.md`](PRE-RUN.md), [`GO.json`](GO.json), and [`evidence/`](evidence/receipts.sha256).
- [`docs/research/notebook/m4-serving.md`](../../../../research/notebook/m4-serving.md) & [`docs/roadmap/items/written-state-count-model.md`](../../../../roadmap/items/written-state-count-model.md): Recorded the Stage B2b `DEV-tune` `P2` negative result, exact equal-byte control numbers, pod spend/archive, and transition to `DECISION.md` Fallback Arm 1 (`VSA SplitMix`).

## Native-space cycle verification (D85)

1. **Single persistent state schema:** 12-bit `UORT` token IDs (`BOS = 0`, `EOS = 1`, `V = 4096`), 32-bit `seed1.d11` `Cl(0,6)` causal `F_b` / `S_b` transport blocks, `E8` 240-root codes (`0..240`), `57,600` product keys, self-describing `UORS3T01` count/total tables (`<= 100,000,000` B per arm), `UORS3F01` capture tapes (`hit_kind`, `predicted_root`, `output_pick_q16`, `mass_q48`), Q32 mixture weights, Q48 integer masses, and PRLL Q12 `bits_q12`. Zero floats, zero matmul, zero softmax.
2. **Shared typed operators & Gate 1 static audit ([`sass-check.txt`](evidence/sass-check.txt)):** `uor_r4_count_runtime::s3_state` on host/verify and mirrored `s3_kernels.cu` compiled with `nvcc -arch=compute_120` (`cubin_sha256 = 9bc3ec4dc03840f7d54a96fa95e65af50e35a849e0c005df958865f865804acb`), `ptx_float_opcode_hits=0`, `float_opcode_hits=0` (`sm120_const_mov_aliases=256` non-arithmetic constant loads).
3. **Bit-identical oracle & transition replay:**
   - **Gate 2 ([`finite.json`](evidence/finite.json), `0.53s`):** All `11` arms checked across `175` probes on `finite-c1.uort.u16` (`bytes <= 8192`, `divergent == 0` on every arm; e.g. `e8`: `8174` B, `divergent: 0`).
   - **Gate 3 (`evidence/build/*.meta.json`, `build/*.time`):** All `11` `UORS3T01` tables built `<= 100,000,000` B inside the `180s` guard (`0:33.51` to `2:17.19`), cold-validated via `S3StateTable::from_bytes`.
   - **Gate 4 (`evidence/verify/*.json`, [`devtune-pick.time`](evidence/devtune-pick.time)):** `devtune-pick` (`3:21.21`, `>= 8,000` native `observe_identity_pick` probes, `0` diffs); shifted batches `65,536 == 65,521` `cmp`-identical; `100%` `packed_sha256` match and `divergent_rows: 0` across all `11` arms on both `R` (`19,315` probes, `15,947,584` rows; [`e8.r.json`](evidence/verify/e8.r.json)) and `DEV-tune` (`13,189` probes, `10,807,417` rows; [`e8.tune.json`](evidence/verify/e8.tune.json)).
4. **Update rule:** Deterministic ascending count cutoff pruning inside `<= 100,000,000` B per `UORS3T01` table and 64-pass Q32 integer EM fitted strictly on reserved `R`, then frozen for `DEV-tune`.
5. **Lossless serialization:** Self-describing `UORS3T01` + `UORS3F01` + sealed [`GO.json`](GO.json) (`13/13` pinned source/doc SHAs matched) + [`receipts.sha256`](evidence/receipts.sha256) (`100%` `OK`).

## Measurements (`DEV-tune`, Vast.ai RTX 5090 `vast-62563028`, est. `~$0.38`)

Baselines on `DEV-tune` ([`devtune-p2.json`](evidence/devtune-p2.json), `eval.time = 5:25.00`, `2,152,408 KiB` RSS):
- `U_count` (`claude-438`, `m4-serving.md:39`): `128,614,178,256` Q12 (`1,431,925` micro-BPB all)
- `P1(A1s)` standalone: `125,750,699,315` Q12 (`1,400,045` micro-BPB all / `1,684,543` novel / `1,506,724` response / `1,687,913` novel_response)
- `e8` `oracle_u_geo_subset`: `125,749,877,450` Q12 (`1,400,035` micro-BPB all; `delta_bits_q12 = -2,864,300,806`, `ratio_milli = 977` vs `U_count`)

| Arm | Table Bytes | Build Time | Standalone All (micro-BPB) | Standalone Novel (micro-BPB) | Combined w/ P1 All (micro-BPB) | Combined w/ P1 Novel (micro-BPB) | Combined All (Q12 bits) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`e8`** | `99,999,994` | `2:17.19` | `4,317,087` | `4,543,972` | `1,400,038` | `1,684,532` | `125,750,080,815` |
| **`tc_3` (best combined ctrl)** | `99,999,995` | `0:51.82` | `2,216,691` | `2,537,949` | **`1,399,617`** | **`1,683,420`** | **`125,712,269,808`** |
| **`tc_2` (best standalone ctrl)** | `99,999,998` | `0:40.91` | **`2,130,029`** | **`2,349,539`** | `1,399,690` | `1,683,722` | `125,718,863,335` |
| `tc_4` | `99,999,999` | `1:05.89` | `2,428,375` | `2,823,989` | `1,399,900` | `1,684,179` | `125,737,677,284` |
| `splitmix` | `99,999,999` | `2:02.13` | `4,301,454` | `4,673,771` | `1,400,029` | `1,684,541` | `125,749,299,335` |
| `tc_1` | `23,037,890` | `0:33.51` | `2,759,062` | `2,860,256` | `1,400,033` | `1,684,519` | `125,749,650,933` |
| `rand` | `100,000,000` | `1:22.13` | `3,048,911` | `3,497,037` | `1,400,036` | `1,684,517` | `125,749,943,201` |
| `tc_6` | `99,999,995` | `1:25.16` | `2,752,862` | `3,213,955` | `1,400,037` | `1,684,518` | `125,750,013,621` |
| `tc_12` | `99,999,988` | `1:40.88` | `3,075,054` | `3,449,850` | `1,400,037` | `1,684,525` | `125,750,060,129` |
| `tc_8` | `99,999,999` | `1:31.14` | `2,923,438` | `3,376,471` | `1,400,039` | `1,684,525` | `125,750,156,060` |
| `tc_10` | `99,999,996` | `1:36.94` | `3,016,294` | `3,435,242` | `1,400,041` | `1,684,528` | `125,750,370,586` |

*(Paired block bootstrap, `2,670` records, `1,000` resamples, `seed = 4390`, 95% empirical CI):*
- `ci_standalone_e8_minus_best_ctrl_all_q12`: `["188298283907", "205308195859"]` (`gate_p2_standalone_1pct = false`)
- `ci_standalone_e8_minus_best_ctrl_novel_q12`: `["137224085354", "151006302059"]`
- `ci_combined_e8_minus_best_ctrl_all_q12`: `["31002630", "44813594"]` (`gate_p2_combined_1pct = false`)
- `ci_combined_e8_minus_best_ctrl_novel_q12`: `["67040913", "78735931"]`
- `ci_combined_e8_minus_p1_all_q12`: `["-1155500", "-195792"]` (`novel`: `["-1194780", "-315887"]`)

## Checks run at the head

- `rustfmt --edition 2021 --check build.rs src/*.rs`, `cargo test --offline`, `cargo clippy --offline --all-targets -- -D warnings` (`p2-s3`) — **PASS**.
- `GO.json` `13/13` pinned SHAs match working tree; `evidence/receipts.sha256` (`54/54` hashed entries, `55` files total in `evidence/`) `100%` `OK`; `scripts/pod/uor-pod status` (`0` running pods, `$0/h`).
- `uor-harness check --staged`, `check --only harness.native_space_contract`, `ratchet d23|d25|d32`, `claim-wording`, `prep-check`, `loop report --check`, `loop pr-check --local` — **PASS**.

## Skills used

- `native-space-cycle`

## Still open

- **Advance `P2` to [`DECISION.md`](../../../../research/reviews/pi-review-2026-11-16/DECISION.md#L293-L294) Fallback Arm 1 (`VSA SplitMix` geometric binding):** Close `e8` `S3` at `100 MB` on `DEV-tune`; keep sealed `DEV-test` untouched.

## Plain summary

At an equal `<= 100,000,000` B table budget on `DEV-tune` (`10.81M` targets, `DEV-test` untouched), `E8Product` + Starling 7-neighbour + affine Q16 Pick (`e8`, `99,999,994` B) passed every Stage 1–4 hardware and replay check (`0` SASS float ops, `0` divergent rows across `11` arms) and then **failed both `P2` gates**: `e8` scored `4,317,087` micro-BPB standalone vs best equal-byte control `tc_2` (`2,130,029`) and `1,400,038` micro-BPB combined with `P1(A1s)` vs best equal-byte control `tc_3` (`1,399,617`, 95% CI `[+31M, +45M]` Q12 worse). Per [`DECISION.md`](../../../../research/reviews/pi-review-2026-11-16/DECISION.md#L293-L294), this closes `e8` `S3` at `100 MB` on `DEV-tune` and advances to Fallback Arm 1 (`VSA SplitMix`).
