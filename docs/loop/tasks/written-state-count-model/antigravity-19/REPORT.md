# REPORT: written-state-count-model, antigravity-19

Lab: antigravity. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P5`, [`DECISION.md` §4 `P5`](../../../../research/reviews/pi-review-2026-11-16/DECISION.md#L335-L346)). Base: `9631cc2fd` (PR `#4306`). Pre-registration & FIT: [`ORDER.md`](ORDER.md), [`SHEET.md`](SHEET.md), [`FIT.md`](FIT.md) (`TEST FITNESS: FIT` R3 `8b37573f` at `62f216a96`, `GO_SHA = 6b41e8321`).

## What did not work, first

1. **Phase `P5` `S1`-Pick mixed-mass Brier training failed the sealed `DEV-test` gate (`verdict = P5_S1_FROZEN_AT_DEFAULT` — configuration closed, mechanism open):**
   - On single-shot sealed `DEV-test` (`5,321` records, `21,349,576` non-BOS targets, `43,326,982` all bytes, `P4_MAX_CONTEXT = 2,048`), `s1_p5_from_seed1` (trained for `3` epochs / `11,682` batches / `11,214,720` candidate trials at `K = 64` on `R` under exact full-vocab Q32 mixed-mass Brier `R1-M6`, `epoch = 420 -> 8,592`) scored **`45,337,664,241,530,629` Q32 (`494,435` micro-Brier)** vs frozen `baseline_seed1` **`45,335,831,539,432,881` Q32 (`494,415` micro-Brier)** — **worsening `DEV-test` Brier by `+40` ppm** (paired 1,000-draw record bootstrap 95% CI **`[+912,062,882,199, +2,777,172,955,450]` Q32**, strictly `> 0`) instead of dropping `>= 1.00%` (`-10,000` ppm).
   - `s1_p5_from_seed1` also **lost to every control arm on sealed `DEV-test` Brier**: worse than equal-trial `ctrl_reseed_canonical` (`45,334,459,266,856,981` Q32 / `494,400` micro-Brier) by `+70` ppm (`95%` CI `[+2,099,036,561,551, +4,293,269,737,739]`), worse than equal-trial `ctrl_reseed_random` (`45,331,326,449,870,045` Q32 / `494,366` micro-Brier) by `+139` ppm (`95%` CI `[+5,508,620,436,260, +7,219,746,585,258]`), and worse than untrained `ctrl_reseed_static` (`45,329,365,953,111,378` Q32 / `494,345` micro-Brier, `-142` ppm vs `baseline_seed1`) by `+183` ppm (`rel_change_ppm`).
   - **All three trained arms degraded out-of-sample relative to their own untrained initial states:** `s1_p5_from_seed1` worsened vs `baseline_seed1` (`+40` ppm, CI `> 0`); `ctrl_reseed_canonical` worsened vs `ctrl_reseed_static` (CI `[+4,382,008,821,233, +5,926,293,910,758]`); and `ctrl_reseed_random` worsened vs `ctrl_reseed_static` (CI `[+1,631,046,422,454, +2,302,688,209,134]`).
   - On `DEV-tune` (`10,807,417` targets, `0` `DEV-test` peeks), `s1_p5_from_seed1` moved `-26` ppm (ep 1) -> `-91` ppm (ep 2) -> `+52` ppm (ep 3; `step_drop = -143` ppm, `total_drop = -52` ppm), tripping the pre-registered `DEV-tune`-only saturation stop rule after Epoch `3` (`dev_tune_saturated_after_epoch_3_step_drop_-143ppm_total_drop_-52ppm`).
   - On `DEV-test` BPB, `s1_p5_from_seed1` scored `220,121,492,078` Q12 (`1,240,349` micro-BPB, `-13` ppm vs `baseline_seed1` `220,124,433,805` Q12 / `1,240,366` micro-BPB, 95% CI `[-7,255,483, +1,291,121]` Q12 straddling `0`), while untrained `ctrl_reseed_static` scored `220,082,283,583` Q12 (`1,240,128` micro-BPB, `-191` ppm). Per `DECISION.md:342`, Gate 5 skipped packing `p5.d11` (`pack_status.json`: `packed = false`), **freezing `S1` at `seed1.d11` (`04652a3d...`)** and leaving Bar A `chat-v0-p2` PRLL at Phase `P4`'s `1.467631` BPB (`antigravity-17`).
   - **Mechanism vs instrument classification (`CHARTER.md §4.1–§4.2`):** **Zero instrument error** (`0` PTX/SASS float or MMA opcodes, `0` diffs across all Gate 2 synthetic & `> 2,048` sliding-window checks, `0 / 640` live CPU `DiscreteCountModel::bind_embedded` + `P4Session::step` spot-check diffs across all `5` arms on sealed `DEV-test`, `0` diffs on `64`-record × `5`-arm determinism repeat, `12/12` `receipts.sha256` OK). This cleanly closes the specific Phase `P5` `S1`-Pick (`16,384`-coord `Cl(0,6)` root) greedy `4,096`-row `BVG-K` configuration under frozen `P4` `UORS6B01` weights (`R1-M6`): (a) `S1` acts only through `code_pick`'s 2-token nearest-earlier-pair pointer inside `P4`'s `22`-expert mixture, where even replacing the entire `16,384`-coord table with random `Cl(0,6)` roots (`ctrl_reseed_static`) shifts full-vocab `DEV-test` Brier by only `-142` ppm (`0.0142%`, `70×` below the `1.00%` bar); and (b) greedy per-`4,096`-row `BVG-K` commits lower single-batch `R` Brier (`-12` to `-26` ppm across `11,682` batches) by overfitting batch-local token co-occurrences (`50,977–84,977` coordinate moves), degrading held-out `DEV-test` Brier on every trained arm.
2. **Note on `DEV-test` BPB at `P4_MAX_CONTEXT = 2,048` vs unbounded-record Stage B5b:**
   - Frozen `p4.d11` (`baseline_seed1`) scored `1,240,366` micro-BPB (`89.44%` `P1(A1s)`, `87.46%` `R*`) on sealed `DEV-test` at the serving `2,048`-token sliding window (`61.2%` of `DEV-test` targets sit at `pos > 2,048`, where `P4Session` drops tokens older than `2,048` from both `code_pick` and `CausalSessionCache`), vs `1,205,738` micro-BPB in Stage B5b (`antigravity-15`) where the session cache spanned full records without a `2,048`-token cap.
3. **Wick power-loss recovery & kernel optimization (`L-002`):**
   - FIT R1 (`feae0449`) caught 3 blocking findings (missing `p5.d11` `fit_json`/`UORS5W02` re-binding + end-to-end test, missing strict `> 2,048` sliding-window parity in Gate 2, and 3 stale Wick byte counts in `SHEET.md`); all 3 were fixed at `8ace40cc8` and cleared FIT R2 (`997efcab`).
   - The first Wick execution into `20261126-antigravity-p5-qual` passed Gates 1–2 (`0` diffs) but was interrupted by a Wick host power loss and reboot (`23:06`) mid-`DEV-tune` Epoch `0` (`0` checkpoints written) because `61.7%` of `DEV-tune` targets exceed `2,048` tokens and the initial stateful kernel rebuilt `CausalSessionCache` serially on thread `0` per slid step while serializing `55,479`-token stragglers.
   - Commit `62f216a96` replaced the stateful hash tables with stateless 256-thread-parallel shared-memory cache evaluation, host-precomputed Order-0 (`UORKNC03` `k = 0`) tables, contiguous `4,096`-row batching (`3,894` uniform batches/epoch, `11,682` batches / `11,214,720` trials for 3 epochs vs `ORDER.md`'s initial `11,679` estimate), and `256`-target tile-balanced multi-arm evaluation with active `pick_class` masking (`> 25×` speedup; `DEV-tune` Epoch `0` dropped from `> 20 min` unfinished to `68 s`). Fresh-context non-author FIT R3 (`8b37573f`) issued `TEST FITNESS: FIT` at `62f216a96`, and per `L-002` the deciding run executed cleanly into fresh sealed root `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p5-qual-r2/`, leaving aborted `20261126-antigravity-p5-qual` untouched.

## What changed

- Added Phase `P5` PR 2 qualification suite under `docs/loop/tasks/written-state-count-model/antigravity-19/` (`0` edits to `crates/` or root `Cargo.toml`/`Cargo.lock`):
  - [`ORDER.md`](ORDER.md), [`SHEET.md`](SHEET.md), [`FIT.md`](FIT.md) (verbatim non-author FIT R3 `8b37573f` at `62f216a96`), and [`dispatch.sh`](dispatch.sh) (`L-002` fresh `OUT_DIR=20261126-antigravity-p5-qual-r2`, `flock -n` Wick GPU lock, `sm_89` & VRAM preflight, `CUDA_DISABLE_PTX_JIT=1`, PTX/SASS zero-float/zero-MMA audit, `cubin_sha256` cross-check across Gates 1–4, `receipts.sha256` sealing).
  - Standalone offline crate [`p5-train/`](p5-train/) (`Cargo.toml`, `Cargo.lock`, `build.rs`, `src/{main.rs,cuda.rs,driver_check.rs,finite.rs,train_eval.rs,p5_kernels.cu}`): pure-integer AOT `sm_89` CUDA kernels (`p5_slot_step_brier_kernel`, `p5_batch_eval_roots_kernel`, `p5_rowsk_score_kernel`, `p5_rowsk_reduce_kernel`, `p5_eval_records_multi_arm_kernel`) reusing `crates/uor-r4-discrete/src/kn_score_division.cuh` and verbatim `crates/uor-r4-integer/src/lawful_likelihood.rs`, with `0` floats, `0` MMA/matmul, `0` hardware variable division, and `#![forbid(unsafe_code)]` on non-CUDA modules.
  - Committed sealed compact evidence in [`evidence/`](evidence/): [`gate1_audit.json`](evidence/gate1_audit.json), [`finite.json`](evidence/finite.json), [`train_report.json`](evidence/train_report.json), [`devtest_p5_report.json`](evidence/devtest_p5_report.json), [`pack_status.json`](evidence/pack_status.json), and Wick [`receipts.sha256`](evidence/receipts.sha256).
- Updated [`docs/research/notebook/m4-serving.md`](../../../../research/notebook/m4-serving.md) and [`docs/roadmap/items/written-state-count-model.md`](../../../../roadmap/items/written-state-count-model.md).

## Measurements

All Wick RTX 4090 artifacts, `5` `UORP5S10` checkpoints, PTX/SASS audits, and reports are sealed at `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p5-qual-r2/` (`receipts.sha256` `12/12 OK`, `$0.00` pod spend, `0` cloud pods).

### Native-space parity & audit summary (`D85`)
- **Compiled cubin:** `p5_kernels.sm_89.cubin` (`cubin_sha256 = 96ce1a34257d2f7519d13af0fc2b4c2829cde77cbfe71cedb24ce541f65996bc`), `5/5` required kernels present, **`0` PTX float/MMA hits, `0` SASS float/MMA hits** (`gate1_audit.json`).
- **Gate 1a (`uor-r4-discrete --features cuda`):** `driver_config_admits_4096_rows_and_refuses_out_of_range` `1/1 ok`; `p5_brier_core` `9/9 ok`.
- **Gate 2 (`finite.json`, `0.90s` on Wick RTX 4090):**
  - `101` causal batch rows vs `P4Session::step`: **`0` diffs**.
  - `rows-K` across `K = 1, 8, 16, 64` (`1,424` lanes): **`0` diffs**.
  - `BVG-K` (`22` steps, `saw_empty`/`saw_single`/`saw_multi`/`saw_joint_accept`/`saw_joint_reject` all `true`) + equal-trial random control: **`0` diffs**.
  - Planted target shift (`433,945,817,201 -> 433,130,418,289` Q32): `changed = true`, **`diff = 0`**.
  - `> 2,048` sliding-window & `64`-token decay boundary check (`record_len = 2,056`, `64/65/128/129/2048` + `7` slid rows `2049..=2055`, `K = 16` `rows-K`): **`0` row diffs, `0` `rows-K` diffs**.
  - Multi-arm tile evaluator (`6` records exhaustive × `5` arms, `275` spots incl. `10` strict `pos > 2,048` spots): **`0` rec diffs, `0` spot diffs, `0` slide tail diffs**.
  - Live `4,096`-row, `K = 64` `uor-r4-discrete` CUDA `DriverConfig` (`0` refused, `4,096` admitted, `4,097` refused) + `Device::{carry_windows, score_rows_k, propose_block}` (`bvgk_cubin_sha256 = fa69822f...`, `rowsk_cubin_sha256 = 9d007de2...`): **`carried_rows = 4,096`, `64/64` `status == 0`, `propose_block` `s_len = 55`, `joint = true` (`4,102,752 -> 4,029,664 -> 2,251,488`), `epoch_untouched = true` (`PASS`)**.
- **Gate 4 live sealed `DEV-test` verification (`devtest_p5_report.json`):**
  - Live CPU `DiscreteCountModel::bind_embedded` + `P4Session::step` spot checks across all `5` arms: **`0 / 640` diffs**.
  - Determinism repeat (`64` records × `5` arms): **`0` diffs**.

### Gate 3 — `R` `rows-K` + `BVG-K` training & `DEV-tune` trajectory (`train_report.json`)
`R`: `4,060` recs / `15,947,584` targets (`3,894` batches/epoch × `3` epochs = `11,682` batches = `11,214,720` equal trials/arm at `K = 64`; `train_wall = 1,107 s`, `single_arm_equiv = 670 s`: `632.3 s` shared materialize + `38.5 s` `s1` `rows-K`/`BVG-K`). `DEV-tune`: `2,670` recs / `10,807,417` targets (`0` `DEV-test` peeks).

| Epoch | `baseline_seed1` `DEV-tune` micro-Brier | `s1_p5_from_seed1` micro-Brier (`ppm` vs base) | `ctrl_reseed_canonical` micro-Brier (`ppm`) | `ctrl_reseed_random` micro-Brier (`ppm`) | `ctrl_reseed_static` micro-Brier (`ppm`) | `s1` step / total drop (`ppm`) |
|---:|---:|---:|---:|---:|---:|---:|
| `0` | `497,587` | `497,587` (`0`) | `497,518` (`-139`) | `497,516` (`-144`) | `497,518` (`-139`) | — |
| `1` | `497,587` | `497,574` (`-26`) | `497,524` (`-127`) | `497,535` (`-105`) | `497,518` (`-139`) | `+26` / `+26` |
| `2` | `497,587` | `497,542` (`-91`) | `497,752` (`+330`) | `497,518` (`-139`) | `497,518` (`-139`) | `+64` / `+91` |
| `3` | `497,587` | `497,613` (`+52`) | `497,574` (`-27`) | `497,543` (`-89`) | `497,518` (`-139`) | `-143` / `-52` (**STOP**) |

| Trained arm (`3` epochs on `R`) | Batches | Exact trials | Commits | Joint accepts / attempts | Coords moved (`distinct vs seed1`) | In-batch `R` Brier delta (`ppm`) | Final `UORP5S10` SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| `s1_p5_from_seed1` | `11,682` | `11,214,720` | `8,172` (`epoch 8,592`) | `3,279 / 7,943` | `51,950` (`7,865`) | `-14` | `9cb3f9c0...` |
| `ctrl_reseed_canonical` | `11,682` | `11,214,720` | `8,194` (`epoch 8,194`) | `3,172 / 8,012` | `50,977` (`15,329`) | `-12` | `f9cb9966...` |
| `ctrl_reseed_random` | `11,682` | `11,214,720` | `11,637` (`epoch 11,637`) | `11,375 / 11,490` | `84,977` (`15,323`) | `-26` | `83930ba3...` |

### Gate 4 — Single-shot sealed `DEV-test` 5-arm qualification (`devtest_p5_report.json`)
`5,321` records, `21,349,576` non-BOS targets (`43,326,982` all bytes), `14,795,227` novel targets (`31,328,319` novel bytes), `P4_MAX_CONTEXT = 2,048`, `eval_wall = 272 s`:

| Arm | Epoch | All Brier Q32 sum | All micro-Brier (`ppm` vs base) | Novel micro-Brier | All Q12 bits | All micro-BPB (`ppm` vs base) | `% P1(A1s)` / `% R*` | Novel micro-BPB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_seed1` | `420` | `45,335,831,539,432,881` | `494,415` (`0`) | `595,008` | `220,124,433,805` | `1,240,366` (`0`) | `89.44%` / `87.46%` | `1,488,134` |
| **`s1_p5_from_seed1`** | `8,592` | `45,337,664,241,530,629` | **`494,435` (`+40`)** | `595,032` | `220,121,492,078` | `1,240,349` (`-13`) | `89.44%` / `87.46%` | `1,488,098` |
| `ctrl_reseed_canonical` | `8,194` | `45,334,459,266,856,981` | `494,400` (`-30`) | `594,985` | `220,100,024,284` | `1,240,228` (`-110`) | `89.43%` / `87.45%` | `1,487,944` |
| `ctrl_reseed_random` | `11,637` | `45,331,326,449,870,045` | `494,366` (`-99`) | `594,940` | `220,091,392,689` | `1,240,179` (`-150`) | `89.43%` / `87.45%` | `1,487,887` |
| `ctrl_reseed_static` | `0` | `45,329,365,953,111,378` | **`494,345` (`-142`)** | `594,911` | `220,082,283,583` | **`1,240,128` (`-191`)** | `89.42%` / `87.44%` | `1,487,823` |

**Paired 1,000-draw record-level bootstrap 95% CIs (`seeds 4393..=4398`):**
- `s1_p5_from_seed1 − baseline_seed1` Brier Q32: **`[+912,062,882,199, +2,777,172,955,450]`** (strictly worse)
- `s1_p5_from_seed1 − ctrl_reseed_canonical` Brier Q32: **`[+2,099,036,561,551, +4,293,269,737,739]`** (strictly worse)
- `s1_p5_from_seed1 − ctrl_reseed_random` Brier Q32: **`[+5,508,620,436,260, +7,219,746,585,258]`** (strictly worse)
- `s1_p5_from_seed1 − baseline_seed1` Q12 bits: `[-7,255,483, +1,291,121]` (straddles `0`)
- `ctrl_reseed_canonical − ctrl_reseed_static` Brier Q32: `[+4,382,008,821,233, +5,926,293,910,758]` (strictly worse)
- `ctrl_reseed_random − ctrl_reseed_static` Brier Q32: `[+1,631,046,422,454, +2,302,688,209,134]` (strictly worse)
- **Verdict:** **`P5_S1_FROZEN_AT_DEFAULT`** (`p5_qualified = false`, `instrument_clean = true`).

## Checks run at the head

- Host `cargo test --offline --locked` in `antigravity-19/p5-train/`: **`7/7` passed** (including `cuda_abi_struct_sizes`, `row_scheduler_and_eval_tiles`, `precompute_uorknc03_o0_matches_compact_count_lookup_exactly`, `bvgk_decide_and_apply`, `non_extrapolating_p5_gate`, `synthetic_bundle_parsers_driver_check_and_bootstrap_ci`, and `rebind_and_pack_p5_deployment_round_trips_modified_s1_end_to_end`).
- Host `cargo clippy --offline --locked --all-targets -- -D warnings` & Wick `cargo clippy --offline --locked --features cuda --all-targets -- -D warnings`: **PASS (`0` warnings)**.
- `bash -n docs/loop/tasks/written-state-count-model/antigravity-19/dispatch.sh`, `python3 scripts/check_no_softmax.py` (`3,828 <= 4,451`), and `python3 scripts/check_no_matmul.py` (`93 <= 93`): **PASS**.
- Wick RTX 4090 `dispatch.sh` (`20261126-antigravity-p5-qual-r2`, `GO_SHA = 6b41e8321`): **ALL GATES 1–5 INSTRUMENT PASS** (`gate1_audit` `0` float / `0` MMA, `finite` `0` diffs + live `4,096`-row `uor-r4-discrete` CUDA check PASS, `train_report` equal-trial invariant `11,214,720 == 11,214,720 == 11,214,720`, `devtest_p5_report` `0 / 640` CPU spots & `0` determinism repeat diffs, `receipts.sha256` `12/12 OK`).
- `uor-harness` checks (`loop report --check`, `claim-wording`, `shard-check`, `check-formats`, `ratchet d23|d25|d32`, `check --only harness`, `loop pr-check --local`): verified clean at exact PR head.

## Skills used

- `native-space-cycle` (`.agents/skills/native-space-cycle/SKILL.md`)
- `uor-research-architect` (`.agents/skills/uor-research-architect/SKILL.md`)

## Still open

- **All lab-scoped phases `P0`–`P5` of `written-state-count-model` (`#4063`) are now complete and sealed.** Per `DECISION.md:342`, `S1` remains frozen at `seed1.d11` (`04652a3d...`), and Phase `P4`'s `p4.d11` (`e0ffbba1...`) remains the current served integer model (`3.988 GB`, `3.978 GB` M1 RSS, `1,239 tok/s` @ `2,048` ctx, `1.240366` BPB on sealed `DEV-test` at `2,048` ctx / `1.205738` unbounded-record, `1.467631` BPB on Bar A `chat-v0-p2`).
- **Phase `P6` (owner-only architecture / charter boundary decision):** Because frozen `P4` missed `0.90` BPB on `chat-v0-p2` and `S1`-Pick Brier training under `P5` closed negative (`+40` ppm Brier vs `seed1.d11`, with `S1` having `< 150` ppm total leverage inside `P4`), closing the remaining `1.468 -> 0.90` BPB gap requires a higher-leverage trainable geometric state (beyond `16,384`-coord 2-token `code_pick` codes) or an owner-only `P6` decision (`CHARTER.md §5.1`).

## Plain summary

Phase `P5` PR 2 executed the full pre-registered, non-author `FIT`-gated Wick RTX 4090 qualification (`$0.00` pod spend) of `16,384`-coordinate `S1` training under exact full-vocab mixed-mass integer Q32 Brier (`R1-M6`) against equal-trial re-seeded controls (`11,214,720` candidate trials/arm across `3` epochs of `R`), alongside live `4,096`-row `uor-r4-discrete` CUDA verification. The instrument passed with zero errors (`0` PTX/SASS floats/MMAs, `0` Gate 2 diffs, `0 / 640` live `P4Session` CPU spot-check diffs, `0` determinism repeat diffs, `12/12` `receipts.sha256` OK), and the scientific verdict is a clean **negative (`P5_S1_FROZEN_AT_DEFAULT`)**: all three trained arms lowered in-batch `R` Brier (`-12` to `-26` ppm) yet worsened held-out sealed `DEV-test` Brier relative to their own untrained starting points (`s1_p5_from_seed1` `+40` ppm vs `baseline_seed1`, 95% CI `[+912B, +2,777B]` Q32 `> 0`, losing to every control including untrained random `ctrl_reseed_static` at `-142` ppm). Per `DECISION.md:342`, `S1` stays frozen at `seed1.d11`, closing the Phase `P5` `S1`-Pick configuration and completing all lab-scoped phases `P0`–`P5` of `written-state-count-model` (`#4063`).
