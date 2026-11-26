# REPORT: written-state-count-model, antigravity-11

Lab: `antigravity`. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P3`). Order: [ORDER.md](ORDER.md). Base: `b2c458e7b`.

**Outcome (`Stage B3c`, `DEV-tune`, `R`, `m2-fresh256`, `R4`): mixed — `R1` (`104/128` pairs) and `R2` (`232/256` rows) PASS both thresholds and paired controls, and `A1s`-warm-start integer EM refit recovers `2,947,074 -> 1,400,063` micro-BPB (`[−15.20M, −8.40M]` Q12 novel CI gain over `P1(A1s)`), while overall `DEV-tune` `>= 1%` bit reduction and `R4 >= 200/240` (`140/240`) remain open (`DEV-test` strictly untouched).**

## What did not work, first

1. **`DEV-tune` `>= 1%` bit reduction and `0.95/0.90/0.80 × R*` missed overall:** None of the 4 trained `P3` arms cleared `<= 99%` of `P1(A1s)` bits on `DEV-tune` (`125,750,699,315` Q12 / `1,400,045` micro-BPB across `10,807,417` targets; `R* DEV-tune = 1,432,318`):
   - `p3c_a1s_em_s3_brier`: `125,752,314,528` Q12 (`1,400,063` micro-BPB, `+18` micro-BPB / `100.00%` (`10000` bps) of `P1(A1s)`, 95% all-target CI `[−2,386,421, +5,543,097]` Q12) — down `-1,547,011` micro-BPB from Stage B3b `p3_s6_s3_brier` (`2,947,074`), monotonically lowering both `R` train mean Brier (`1,941,322,308 -> 1,939,299,812`) and `R` train bits (`185,024,291,823 -> 184,902,336,390` Q12) every epoch and beating `P1(A1s)` on held novel targets (`1,684,365` vs `1,684,543` micro-BPB, 95% CI `[−15,198,682, −8,398,877]` Q12).
   - `p3c_a1s_brier_s6`: `126,133,212,698` Q12 (`1,404,303` micro-BPB, `10030` bps, 95% CI `[+346,609,719, +422,277,033]` Q12). Lowered `R` mean Brier across all 3 epochs (`1,941,322,308 -> 1,923,589,614`), while `R` train bits rose slightly, isolating why the $L^2$ Brier objective diverges slightly from bits even under warm start.
   - `p3c_a1s_brier_s2_multi`: `126,716,085,355` Q12 (`1,410,793` micro-BPB, `10076` bps, CI `[+900,065,146, +1,032,939,370]`). Lowered both `R` mean Brier (`1,946,586,037 -> 1,899,056,379`) and `R` train bits (`173,197,170,009 -> 165,492,012,481`) every epoch.
   - `p3c_a1s_em_s2_multi_brier`: `138,642,772,338` Q12 (`1,543,578` micro-BPB, `11025` bps, CI `[+12,233,274,545, +13,582,317,977]`). Lowered `R` train bits steeply (`173,197,170,009 -> 141,245,652,973` Q12), exposing in-sample `R` self-memorization when `S2(1..=4)` lookup tables are built on `R` and fitted via EM on the same `R` stream rather than built on disjoint `C` or cross-fitted folds.
2. **`R4` held-out phrasing threshold (`>= 200/240`) missed:** Pooled active accuracy reached `140/240` (`40/120` both-role pairs; beating `NoMemory` `0/240` by `+140` [ceil `24`] and `ShuffledRole` `80/240` by `+60` [ceil `30`]). In-stem paraphrasing reached `100/120` and OOV-synonym rose from `20/120` (Stage B3b) to `40/120`, still short of `200/240` under single-hop `D87 R` `±8` co-occurrence propagation.
3. **Initial pre-run FIT iterations (`R1`–`R3`) and PR `#4287` review check:** Pre-run FIT iterations caught host/pod manifest alignment issues before GPU execution (`R4` issued `TEST FITNESS: FIT`). At commit `f796ae906`, exact-head review flagged that an older `uor-harness` on `PATH` had missed the 7 required `## ` sections in `REPORT.md`; this document was reformatted to satisfy `uor-harness loop report --check` and `loop pr-check`.
4. **`DEV-test` kept untouched:** All 4 sealed `DEV-test` checks (`dev-test`, `rstar-devtest`, `023784b7`, `ec048217`) were hard-locked via `dispatch.sh` exit `99` and `DevTuneData::load_checked`.

## What changed

1. **Task package & standalone GPU runner (`docs/loop/tasks/written-state-count-model/antigravity-11/`):**
   - Added [`ORDER.md`](ORDER.md), [`SHEET.md`](SHEET.md), [`FIT.md`](FIT.md) (non-author `pro` `TEST FITNESS: FIT`), [`dispatch.sh`](dispatch.sh), and `evidence/` (`GO.json`, `sass-check.txt`, `finite.json`, `brier_report.json`, `reply_report.json`, `finite.time`, `brier.time`, `reply.time`, `receipts.sha256`).
   - Added standalone `p3c-eval/` crate (`Cargo.toml`, `Cargo.lock`, `build.rs`, `src/{main,cuda,s3_decode,brier_eval,reply_eval}.rs`, `src/p3_brier_kernels.cu`) reusing `uor_r4_count_runtime::p3_state` with mirrored pure-integer CUDA kernels (`compute_89`, zero floats/matmul/softmax): `A1s`-warm-started `S6`, bounded `S2(1..=4)` injective 64-bit `i8` error-written tables (`77,750,708` B `<= 100,000,000` B budget), integer EM (`p3_em_step_batch`), balanced `TRAIN_MWORLD` (`24` episodes, `0` pre-stuffing), full `15,951,644`-token `D87 R` `±8` co-occurrence propagation, and active `s4_hinge_step` (`2,813` updates).
2. **Notebook & roadmap updates:**
   - Recorded Stage B3c row in [`m4-serving.md`](../../../../research/notebook/m4-serving.md) (`negative` verdict prefix) and Stage B3c summary line in [`written-state-count-model.md`](../../../../roadmap/items/written-state-count-model.md).
3. **Diff size breakdown (`uor-harness loop pr-check` & `git diff --numstat b2c458e7b..HEAD`, `25` files, `+7,382/-0`, item has no size estimate):**
   - `uor-harness loop pr-check` classes: `tests 0 file(s) +0/-0, code 8 file(s) +5631/-0, docs 6 file(s) +238/-0, generated 0 file(s) +0/-0, other 11 file(s) +1513/-0` (where `code` counts `.rs`/`.toml`/`.sh`, `docs` counts `.md`, and `other` counts `.cu`, `Cargo.lock`, and `evidence/*`).
   - File-group `git diff --numstat` breakdown: `8` implementation/script files (`p3c-eval/src/*.rs`, `p3c-eval/src/p3_brier_kernels.cu`, `p3c-eval/build.rs`, `dispatch.sh`) = `+6,042/-0`; `6` `.md` docs (`ORDER.md`, `SHEET.md`, `FIT.md`, `REPORT.md`, `m4-serving.md`, `written-state-count-model.md`) = `+238/-0`; `11` manifest/evidence files (`p3c-eval/Cargo.toml`, `p3c-eval/Cargo.lock`, `9` `evidence/*` files) = `+1,102/-0`.
   - Diff stays strictly within the `3` paths scoped in [`ORDER.md`](ORDER.md).

## Measurements

All GPU runs executed on `vast-62629180` (RTX 4090 `compute_89`, `$0.736/h`, torn down after Wick verification) under [`dispatch.sh`](dispatch.sh) Gates 1–5 and archived to `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p3c-eval/` (`sha256sum --strict -c receipts.sha256` -> `16/16 OK`).

- **Gate 1 (PTX/SASS 0-float audit):** `cubin_sha256 = d34d2f5d04e8e15754301235ab494330fe047f6d5a933fbf44b512b1afaad307`, `nvcc_arch = compute_89`, `compute_cap = 8.9`, `ptx_float_opcode_hits = 0`, `float_opcode_hits = 0`, `sm120_const_mov_aliases = 0` ([`sass-check.txt`](evidence/sass-check.txt)).
- **Gate 2 (`p3c-eval finite`, `0:00.32`, `113,328` KiB RSS):** `rows_checked = 525` (`brier = 256`, `em = 256`, `s3 = 5`, `s2 = 4`, `s4 = 4`), `divergent = 0` ([`finite.json`](evidence/finite.json)).
- **Gate 3 (`p3c-eval brier`, `1:08.26`, `10,843,572` KiB RSS):** `dev_tune_targets = 10,807,417`, `s2_multi_table` `77,750,708` B (`[3,501,302, 24,249,421, 24,999,989, 24,999,996]`), `shifted_batch_parity = true` (`65,536 == 65,521`), `spot_check_probes = 13,153`, `0` divergences across S6/EM/S3/S2 ([`brier_report.json`](evidence/brier_report.json)):

| Arm | Experts | Optimizer / Tables | All `bits_q12` | All micro-BPB | Novel micro-BPB | Response micro-BPB | 95% CI vs `P1(A1s)` All (`Q12`) | `<= 99% P1` |
|---|---|---|---|---|---|---|---|---|
| `P1(A1s)` baseline | 14 | Frozen Stage B1a | `125,750,699,315` | `1,400,045` | `1,684,543` | `1,506,724` | — | baseline |
| `p3c_a1s_brier_s6` | 15 | Tangent Brier (`lr_shift=7`) | `126,133,212,698` | `1,404,303` | `1,690,470` | `1,511,545` | `[+346,609,719, +422,277,033]` | `false` |
| `p3c_a1s_em_s3_brier` | 15 | Integer EM (`prior=64`) + `S3` (`lr_shift=9`) | `125,752,314,528` | `1,400,063` | `1,684,365` | `1,506,692` | `[−2,386,421, +5,543,097]` (Novel `[−15.20M, −8.40M]`) | `false` |
| `p3c_a1s_em_s2_multi_brier` | 19 | Integer EM (`prior=64`) + `S3` + `S2(1..=4)` on `R` | `138,642,772,338` | `1,543,578` | `1,868,824` | `1,659,161` | `[+12,233,274,545, +13,582,317,977]` | `false` |
| `p3c_a1s_brier_s2_multi` | 19 | Tangent Brier (`lr_shift=7`) + `S3` + `S2(1..=4)` on `R` | `126,716,085,355` | `1,410,793` | `1,698,653` | `1,518,266` | `[+900,065,146, +1,032,939,370]` | `false` |

- **Gate 4 (`p3c-eval reply`, `0:46.68`, `3,983,380` KiB RSS):** `s4_artifact` `1,318,602` B (`50,289` contexts), `d87_r_tokens_scanned = 15,951,644` (`15,947,584` non-BOS; `256` job / `256` home tokens), `total_hinge_updates = 2,813` (`2,237 + 288 + 288`), `s4_gpu_vs_cpu_divergences = 0` ([`reply_report.json`](evidence/reply_report.json)):

| Panel / Mode | Rows (`OK`) | Distinct Pairs | Net vs `NoMemory` (Ceil) | Net vs `ShuffledRole` (Ceil) | Gate |
|---|---|---|---|---|---|
| `m2-fresh256` Active | `232 / 256` (`90.6%`) | `104 / 128` (`81.3%`) | `+232` (`32`) PASS | `+208` (`32`) PASS | `R1` PASS, `R2` PASS |
| `m2-fresh256` Controls (`NoMemory` / `ShuffledRole`) | `0 / 256` / `24 / 256` | `0 / 128` / `0 / 128` | — | — | controls separated |
| `R4` Held-Out Phrasing Active | `140 / 240` (`100/120` stem, `40/120` OOV-synonym) | `40 / 120` | `+140` (`24`) PASS | `+60` (`30`) PASS | `R4 >= 200` `false` |
| `R4` Controls (`NoMemory` / `ShuffledRole`) | `0 / 240` / `80 / 240` | — | — | — | controls separated |

## Checks run at the head

Run at the edited head in `/Users/casey.allard/uor-r4-b3c-p3-repair`:
- `uor-harness loop report --check` (using `/Users/casey.allard/uor-r4/local/harness-target/debug/uor-harness`): `PASS`
- `uor-harness loop pr-check --local --base b2c458e7b --head HEAD`: `verdict: PASS` (`order`, `scope`, `report`, `banned-presence`, `commit-order`, `owner-paths`, `rule-sources`, `rendered-lock`)
- `uor-harness check --only harness.native_space_contract`: `1 rules, 0 failed` (`0` errors, `0` warnings)
- `uor-harness ratchet d23 --quiet && d25 --quiet && d32 --quiet && claim-wording && shard-check && check-formats`: `PASS`
- `cargo fmt --check`, `cargo test --offline --locked` (`6 passed; 0 failed`), `cargo clippy --offline --locked --no-deps --all-targets -- -D warnings`, `bash -n dispatch.sh`: `PASS`
- `GO.json` (`38/38 OK`), Wick archive `sha256sum --strict -c receipts.sha256` (`16/16 OK`), and `(cd /Users/casey.allard/uor-r4 && scripts/pod/uor-pod status)` (`0` pods owned by `antigravity`).

## Skills used

- `native-space-cycle` (`.agents/skills/native-space-cycle/SKILL.md`): verified single native integer state, shared `p3_state` operator path, bit-identical replay oracles, and lossless payload hashes before GPU execution.
- `uor-research-architect` (`.agents/skills/uor-research-architect/SKILL.md`): enforced pre-run non-author FIT review, `DEV-test` isolation, paired memory controls, Wick archival, and immediate pod teardown.
- `verification-before-completion` (`verification-before-completion/SKILL.md`): enforced fresh post-edit verification of all `uor-harness`, Rust, and PR checks prior to reporting completion.

## Still open

1. **Overall `DEV-tune` `>= 1%` bit reduction over `P1(A1s)` (`#4063`, Stage B4):** Build bounded `S2(1..=4)` lookup tables from disjoint `C` (or cross-fitted `R` folds) rather than the same `R` stream fitted by `S6` EM, while retaining `S3` + pure-integer EM (`1,684,365` novel micro-BPB).
2. **`R4` held-out OOV-synonym phrasing (`140/240 -> >= 200/240`):** Expand pure-integer `D87 R` multi-hop/second-order co-occurrence propagation (bridge high-PMI intermediate tokens in addition to direct `±8` anchor neighbors) without touching eval-panel strings, preserving `R1` (`104/128`) and `R2` (`232/256`).
3. **`DEV-test` sealed evaluation:** Strictly deferred until a candidate first clears `DEV-tune` `>= 1%` bit reduction and `R4 >= 200/240`.

## Plain summary

This task tested targeted integer-only repairs to the UOR-R4 predictive count model (`P3`, which estimates next-token probabilities as exact integers without floating point, matrix multiplications, or softmax) on an RTX 4090 GPU before tearing the machine down cleanly. On the conversational fact-memory tests (`m2-fresh256`), balancing the training episodes and training a margin-based hinge rule over `15.9` million text tokens made exact role-bound recall pass both project bars (`104/128` distinct pairs for `R1`, `232/256` rows for `R2`) while scoring `0/256` without memory and `24/256` when roles are shuffled. On text prediction accuracy (`DEV-tune`, measured in bits per byte), initializing the mixture combiner from a strong pre-trained baseline (`A1s`) and using integer expectation-maximization (`EM`) cut the prior stage's error roughly in half (`2,947,074` down to `1,400,063` millionths of a bit per byte) and beat the baseline on novel words (`1,684,365` vs `1,684,543`), though adding lookup tables built from the same training stream caused over-memorization on seen text and prevented clearing the overall `1%` bit-reduction goal or the unseen-synonym threshold (`140/240` vs `200/240`), leaving those two items for the next stage.
