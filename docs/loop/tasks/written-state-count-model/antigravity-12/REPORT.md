# REPORT: written-state-count-model, antigravity-12

Lab: `antigravity`. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P3`). Order: [ORDER.md](ORDER.md). Base: `aecd3675a`.

**Outcome (`Stage B4`, `DEV-tune`, `R`, `m2-fresh256`, `R4`): mixed (combined `SHEET.md` §4 gate remains `negative`) — Part A PASSES `DEV-tune` `<= 99% P1(A1s)` overwhelmingly (`p4_full_errwrite` `1,209,985` micro-BPB = `86.42%` of `P1(A1s)` [`−13.58%` fewer bits] and `0.8447 × R*`, CI `[−18.18B, −16.00B]` Q12, error-write strictly beats static `[−622.95M, −530.64M]` Q12), and `m2-fresh256` reaches `100%` (`R1 = 128/128` pairs PASS, `R2 = 256/256` rows PASS vs `0/256` `NoMemory` and `0/256` `ShuffledRole`), while `R4` held-out phrasing improves to `160/240` (`120/120` in-stem, `40/120` OOV-synonym, missing `>= 200/240`), leaving sealed `DEV-test` strictly untouched.**

## What did not work, first

1. **`R4` held-out phrasing threshold (`>= 200/240`) missed:** Active `S4` reached `160/240` (`60/120` both-role pairs; `+160` over `NoMemory` `0/240` [ceil `26`] and `+100` over `ShuffledRole` `60/240` [ceil `30`]), with `120/120` (`100%`) in-stem paraphrasing and `40/120` OOV-synonym accuracy (`s4_gpu_vs_cpu_divergences = 0`, `2,168` active hinge updates). Mechanism/corpus classification: `D87 R` (`15.95M` tokens of TinyStories children's prose) 2-hop 512-bit PPMI propagation cleanly extracts narrative associates (`" walking"`, `" rain"`, `" tea"`, `" city"`, `" happened"`, `" know"`) rather than formal adult synonyms (`"occupation"`, `"employment"`, `"residence"`, `"domicile"`) that are rare or absent in `D87 R`; broadening non-panel synonym coverage requires integer multi-hop co-occurrence over general-domain `C` rather than `D87 R` alone.
2. **Disjoint-`C` `S2-skip` + `S3` without causal session cache missed `<= 99% P1(A1s)`:** Building non-contiguous skip-gram tables on `4.15 GB` disjoint `C` (`79,999,990` B payload) fixed Stage B3c's in-sample `R` overfit (`1,543,578` -> `1,400,664` micro-BPB `p4_c_skip_static` / `1,400,486` `p4_c_skip_errwrite`), and `R` Brier error-write lowered held bits strictly vs static (`controlled p4_c_skip_errwrite - p4_c_skip_static` = `[−18,175,875, −14,065,274]` Q12), yet `S2-skip` + `S3` alone without within-record session caching still trailed `P1(A1s)` slightly (`100.03%`, CI `[+32,075,955, +47,367,718]` Q12).
3. **Operational retries & Wick-to-Vast fallback (`L-002` honored):**
   - Pre-run FIT R1 caught missing `#[cfg(feature = "cuda")]` `p_errs`/`b_errs`/`upload_gpu` accessors on `GoldenShellStateTable` in `s3_decode.rs` (restored and verified cleanly on Wick `--features cuda` clippy `-D warnings` & tests `4/4` -> FIT R2).
   - Wick Run `r1` (`20261126-antigravity-p4-eval`) passed Gates 1–2 (`0` SASS floats, `0/525` finite diffs) then exited `124` at Gate 3 (`timeout 300` expired while single-threaded `build-skip` streamed `2.075B` tokens); updated `dispatch.sh:162` to `timeout 1800` + FIT R3 and wrote into fresh `run_root` `r2`.
   - Wick Run `r2` (`20261126-antigravity-p4-eval-r2`) passed Gates 1–4 (`build-skip` `9:07.21`; `brier_report.json` bit-identical to final Run `r3`: `p4_full_errwrite` `108,679,751,739` Q12) before another session's GPU process (`PID 1787751`, `conjunction-probe-v105`) started on Wick and blocked Gate 5 `admit_gpu`. Per `ORDER.md`/`SHEET.md` fallback policy, provisioned single-GPU Vast.ai RTX 4090 pod `vast-62650633` (`$0.869/h`), ran complete end-to-end Gates 1–5 cleanly into `/workspace/p4-run`, archived full evidence + binary `.uors*` artifacts to Wick `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p4-eval-r3/` (`19/19 OK`), and deleted `vast-62650633` (`0` pods owned by `antigravity`).
4. **`DEV-test` kept strictly untouched:** Hard-locked via `dispatch.sh` exit `99` and `DevTuneData::load_checked`; Stage B4 closes negative on the combined §4 gate solely due to `R4` (`160/240 < 200/240`), deferring `DEV-test` to a future pre-registered run.

## What changed

1. **Task package & standalone GPU runner (`docs/loop/tasks/written-state-count-model/antigravity-12/`):**
   - Added [`ORDER.md`](ORDER.md), [`SHEET.md`](SHEET.md), [`FIT.md`](FIT.md) (non-author `pro` R3 `TEST FITNESS: FIT`, preserved byte-identical to sealed `GO.json`/`receipts.sha256` `cac079d8...`; `FIT.md:65` drops trailing `2` on `p4-eval/src/main.rs` vs `e2f0b7b3...eb2` in `GO.json`), [`dispatch.sh`](dispatch.sh), and `evidence/` (`GO.json`, `sass-check.txt`, `finite.json`, `skip_build.json`, `brier_report.json`, `reply_report.json`, `finite.time`, `skip_build.time`, `brier.time`, `reply.time`, `receipts.sha256`).
   - Added standalone `p4-eval/` crate (`Cargo.toml`, `Cargo.lock`, `build.rs`, `src/{main,cuda,cache_eval,s3_decode,brier_eval,reply_eval}.rs`, `src/p4_brier_kernels.cu`) reusing `uor_r4_count_runtime::p3_state` with `MAX_EXPERTS = 24` pure-integer CUDA kernels (`compute_89`, zero floats/matmul/softmax/runtime CUDA divides): `<= 80 MB` disjoint-`C` `S2-skip` streaming builder (`skip_lag2`, `skip_3_1`, `skip_4_1`), causal within-record written-state session cache (`cache_uni_recency`, `cache_bi_local`, `cache_tri_local`, `cache_lcp_local`, `0` B on disk, `BOS = 0` reset, `2,640` cache-hit-bucketed `S6`), word-initial non-top-64 clean `TRAIN_MWORLD` anchors + `4096 × 512` 2-hop 512-bit PPMI popcount `D87 R` propagation (`15.95M` tokens), top-`192` scaffold pairing, active `s4_hinge_step`, and GPU-voted `m2-fresh256` & `R4` panels.
2. **Notebook & roadmap updates:**
   - Recorded Stage B4 row in [`m4-serving.md`](../../../../research/notebook/m4-serving.md) (`negative` verdict prefix) and Stage B4 summary paragraph in [`written-state-count-model.md`](../../../../roadmap/items/written-state-count-model.md).
3. **Diff size breakdown (`uor-harness loop pr-check` & `git diff --numstat aecd3675a..HEAD`, `28` files, `+8,320/-0`, item has no size estimate):**
   - `uor-harness loop pr-check` classes: `tests 0 file(s) +0/-0, code 9 file(s) +6452/-0, docs 6 file(s) +241/-0, generated 0 file(s) +0/-0, other 13 file(s) +1627/-0` (where `code` counts `.rs`/`.toml`/`.sh`, `docs` counts `.md`, and `other` counts `.cu`, `Cargo.lock`, and `evidence/*`).
   - File-group `git diff --numstat` breakdown: `9` implementation/script files (`p4-eval/src/*.rs`, `p4-eval/src/p4_brier_kernels.cu`, `p4-eval/build.rs`, `dispatch.sh`) = `+6,863/-0`; `6` `.md` docs (`ORDER.md`, `SHEET.md`, `FIT.md`, `REPORT.md`, `m4-serving.md`, `written-state-count-model.md`) = `+241/-0`; `13` manifest/evidence files (`p4-eval/Cargo.toml`, `p4-eval/Cargo.lock`, `11` `evidence/*` files) = `+1,216/-0`.
   - Diff stays strictly within the `3` paths scoped in [`ORDER.md`](ORDER.md) (`0` edits to `crates/**`).

## Measurements

All final end-to-end GPU measurements ran on `vast-62650633` (RTX 4090 `compute_89`, `$0.869/h`, released & deleted after verified Wick archival at `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p4-eval-r3/`, `sha256sum --strict -c receipts.sha256` -> `19/19 OK`; Gate 4 `brier_report.json` also replicated bit-identically on Wick RTX 4090 in Run `r2`).

- **Gate 1 (PTX/SASS 0-float audit):** `cubin_sha256 = 0e661e6c8ce95fb674439e2e954cff207cbc68584352b9cbca5bdaef9a855360`, `nvcc_arch = compute_89`, `compute_cap = 8.9`, `ptx_float_opcode_hits = 0`, `float_opcode_hits = 0`, `sm120_const_mov_aliases = 0` ([`sass-check.txt`](evidence/sass-check.txt)).
- **Gate 2 (`p4-eval finite`, `0:00.31`, `111,656` KiB RSS):** `rows_checked = 525` (`brier = 256`, `em = 256`, `s3 = 5`, `s2 = 4`, `s4 = 4`), `divergent = 0` ([`finite.json`](evidence/finite.json)).
- **Gate 3 (`p4-eval build-skip`, `19:00.04`, `3,564,524` KiB RSS):** Disjoint-`C` (`cd3adfe9...`, `4.15 GB`) streaming `s2_skip.uors2s01`: `79,999,990` B payload (`80,106,901` B file `<= 80,000,000` B payload cap), `[19,999,994, 29,999,998, 29,999,998]` B, `[1,113, 6,411, 5,827]` ctxs, `[2,854,916, 4,272,892, 4,274,060]` entries ([`skip_build.json`](evidence/skip_build.json)).
- **Gate 4 (`p4-eval brier`, `1:01.28`, `11,982,364` KiB RSS):** `dev_tune_targets = 10,807,417`, `shifted_batch_parity = true` (`65,536 == 65,521` on both `880` and `2,640` bucket layouts), `spot_check_probes = 13,153`, `0` divergences across S6/EM/S3/S2 ([`brier_report.json`](evidence/brier_report.json)):

| Arm | Experts / Buckets | Tables / Optimizer | All `bits_q12` | All micro-BPB | Novel micro-BPB | Response micro-BPB | 95% CI vs `P1(A1s)` All (`Q12`) | `% P1(A1s)` / `vs R*` | `<= 99% P1` |
|---|---|---|---|---|---|---|---|---|---|
| `P1(A1s)` baseline | 14 / 880 | Frozen Stage B1a (`R* = 1,432,318`) | `125,750,699,315` | `1,400,045` | `1,684,543` | `1,506,724` | — | `100.00%` / `97.75%` | baseline |
| `p4_c_skip_static` | 18 / 880 | Disjoint-`C` `S2-skip` + `S3` + EM (`prior=64`) | `125,806,352,124` | `1,400,664` | `1,684,806` | `1,507,251` | `[+47,590,050, +63,406,799]` | `100.04%` / `97.79%` | `false` |
| `p4_c_skip_errwrite` | 18 / 880 | Disjoint-`C` `S2-skip` + `S3` + `errwrite` (`lr=7`) | `125,790,323,877` | `1,400,486` | `1,684,687` | `1,507,083` | `[+32,075,955, +47,367,718]` | `100.03%` / `97.77%` | `false` |
| `p4_cache_static` | 18 / 2,640 | Causal Session Cache (`0` B disk) + EM (`prior=64`) | `109,253,948,368` | `1,216,378` | `1,445,334` | `1,300,915` | `[−17,576,054,075, −15,463,612,312]` | **`86.88%` / `84.92%`** | **`true`** |
| `p4_full_errwrite` | 22 / 2,640 | Full (`S2-skip` + `S3` + Cache + `errwrite`, `lr=7`) | **`108,679,751,739`** | **`1,209,985`** | **`1,436,954`** | **`1,293,525`** | **`[−18,183,857,041, −16,002,208,832]`** (Novel `[−17.29B, −15.19B]`) | **`86.42%` (`-13.58%`) / `84.47%`** | **`true`** |

  - Controlled error-write deltas (95% bootstrap CI): `p4_c_skip_errwrite - p4_c_skip_static` = `[−18,175,875, −14,065,274]` Q12 (`R` mean Brier `1,956,742,338 -> 1,940,830,640`); `p4_full_errwrite - p4_cache_static` = `[−622,950,243, −530,636,089]` Q12 (`R` mean Brier `1,959,327,005 -> 1,781,316,530`). Both `p4_cache_static` and `p4_full_errwrite` clear `0.95 × R*` and `0.90 × R*` on `DEV-tune`.
- **Gate 5 (`p4-eval reply`, `0:40.08`, `4,022,768` KiB RSS):** `s4_train.uors4p01` (`4,402,792` B, `168,948` contexts, `371` job / `384` home `D87 R` 2-hop propagated tokens, `2,168` active hinge updates across `3` epochs, `48/48` train exact top-1, hinge loss `9,304,702 -> 4,249,432` Q16, `s4_gpu_vs_cpu_divergences = 0`) ([`reply_report.json`](evidence/reply_report.json)):

| Panel / Mode | Rows (`OK`) | Distinct Pairs | Net vs `NoMemory` (Ceil) | Net vs `ShuffledRole` (Ceil) | Gate |
|---|---|---|---|---|---|
| `m2-fresh256` Active | **`256 / 256` (`100%`)** | **`128 / 128` (`100%`)** | `+256` (`32`) PASS | `+256` (`32`) PASS | `R1` PASS, `R2` PASS |
| `m2-fresh256` Controls (`NoMemory` / `ShuffledRole`) | `0 / 256` / `0 / 256` | `0 / 128` / `0 / 128` | — | — | controls separated |
| `R4` Held-Out Phrasing Active | `160 / 240` (`120/120` stem, `40/120` OOV-synonym) | `60 / 120` | `+160` (`26`) PASS | `+100` (`30`) PASS | `R4 >= 200` `false` |
| `R4` Controls (`NoMemory` / `ShuffledRole`) | `0 / 240` / `60 / 240` | — | — | — | controls separated |

## Checks run at the head

Run at the edited head in `/Users/casey.allard/uor-r4/.worktrees/antigravity-b4-p3`:
- `uor-harness loop report --check` (`/Users/casey.allard/uor-r4/local/harness-target/debug/uor-harness`): `PASS`
- `uor-harness loop pr-check --local --base aecd3675a --head HEAD`: `verdict: PASS`
- `uor-harness check --only harness.native_space_contract`: `PASS` (`0` errors, `0` warnings)
- `uor-harness ratchet d23 --quiet && d25 --quiet && d32 --quiet && claim-wording && shard-check && check-formats`: `PASS`
- `cargo fmt --check`, `cargo test --offline --locked` (`4 passed; 0 failed`), `cargo clippy --offline --locked --no-deps --all-targets -- -D warnings`, `bash -n dispatch.sh`: `PASS`
- Wick `--features cuda` `cargo clippy -- -D warnings && cargo test --offline --locked --features cuda` (`4 passed; 0 failed`), `vast-62650633` `GO.json` (`40/40 OK`), Wick archive `sha256sum --strict -c receipts.sha256` (`19/19 OK`), and `(cd /Users/casey.allard/uor-r4 && scripts/pod/uor-pod status)` (`0` pods owned by `antigravity`).

## Skills used

- `native-space-cycle` (`.agents/skills/native-space-cycle/SKILL.md`): verified single native integer state, shared `p3_state` operator path, bit-identical GPU/CPU replay oracles, and lossless payload hashes before GPU execution.
- `uor-research-architect` (`.agents/skills/uor-research-architect/SKILL.md`): enforced pre-run non-author FIT reviews, `DEV-test` isolation, paired memory controls, Wick archival, and immediate Vast pod teardown.
- `verification-before-completion` (`verification-before-completion/SKILL.md`): enforced fresh post-edit verification of all `uor-harness`, Rust, and PR checks prior to reporting completion.

## Still open

1. **`R4` OOV-synonym phrasing (`160/240 -> >= 200/240`):** Extract 2-hop 512-bit PPMI context signatures over broad-domain `C` (plus `D87 R`) rather than TinyStories `D87 R` alone, so adult occupational/residential synonyms link to role anchors while preserving `R1` (`128/128`) and `R2` (`256/256`).
2. **Pre-registered single-shot `DEV-test` & client integration for `p4_full_errwrite`:** Once `R4 >= 200/240` clears, run the separate pre-registered `DEV-test` evaluation for causal session cache + `S2-skip` + `S3` (`1,209,985` micro-BPB / `0.8447 × R*` on `DEV-tune`) and wire the integer causal session cache into the serving client.

## Plain summary

This stage tested two integer-only improvements to the UOR-R4 language predictor (`P3`) and reply memory (`S4`) on an RTX 4090 GPU without floating point, matrix multiplications, or softmax: (1) adding a zero-disk causal within-document session cache and disjoint-corpus skip-gram error tables lowered out-of-sample prediction cost on `DEV-tune` by **13.58%** below the frozen `P1(A1s)` baseline (`1,209,985` vs `1,400,045` millionths of a bit per byte, and `0.8447×` the unpruned reference `R*`, clearing `0.90 × R*` with tight bootstrap confidence bounds), and (2) filtering query anchors and propagating 2-hop context signatures over `15.95` million tokens pushed fresh conversational fact-recall (`m2-fresh256`) to **100%** (`128/128` distinct pairs for `R1`, `256/256` rows for `R2`, vs `0/256` without memory or with shuffled roles). However, unseen adult-synonym phrasing (`R4`) reached `160/240` instead of the `200/240` target because the TinyStories `D87 R` corpus lacks formal adult synonyms, so the combined Stage B4 bar stays negative and sealed `DEV-test` remains untouched.
