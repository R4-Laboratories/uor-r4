# REPORT: written-state-count-model, antigravity-10 (Stage B3b `P3` GPU Brier Error-Write & GPU-Voted `S4` Reply Panels)

Lab: antigravity. Item: [`written-state-count-model`](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P3`). Order: [`ORDER.md`](ORDER.md) (`d4dc4fa91146d15ad9564b8aa9ff2816eb03b74c743cf1b4b633b9042cd3130f`). Sheet: [`SHEET.md`](SHEET.md) (`e96c372301378e99d18053b5025b4eaa5b489798f84de5207c012f6fb501e95a`). Non-author FIT: [`FIT.md`](FIT.md) (`48a07d404f874d4bfdcf13b41e678b826c308f8dd288aafa83377ed5ba0e50f6`, `TEST FITNESS: FIT` R6). GO manifest: [`GO.json`](GO.json) (`209ce54be1832894a0f4f9e886d372dd7dde56e402b5ebea8a81737299f0551a`). Evidence: [`evidence/receipts.sha256`](evidence/receipts.sha256).

Wick archive: `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-p3-eval/` (`sha256sum --strict -c receipts.sha256` verified `15/15 OK`, including all four binary artifacts `artifacts/{p3_s6_brier,p3_s6_s3_brier,p3_s6_s2_err_brier}.uors6b01` and `artifacts/s4_train.uors4p01`, before pod deletion).

## What did not work, first

- **All three GPU Brier error-write arms strictly lowered `R` mean Brier across all 3 epochs (`fell = true`), yet NONE cleared the `>= 1%` bit-reduction gate (`gate_brier_1pct_fewer_bits = false`) vs log-likelihood-fitted `P1(A1s)` (`125,750,699,315` Q12 / `1,400,045` micro-BPB all) or `0.95 / 0.90 / 0.80 × R*` (`DEV-tune` `R* = 1,432,318` micro-BPB) on `DEV-tune` (`10,807,417` targets, `21,928,475` B):**
  - **`p3_s6_brier` (15 experts):** `DEV-tune` = `266,763,100,419` Q12 (`2,970,006` micro-BPB all / `3,791,349` novel / `3,212,781` response / `3,793,030` novel_response) — **`212.13%` of `P1(A1s)`**, **`207.35%` of `R*`**, 95% block-bootstrap CI minus `P1`: `[+133,889,856,028, +149,144,211,487]` Q12 all / `[+131,022,574,151, +146,121,388,830]` novel.
  - **`p3_s6_s3_brier` (15 experts + `golden_shell` `i8` error cells; `4.28M / 4.20M / 4.10M` cells updated):** `DEV-tune` = `264,703,382,020` Q12 (`2,947,074` micro-BPB all / `3,760,059` novel / `3,188,401` response / `3,762,382` novel_response) — **`210.49%` of `P1(A1s)`**, **`205.75%` of `R*`** (`-22,932` micro-BPB vs `p3_s6_brier` alone; CI minus `P1`: `[+131,871,526,966, +146,956,930,336]` Q12 all).
  - **`p3_s6_s2_err_brier` (16 experts + `22,281,057` B / `2,480,021`-entry `S2` order-2 `i8` error table; `1.38M` cells updated/epoch):** `DEV-tune` = `253,232,957,863` Q12 (`2,819,368` micro-BPB all / `3,583,466` novel / `3,047,375` response / `3,585,376` novel_response) — **`201.37%` of `P1(A1s)`**, **`196.83%` of `R*`** (`-150,638` micro-BPB vs `p3_s6_brier`; CI minus `P1`: `[+121,209,818,234, +134,777,343,214]` Q12 all).
  - *Mechanism diagnosis (not an instrument bug):* All three Brier arms minimized quadratic $L^2$ Brier objective monotonically on `R` across 3 epochs (`100%` GPU/CPU & shift parity), and adding error-written `S3` (`-22,932` micro-BPB) and `S2` (`-150,638` micro-BPB) materially improved the Brier-fitted family, **yet optimizing Brier score ($L^2$ linear mass residual) rather than log-loss allocates much flatter simplex weights across experts than `P1(A1s)`'s EM log-loss optimum**, doubling `-log2 p` bits on `DEV-tune`.
- **Single-shot GPU-voted `S4` reply gates `R2` (`192/256`, `75.0%` vs `>= 200/256`) and `R4` (`140/240`, `58.3%` vs `>= 200/240`) missed their `>= 200` thresholds (while **`R1` PASSED at `64/128`** and both `m2-fresh256` and `R4` crushed `NoMemory` and `ShuffledRole` paired controls):**
  - On `m2-fresh256` (`256` rows, `128` pairs): `Active` scored **`64/128` pairs (`50.0%`, `gate_r1_ge_52_of_128 = true` PASS)** and **`192/256` rows (`75.0%`, 8 rows below `200`, `gate_r2_ge_200_of_256 = false`)** — diagnostic inspection of `reply_report.json` samples shows paired failures occur on rows where one role wins both questions (`basalt loggia.` emitted for both `-job` and `-home` on `q0`), leaving `64` pairs `2/2` (`128`) and `64` pairs `1/2` (`64`).
  - On `R4` (`240` held-out-phrasing rows, `120` pairs): `Active` scored **`120/120` (`100%`) in-stem rows**, but only **`20/120` (`16.7%`) OOV-synonym rows** -> pooled **`140/240` (`58.3%`, `gate_r4_ge_200_of_240 = false`)**.
  - On `s4_hinge_step`: `8` `M-World` train episodes produced `16/16` `exact_top1` on epoch 1 with `total_hinge_loss_q16 = 0` (`updates_applied = 0` across all 3 epochs), indicating zero training pressure on the unigram/2-gram OOV-synonym gap from the `TRAIN_MWORLD` margin hinge step alone.
- **Pod / pre-run issues resolved cleanly via fresh `run_root` and targeted FIT updates (L-002 preserved):**
  - First KR offer (`vast-62600834`) timed out SSH and was deleted by `uor-pod`; shared `vast-62591945` went away mid-stream; execution completed cleanly on dedicated RTX 4090 pod `vast-62604974` (`compute_cap 8.9`, `$0.736/h`, deleted cleanly via `uor-pod down --confirm-archived`, `0` pods remaining).
  - Pre-execution compiler/audit catches resolved with 3 targeted R4/R5/R6 `FIT.md` updates before `run-v3`: (R4) `clone_htod_or` slice coercion in `src/cuda.rs`, (R5) pure-integer `mul_div_u64` replacing runtime `/` at `p3_brier_kernels.cu:374` (`float_opcode_hits: 6 -> 0` on `sm_89`), (R6) `seal["manifest_sha256"]` lookup in `brier_eval.rs:497` and `reply_eval.rs:854`; failed `run-v1`/`run-v2` directories were never reused.
- **`DEV-test` stayed strictly sealed and untouched** (`dispatch.sh:15-18` exit `99`, `brier_eval.rs:121-144` hard-locked to `dev-tune`).

## What changed

- [`docs/loop/tasks/written-state-count-model/antigravity-10/`](.): Pre-registered CUDA crate [`p3-eval/`](p3-eval/Cargo.toml) (`p3_brier_kernels.cu`, `cuda.rs`, `s3_decode.rs`, `brier_eval.rs`, `reply_eval.rs`, `main.rs`, `build.rs`), [`ORDER.md`](ORDER.md), [`SHEET.md`](SHEET.md), [`FIT.md`](FIT.md) (`R1`–`R6`, final `TEST FITNESS: FIT`), [`dispatch.sh`](dispatch.sh), [`GO.json`](GO.json), compact [`evidence/`](evidence/receipts.sha256) (`receipts.sha256`, `sass-check.txt`, `finite.json`, `brier_report.json`, `reply_report.json`, `*.time`), and this [`REPORT.md`](REPORT.md).
- [`docs/research/notebook/m4-serving.md`](../../../../research/notebook/m4-serving.md) & [`docs/roadmap/items/written-state-count-model.md`](../../../../roadmap/items/written-state-count-model.md): Logged Stage B3b measured outcomes (`R1 PASS`, Brier monotonicity + `S3`/`S2` delta gains + quadratic-vs-log-loss diagnosis, `R2`/`R4` controls vs threshold misses, Wick archive).

## Native-space cycle verification (D85)

1. **Single persistent state schema:** 12-bit `UORT` token IDs (`V = 4096`), `UORKNC03` reversed trie (`full-c.v3`, `75ccb234...`), `UORS3T02` `golden_shell` (`e4c4d1a8...`), `UORS5W02` frozen `A1s` (`e2fbf41d...`), `UORS6B01` Brier state (`16` buckets × `15`/`16` Q32 weights + `i8` error cells), and `UORS4P01` pointer-valued `S4` (`35,994` contexts, `941,100` B) with shared `apply_lawful_stop_and_rep_guard` & PRLL Q12 scoring. Zero floats, zero matmul, zero softmax.
2. **Gate 1 PTX & SASS 0-float audit ([`sass-check.txt`](evidence/sass-check.txt)):** `nvcc -arch=compute_89` (`cubin_sha256 = b0792044b35cd164b5a572b1904b770ef714c521d5b8350e3d17ee4a4c97e54d`), `ptx_float_opcode_hits=0`, `float_opcode_hits=0`, `sm120_const_mov_aliases=0`.
3. **Gate 2 multi-kernel finite bit-exactness ([`finite.json`](evidence/finite.json), `0:00.35`, `113,668 KiB` RSS):** `brier_checked=256 (0 div)`, `s3_checked=5 (0 div)`, `s2_checked=4 (0 div)`, `s4_checked=4 (0 div)`, `rows_checked=269`, **`divergent=0`**.
4. **Gate 3 & Gate 4 runtime GPU vs CPU parity:**
   - `brier` (`0:34.55`, `6,808,216 KiB` RSS): `shifted_batch_parity = true`, `spot_check_probes = 13,153`, `spot_check_divergences = 0`, `s3_spot_check_divergences = 0`, `s2_spot_check_divergences = 0`.
   - `reply` (`0:47.43`, `3,977,276 KiB` RSS): `s4_gpu_vs_cpu_divergences = 0`.
5. **Gate 5 receipt manifest & Wick archive:** `receipts.sha256` verified `15/15 OK` on Wick `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-p3-eval/`.

## Measurements (`DEV-tune`, `m2-fresh256`, `R4`, Vast.ai RTX 4090 `vast-62604974`)

### 1. Brier Error-Write Panel (`R` -> `DEV-tune`, `10,807,417` targets / `21,928,475` B)

Baselines on `DEV-tune` ([`brier_report.json`](evidence/brier_report.json)): `P1(A1s)` = `125,750,699,315` Q12 (`1,400,045` micro-BPB all / `1,684,543` novel / `1,506,724` resp / `1,687,913` nov-resp); `DEV-tune R*` = `1,432,318` micro-BPB.

| Arm | Experts / State | `R` Mean Brier Q32 (`E0 -> E1 -> E2 -> E3`) | `DEV-tune` All (Q12 bits) | All (micro-BPB) | Novel (micro-BPB) | `% P1` (bps) | `% R*` (bps) | `>= 1%` Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **`p3_s6_brier`** | 15 experts (`11,658` bkts/ep) | `2,717,923,805 -> 1,822,914,699 -> 1,813,733,258 -> 1,810,675,254` | `266,763,100,419` | `2,970,006` | `3,791,349` | `21213` | `20735` | **false** |
| **`p3_s6_s3_brier`** | 15 + `golden_shell` `i8` (`4.28M/4.20M/4.10M` cells) | `2,717,923,805 -> 1,822,859,918 -> 1,813,616,868 -> 1,810,519,248` | `264,703,382,020` | `2,947,074` | `3,760,059` | `21049` | `20575` | **false** |
| **`p3_s6_s2_err_brier`** | 16 + `22,281,057` B `S2` `i8` (`1.38M` cells/ep) | `2,691,884,819 -> 1,755,971,764 -> 1,750,376,945 -> 1,750,121,343` | `253,232,957,863` | `2,819,368` | `3,583,466` | `20137` | `19683` | **false** |

### 2. GPU-Voted Pointer-Valued `S4` + Single-Shot Reply Panels ([`reply_report.json`](evidence/reply_report.json))

- **`S4` artifact (`s4_train.uors4p01`, `51503f76...`):** `941,100` B (`35,994` contexts, `8` `M-World` train episodes, `8` extracted spans; `3` hinge epochs `16/16` exact_top1, `0` loss, `0` updates).
- **`m2-fresh256` (`128` pairs, `256` rows):**
  - `Active`: **`64/128` both-complete-distinct pairs (`gate_r1_ge_52_of_128 = true` PASS)**, **`192/256` rows (`75.0%`, `gate_r2_ge_200_of_256 = false`)**.
  - Controls: `NoMemory` `0/128` pairs, `0/256` rows (`paired_vs_no_memory = +192`, `band_ceil = 28`, `beats_control = true`); `ShuffledRole` `0/128` pairs, `44/256` rows (`paired_vs_shuffled_role = +148`, `band_ceil = 32`, `beats_control = true`).
- **`R4` held-out phrasing (`120` pairs, `240` rows):**
  - `Active`: **`140/240` rows (`60/120` pairs; `gate_r4_ge_200_of_240 = false`)** — **`120/120` (`100%`) in-stem rows**, **`20/120` (`16.7%`) OOV-synonym rows**.
  - Controls: `NoMemory` `0/240` rows (`paired_vs_no_memory = +140`, `band_ceil = 24`, `beats_control = true`); `ShuffledRole` `60/240` rows (`paired_vs_shuffled_role = +80`, `band_ceil = 30`, `beats_control = true`).

## Checks run at the head

- `cargo fmt --check`, `cargo test --offline --locked` (`5 passed`), `cargo clippy --offline --locked --no-deps --all-targets -- -D warnings` (`0` warnings/errors), `bash -n dispatch.sh` — **PASS**.
- `GO.json` `13/13` doc/source fields (`12` FIT-audited package files + `FIT.md`) + `GO.json` CLI digest (`$2`) + `2/2` binary digests verified on pod (`run-v3`); Wick archive `sha256sum --strict -c receipts.sha256` `15/15 OK`; `scripts/pod/uor-pod status` `0` running pods (`$0/h`).
- `uor-harness ledger-sync`, `check --staged`, `check --only harness.native_space_contract`, `ratchet d23|d25|d32`, `claim-wording`, `loop report --check`, `loop pr-check --local` — **PASS**.

## Skills used

- `native-space-cycle`

## Still open

- Keep sealed `DEV-test` untouched.
- Open B3b follow-up paths: (1) bridge the `-log2 p` vs $L^2$ Brier objective gap by driving `S3`/`S2`/`S6` updates with integer log-ratio / bit-loss residuals or EM mixture updates without softmax/cross-entropy rather than squared $L^2$ Brier simplex projection; (2) resolve `m2-fresh256` role collision (`64` 1/2-hit pairs on `q0`) and `R4` OOV-synonym phrasing (`20/120`) inside pointer-valued `S4` via stem-invariant role/synonym keying and positive margin hinge margins above `0`.

## Plain summary

On a dedicated RTX 4090 pod (`vast-62604974`, archived to Wick and deleted cleanly), Stage B3b's pure-integer CUDA kernels passed all 5 static, finite (`0/269` divergent), runtime GPU-vs-CPU parity (`0/13,153` Brier probes, `0` S4 diffs), and SASS 0-float audits (`0` float/MMA opcodes on `sm_89`). All three Brier error-write arms monotonically reduced `R` Brier across 3 epochs (`S3` `-22,932` micro-BPB and `S2` `-150,638` micro-BPB over `S6` Brier alone), yet scored `2.819–2.970` BPB on `DEV-tune` vs log-likelihood `P1(A1s)` `1.400` BPB because quadratic Brier optimization under-sharpened expert weights. On single-shot GPU-voted `S4` free generation, **`R1` passed (`64/128` both-complete-distinct pairs, gate `>= 52/128`)** and crushed `NoMemory` (`+192` / `+140`) and `ShuffledRole` (`+148` / `+80`) controls on both `m2-fresh256` and `R4`, while **`R2` (`192/256` rows vs `>= 200/256`)** and **`R4` (`140/240`: `120/120` in-stem, `20/120` OOV-synonym)** missed their `>= 200` row thresholds.
