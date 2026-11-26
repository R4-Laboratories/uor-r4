# Pre-Registration Sheet: `written-state-count-model`, `antigravity-10` (Stage B3b `P3` GPU Brier Error-Write + `S4` Reply Panels)

- **Lab:** `antigravity`. **Milestone / Item:** `M2+` / [`written-state-count-model`](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `P3`, Base: `8a8b0cf3b`).
- **Compute Host:** Dedicated single-GPU RTX 5090 (`sm_120`) Vast.ai pod (`uor-pod up --lab antigravity --count 1`, `<= $1/h` inside the `4 pods / $8.00/h` cap, torn down immediately after archiving `evidence/` + `receipts.sha256` to `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-p3-eval/`). `GO.json` pins `expected_host`, `expected_compute_cap`, `device_uuid`, `gpu_lock`, `UOR_NVCC_ARCH`, and compiled `p3-eval` + `p3_brier_kernels.cubin` + all source/input SHA-256s; `admit_gpu` (`nvidia-smi` compute apps empty, `>= 8,192 MiB` free VRAM, `>= 10 GiB` free host RAM) runs before every live GPU step.
- **HARD LOCK ON `DEV-test`:** `dev-test.tokens.u16` (`023784b74c98a796401a064d50bf2f83f2f8eb95ab8bbead1cbe2aec6e7a176b`), `dev-test.json` (`ec048217ac64f05a6519d7e38dd2fabf6e244eca8009b24476b3870cbc0f73f2`), and `rstar-devtest` paths MUST NOT be opened, read, referenced in `GO.json`, or scored in `antigravity-10` (enforced by both `dispatch.sh` exit `99` and `DevTuneData::load_checked`).

## 1. Frozen Inputs (verified by SHA-256 before execution)

| Artifact | Origin / Role | Pinned SHA-256 / Size |
|---|---|---|
| Compact count/Pick table (`full-c.v3`) | `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/gate3-fullc/full-c.v3` | `75ccb234e669cb6b707ffe4091822a91a225e0875a0e99edf9f21f8f2ec3f233` |
| Reserved `R` stream (`r.tokens.u16`) | `/home/callard/uor-codex/20261119-rival-admission-29/count/cr-v1/r.tokens.u16` (`4,060` records, `15,947,584` targets) | `5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a` (`31,903,352` B) |
| Stage B1a `R` & `DEV-tune` `UORCBF01` | `r-65536.tape` / `tune-65536.tape` | `8b6b39d25ace5d1dadfe50fd05d8638b6b5b3765784f20e8714f3832935d8c0a` / `dc0785bf0002a17f9e416235e7b4aeb1308b218574ce6b5145b808244fc4f24d` |
| Stage B1a `A1s` weights, manifest & seal | `a1s.weights` / `a1s.manifest.json` / `a1s.seal.json` | `e2fbf41d38d9dbe3449cdbb85c8bd335c5ee1f803e35bc7993070d65c0d33293` / `26f8f2225a8a0735400a647e37f2f97d86cde9f2de8d847ef791469a6d6b02ec` / `b6e9ac04006100922873d86562d6f525a9e5b73c713ca15d2dd329294d798071` |
| Stage B1a `R` Pick & Stage B2 `DEV-tune` Pick | `pick.rows` (`R`) / `devtune-pick.rows` (`DEV-tune`) | `ba9e646a99ed1507235e95715c6e959469cd830085b6b492b01b2a083769bba9` / `e2e3cf84db615517c84724700396fa6da2e4861fe92a5371149a0e593d3e5496` |
| Stage B2 `golden_shell.uors3t02` (`S3`) | `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-p2-fallback-devtune/tables/golden_shell.uors3t02` (`<= 100` MB) | `e4c4d1a8c571828221271c76f9062c05e87d0a7f3cc4e1922ff62dd1d8b2c42c` |
| Relation trunk (`seed1.d11`) | `/mnt/plexpool/uor-r4-corpus/runs/20261120-complete-c1-31/normal-5ea09ea47/fixtures/seed1.d11` | `04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d` |
| `DEV-tune` census & cohorts | `/mnt/plexpool/uor-r4-corpus/runs/20261120-pruned-r-fit-37/tune/c420/` (`2,670` records, `10,807,417` targets) | `dev-tune.json` `90fbc73a...`, `tokens` `75c6d1ac...`, `novel` `7df42437...`, `mask` `cf359a0e...`, `segment` `202efde4...`, `id-bytes.u8` `befb22f6...` |
| Pinned ByteBPE tokenizer (`tokenizer.json`) | `.uor-models/research/issue-1019/tokenizer/tokenizer.json` (`V = 4096`) | `d36d3e8700a123e620012df77de195f244fbdb4d05d9e1aa7e77fa9407590f89` |
| Sealed `m2-fresh256` panel (`256` rows, `128` pairs) | Emitted by `docs/labs/m2-fresh-heldout-panel/instrument` (`--emit-dir`) | `plan.json` `0ed1ab98...`, `inputs.json` `1de2fc28...`, `labels.json` `3b5c5477...`, `manifest.json` `7c33f3f7...` |

## 2. Evaluated Mechanisms & Four Pure-Integer GPU Kernels

All four AOT CUDA kernels in `p3_brier_kernels.cu` use strictly pure-integer types (`U8`/`I8`/`U16`/`U32`/`U64`/`I64`), unsigned 64-bit atomics (`unsigned long long*`), and bit-exact 128-bit integer multiply/divide (`__umul64hi` + restoring binary division), with **zero** float types/ops, **zero** MMA/Tensor Core instructions, **zero** softmax/squashing, and **zero** matrix multiplication:

1. `p3_brier_step_batch`: `Q32 x Q48` simplex mixture + target-token Brier score `(1 - p(y))^2` + gradient accumulation.
2. `p3_s3_golden_lookup_and_accum`: Flat-CSR primary -> backoff -> Starling-7 `GoldenShell` lookup using exact normalized `1 + ((eff_hit * (2^48 - 4096)) / eff_total)` (committing on context hit with floor `1` on absent tokens so `sum_{v=0..4095} p(v) <= 2^48`, falling back to `2^36` only on full miss) + GPU target error accumulation.
3. `p3_s2_lookup_and_accum`: Flat-CSR order-2 `(prev2, prev1)` lookup with exact record-boundary reset across `0` delimiters, strict `<= 50,000,000` B header+row byte budget (`10 * ctxs + 7 * rows <= 50,000,000`), and exact normalized `1 + ((eff_hit * (2^48 - 4096)) / eff_total)` so `sum_{v=0..4095} p(v) <= 2^48` + GPU target error accumulation.
4. `p3_s4_vote_batch`: Flat-CSR `UORS4P01` pointer voting kernel computing exact Q16 + `err_i8 * 512` candidate votes on GPU.

### Part A — Pure-Integer GPU Brier Error-Write (`R` -> `DEV-tune`, `UORS6B01`)

- **Objective & Baseline:** Pure-integer target-token Brier LMS (`(1 - p(y))^2` with `rival_sq_q32 = 0`, because sealed `r-65536.tape` stores target-token masses only; `DEV-tune` scored strictly in exact lawful `bits_q12` and compared against count-only `P1(A1s)` = `125,750,699,315` Q12 / `1,400,045` micro-BPB and `R*` `DEV-tune` = `1,432,318` micro-BPB via `1,000`-resample complete-record block bootstrap, `seed = 4390`).
- **Four Pre-Registered Arms:**
  1. `P1(A1s)` (`14` count/Pick experts, `125,750,699,315` Q12 baseline).
  2. `p3_s6_brier`: `15` experts (`14` P1 experts + initial GPU-evaluated `golden_shell` `UORS3T02`), `880` Pick/horizon buckets initialized uniform on `Q32`, trained `3` epochs on `R` in canonical fixed-boundary macro-batches of `524,288` rows (`31` macro-batches/epoch = `93` simplex LMS steps, CUDA chunk `65,536`).
  3. `p3_s6_s3_brier`: Same `15` experts with per-epoch post-S6 GPU `p3_s3_golden_lookup_and_accum` accumulation and zero-centered per-context `S3` `i8` error-cell writes (`lr_shift = 8`, bounded `[-127, 127]`, `sum_v p(v) <= 2^48`) + GPU mass refresh across `3` epochs.
  4. `p3_s6_s2_err_brier`: `16` experts (`15` above + `<= 50,000,000` B `S2` order-2 table built strictly on `R` with record-boundary reset) with per-epoch post-S6 GPU `p3_s2_lookup_and_accum` accumulation + zero-centered `i8` error-cell writes + GPU mass refresh across `3` epochs.
- **Parity Guarantees:** Epoch 1 macro-batch 0 asserts bit-exact identity between CUDA chunks `65,536` and `65,521`; `DEV-tune` asserts `>= 12,500` CPU vs GPU probes with `0` divergences across `S6` Brier mix (`spot_check_divergences == 0`), `S3` lookup (`s3_spot_check_divergences == 0`), and `S2` lookup (`s2_spot_check_divergences == 0`).

### Part B — Pointer-Valued `S4` (`UORS4P01`) + Hinge + `R1` / `R2` / `R4` Reply Panels

- **Training & Overlap Isolation:** `S4` (`UORS4P01`) counts background co-occurrences from `D87 R` (`<= 60,000` tokens, neutral slot `(3, 0)`) mixed with declarative + query `ft/M-World TRAIN` episodes (`8` job literals, `8` home literals, `6` job queries, `6` home queries) followed by `3` epochs of `s4_hinge_step` (`margin = 2 * P3_Q16_ONE`, `step_i8 = 4`). Fail-closed `verify_disjoint_sets` enforces `0` exact-literal and `0` word-2-gram collisions against ALL THREE held-out sets: historical `ft/dev` (`74` literals + `12` `dev512` queries), `m2-fresh256` (`256` rows), and `R4` (`240` rows).
- **GPU Voting & Inference:** On GPU builds, `p3_s4_vote_batch` computes pointer candidate scores on GPU for hinge validation and `m2-fresh256` / `R4` (`Active` & `ShuffledRole`), verifying `0` GPU-vs-CPU score divergences (`s4_gpu_vs_cpu_divergences == 0`). Declarative role-slot spans (`(1, 0)` job, `(2, 0)` home) come solely from prior `Context` turns (`extract_causal_role_spans`), never reading query text or labels, combined with `UORKNC03` `full-c.v3` + `S5` Pick + `apply_lawful_stop_and_rep_guard`. Single-shot free generation requires `eos_stopped && FrozenAnswers::accepts(&text)`.
- **Panels & Controls:** Both `m2-fresh256` (`256` rows, `128` pairs via `pair_id` distinctness) and `R4` (`240` held-out-phrasing rows, reported pooled and split into `120` in-stem + `120` OOV-synonym rows) are scored in `Active`, `NoMemory` (disables both `S4` pointer choice and `S5` Pick per `DECISION.md:84`), and `ShuffledRole` modes with `paired_band_compare`.

## 3. Five Sequential Execution Gates

- **Gate 1 — PTX + SASS 0-Float / 0-Tensor-Core Static Audit (CPU):** Inspect `nvcc -ptx -arch=$UOR_NVCC_ARCH` (`0` PTX float ops) and `cuobjdump --dump-sass "$CUBIN"` (`0` FP32/FP64/FP16/MUFU/MMA ops, excluding exact `sm_120` `HFMA2 R<n>, -RZ, RZ,` constant-load aliases).
- **Gate 2 — Finite Multi-Kernel GPU vs CPU Bit-Exactness (`p3-eval finite`, `flock -n`):** Verifies ALL 4 CUDA kernels (`p3_brier_step_batch`, `p3_s3_golden_lookup_and_accum`, `p3_s2_lookup_and_accum`, `p3_s4_vote_batch`) match CPU bit-for-bit (`brier_divergent == 0`, `s3_divergent == 0`, `s2_divergent == 0`, `s4_divergent == 0`, `divergent == 0`) with `cubin_sha256 == $(field cubin_sha256)`.
- **Gate 3 — Brier Error-Write Evaluation (`p3-eval brier`, `flock -n`):** Checks pinned `seed1.d11` and `golden_shell.uors3t02` hashes, `dev_tune_targets == 10807417`, `p1_a1s_baseline.all.bits_q12 == "125750699315"`, `shifted_batch_parity == true`, `spot_check_probes >= 12500`, `spot_check_divergences == 0`, `s3_spot_check_divergences == 0`, `s2_spot_check_divergences == 0`, monotonic `R` Brier drop (`ep 1..3` `.fell == true` for all 3 trained arms), and writes `p3_s6_*.uors6b01` + `brier_report.json`.
- **Gate 4 — GPU-Voted `S4` + `R1` / `R2` / `R4` Reply Panels (`p3-eval reply`, `flock -n`):** Checks pinned `R_UORT_SHA256`, `0` overlap vs `ft/dev` + `m2-fresh256` + `R4`, reloads `s4_train.uors4p01`, scores `m2-fresh256` (`256` rows / `128` pairs) and `R4` (`240` rows / `120` pairs) across `active`, `no_memory`, and `shuffled_role` via `p3_s4_vote_batch`, asserts `s4_gpu_vs_cpu_divergences == 0`, and emits `reply_report.json`.
- **Gate 5 — Receipt Manifest (`receipts.sha256`):** Hash all `evidence/` JSONs, `.uors6b01`, `.uors4p01`, `sass-check.txt`, logs, `GO.json`, `ORDER.md`, `SHEET.md`, `FIT.md` and verify cleanly on Wick archival before `uor-pod down`.

## 4. Branch Outcomes & Stop Rules

- **Brier Gate (`DEV-tune`):** Record honestly whether any of `p3_s6_brier`, `p3_s6_s3_brier`, or `p3_s6_s2_err_brier` clears `<= 99%` of `P1(A1s)` bits (`95%` paired bootstrap CI upper bound `< 0`) and where each stands vs `0.95 / 0.90 / 0.80 × R*` (`1,432,318` micro-BPB). Even a negative or sub-1% result advances honestly to `REPORT.md`, keeping `DEV-test` untouched.
- **Reply Gate:** Record exact `R1` (`>= 52/128` pairs), `R2` (`>= 200/256` rows), and `R4` (`>= 200/240` rows) counts against both `NoMemory` and `ShuffledRole` paired bands, plus explicit `R4` in-stem (`120` rows) vs OOV-synonym (`120` rows) breakdown.
- **Stop immediately if:** any non-alias PTX/SASS float or MMA opcode appears, any `DEV-test` / `rstar-devtest` path appears, any kernel finite or spot-check or `65536 == 65521` or `s4_gpu_vs_cpu` parity diverges, `P1(A1s)` baseline differs from `125,750,699,315` Q12, monotonic Brier decrease fails across epochs on `R`, `S2` exceeds `50,000,000` B, or any source/binary/input SHA-256 mismatches `GO.json`.
