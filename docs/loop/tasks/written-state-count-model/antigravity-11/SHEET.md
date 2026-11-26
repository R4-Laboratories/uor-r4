# Pre-Registration Sheet: `written-state-count-model`, `antigravity-11` (Stage B3c `P3` GPU Repair + `S4` Reply Panels)

- **Lab:** `antigravity`. **Milestone / Item:** `M2+` / [`written-state-count-model`](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `P3`, Base: `b2c458e7b`).
- **Compute Host:** Dedicated single-GPU RTX 4090 (`sm_89`) Vast.ai pod (`uor-pod up --lab antigravity --count 1`, `<= $1/h` inside the `4 pods / $8.00/h` cap, torn down immediately after archiving `evidence/` + `receipts.sha256` to `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p3c-eval/`). `GO.json` pins `expected_host`, `expected_compute_cap`, `device_uuid`, `gpu_lock`, `UOR_NVCC_ARCH`, and compiled `p3c-eval` + `p3_brier_kernels.cubin` + all source/input SHA-256s; `admit_gpu` (`nvidia-smi` compute apps empty, `>= 8,192 MiB` free VRAM, `>= 10 GiB` free host RAM) runs before every live GPU step.
- **HARD LOCK ON `DEV-test`:** `dev-test.tokens.u16` (`023784b74c98a796401a064d50bf2f83f2f8eb95ab8bbead1cbe2aec6e7a176b`), `dev-test.json` (`ec048217ac64f05a6519d7e38dd2fabf6e244eca8009b24476b3870cbc0f73f2`), and `rstar-devtest` paths MUST NOT be opened, read, referenced in `GO.json`, or scored in `antigravity-11` (enforced by both `dispatch.sh` exit `99` and `DevTuneData::load_checked`).

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

## 2. Evaluated Mechanisms & Five Pure-Integer GPU Kernels

All five AOT CUDA kernels in `p3_brier_kernels.cu` use strictly pure-integer types (`U8`/`I8`/`U16`/`U32`/`U64`/`I64`), unsigned 64-bit atomics (`unsigned long long*`), and bit-exact 128-bit integer multiply/divide (`__umul64hi` + restoring binary division `mul_div_u64`), with **zero** float types/ops, **zero** MMA/Tensor Core instructions, **zero** softmax/squashing, **zero** matrix multiplication, and **zero** hardware runtime division instructions:

1. `p3_brier_step_batch`: `Q32 x Q48` simplex mixture + target-token Brier score `(1 - p(y))^2` + gradient accumulation.
2. `p3_em_step_batch`: `Q32 x Q48` simplex mixture + exact Q32 EM responsibility accumulation `mul_div_u64(wm_q48, P3_Q32_ONE, mixed)` per bucket/expert.
3. `p3_s3_golden_lookup_and_accum`: Flat-CSR primary -> backoff -> Starling-7 `GoldenShell` lookup using exact normalized `1 + ((eff_hit * (2^48 - 4096)) / eff_total)` + GPU target error accumulation.
4. `p3_s2_lookup_and_accum`: Flat-CSR injective 64-bit `S2` order `1..=4` lookup (`bsearch_u64`) with exact record-boundary reset across `0` delimiters, strict `<= 25,000,000` B per order (`<= 100,000,000` B total via `14 * ctxs + 7 * rows`, selected descending `context_total`), and exact normalized `1 + ((eff_hit * (2^48 - 4096)) / eff_total)` + GPU target error accumulation.
5. `p3_s4_vote_batch`: Flat-CSR `UORS4P01` pointer voting kernel computing exact Q16 + `err_i8 * 512` candidate votes on GPU.

### Part A — `A1s`-Warm-Started `S6` + EM/Brier + Bounded Multi-Order `S2` (`R` -> `DEV-tune`, `UORS6B01`)

- **Objective & Warm-Start:** Every trained arm warm-starts `S6` (`880` buckets) from frozen Stage B1a `P1(A1s)` weights (reserving `P3_Q32_ONE >> 7` per newly added expert while keeping proportional `A1s` structure). `DEV-tune` is scored strictly in exact lawful `bits_q12` vs `P1(A1s)` = `125,750,699,315` Q12 / `1,400,045` micro-BPB and `R*` = `1,432,318` micro-BPB (`1,000`-resample complete-record block bootstrap, `seed = 4390`).
- **Four Pre-Registered Arms (`3` epochs each on `R`):**
  1. `p3c_a1s_brier_s6` (`15` experts: `14` P1 + static `S3`): `A1s`-warm-started conservative tangent-centered Brier `S6` updates (`lr_shift = 7`, floor `1 << 10`), checking `65,536 == 65,521` Brier batch parity on Epoch 1 macro-batch 0.
  2. `p3c_a1s_em_s3_brier` (`15` experts: `14` P1 + dynamic `S3`): `A1s`-warm-started pure-integer EM `S6` refit (`prior_weight = 64`, `65,536 == 65,521` EM batch parity, final epoch-3 S6 refit after S3 write) alternating with GPU `p3_s3_golden_lookup_and_accum` zero-centered `i8` error-cell writes (`lr_shift = 9`).
  3. `p3c_a1s_em_s2_multi_brier` (`19` experts: `14` P1 + `S3` + `S2` orders `1..=4`): `A1s`-warm-started pure-integer EM `S6` refit (with final epoch-3 S6 refit after S3/S2 write) alternating with GPU `S3` + GPU multi-order `S2` (`1..=4`, `<= 100 MB`) `i8` error-cell writes (`lr_shift = 9`).
  4. `p3c_a1s_brier_s2_multi` (`19` experts: `14` P1 + `S3` + `S2` orders `1..=4`): `A1s`-warm-started conservative tangent-centered Brier `S6` (`lr_shift = 7`) alternating with GPU `S3` + GPU multi-order `S2` (`1..=4`, `<= 100 MB`) `i8` error-cell writes (`lr_shift = 9`).
- **Parity Guarantees:** `>= 12,500` CPU vs GPU spot-checks on `DEV-tune` with `0` divergences across `S6` Brier, EM, `S3`, and `S2(1..=4)`.

### Part B — Balanced `S4` (`UORS4P01`) + `D87 R` `15.9M`-Token Co-Occurrence Propagation + Active Hinge

- **Training & Overlap Isolation:** Replace `ep_i % 6` query duplication and `4:1` pre-stuffing with symmetric balanced episodes (`24` episodes: `3x8` literals, `4x6` job/home queries). Derive role anchors strictly from `TRAIN_MWORLD` queries (excluding top-48 universal `D87 R` stopwords), scan all `15,947,584` `D87 R` tokens (`r.tokens.u16`) with record-bounded `±8` co-occurrence (`(co << 16) / (freq + 64)`) to select top-256 non-function tokens per role paired against top-64 `D87 R` function tokens and shared `TRAIN` scaffold, compress `D87 R` neutral `(3, 0)` background counts for `TRAIN` pair keys, and run `3` epochs of `s4_hinge_step` (`margin = 16 * P3_Q16_ONE`, `step_i8 = 8`, asserting `total_hinge_updates > 0`) with GPU voting parity (`0` diffs). Fail-closed `verify_disjoint_sets` enforces `0` collisions against `ft/dev`, `m2-fresh256`, and `R4`.
- **Panels & Controls:** Score `m2-fresh256` (`256` rows / `128` pairs) and `R4` (`240` rows: `120` in-stem + `120` OOV-synonym) via GPU `p3_s4_vote_batch` across `Active`, `NoMemory`, and `ShuffledRole` with `paired_band_compare`.

## 3. Five Sequential Execution Gates

- **Gate 1 — PTX + SASS 0-Float / 0-Tensor-Core Static Audit (CPU):** Inspect `nvcc -ptx -arch=$UOR_NVCC_ARCH` (`0` PTX float ops) and `cuobjdump --dump-sass "$CUBIN"` (`0` FP32/FP64/FP16/MUFU/MMA ops).
- **Gate 2 — Finite Multi-Kernel GPU vs CPU Bit-Exactness (`p3c-eval finite`, `flock -n`):** Verifies ALL 5 CUDA kernels (`p3_brier_step_batch`, `p3_em_step_batch`, `p3_s3_golden_lookup_and_accum`, `p3_s2_lookup_and_accum`, `p3_s4_vote_batch`) match CPU bit-for-bit (`divergent == 0`) with `cubin_sha256 == $(field cubin_sha256)`.
- **Gate 3 — Brier & EM Evaluation (`p3c-eval brier`, `flock -n`):** Checks pinned hashes, `dev_tune_targets == 10807417`, `P1(A1s)` baseline `125750699315`, `s2_multi_table.total_bytes <= 100000000`, `shifted_batch_parity == true`, `spot_check_probes >= 12500`, `0` divergences (`spot_check`, `em_spot_check`, `s3_spot_check`, `s2_spot_check`), net 3-epoch `R` improvement (Brier drop on `p3c_a1s_brier_*`, bits drop on `p3c_a1s_em_*`) while logging per-epoch `brier_fell` and `bits_fell`, and writes 4 `p3c_*.uors6b01` files + `brier_report.json`.
- **Gate 4 — Balanced `S4` + `D87 R` Propagation + `R1` / `R2` / `R4` Reply Panels (`p3c-eval reply`, `flock -n`):** Checks `0` overlap vs `ft/dev` + `m2-fresh256` + `R4`, `s4_artifact.total_hinge_updates > 0`, `s4_gpu_vs_cpu_divergences == 0`, scores `m2-fresh256` & `R4` (`Active`/`NoMemory`/`ShuffledRole`), and writes `s4_train.uors4p01` + `reply_report.json`.
- **Gate 5 — Receipt Manifest (`receipts.sha256`):** Hash all `evidence/` JSONs, `.uors6b01`, `.uors4p01`, `sass-check.txt`, logs, `GO.json`, `ORDER.md`, `SHEET.md`, `FIT.md` and verify cleanly on Wick archival before `uor-pod down`.

## 4. Branch Outcomes & Stop Rules

- **Brier / EM Gate (`DEV-tune`):** Record exact `DEV-tune` `bits_q12` and micro-BPB of all 4 arms vs `P1(A1s)` (`<= 99%`, `95%` bootstrap CI upper bound `< 0`) and `0.95 / 0.90 / 0.80 × R*` (`1,432,318`). Even a negative or sub-1% result advances honestly to `REPORT.md`, keeping `DEV-test` untouched.
- **Reply Gate:** Record exact `R1` (`>= 52/128` pairs), `R2` (`>= 200/256` rows), and `R4` (`>= 200/240` rows) counts + `R4` in-stem vs OOV-synonym breakdown vs `NoMemory` & `ShuffledRole`.
- **Stop immediately if:** any PTX/SASS float or MMA opcode appears, any `DEV-test` / `rstar-devtest` path appears, any finite/spot-check/batch parity diverges, `P1(A1s)` baseline differs from `125,750,699,315` Q12, `S2` exceeds `25,000,000` B/order or `100,000,000` B total, `total_hinge_updates == 0`, or any source/binary/input SHA-256 mismatches `GO.json`.
