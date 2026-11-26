# Pre-Registration Sheet: `written-state-count-model`, `antigravity-12` (Stage B4 `P3` Disjoint-`C` `S2-skip` + Causal Session Cache + 2-Hop `S4`)

- **Lab:** `antigravity`. **Milestone / Item:** `M2+` / [`written-state-count-model`](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `P3`, Base: `aecd3675a`).
- **Compute Host:** Primary Wick workstation RTX 4090 (`sm_89`, `GPU-7c813ad0-2151-bb1d-5efb-a3340b9a6a41`, `$0.00` pod spend, lock `/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock` held with `flock -n`); emergency single-GPU Vast.ai RTX 4090 fallback ONLY if Wick GPU is occupied by a foreign job (archived to Wick before `uor-pod down`). `GO.json` pins `expected_host`, `expected_compute_cap`, `device_uuid`, `gpu_lock`, `UOR_NVCC_ARCH`, and compiled `p4-eval` + `p4_brier_kernels.cubin` + all source/input SHA-256s; `admit_gpu` (`nvidia-smi` compute apps empty, `>= 8,192 MiB` free VRAM, `>= 10 GiB` free host RAM) runs before every live GPU step.
- **HARD LOCK ON `DEV-test`:** `dev-test.tokens.u16` (`023784b74c98a796401a064d50bf2f83f2f8eb95ab8bbead1cbe2aec6e7a176b`), `dev-test.json` (`ec048217ac64f05a6519d7e38dd2fabf6e244eca8009b24476b3870cbc0f73f2`), and `rstar-devtest` paths MUST NOT be opened, read, referenced in `GO.json`, or scored in `antigravity-12` (enforced by both `dispatch.sh` exit `99` and `DevTuneData::load_checked`).

## 1. Frozen Inputs (verified by SHA-256 before execution)

| Artifact | Origin / Role | Pinned SHA-256 / Size |
|---|---|---|
| Disjoint `C` stream (`c.tokens.u16`) | `/mnt/plexpool/uor-r4-corpus/runs/20261120-whole-context-36/input/c.tokens.u16` | `cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a` (`4,151,216,778` B) |
| Compact count/Pick table (`full-c.v3`) | `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/gate3-fullc/full-c.v3` | `75ccb234e669cb6b707ffe4091822a91a225e0875a0e99edf9f21f8f2ec3f233` |
| Reserved `R` stream (`r.tokens.u16`) | `/home/callard/uor-codex/20261119-rival-admission-29/count/cr-v1/r.tokens.u16` (`4,060` records, `15,947,584` targets) | `5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a` (`31,903,352` B) |
| Stage B1a `R` & `DEV-tune` `UORCBF01` | `r-65536.tape` / `tune-65536.tape` | `8b6b39d25ace5d1dadfe50fd05d8638b6b5b3765784f20e8714f3832935d8c0a` / `dc0785bf0002a17f9e416235e7b4aeb1308b218574ce6b5145b808244fc4f24d` |
| Stage B1a `A1s` weights, manifest & seal | `a1s.weights` / `a1s.manifest.json` / `a1s.seal.json` | `e2fbf41d38d9dbe3449cdbb85c8bd335c5ee1f803e35bc7993070d65c0d33293` / `26f8f2225a8a0735400a647e37f2f97d86cde9f2de8d847ef791469a6d6b02ec` / `b6e9ac04006100922873d86562d6f525a9e5b73c713ca15d2dd329294d798071` |
| Stage B1a `R` Pick & Stage B2 `DEV-tune` Pick | `pick.rows` (`R`) / `devtune-pick.rows` (`DEV-tune`) | `ba9e646a99ed1507235e95715c6e959469cd830085b6b492b01b2a083769bba9` / `e2e3cf84db615517c84724700396fa6da2e4861fe92a5371149a0e593d3e5496` |
| Stage B2 `golden_shell.uors3t02` (`S3`) | `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-p2-fallback-devtune/tables/golden_shell.uors3t02` (`<= 100` MB) | `e4c4d1a8c571828221271c76f9062c05e87d0a7f3cc4e1922ff62dd1d8b2c42c` |
| Relation trunk (`seed1.d11`) | `/home/callard/uor-codex/20261118-diagnostics/inputs/seed1.d11` | `04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d` |
| `DEV-tune` census & cohorts | `/mnt/plexpool/uor-r4-corpus/runs/20261120-pruned-r-fit-37/tune/c420/` (`2,670` records, `10,807,417` targets) | `dev-tune.json` `90fbc73a...`, `tokens` `75c6d1ac...`, `novel` `7df42437...`, `mask` `cf359a0e...`, `segment` `202efde4...`, `id-bytes.u8` `befb22f6...` |
| Pinned ByteBPE tokenizer (`tokenizer.json`) | `.uor-models/research/issue-1019/tokenizer/tokenizer.json` (`V = 4096`) | `d36d3e8700a123e620012df77de195f244fbdb4d05d9e1aa7e77fa9407590f89` |
| Sealed `m2-fresh256` panel (`256` rows, `128` pairs) | Emitted by `docs/labs/m2-fresh-heldout-panel/instrument` (`--emit-dir`) | `plan.json` `0ed1ab98...`, `inputs.json` `1de2fc28...`, `labels.json` `3b5c5477...`, `manifest.json` `7c33f3f7...` |

## 2. Evaluated Mechanisms & Pure-Integer GPU Kernels

All five AOT CUDA kernels in `p4_brier_kernels.cu` (`MAX_EXPERTS = 24`) use strictly pure-integer types, unsigned 64-bit atomics (`unsigned long long*`), and bit-exact 128-bit integer multiply/divide (`__umul64hi` + `mul_div_u64`), with **zero** float ops, **zero** MMA ops, **zero** softmax/squashing, **zero** matmul, and **zero** hardware runtime divides.

### Part A — Disjoint-`C` `S2-skip` + Causal Session Cache + Error-Write (`R` -> `DEV-tune`)

- **Disjoint-`C` `S2-skip` (`UORS2S01`, `<= 80,000,000` B):** `build-skip` streams `c.tokens.u16` (`2.075B` tokens) in 2 passes (`~400 MB` RAM) to build injective 64-bit skip tables `skip_lag2 <= 20,000,000` B (`t_{n-2} -> t_n`), `skip_3_1 <= 30,000,000` B (`(t_{n-3}, t_{n-1}) -> t_n`), and `skip_4_1 <= 30,000,000` B (`(t_{n-4}, t_{n-1}) -> t_n`), bound cleanly to `c_corpus_sha256`.
- **Causal Within-Record Cache (`cache_eval.rs`, `0` B stored on disk):** Computes `cache_uni_recency`, `cache_bi_local`, `cache_tri_local`, and `cache_lcp_local` strictly causally and reset at every `BOS = 0`, emitting `cache_hit_state in 0..=2` (`2,640` `S6` buckets).
- **Four Pre-Registered Arms (`DEV-tune` `1,000`-resample block bootstrap `seed = 4390` vs `P1(A1s)` `125,750,699,315` Q12 / `1,400,045` micro-BPB and `R*` `1,432,318` micro-BPB, plus controlled `errwrite - static` deltas):**
  1. `p4_c_skip_static` (`18` experts, `880` buckets): `14` P1 + static `S3` + static `S2-skip[0..3]`, EM `S6` refit (`prior = 64`), `err_i8 = 0`.
  2. `p4_c_skip_errwrite` (`18` experts, `880` buckets): `3` epochs of EM `S6` + GPU Brier error-write (`lr_shift = 7`) on `S3` + `S2-skip[0..3]`, plus final epoch-3 EM `S6` refit.
  3. `p4_cache_static` (`18` experts, `2,640` cache buckets): `14` P1 + `4` causal session cache experts, EM `S6` refit (`prior = 64`).
  4. `p4_full_errwrite` (`22` experts, `2,640` cache buckets): `14` P1 + `S3` + `S2-skip[0..3]` + `4` causal cache experts, `3` epochs of EM `S6` + GPU Brier error-write (`lr_shift = 7`), plus final epoch-3 EM `S6` refit.

### Part B — Clean Word-Initial Anchors + 2-Hop 512-Bit PPMI Propagation + Active `S4` Hinge

- Filter `TRAIN_MWORLD` seed anchors strictly to word-initial tokens (`starts_with(' ')`, len `>= 4`, not top-64 `D87 R` stopwords), build a `4096 × 512` medium-frequency context matrix (`±6` record-bounded window over all `15,947,584` `D87 R` tokens), extract exact 512-bit PPMI context signatures (`[u64; 8]`), score word-initial candidates via 1-hop + 2-hop `AND`/`popcount` under `>= 1.2x` (`5*win >= 6*lose`) contrast (top-`384` per role, tiered counts `4, 3, 2, 1`), pair with top-`192` frequent scaffold tokens + `shared_frame`, run `3` epochs active `s4_hinge_step`, and evaluate GPU-voted `R1`/`R2` (`m2-fresh256`) + `R4` (`240` rows) against `NoMemory` and `ShuffledRole`.

## 3. Five Sequential Execution Gates

- **Gate 1 (PTX + SASS):** `ptx_float_opcode_hits == 0`, `float_opcode_hits == 0` on `sm_89`.
- **Gate 2 (`p4-eval finite`, `flock -n`):** All 5 CUDA kernels match CPU (`22` experts / `12` buckets, `rows_checked = 525`, `divergent == 0`).
- **Gate 3 (`p4-eval build-skip`, `timeout 300` CPU):** Materialize disjoint-`C` `s2_skip.uors2s01` (`per_family_bytes <= [20M, 30M, 30M]`, `total <= 80M`), verifying `c_corpus_sha256` + reload parity.
- **Gate 4 (`p4-eval brier`, `flock -n`):** Check pinned SHAs, `DEV-tune == 10,807,417`, `P1(A1s) == 125,750,699,315`, `shifted_batch_parity == true` (`880` and `2,640` buckets, Brier & EM), `spot_check_probes >= 12,500`, `0` divergences (`mix`, `s3`, `s2`), and net 3-epoch `R` Brier drop on `arms[1]` & `arms[3]`.
- **Gate 5 (`p4-eval reply`, `flock -n` + `receipts.sha256`):** Check `0` overlap vs `ft/dev` + `m2-fresh256` + `R4`, `s4_gpu_vs_cpu_divergences == 0`, `total_hinge_updates > 0`, score `m2-fresh256` + `R4`, and seal `receipts.sha256` archived cleanly on Wick.

## 4. Branch Outcomes & Stop Rules

- **Brier / Cache / Skip Gate (`DEV-tune`):** Report exact `DEV-tune` `bits_q12`, micro-BPB, 95% CIs vs `P1(A1s)` & `errwrite - static` controls, and `0.95/0.90/0.80 × R*` honestly (`DEV-test` stays untouched here). **If PASS (`<= 99%` of `P1(A1s)` bits with 95% CI upper bound `< 0` and Brier falling) AND `R4 >= 200/240`:** unlocks a separate pre-registered Stage B4b sealed `DEV-test` qualification run before advancing to `P4`. **If negative / missed:** closes this specific Stage B4 configuration (`REPORT.md` + `m4-serving.md`), preserves `DEV-test` untouched, and records the measured `cache` / `S2-skip` / `errwrite - static` attribution to scope Stage B5.
- **Reply Gate (`m2-fresh256` + `R4`):** Report exact `R1` (`>= 52/128`), `R2` (`>= 200/256`), and `R4` (`>= 200/240`) with in-stem vs OOV-synonym breakdown vs paired `NoMemory` & `ShuffledRole` controls.
- **Stop immediately if:** any PTX/SASS float/MMA op appears, any `DEV-test` path/SHA appears, any finite/batch/spot-check diverges, `s2_skip` exceeds budgets, `s4_gpu_vs_cpu_divergences > 0`, `total_hinge_updates == 0`, or any SHA mismatches `GO.json`.
