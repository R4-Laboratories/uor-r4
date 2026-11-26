# REPORT: written-state-count-model, antigravity-13

Lab: `antigravity`. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P3`). Order: [ORDER.md](ORDER.md). Base: `bf4d9747a`.

**Outcome (`Stage B5`, `m2-fresh256`, `R4`): PASS against all Stage B5 `DECISION.md` §4 `S4` panel bars (`R1 = 128/128` pairs [`100.0%` >= `52/128`], `R2 = 256/256` rows [`100.0%` >= `200/256`], and `R4 = 200/240` rows [`83.3%` >= `200/240`; `120/120` in-stem + `80/120` OOV-synonym], beating `NoMemory` and `ShuffledRole` paired controls by `+256` on `m2-fresh256` and `+200` / `+160` on `R4`), unlocking Stage B5b sealed `DEV-test` qualification while leaving `DEV-test` strictly untouched (`0` active pods, `$0.00` pod spend).**

## What did not work, first

1. **Two of six `R4` OOV-synonym phrasings (`"State my dwelling."` and `"Clarify my quarters."`) scored `0/20` each (`40/120` OOV miss):** While `"Clarify my profession."`, `"State my occupation."`, `"Which trade do I practice?"`, and `"Which town shelters me?"` all scored `20/20` (`80/120` OOV-synonym vs `40/120` in `antigravity-12` Stage B4 / `antigravity-11` Stage B3c and `20/120` in `antigravity-10` Stage B3b), `"dwelling"` and `"quarters"` missed `prop_home` (top `1,024`, `>= 1.25x` contrast `4 * win >= 5 * lose`, `top_words = 32,768`). Pass 1 hash capacity did not drop any atoms (`c_unique_tracked_atoms = 279,602 << PASS1_LOAD_LIMIT = 3,145,728`); rather, `"quarters"` is polysemous across domains (financial/calendar/spatial quarters) while `"dwelling"` fell outside top-`32,768` tracked word atoms or top-`1,024` `prop_home` margin ranking. *(Provenance note [`L-001`]: [`SHEET.md`](SHEET.md#L16) §1 mis-cited Stage B4 as `R1 = 109/128`, `R2 = 228/256`, `R4_oov = 2/120`, and `2,420,971` coding/dialogue tokens; the actual `antigravity-12` [`REPORT.md`](../antigravity-12/REPORT.md#L5) scored `R1 = 128/128`, `R2 = 256/256`, `R4_oov = 40/120` [`160/240`] on `15,951,644`-token `D87 R` `r.tokens.u16`. [`SHEET.md`](SHEET.md) is kept byte-identical here so its sealed SHA-256 `ed9ca433...` in [`FIT.md`](FIT.md) and [`evidence/GO.json`](evidence/GO.json) remains valid.)*
2. **Non-English / cross-domain noise in high-margin 2-hop PPMI candidates and `ShuffledRole` `40/240` score on `R4`:** `top_prop_job_preview` (`c_ppmi_s4`) surfaced noisy non-English web-crawl tokens (`" dailctic"`, `" arabic"`, `" tunisian"`, `" derja"`, `b = 47..50`) alongside genuine occupational nouns (`" fiduciary"`, `" internship"`, `" duties"`, `" translator"`, `" jobs"`, `" career"`, `" employers"`, `" roles"`, `" hiring"`, `" wages"`, `" unemployment"`). Meanwhile, `NoMemory` scored `0/240` (zero fallback predictions exist without `S4`), whereas `ShuffledRole` flips `[(1, 0), (2, 0)] -> [(2, 0), (1, 0)]`: `ShuffledRole` scored `40/240` on `R4` (`0/256` on `m2-fresh256`) because `Active` actively mis-voted role `1` (`job`) on the two missed `home` queries (`"State my dwelling."` and `"Clarify my quarters."`, `0/20` each under `Active`), which role inversion flipped to role `2` (`home`) for `20 + 20 = 40/240` (`Active` still beat `ShuffledRole` by `+160` rows `>> band_ceil = 32`).
3. **Standalone `R4`/`m2-fresh256` Stage B5 run did not re-run `DEV-tune` BPB or sealed `DEV-test`:** Per `ORDER.md` and `SHEET.md`, Stage B5 isolated and solved the `S4` OOV-synonym bottleneck first (unlocking Stage B5b `DEV-test` qualification alongside Stage B4's `p4_full_errwrite`); Stage B5 does not itself claim `DEV-test` BPB pass.
4. **Operational retries fixed pre-gate under `L-002`:** An initial local clippy/test iteration and a pre-Gate-1 `dispatch.sh` run hit a `seed1.d11` checksum typo (`run_dir = 20261126-antigravity-p5-s4`) and a Gate 1 `grep -q` SIGPIPE (`141`, `run_dir = 20261126-antigravity-p5-s4-r2`), each fixed and re-reviewed (`FIT` R4) before executing the fresh sealed run into `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p5-s4-r3`.

## What changed

All edits live strictly inside the 3 ordered paths (`docs/loop/tasks/written-state-count-model/antigravity-13/**`, `docs/research/notebook/m4-serving.md`, and `docs/roadmap/items/written-state-count-model.md`); zero edits to `crates/**`, zero floats/matmul/softmax/runtime CUDA divides:

1. **Pre-registration, R4 non-author FIT review & hardened dispatch (`antigravity-13/`):**
   - [`ORDER.md`](ORDER.md), [`SHEET.md`](SHEET.md), [`FIT.md`](FIT.md) (`TEST FITNESS: FIT` R4 across all 11 package SHA-256s), [`dispatch.sh`](dispatch.sh) (`DEV-test` `exit 99` guard, `L-002` fresh `RUN_DIR` check, `CUDA_DISABLE_PTX_JIT=1`, `sm_89` VRAM check, non-vacuous `nvcc -ptx` + `cuobjdump --dump-sass` Gate 1, Gate 2 embedded `cubin_sha256` bind, Gate 3 parity + `receipts.sha256`, `checks_verified=32`).
2. **Standalone `p5-eval` + pure-integer CUDA kernel (`antigravity-13/p5-eval/`):**
   - [`p5_s4_kernels.cu`](p5-eval/src/p5_s4_kernels.cu): Pure-integer `p3_s4_vote_batch` kernel using 128-by-64 restoring `mul_div_u64` and `bsearch_u32` (zero floats, zero MMA, zero softmax, zero runtime `/` or `%`).
   - [`cuda.rs`](p5-eval/src/cuda.rs) & [`main.rs`](p5-eval/src/main.rs): Loads embedded AOT `.sm_89.cubin` via `cudarc::driver::Ptx::from_binary`, executes `finite` (`64` probes) and `reply`, with `#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]`.
   - [`c_ppmi.rs`](p5-eval/src/c_ppmi.rs): `#![forbid(unsafe_code)]`; injective 64-bit ByteBPE word-atom packing (`1..=5` lowercase ASCII tokens, trimmed byte len `>= 3`); streaming Pass 1 over `c.tokens.u16` (`2,075,608,357` targets, SHA-256 `cd3adfe9...` verified on the stream); Pass 2 record-bounded `±6` word-window `32,768 × 1,024` co-occurrence + `TOP_PPMI_BITS = 128` packed `[u64; 16]` signatures + `2 * h2 + h1` `>= 1.25x` contrast (`1,024` Job / `1,024` Home atoms).
   - [`reply_eval.rs`](p5-eval/src/reply_eval.rs): `#![forbid(unsafe_code)]`; frame-matched `TRAIN_{JOB,HOME}_QUERIES`, `verify_disjoint_sets` (`0` collisions vs `ft_dev`, `m2-fresh256`, and `R4`), `refuse_sealed_devtest_sha`, contrastive multi-token internal stem-pair emission (`0 <= a < b < len`, `4 * win >= 5 * lose`, `5,637` pairs, zero `t0 × scaffold` leakage) + single-token emissions (`249`), `3` epochs `s4_hinge_step`, and GPU-voted panel grading.
3. **Notebook & roadmap records:** Prepended Stage B5 row in [`m4-serving.md`](../../../../research/notebook/m4-serving.md) and appended Stage B5 summary in [`written-state-count-model.md`](../../../../roadmap/items/written-state-count-model.md).

## Measurements

Executed on Wick RTX 4090 (`GPU-7c813ad0-2151-bb1d-5efb-a3340b9a6a41`, `sm_89`, `CUDA_DISABLE_PTX_JIT=1`, `$0.00` pod spend), archived permanently on Wick at `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p5-s4-r3/` (`receipts.sha256` `7/7 OK`):

| Gate / Metric | Stage B4 (`antigravity-12`) | Stage B5 (`antigravity-13`) | Threshold / Bar | Source Evidence |
|---|---|---|---|---|
| Gate 1 PTX / SASS Float & MMA | `0 / 0` | **`0 / 0`** (`p5_s4_kernels.sm_89.cubin` `6afebd96`) | `0` | [`sass-check.txt`](evidence/sass-check.txt) |
| Gate 2 Finite GPU vs CPU | `525/525` (`0` diffs) | **`64/64` (`0` diffs, `0.13s`, `175,696 KiB`)** | `64/64`, `0` diffs | [`finite.json`](evidence/finite.json), [`finite.time`](evidence/finite.time) |
| Gate 3 Wall Time & Peak RSS | `9:07` (Wick `r2`) | **`1:02.10` (`4,043,496 KiB` / `3.86 GiB`)** | `< 2400s` | [`reply.time`](evidence/reply.time) |
| `C` Words / Unique / Active | — | **`333,859,944` words / `279,602` unique / `32,768` active** | `2,075,608,357` toks | [`reply_report.json`](evidence/reply_report.json) |
| `S4` `UORS4P01` Table | `D87 R` | **`60,946` ctx / `1,590,748 B` (`c6fd3f2f`)**; `4,505` hinge updates | `> 0` hinge updates | [`reply_report.json`](evidence/reply_report.json) |
| **`m2-fresh256` (`R1` Pairs)** | `128/128` | **`128/128` (`100.0%`)** (`NoMem 0`, `Shuf 0`, `+256` rows > ceil `32`) | `>= 52/128` + ctrls | [`reply_report.json`](evidence/reply_report.json) |
| **`m2-fresh256` (`R2` Rows)** | `256/256` | **`256/256` (`100.0%`)** (`NoMem 0`, `Shuf 0`, `+256` rows > ceil `32`) | `>= 200/256` + ctrls | [`reply_report.json`](evidence/reply_report.json) |
| **`R4` Total (`240` Rows)** | `160/240` | **`200/240` (`83.3%`, `80/120` pairs)** | **`>= 200/240`** | [`reply_report.json`](evidence/reply_report.json) |
| `R4` In-Stem (`120` Rows) | `120/120` | **`120/120` (`100.0%`)** (`job`/`home`/`live`/`residence` = `20/20` each) | `—` | [`reply_report.json`](evidence/reply_report.json) |
| `R4` OOV-Synonym (`120` Rows) | `40/120` | **`80/120` (`66.7%`)** (`profession`/`occupation`/`trade`/`town shelters` `20/20`; `quarters`/`dwelling` `0/20`) | `—` | [`reply_report.json`](evidence/reply_report.json) |
| `R4` Paired Controls | `NoMem 0`, `Shuf 60` | **`NoMem 0/240` (`net +200` > ceil `30`); `Shuf 40/240` (`net +160` > ceil `32`)** | Strictly beats both | [`reply_report.json`](evidence/reply_report.json) |
| Runtime GPU vs CPU Parity | `0` diffs | **`0` diffs (`s4_gpu_vs_cpu_divergences = 0`)** | `0` diffs | [`reply_report.json`](evidence/reply_report.json) |

## Checks run at the head

Run in `/Users/casey.allard/uor-r4/.worktrees/antigravity-b5-s4`:
- `cargo fmt --check`, `cargo test --offline --locked` (`3 passed; 0 failed`), `cargo clippy --offline --locked --no-deps --all-targets -- -D warnings`, `bash -n dispatch.sh`: `PASS`
- Wick CUDA `cargo clippy --offline --locked --features cuda --no-deps --all-targets -- -D warnings` & `cargo test --offline --locked --features cuda` (`3 passed; 0 failed`): `PASS`
- Wick `GO.json` (`checks_verified=32`, `STAGE_B5_P3_S4_COMPLETE`) & Wick archive `sha256sum --strict -c receipts.sha256` (`7/7 OK`): `PASS`
- `(cd /Users/casey.allard/uor-r4 && scripts/pod/uor-pod status)`: `0` pods owned by `antigravity` (`$0/h`)
- `/Users/casey.allard/uor-r4/local/harness-target/debug/uor-harness`: `loop report --check`, `claim-wording`, `shard-check`, `check-formats`, `ratchet d23|d25|d32 --quiet`, `check --only harness.native_space_contract`, `loop pr-check --local`: run at final Commit 2 head (verified before push/PR).

## Skills used

- `native-space-cycle` (`.agents/skills/native-space-cycle/SKILL.md`): verified shared `p3_state` integer operators, bit-identical GPU/CPU finite + runtime parity (`0` divergences), and lossless `UORS4P01` + `receipts.sha256` hashing.
- `uor-research-architect` (`.agents/skills/uor-research-architect/SKILL.md`): enforced R1–R4 non-author `pro` FIT reviews, `DEV-test` isolation, `L-002` fresh run roots, paired memory controls, and permanent Wick archival.
- `verification-before-completion` (`verification-before-completion/SKILL.md`): enforced fresh post-edit local, CUDA, Wick archive, and `uor-harness` verification before PR/merge.

## Still open

1. **Stage B5b — Single-shot sealed `DEV-test` `P3` qualification (`m2_devtest` + `p4_full_errwrite`):** With `R1 = 128/128`, `R2 = 256/256`, and `R4 = 200/240` cleared against both controls, execute the pre-registered Stage B5b sealed `DEV-test` run combining Stage B4's causal session cache + `S2-skip` + `S3` (`1,209,985` micro-BPB / `0.8447 × R*` on `DEV-tune`) with Stage B5's `C`-corpus PPMI `S4` under `DECISION.md` §4 (`#4063`).
2. **Optional OOV refinement beyond `200/240` (`"quarters"`, `"dwelling"`):** Clean non-English tokens in `c_ppmi.rs` (`dailctic`/`tunisian`/`derja`) or expand `top_words > 32,768` / `prop_cap > 1,024` if later `M3`/`M4` conversational panels need higher lexical recall on rare residential synonyms.

## Plain summary

This stage replaced tiny-corpus word association (`D87 R`) with a pure-integer streaming pass over all `2.075 billion` tokens of general-domain `C` (`c.tokens.u16`), packing `1..=5`-subword lowercase English words into 64-bit integer keys, computing `1024`-bit excess-PMI signatures in `62.1` seconds on an idle Wick RTX 4090 GPU (`$0.00` pod cost), and emitting contrastive internal subword pairs for active `S4` hinge training and GPU-voted reply selection — with zero floating-point instructions, zero matrix multiplications, zero softmax, and zero GPU-vs-CPU divergences. Held-out fact-recall on `m2-fresh256` stayed at **100%** (`R1 = 128/128` pairs, `R2 = 256/256` rows), and unseen-phrasing `R4` jumped from `160/240` (`40/120` OOV-synonym) to **`200/240`** (`120/120` in-stem + **`80/120` OOV-synonym**), clearing the `>= 200/240` bar against both `NoMemory` (`0/240`) and `ShuffledRole` (`40/240`) controls and unlocking sealed `DEV-test` qualification while keeping `DEV-test` untouched.
