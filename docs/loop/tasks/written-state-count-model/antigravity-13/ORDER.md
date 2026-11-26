# ORDER: written-state-count-model, antigravity-13

Lab: antigravity. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P3`). Base: `bf4d9747a`.

## Goal

Pre-register and execute **Stage B5 (`P3` broad-domain `C`-corpus ByteBPE word-atom 1024-bit 2-hop PPMI `S4`, [`DECISION.md §4`](../../../../research/reviews/pi-review-2026-11-16/DECISION.md#L304-L320))** to clear **`R4 >= 200/240`** while holding **`R1 >= 52/128`** and **`R2 >= 200/256`** on `m2-fresh256` against paired controls, keeping sealed `DEV-test` strictly untouched:

1. **Injective ByteBPE Word-Atom Streaming over `c.tokens.u16` (`2,075,608,357` targets, SHA `cd3adfe9...`):**
   - Pack `1..=5` subword token IDs of each lowercase ASCII alphabetic word atom (`decoded trimmed len >= 3`) into an injective 64-bit key `(len << 60) | ...` across record boundaries (`0` / `1`).
   - Stream `c.tokens.u16` in 2 CPU passes: select top `W = 32,768` word atoms (`+` `TRAIN` anchor word atoms) and `1,024` medium-frequency context basis atoms (ranks `128..1152`), populate a record-bounded `±6` word-window matrix (`32,768 × 1,024` `u32`), extract top-`128` excess-PMI bits (`[u64; 16]`), and combine 1-hop `anchor_idf`-weighted co-occurrence with top-`3` `popcount_1024` 2-hop similarity under `>= 1.25x` (`4*win >= 5*lose`) contrast (`top-1,024` Job / `top-1,024` Home, tiered `8/6/4/2`).
2. **Frame-Matched `TRAIN` Anchors, Contrastive Stem-Pair Emission, Active `S4` Hinge & GPU-Voted Panels:**
   - Use frame-matched `TRAIN` queries (`job/employment/role/work/task/duty/status` vs `home/street/address/house/residence/live/location`), verify `0` literal/2-gram overlap vs `ft/dev`, `m2-fresh256`, and `R4`, emit single-token × scaffold + contrastive multi-token stem internal pairs, train `3` epochs `s4_hinge_step`, and score `m2-fresh256` (`R1`/`R2`) + `R4` (`240` rows) via GPU `p3_s4_vote_batch` against `NoMemory` and `ShuffledRole`.

## Native-space cycle contract (D85)

1. **State & carried fields:** Original 12-bit `UORT` tokens (`V = 4096`), `UORKNC03` (`full-c.v3`), `A1s` `UORS5W02`, `seed1.d11`, `UORS4P01`, and `LawfulStopConfig`. Zero floats, zero matmul, zero softmax.
2. **Shared operators:** Standalone `p5-eval` depends on `uor_r4_count_runtime::p3_state` (`#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]`), with mirrored pure-integer `p5_s4_kernels.cu` verified via PTX/SASS 0-float audit.
3. **Transition parity:** GPU vs CPU finite `p3_s4_vote_batch` (`64` probes, `0` diffs) and runtime panel parity (`s4_gpu_vs_cpu_divergences == 0`).
4. **Update rule:** Balanced `TRAIN_MWORLD` (`24` episodes), `3` epochs `s4_hinge_step` (`total_hinge_updates > 0`).
5. **Lossless serialization:** Whole-payload SHA-256 + `train_sha256` on `UORS4P01`, sealed `GO.json`, and Wick `evidence/receipts.sha256`.

## Files in scope

- `docs/loop/tasks/written-state-count-model/antigravity-13/**`
- `docs/research/notebook/m4-serving.md`
- `docs/roadmap/items/written-state-count-model.md`

## Done when

1. Non-author `pro` `FIT.md` (`TEST FITNESS: FIT`), `GO.json`, `rustfmt --check`, `cargo test`, `clippy -D warnings`, PTX/SASS 0-float audit, and Wick `receipts.sha256` pass with `DEV-test` untouched and `0` pods owned by `antigravity`.
2. `REPORT.md` (all 7 canonical `## ` headings, negatives first), `m4-serving.md`, `written-state-count-model.md`, fresh `pro` exact-head review (`VERDICT: APPROVE`), and CI merge to `main` pass.
