# Pre-Registration Sheet: Gate 1 `UORBFY03` Zero-Alloc Serving, `Entmax15` Loss & `L=8` Breakthrough (`antigravity-6`)

Lab: antigravity. Item: [End-to-end integration](../../../../roadmap/items/end-to-end-integration.md). References #4064 and #2030.

## 1. Whole-Project Context & Why This Exact Run Now

In `antigravity-5` (PR `#4347`), `UORBFY03 v1` (`D=512, L=6, ffn4, k=64`, `30.27 MB`) delivered our largest pure-integer leap (`chat-v0-p2 = 2.2980 BPB`, `-0.2362 BPB`; `DEV-tune E1_all = 2.6166 BPB`, `-0.3843 BPB`, `316‰` top-1; `shuf_delta = +8,693`, `3.38×`), with `0` PTX/SASS float/div/MMA and `0` RTX 5090 `sm_120` == Apple M1 replay diffs, while honestly classifying `V3_PLATEAU_AT_RESCALED_SQUARE` (Rule (e)) because it remained `>= 2.00 BPB` and fell below `1,000 tok/s` on single-core M1 (`696 tok/s`).

Forensic X-ray of `antigravity-5`'s saved `.uorbfy03` artifacts on `chat-v0-p2` isolated four concrete, quantified bottlenecks:
1. **Serving implementation overhead (already solved locally, `0` diffs):** `step()` allocated `> 300` training-cache `Vec`s/token and ran `331K` runtime `givens_beta_q15` divisions/token. `ButterflyV3Prepared` (zero steady-state allocs, load-time precomputed `beta_q15`, exact Euclidean-norm `SAFE_NORM2 = (ACT_CLAMP >> 1)^2` fast path, unrolled `s = 0..2`, `u64::isqrt`, `stereographic_gate_forward_q15`) lifts trained `antigravity-5` Arm 2 from `696 -> 1,699 tok/s` (`0` diffs) and `D=512, L=8, ffn4, sk=32` (`13.73 MB`) to **`1,329 tok/s`** single-core Apple M1.
2. **Backbone vs PKM imbalance & tail-cell starvation:** In `antigravity-5` Arm 2, `12.58M` (`83.2%`) params sat in `64×64` PKM values (`-0.034 BPB` ablation) while only `0.39M` (`2.6%`) sat in the 6-layer Butterfly-Rotor + 4× FFN trunk (`-2.165 BPB` ablation); coarse cells `32..63` barely moved from random init (`emb_rms = 2,390` vs `2,366` init) and their centroids shrank to `981` RMS under negative MSE push. Right-sizing PKM to `32×32 = 1,024` slots frees `18.9 MB`, gives `4×` denser slot training, and funds `L = 8` (`+33%` trunk depth) inside a **`13.73 MB`** artifact (`> 1,300 tok/s` M1).
3. **Independent Hui-Belkin loss vs Coupled Shift-Invariant `1.5-Entmax`:** On `Arm 2`, `21.0%` of coarse targets and `18.4%` of fine targets slammed into the `0.999` score cap while 63 negative classes drifted to `+0.16..+0.19`. Pure-integer Tsallis `α = 1.5` Fenchel-Young (`LossKind::Entmax15`, `p_i = [(z_i - tau)/2]_+^2`, `dz_i = 8*(p_i - 1[i=t])`, 20-step shift bisection, `0` `exp`/`log`/`softmax`/div, explicitly approved in `expert_panel_architecture_review.md`) is strictly zero-sum, shift-invariant, protects rare cells (`dz_j = 0` outside active support), and achieves `+0.0002 bits` excess KL across sharp, Zipf, and 95%-peak distributions in Q15.
4. **Under-trained 1-epoch horizon & missing SFT `EOS` supervision:** `Arm 2` was still descending at step `3,333` (`1` epoch, low `lr_q8 = 4,096`) and never ran answer-only SFT (`gold_eos = 0/82`).

## 2. Exact Controlled Design (`1`-Variable Loss Head-to-Head, Zero Float, Zero Div in CUDA)

- **Shared Architecture (both arms identical, `UORBFY03 v1`, `6,863,360` params, `13,734,980 B` = `13.73 MB`):**
  `D = 512`, `L = 8`, `ffn_branches = 4`, `pkm_sub_keys = 32` (`1,024` slots/layer), `pkm_key_dim = 32`, `freq_sorted_perm = true` (`TRAIN` only), `learn_pkm_keys = true`, `batch = 192` (`> 170` RTX 5090 SMs), `seq = 256`, `2` Pretrain epochs on `train-100m` (`200M` tok, `200`-step linear warmup `512 -> 8,192`, smooth linear decay `8,192 -> 512`) + `1` mandatory Answer-Only SFT epoch (`mask == 1`, `2,048 -> 256`), `wd_shift = 20`, `eps = 16`, `seed = 20261128`.
- **Arm 1 (GPU 0): `v3b_d512_l8_ffn4_sk32_entmax15`** — **`LossKind::Entmax15`** (`code = 3, param = 0`).
- **Arm 2 (GPU 1): `v3b_d512_l8_ffn4_sk32_hb64`** — **`LossKind::HuiBelkinRescaledSquare { pos_weight_k: 64 }`** (`code = 1, param = 64`) — exact 1-variable control.
- **Pinned Inputs (unchanged from `antigravity-5`; `dev-test.*` NEVER copied or opened):**
  - `train-100m.{tokens.u16,mask.u8}`, `dev-tune.{tokens.u16,mask.u8,novel.u8,absent.u8}`, `id-bytes.u8`, `chat-v0-p2/heldout.u16` (`512`w, `196,608` targets), `tokenizer.json`, `m2-fresh256/{inputs,labels}.json`.

## 3. Gates & Compute Budget

- **Gate 1 (AOT `sm_<CC>` Cubin Audit):** `0` PTX float/div/MMA, `0` arithmetic SASS float/MMA (`sm_120` `HFMA2 -RZ, RZ,` mov aliases logged separately), `3/3` kernels present. Fail-closed.
- **Gate 2 (Finite GPU-vs-CPU Exact Parity):** `0` diffs across all `4` configs (`D=32/256/512`, `ffn4`, `sk=4/8/32`, `kd=8/32`, `Entmax15` & `HB k=64`, `freq_sorted`, `reset_on_bos`, multi-step Pure-Int AdamW, `UORBFY03 v1` reload, and `ButterflyV3Prepared` `0`-diff parity). Fail-closed.
- **Gate 3 & Serving Bench:** Parallel `2 × RTX 5090` Vast.ai pod (`<= $2.50/h`, within `<= 4` pods / `<= $8/h` cap, expected `~2.5–3.0 h`, `~$4.50–$6.00`). Archive all receipts to Wick ZFS (`receipts.sha256 OK`) and **delete pod immediately upon completion (`L-004`)**. Run Apple M1 `bench-cpu` (`ButterflyV3Prepared`, `7`-run median, `/usr/bin/time -l`, `2,048` ctx, `128` gen, fail-closed `0`-diff vs RTX 5090 `sm_120` fixture) on both new artifacts AND re-bench `antigravity-5` Arm 2 (`v3_d512_l6_ffn4_k64.uorbfy03`).

## 4. Pre-Registered Decision Rules (Exhaustive, Mutually Exclusive, Zero Goalpost Shifting)

Every rule below is evaluated on a **single saved `.uorbfy03` artifact** (`best_model`, selected strictly by lowest `DEV-tune E1_all` across Ep 1–3; `unwrapped_eval_sft_cold` is additionally logged as diagnostic telemetry when `best_epoch != 3` and NEVER mixed across artifacts to satisfy a gate):

- **(a) `INSTRUMENT_FAILURE`:** Gate 1 fails, Gate 2 has `> 0` diffs, any live Bar A CPU spot-check diff `!= 0`, or any Apple M1 `bench-cpu` CUDA=M1 replay diff `!= 0`. Stop and repair instrument before any claim.
- **(b) `GATE1_FULL_PASS`:** Instrument passes AND one single `< 50 MB` saved `.uorbfy03` artifact (`best_model`) clears ALL FIVE full Gate 1 bars:
  1. `chat-v0-p2 <= 1,600,000` (`<= 1.60 BPB`),
  2. `DEV-tune E1_nov < 1,717,007` (strictly beating `3.99 GB p4.d11`) with `shuf_delta CI_lo > 0`,
  3. Unwrapped `E2` `gate_e2_pass == true` on that same `best_model`,
  4. Unwrapped `E3` (`m2-fresh256` active) `>= 52/128` pairs on that same `best_model`, AND
  5. Single-core Apple M1 `>= 1,000 tok/s` (`0` diffs) on that same `best_model`.
  *Unlocks:* Gate 2 (`737.5M` `TRAIN` full-scale run + sealed `DEV-test` qualification).
- **(c) `GATE1_BPB_AND_SPEED_BREAKTHROUGH`:** Rule (b) not met, instrument passes, AND one single `< 50 MB` saved `.uorbfy03` artifact (`best_model`) clears single-core Apple M1 `>= 1,000 tok/s` (`0` diffs), `chat-v0-p2 <= 1,900,000` (`<= 1.90 BPB`, `>= -0.398 BPB` below `antigravity-5`), and `shuf_delta CI_lo > 0`.
  *Unlocks:* Scaling the winning loss (`Entmax15` vs `HB64`) directly to full `737.5M` `TRAIN` + wider/deeper `O(D log D)` trunk to clear `< 1.4676 BPB` and unwrapped `E2–E4` (sealed `DEV-test` remains untouched).
- **(d) `LOSS_OR_HORIZON_PROGRESS_BELOW_BAR`:** Rules (b)–(c) not met, instrument passes, AND one single `< 50 MB` saved `.uorbfy03` artifact (`best_model`) clears single-core Apple M1 `>= 1,000 tok/s` (`0` diffs), `shuf_delta CI_lo > 0`, AND at least one of:
  1. `chat-v0-p2 < 2,277,858` (strictly below `antigravity-5` Arm 2's 95% CI lower bound `2,277,858`), OR
  2. `DEV-tune E1_all <= 2,566,579` (`>= 50,000` micro-BPB / `>= 0.050 BPB` strictly below `antigravity-5` Arm 2's `2,616,579`) with `chat-v0-p2 <= 2,298,030`, OR
  3. Non-degenerate E2 breakthrough on that same `best_model`: `natural_eos_permille >= 100‰`, `premature_eos == 0`, `distinct_first_nonspace_tokens >= 10`, `fourgram_loop_permille < 500‰`, AND `chat-v0-p2 <= 2,298,030`.
  *Unlocks:* Winning loss locked in (`Entmax15` vs `HB64`); targeted capacity/supervision upgrade (`ReGLU` bilinear butterfly FFN / `8`-tap shift-mix / `737.5M` `TRAIN`) in `antigravity-7` (sealed `DEV-test` remains untouched).
- **(e) `V3B_PLATEAU`:** Instrument passes, but Rules (b)–(d) are not met on any single saved artifact (no statistically clear improvement over `antigravity-5` Arm 2 on `chat-v0-p2`/`E1_all`/non-degenerate E2, OR single-core Apple M1 `< 1,000 tok/s`).
  *Unlocks:* Pivot note (`CHARTER §4.4`) & forensic X-ray comparing `Entmax15` vs `HB64` before any further GPU spend.
