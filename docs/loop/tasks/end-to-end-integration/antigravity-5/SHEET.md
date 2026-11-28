# Pre-Registration Sheet: Gate 1 `UORBFY03` Capacity, Stereographic Selectivity & Loss Breakthrough (`antigravity-5`)

- **Lab:** `antigravity`
- **Item / Issue / Phase:** `end-to-end-integration` (`#4064`, referencing `#2030`), **Gate 1 `UORBFY03` Architecture Breakthrough** (unlocked by `antigravity-4` PR `#4339` verdict `DIAG_CAPACITY_OR_LOSS_BOTTLENECK`, which proved Pure-Int Q15 Lazy-Sparse AdamW matches Diagnostic F32 within `+0.0026 BPB` on `chat-v0-p2` and beats F32 by `-0.0016 BPB` on `E1_all`, isolating the `2.5342 BPB` floor to backbone capacity, non-selective `S³` decay, narrow frozen PKM keys, and `64×64` readout routing).
- **Objective:** Test the upgraded pure-integer `UORBFY03` geometry targeting all 5 X-rayed bottlenecks simultaneously across two complementary scale/loss configurations on Gate 0's `train-100m`:
  1. **`4×` parallel `O(D log D)` Butterfly FFN** (`ffn_branches = 4`) + deeper/wider scale (`D=256, L=8` and `D=512, L=6`).
  2. **Pure-`S¹` rational stereographic selective `S³` gate** (`g(z) = z^2 / (1 + z^2) ∈ [0, 1)` on `rot_y[b+0] + gate_bias_q15[q]`, zero `exp`/sigmoid/squashing) + **4-tap causal depthwise `O(D)` shift-mix** + **widened `±8.0` `S³` tangent clamp** (`0%` dead clipped rotors).
  3. **Wide `64`-dim (`32 + 32`) learned Sparse PKM keys** + tiny non-zero `±256` `pkm.values` init for live Step-1 key gradients.
  4. **Dedicated `head_proj: ButterflyLinear`** + **`TRAIN`-only descending frequency-sorted `vocab_perm`** (minimizing coarse routing entropy) + **class-balanced Hui-Belkin (`k = 32`, `k = 64`)** + gentler `wd_shift = 19` (`lambda = 0.0625`).
  5. **Scalar-reciprocal exact integer RMSNorm (`inv_rms_q40`, `dot_over_denom3_q40`) & `unit_quat_wide_q15` (`inv_norm_q40`):** replaces per-element software divisions with `1` scalar division per vector (`0` float, `0` PTX `div`, `0` MUFU SASS).
- **Charter §3 & D85 Constraints:**
  - **100% Pure-Integer Default Path ONLY (no F32 kernels):** `0` PTX float/div/MMA, `0` arithmetic SASS float/MMA (excluding logged Blackwell `sm_120` `HFMA2 -RZ, RZ,` constant-load aliases), `0` normalized exponential/squashing, `0` dense matmul, single persistent `i16` Q15 `UORBFY03 v1` state, `0`-diff CUDA = CPU = Apple M1 = WASM32 parity.
  - **Zero `DEV-test` peeks:** `dev-test.*` is explicitly refused and never copied to the pod.

---

## 1. Hardware, Compute Budget & Safety

- **GPU:** `1` Vast.ai `2 × RTX 5090` datacenter pod (`~$1.95/h`, strictly inside global caps `<= 4 pods`, `<= $8.00/h`; fallback `1 × RTX 5090` sequential if `2×` unavailable). Wick RTX 4090 is left untouched.
- **Parallel Execution (`~35–50 min` wall for both Epoch-1 arms in parallel, `~$1.60` base; up to `~1.8 h` / `<= $3.60` max if Epoch 2–3 auto-SFT triggers):**
  - `GPU 0`: Arm 1 (`v3_d256_l8_ffn4_k32`, `9.72M` params, `19.45 MB` `.uorbfy03`) + cold unwrapped `E2–E4` + replay fixture.
  - `GPU 1`: Arm 2 (`v3_d512_l6_ffn4_k64`, `15.13M` params, `30.27 MB` `.uorbfy03`) + cold unwrapped `E2–E4` + replay fixture.
- **Teardown (`L-004`):** Archive all evidence, `.uorbfy03` checkpoints, PTX/SASS audits, and `receipts.sha256` to Wick ZFS (`/mnt/plexpool/uor-r4-corpus/runs/20261128-antigravity-gate1-v3-r1/`), verify `sha256sum -c`, and immediately `release` + `down` the pod.

---

## 2. Frozen Inputs (Fail-Closed SHA-256 Pins — Identical to `antigravity-3`/`antigravity-4`)

| Role | Exact Bytes | Exact SHA-256 |
|---|---:|---|
| `tokenizer.json` | `265,393` | `a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377` |
| `train-100m.tokens.u16` (`71,876` recs, `100,472,502` tok) | `200,945,068` | `c804c3b084a49d11f2bfcd2a5cf741862df29b42e3442ccb609305e049aba590` |
| `train-100m.mask.u8` (`65,499,244` response tgts) | `100,472,502` | `9e8d0cb6302ea890be5847c76133a731e7b4a080cfa096049cf37323072ae1d5` |
| `dev-tune.tokens.u16` (`2,670` recs, `3,805,298` tok) | `7,610,660` | `99b714a67415d92e42750577110d9ce405625d93adabbd906739bbfca60d9c78` |
| `dev-tune.{mask,novel,absent}.u8` | `3,805,298` ea | `643fccbb...` / `6239ac46...` / `8860f193...` |
| `id-bytes.u8` | `4,096` | `befb22f6908751ae522a80336afa584eb2c24e36a10b90ad3105e39271a48ac9` |
| `chat-v0-p2/heldout.u16` (`512`w × `384` = `196,608` tgts) | `12,389,242` | `e5f400b0e676791fd0458ce810163c5d2542adda80dccc8282f37d9ca8904cef` |
| `m2-fresh256` (`inputs.json` / `labels.json`) | `524,850` / `68,000` | `1de2fc28...` / `3b5c5477...` |

---

## 3. Experimental Arms (`UORBFY03 v1`, Pure-Int Lazy-Sparse AdamW, `seed = 20261128`)

Both arms train on the exact same shuffled `train-100m` Epoch 1 stream (`seq_len = 256`, `batch_size = 128`, `3,333` steps = `100.5M` tokens, `200`-step linear warmup, linear decay `lr_q8: 4,096 -> 512`, `beta1 = 7/8`, `beta2 = 63/64`, `wd_shift = 19` (`lambda = 0.0625`), `eps = 16`, `no_wd` on `shift_mix`/`gate_bias`, lazy sparse skip on `embeddings`/`pkm.values`, Q12 anchor decay toward `4,096`, `TRAIN`-only descending frequency-sorted `vocab_perm`, `learn_pkm_keys = true`, `pkm_sub_keys = 64`, `pkm_key_dim = 32`, `ffn_branches = 4`):

| Arm | Width `D` | Depth `L` | FFN Branches | PKM Key Dim | Loss `k` | Backbone Params | Total Params | `.uorbfy03` Size | Hypothesis Tested |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`antigravity-4` Anchor** | `256` | `4` | `1` | `8 + 8` (frozen) | `8` | `57K` | `5.32M` | `10.65 MB` | Baseline: `chat = 2.5342`, `E1_all = 3.0009`, `4,146 tok/s` M1 |
| **Arm 1: `v3_d256_l8_ffn4_k32`** | **`256`** | **`8`** | **`4`** | **`32 + 32`** (learned) | **`32`** | **`236K`** (`4.1×`) | **`9.72M`** | **`19.45 MB`** | Deep `L=8` + high M1 serving headroom (`~1.6–2.4K tok/s`) + `k=32` |
| **Arm 2: `v3_d512_l6_ffn4_k64`** | **`512`** | **`6`** | **`4`** | **`32 + 32`** (learned) | **`64`** | **`393K`** (`6.9×`) | **`15.13M`** | **`30.27 MB`** | Wide `D=512` state/embeddings (`2×`) + `1:1` balanced `k=64` |

**Pre-registered Auto-Continuation:** If either arm breaks **`chat-v0-p2 < 2,000,000` (`< 2.00 BPB`)** in Epoch 1 with monotone descent (`win_loss_last <= win_loss_2000`), `butterfly-v3` automatically runs **Epochs 2–3 Protocol-2 SFT (`mask == 1`)** (`lr_q8: 1,024 -> 128`) on that arm before cold `UORBFY03` reload, unwrapped `E2–E4`, and `*_replay_fixture.json` emission for Apple M1 `bench-cpu`.

---

## 4. Execution Gates & Decision Table

1. **Gate 1 — Pure-Int AOT PTX/SASS Audit (`butterfly_kernels_v3.sm_<CC>.cubin`):** `0` PTX float, `0` PTX `div.`/`rem.`, `0` PTX/SASS MMA, `0` arithmetic SASS float (excluding logged `sm_120` `HFMA2 -RZ, RZ,` constant-load aliases).
2. **Gate 2 — Finite Pure-Int GPU-vs-CPU Exact Parity (`0` diffs across 4 configs):** Verify **`0` raw-word diffs** between CUDA `butterfly_kernels_v3.cu` and Rust `butterfly_rotor_v3.rs` across:
   - Config 1: `D = 32, L = 2, ffn_branches = 4, pkm_sub_keys = 4, pkm_key_dim = 8, k = 32`, `freq_sorted` perm, `learn_pkm_keys = true`, `3` multi-step `apply_adamw_step` updates + `UORBFY03` reload.
   - Config 2: `D = 256, L = 2, ffn_branches = 4, pkm_sub_keys = 8, pkm_key_dim = 32, k = 32`, `freq_sorted` perm, `learn_pkm_keys = true`, `2` AdamW steps.
   - Config 3: `D = 512, L = 2, ffn_branches = 4, pkm_sub_keys = 8, pkm_key_dim = 32, k = 64`, `freq_sorted` perm, `learn_pkm_keys = true`, `2` AdamW steps.
   - Config 4: `D = 32, L = 2, ffn_branches = 2, pkm_sub_keys = 4, pkm_key_dim = 8, Sparsemax64`, identity perm, `learn_pkm_keys = false`, `2` AdamW steps + `reset_on_bos = true` eval check (`carry` + `shift_hist` reset).
   Require `0` diffs across every loss, `i64` grad word, `i16` weight, `i32` `m`, `u64` `v`, Q48 target mass, and greedy token ID.
3. **Decision Table (Ordered, Mutually Exclusive, 100% Exhaustive — First Matching Rule Wins; Epoch 1 `win_loss_last <= win_loss_2000` checks monotone pretrain descent and BPB/E1–E4/M1 metrics evaluate each arm's final cold-reloaded best `UORBFY03` artifact):**
   - **(a) `INSTRUMENT_FAILED`:** Any Gate 1 AOT PTX/SASS check, Gate 2 finite `0`-diff parity check, or Apple M1 `0`-diff CUDA=M1 replay check fails.
   - **(b) `GATE1_CLEARED`:** At least one `UORBFY03` arm clears the full Gate 1 bar (`chat-v0-p2 < 1,600,000` micro-BPB, `E1_nov < 1,717,000`, `shuf_ci95_lo > 0`, unwrapped `E2` pass & `E3 >= 52/128` pairs, Apple M1 single-core `>= 1,000 tok/s`, artifact `< 50 MB`, `0` CUDA=M1 diffs). Unlocks immediate **Gate 2 (`737.5M` `TRAIN` + sealed `DEV-test`)**.
   - **(c) `V3_BREAKTHROUGH_NEEDS_SFT_OR_SCALE`:** Rule (b) does not match, AND at least one arm breaks **`chat-v0-p2 < 2,000,000` (`< 2.00 BPB`)** with Epoch 1 monotone descent (`win_loss_last <= win_loss_2000`), Apple M1 single-core `>= 1,000 tok/s`, and artifact `< 50 MB`. Confirms `UORBFY03` broke the `2.53 BPB` ceiling by `> 0.53 BPB` -> unlocks targeted `antigravity-6` (multi-epoch pretrain / SFT / scale-up to clear `< 1.60 BPB` & unwrapped `E2–E4`).
   - **(d) `V3_PARTIAL_GAIN_NEEDS_LOSS_OR_HEAD_SURGERY`:** Rules (b)–(c) do not match, AND at least one arm achieves **`chat-v0-p2 <= 2,384,184`** (`>= 150,000` micro-BPB = `>= 0.15 BPB` improvement over `antigravity-4`'s `2,534,184`) with Epoch 1 monotone descent (`win_loss_last <= win_loss_2000`), Apple M1 single-core `>= 1,000 tok/s`, and artifact `< 50 MB`. (Note: since (c) did not match, any qualifying arm here sits strictly in `[2,000,000, 2,384,184]`.) Isolates remaining gap to `64×64` square-loss head/readout -> unlocks lawful non-square geometric loss (rank/margin or rational-log surrogate without `exp`) in `antigravity-6`.
   - **(e) `V3_PLATEAU_AT_RESCALED_SQUARE`:** All remaining Gate-passing outcomes (neither arm achieves `<= 2,384,184` micro-BPB with Epoch 1 monotone descent, `>= 1,000 tok/s` Apple M1 single-core, and `< 50 MB`, OR an arm reaches `<= 2,384,184` / `< 2.00 BPB` only by falling below `1,000 tok/s` on Apple M1, exceeding `50 MB`, or failing Epoch 1 monotone descent). Closes this `UORBFY03` rescaled-square configuration and triggers targeted loss-geometry / throughput redesign in `antigravity-6`.
