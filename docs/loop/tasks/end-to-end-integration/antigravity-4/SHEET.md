# Pre-Registration Sheet: Gate 1 `2×2` Diagnostic & Pure-Integer AdamW Repair (`antigravity-4`)

- **Lab:** `antigravity`
- **Item / Issue / Phase:** `end-to-end-integration` (`#4064`, referencing `#2030`), **Gate 1 Diagnostic & Repair** (unlocked by `antigravity-3` PR `#4331` verdict `GATE1_DIAGNOSTIC_NEEDED`).
- **Objective:** Decompose the `antigravity-3` `r4` plateau (`hb_l4_d256 = 2.6873 BPB`, universal step-`2,000` turnaround across all 3 Sign-Lion arms) into three orthogonal factors — **(1) Optimizer** (Sign-Lion vs Pure-Integer Lazy-Sparse AdamW), **(2) Head/Loss/PKM geometry** (`k=8` + identity `vocab_perm` + frozen PKM keys vs `k=1` calibrated linear Brier + `TRAIN`-balanced `vocab_perm` + learned PKM keys), and **(3) Numeric Precision** (Persistent Pure-Integer Q15 `UORBFY01 v2` vs `CHARTER §3.7` Diagnostic-Only F32 Twin) — in one fast, low-cost `2×2` factorial on Gate 0's `train-100m`.
- **Charter §3 & D85 Constraints:**
  - **Pure-Integer Arms (Arms 1 & 2, default path):** Zero floats (`0` PTX float/div/MMA, `0` arithmetic SASS float/MMA outside logged Blackwell `sm_120` `HFMA2 -RZ, RZ,` constant-load aliases), zero softmax/squashing, zero dense matmul, single persistent `i16` Q15 `UORBFY01 v2` state, `0`-diff CUDA = CPU = Apple M1 parity.
  - **Diagnostic F32 Twin Arms (Arms 3 & 4, `CHARTER §3.7` diagnostic only):** Strictly outside `crates/`, zero softmax/log-softmax/cross-entropy/squashing (§3.2), zero dense matmul/GEMM/MMA (§3.3), never serialized as a served artifact, used solely to measure the exact Q15-vs-F32 precision gap at matched hyperparameters.
  - **Zero `DEV-test` peeks:** `dev-test.*` is explicitly refused and never copied to the pod.

---

## 1. Hardware, Compute Budget & Safety

- **GPU:** `1` Vast.ai `2 × RTX 5090` datacenter pod (`~$1.95/h`, strictly inside global caps `<= 4 pods`, `<= $8.00/h`; fallback `1 × RTX 5090` if `2×` unavailable). Wick RTX 4090 is occupied and left untouched.
- **Parallel Execution (`~45 min` wall for 4 Epoch-1 arms, `~$1.50` base; up to `~1.5 h` / `<= $3.50` max if Epoch 2–3 auto-continuation triggers):**
  - `GPU 0`: Pure-Integer Arm 1 (`int_adamw_hb8`) -> Pure-Integer Arm 2 (`int_adamw_brier1_bal`) -> optional Epochs 2–3 SFT + unwrapped `E2–E4` on winning pure-integer arm.
  - `GPU 1`: Diagnostic F32 Arm 3 (`diag_f32_adamw_hb8`) -> Diagnostic F32 Arm 4 (`diag_f32_adamw_brier1_bal`).
- **Teardown (`L-004`):** Archive all evidence, pure-integer `.uorbfy01` checkpoints, PTX/SASS audits, and `receipts.sha256` to Wick ZFS (`/mnt/plexpool/uor-r4-corpus/runs/20261128-antigravity-gate1-diag-r1/`), verify `sha256sum -c`, and immediately `release` + `down` the pod.

---

## 2. Frozen Inputs (Fail-Closed SHA-256 Pins — Identical to `antigravity-3`)

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

## 3. `2×2` Controlled Factorial Design (`L = 4, D = 256, pkm_sub_keys = 64, seed = 20261127`)

All 4 arms train on the **exact same shuffled `train-100m` Epoch 1 stream** (`seq_len = 256`, `batch_size = 128`, `3,333` steps = `100.5M` tokens, `200`-step linear warmup, linear decay from `lr = 4.88e-4` (`lr_q8 = 4,096`) to `lr = 6.10e-5` (`lr_q8 = 512`), `beta1 = 7/8 = 0.875`, `beta2 = 63/64 = 0.984375`, `wd_shift = 17` (`lambda = 0.25` in both Q15 and Q12, matched `wd = 0.25` in F32), `eps = 16` (`4.8828e-4` in both Q15 and F32), gradient clamp `±4,096.0` (`±2^27` Q15), lazy sparse update on `embeddings`/`pkm.values`, Q12 norm anchor decay toward `1.0` = `4,096`):

| Arm | Numeric Space | Optimizer | Loss `k` | `vocab_perm` | PKM Keys | Single-Variable Question |
|---|---|---|---|---|---|---|
| **`r4` Anchor** (`#4331`) | Pure-Int Q15 (`UORBFY01 v2`) | Sign-Lion (`lr_q8 = 2048 -> 1025`) | `k = 8` | Identity | Frozen | Baseline: `chat = 2.6873`, `E1_all = 3.2269`, step-`2000` turnaround |
| **Arm 1: `int_adamw_hb8`** | **Pure-Int Q15** (`UORBFY01 v2`) | **Pure-Int Lazy-Sparse AdamW** | `k = 8` | Identity | Frozen | **Optimizer effect:** Pure-Int AdamW vs Sign-Lion (`Arm 1` vs `r4`) |
| **Arm 2: `int_adamw_brier1_bal`** | **Pure-Int Q15** (`UORBFY01 v2`) | **Pure-Int Lazy-Sparse AdamW** | **`k = 1` (linear Brier)** | **`TRAIN`-balanced** | **Learned** | **Head/Loss/PKM effect in Pure-Int:** `Arm 2` vs `Arm 1` |
| **Arm 3: `diag_f32_adamw_hb8`** | **Diagnostic F32** (§3.7, `0` softmax/matmul) | **F32 Lazy-Sparse AdamW** | `k = 8` | Identity | Frozen | **Q15-vs-F32 precision gap:** `Arm 3` vs `Arm 1` |
| **Arm 4: `diag_f32_adamw_brier1_bal`** | **Diagnostic F32** (§3.7, `0` softmax/matmul) | **F32 Lazy-Sparse AdamW** | **`k = 1` (linear Brier)** | **`TRAIN`-balanced** | **Learned** | **F32 ceiling of calibrated Brier + balanced perm + learned PKM:** `Arm 4` vs `Arm 2` & `Arm 3` |

**Pre-registered Auto-Continuation:** Both pure-integer arms (`Arm 1` and `Arm 2`) on `GPU 0` cold-reload their best `UORBFY01 v2` checkpoint for unwrapped `E2–E4` and emit `*_replay_fixture.json` for Apple M1 `bench-cpu` (`0`-diff CUDA=M1). If either pure-integer arm breaks **`chat-v0-p2 < 2,000,000` (`< 2.00 BPB`)** in Epoch 1 with monotone descent (`win_loss_last <= win_loss_2000`), `butterfly-diag` automatically runs **Epochs 2–3 Protocol-2 SFT (`mask == 1`)** (`lr_q8: 1,024 -> 128`) on it before final cold `E2–E4`!

---

## 4. Execution Gates & Decision Table

1. **Gate 1a — Pure-Int AOT PTX/SASS Audit (`butterfly_kernels.cu`):** `0` PTX float, `0` PTX `div.`/`rem.`, `0` PTX/SASS MMA, `0` arithmetic SASS float (excluding logged `sm_120` `HFMA2 -RZ, RZ,` aliases).
2. **Gate 1b — Diagnostic F32 AOT PTX/SASS Audit (`diag_f32_kernels.cu`):** `0` softmax/exp/log/squashing instructions (`ex2`/`lg2` strictly `0`!) and `0` PTX/SASS MMA.
3. **Gate 2 — Finite Pure-Int GPU-vs-CPU Exact Parity & Diag F32 Adjoint Check:** `4` pure-integer configs (`D ∈ {32, 256}`, `L ∈ {2, 4}`, `k ∈ {1, 8}`, `learn_pkm_keys ∈ {false, true}`, balanced & identity `vocab_perm`, `3` multi-step `apply_adamw_step` updates + `UORBFY01 v2` reload): require **`0` diffs** across every loss, `i64` grad word, `i16` weight, `i32` `m`, `u64` `v`, Q48 target mass, and greedy token ID. Also verify F32 twin passes central finite-difference gradient check across all active slices (`cosine >= 0.995`) and `> 50%` 20-step finite descent with zero NaNs on both `(k=8, frozen)` and `(k=1, learn_pkm)`.
4. **Decision Table (Ordered, Mutually Exclusive, 100% Exhaustive over Matched Twins `(Arm 1, Arm 3)` and `(Arm 2, Arm 4)` on Epoch 1 `chat-v0-p2` `stream_basis_micro_bpb`):**
   Evaluate rules (a)–(d) strictly in order; the first matching rule sets the verdict:
   - **(a) `INSTRUMENT_FAILED`:** Any Gate 1a, Gate 1b, Gate 2, or Apple M1 `0`-diff replay check fails.
   - **(b) `DIAG_INT_ADAMW_BREAKTHROUGH`:** At least one pure-integer arm (`Arm 1` or `Arm 2`) clears **`< 2.00 BPB`** (`< 2,000,000` micro-BPB) on Epoch 1 `chat-v0-p2` with monotone descent (`win_loss_last <= win_loss_2000`), AND its matched F32 twin (`Arm 3` for `Arm 1`, `Arm 4` for `Arm 2`) does not beat it by `> 0.15 BPB` (`Arm_int - Arm_f32 <= 0.15 BPB`). Unlocks immediate `L=8` & `L=12` pure-integer scaling in `antigravity-5` to clear Gate 1 (`< 1.60` / `< 1.4676` BPB + unwrapped E2–E4).
   - **(c) `DIAG_PRECISION_GAP_ISOLATED`:** Rule (b) does not match, AND at least one F32 arm (`Arm 3` or `Arm 4`) clears **`< 2.00 BPB`** on Epoch 1 `chat-v0-p2` AND beats its matched pure-integer twin (`Arm 1` for `Arm 3`, `Arm 2` for `Arm 4`) by **`> 0.15 BPB`** (`Arm_int - Arm_f32 > 0.15 BPB`) — OR a pure-integer arm clears `< 2.00 BPB` with `Arm_int - Arm_f32 <= 0.15 BPB` but exhibits a non-monotone turnaround (`win_loss_last > win_loss_2000`) while its F32 twin descends monotonically. Isolates the remaining barrier to Q15 forward/backward/step precision -> unlocks Q23/Q15 higher-dynamic-range integer activations (`i32` Q23 carry/residual + `i16` Q15 weights) in `antigravity-5`.
   - **(d) `DIAG_CAPACITY_OR_LOSS_BOTTLENECK`:** All remaining Gate-passing outcomes (specifically: both `Arm 3` and `Arm 4` are `>= 2.00 BPB`, OR an F32 arm is `< 2.00 BPB` without beating its pure-integer twin by `> 0.15 BPB` and without monotone pure-int breakthrough). Proves `D=256, L=4` (`57K` butterfly params + `64×64` square-loss head) is architecture/loss-limited rather than Q15-precision-limited -> unlocks `4×` butterfly FFN expansion (`O(D log D)`), wider `D=512`, and lawful non-square geometric losses (e.g., polynomial/spherical margin or piecewise-linear rational log-surrogate without `exp`) before any further `D=256` integer tuning!
