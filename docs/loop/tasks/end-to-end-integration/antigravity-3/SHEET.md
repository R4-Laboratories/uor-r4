# Pre-Registration Sheet: Gate 1 Pure-Integer Butterfly-Rotor Pilot (`antigravity-3`)

- **Lab:** `antigravity`
- **Item / Issue / Phase:** `end-to-end-integration` (`#4064`, referencing `#2030`), **Gate 1 PR 2 of 2** (owner-approved six-expert architectural review, 27 Nov 2026).
- **Prerequisites:**
  - Gate 0 (`antigravity-1`, PR `#4316`, merged at `6c78eeb0c`): Clean `DialogueProtocol::literal_roles_v2` D87 answer-only corpus (`train-100m`, `dev-tune`, sealed `dev-test`) and unwrapped `E1–E4` frozen `p4.d11` baseline.
  - Gate 1 PR 1 (`antigravity-2`, PR `#4317`): Pure-integer Reversible Lifting-Givens Butterfly + Non-Abelian `S³` Holographic Quaternion Rotor + Sparse PKM + Factored `64×64` Hierarchical Readout + Exact Integer Adjoint & Lion engine (`UORBFY01`) in `crates/uor-r4-native-vector/src/butterfly_rotor.rs`.
- **Charter §3 & D85 Constraints:** Zero floating point (`f16`/`bf16`/`f32`/`f64`, zero PTX/SASS float ops), zero hardware division in CUDA (exact binary restoring integer `div_*_nodiv` & `isqrt_*_nodiv`), zero MMA/dense GEMM, zero normalized exponential/squashing (`exp`/`log`/`sigmoid`/`tanh`/`GELU`/`SiLU`), single persistent `i16` Q15 state updated directly on GPU and serialized losslessly as `UORBFY01`. **Zero `DEV-test` peeks** during Gate 1 training, stopping, or selection (`DEV-test` remains strictly sealed).

---

## 1. Hardware, Compute Budget & Safety

- **Primary GPU:** Wick RTX 4090 (`sm_89`, `24 GB` VRAM, `$0.00` pod spend) if idle (`flock` non-blocking check, zero disturbance to any active lab job).
- **Pre-registered Fallback GPU (if Wick is occupied):** 1 Vast.ai datacenter RTX 5090 (`sm_120`, `32 GB` VRAM, `~$0.52/h` via `scripts/pod/uor-pod`, strictly inside caps `<= 4 pods`, `<= $8.00/h`), leased exclusively to `antigravity`, torn down immediately after saving all artifacts & receipts to Wick ZFS.
- **Hard Compute Cap:** `< 1` GPU-day (`<= 6` hours wall / `<= $4` max cloud spend if Vast fallback is used). All `3` controlled arms at `D = 256` on `train-100m` (`3` epochs × `100.5M` tokens) are estimated at `~ 1–2` hours total.
- **Native Serving Bench:** Apple M1 (`16 GB` laptop) native Rust `bench-cpu` (`2,048`-context prefill + `128`-token warm decode, `7` runs median) on cold-reloaded trained `UORBFY01` artifact, plus `0`-diff CUDA=M1 replay verification.

---

## 2. Frozen Inputs (Fail-Closed SHA-256 Pins)

Every input file is verified against its exact SHA-256 before compute; any mismatch halts with non-zero exit:

| Role | Path on Wick ZFS | Exact Bytes | Exact SHA-256 |
|---|---|---:|---|
| Bound tokenizer | `/home/callard/uor-codex/20261118-diagnostics/inputs/tokenizer.json` | `265,393` | `a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377` |
| Gate 0 `train-100m` tokens (`71,876` recs, `100,472,502` tok) | `/mnt/plexpool/uor-r4-corpus/runs/20261127-antigravity-gate0-d87-proto2/corpus/train-100m.tokens.u16` | `200,945,068` | `c804c3b084a49d11f2bfcd2a5cf741862df29b42e3442ccb609305e049aba590` |
| Gate 0 `train-100m` mask (`65,499,244` response tgts) | `/mnt/plexpool/uor-r4-corpus/runs/20261127-antigravity-gate0-d87-proto2/corpus/train-100m.mask.u8` | `100,472,502` | `9e8d0cb6302ea890be5847c76133a731e7b4a080cfa096049cf37323072ae1d5` |
| Gate 0 `dev-tune` tokens (`2,670` recs, `3,805,298` tok) | `/mnt/plexpool/uor-r4-corpus/runs/20261127-antigravity-gate0-d87-proto2/corpus/dev-tune.tokens.u16` | `7,610,660` | `99b714a67415d92e42750577110d9ce405625d93adabbd906739bbfca60d9c78` |
| Gate 0 `dev-tune` mask / novel / absent | `.../dev-tune.{mask,novel,absent}.u8` | `3,805,298` ea | `643fccbb...` / `6239ac46...` / `8860f193...` |
| Token UTF-8 byte lengths | `.../id-bytes.u8` | `4,096` | `befb22f6908751ae522a80336afa584eb2c24e36a10b90ad3105e39271a48ac9` |
| Bar A `chat-v0-p2` `heldout.u16` (`512`w × `384` = `196,608` tgts) | `/mnt/plexpool/uor-r4-corpus/inputs/chat-v0-p2/heldout.u16` | `12,389,242` | `e5f400b0e676791fd0458ce810163c5d2542adda80dccc8282f37d9ca8904cef` |
| Unwrapped `m2-fresh256` (`inputs.json` / `labels.json`) | Generated deterministically via `docs/labs/m2-fresh-heldout-panel/instrument` | `524,850` / `68,000` | `1de2fc28...` / `3b5c5477...` |

**Sealed `DEV-test` (`dev-test.tokens.u16`, `39aa7376...`) is explicitly refused by `load_proto2_dev_records` and NEVER opened in Gate 1.**

---

## 3. Experimental Arms & Controlled Schedule

All three pure-integer arms train on the **exact same** `train-100m` stream (`seq_len = 256`, `batch_size = 128` -> `32,768` tok/step, `3` epochs: **Epoch 1** full causal non-BOS LM to train all embeddings and syntactic/world geometry; **Epochs 2–3** Protocol-2 answer-only `mask == 1` SFT; `lr_q8 = 2,048` (`8.0` Q15 LSB/step = `2.44e-4` sign-Lion LR) with `200`-step linear warmup and smooth per-batch decay `2,048 -> 1,024 -> 512 -> 128`, LR-scaled decoupled Lion weight decay `wd_term = (w * step_lr_q8) / 2^17` (`wd_shift = 17`, `lambda = 0.25` in Q15), `24`-bit NITI, exact `SO(2)` orthogonal Givens lifting `beta(t) = -2t/(1+t^2)`, exact rational Hui-Belkin Bayes posterior `p_i = z_i/(k - (k-1)*z_i)`, `seed = 20261127`):

*(Amendment provenance — L-007:*
- *`r2` (`vast-62893144`, `28 min`, `$0.24`): Passed Gate 1a (`0` PTX/SASS float/div/MMA) and Gate 2 (`4 configs, 0 diffs`), dropped loss `262K -> 163.5K` in 750 steps before diverging back to `172K`. Fixed hyperbolic shears (`SO(2)` Givens), `2.86 BPB` Hui-Belkin floor (rational Bayes posterior), 14-bit NITI freezing (`24`-bit NITI), and `lr_q8 = 16,384`.*
- *`r3` (`vast-62899157`, `66 min`, `$0.57`): Passed Gate 1a & Gate 2 (`0` diffs), completed `hb_l4_d256` (`10.65 MB` `UORBFY01 v2`, **`4,557 tok/s`** single-core Apple M1 decode in `22.7 MB` RSS, `0` diffs RTX 5090 = Apple M1 replay, `chat-v0-p2 = 2.709 BPB`, `E1 top1 = 227‰`, `prompt-shuffle ΔBPB = +3,130` CI `[+1,995, +4,189]`, `E2 natural_eos = 12.5%` vs `0%` on `p4.d11`). Stopped `hb_l8_d256` early after Epoch 3 & weight X-ray proved unscaled `wd_term = w * 256 / 2^12` crushed `embeddings rms` from `2,365 -> 797` and drove Epoch 3 loss upward (`113.7K -> 132.0K`) as `lr_q8` annealed. Fixed to exact LR-scaled Lion weight decay `(w * step_lr_q8) / 2^17` identically in `butterfly_rotor.rs` and `butterfly_kernels.cu` before relaunching full 3-arm `r4`.)*

| Arm | Width `D` | Depth `L` | PKM Sub-Keys | Loss Function | Artifact Size (`UORBFY01 v2`) | Role in Design |
|---|---:|---:|---:|---|---:|---|
| **`hb_l4_d256`** | `256` | `4` | `64` (`4,096` slots/L) | `HuiBelkinRescaledSquare { k: 8 }` | `~ 10.6 MB` | Depth-scaling baseline (`L = 4`) |
| **`hb_l8_d256`** | `256` | `8` | `64` (`4,096` slots/L) | `HuiBelkinRescaledSquare { k: 8 }` | `~ 19.2 MB` | Depth-scaling test (`L = 8` vs `L = 4`) |
| **`sp_l8_d256`** | `256` | `8` | `64` (`4,096` slots/L) | `Sparsemax64` | `~ 19.2 MB` | Controlled A/B loss test vs `hb_l8_d256` |
| **`p4.d11` (Anchor)** | — | — | — | Frozen static count (`Gate 0` / `P4`) | `3,988.2 MB` | Unwrapped baseline (`chat-v0-p2 = 1.4676`, E1 `1.4314` / nov `1.7170`, E2 `955‰` `" We"` / `0/200` EOS, E3 `0/128`, E4 `0/100`) |

---

## 4. Execution Gates & Non-Extrapolating Verdict Criteria

1. **Gate 1 — AOT PTX/SASS Audit:** Compile `butterfly_kernels.cu` AOT (`nvcc -cubin`, `CUDA_DISABLE_PTX_JIT=1`), audit `nvcc -ptx` and `cuobjdump --dump-sass` on all `3` required kernels (`bfy_batch_fwd_bwd_kernel`, `bfy_lion_update_kernel`, `bfy_eval_records_kernel`): require **`0` float opcodes** (excluding exact Blackwell `sm_120` `HFMA2 R<n>, -RZ, RZ,` integer constant-load aliases, logged separately, matching `antigravity-10/11/12`), **`0` PTX integer `div.`/`rem.` opcodes**, and **`0` MMA opcodes**.
2. **Gate 2 — Finite GPU-vs-CPU Exact Parity:** Run `butterfly-pilot finite` across `4` configs (`D ∈ {32, 256}`, `L ∈ {2, 4}`, both `HuiBelkinRescaledSquare` and `Sparsemax64`, `3` multi-step Lion updates + mid-record `BOS=0` reset eval + `UORBFY01` reload): require **`0` diffs** across loss, every `i64` gradient word, every `i16` weight, every `i32` momentum word, every Q48 target mass, and every greedy token ID. Any non-zero diff halts before training (`INSTRUMENT_FAILED`).
3. **Gate 3 — Controlled Training & Epoch Curves (`train-100m`):** Train `hb_l4_d256`, `hb_l8_d256`, and `sp_l8_d256` (`3` epochs each), logging `chat-v0-p2` Bar A PRLL micro-BPB (with live CPU spot-check `0` diffs!) and Gate 0 `DEV-tune` E1 (stratified `256` records, intact `\nAssistant:`-terminated true & swapped prompts, byte-weighted paired bootstrap CI95 on `ΔBPB`, `all_answer`, `novel_absent_answer`, `in_context_answer`, `gold_eos`) after epochs `0, 1, 2, 3`.
4. **Gate 4 — Strict Unwrapped Evaluation (`E2`, `E3`, `E4`) & M1 Native Bench:** Cold-reload each arm's best `UORBFY01` checkpoint and run:
   - **E2** (`200` stratified `DEV-tune` intact prompts, cap `128`, `64` counterfactual pairs at stride `take/2`, Gate 0 `>= 3` 4-gram repeat threshold, tracking both raw token-1 and non-space first token diversity).
   - **E3** (strict unwrapped `m2-fresh256` verified upfront before training, requiring `hit_eos`, `DialogueProtocol::reply_text`, distinct token-ID pair completion vs `no_memory` & `role_swapped`).
   - **E4** (`100`-item Gate 0 checkable micro-suite reporting both strict `hit_eos` + terminal/exact match and Gate 0-compatible count).
   - **Apple M1 `bench-cpu`** (`2,048`-context prefill, `128`-token warm decode `tok/s`, cold open ms, `0`-diff determinism replay AND `0`-diff CUDA=M1 replay verification against each arm's GPU-emitted `*_replay_fixture.json`).
5. **Pre-Registered Verdict Rules:**
   - **`GATE1_PASS_UNLOCK_GATE2`** iff:
     1. Gate 1 (`0` PTX/SASS float/div/MMA) and Gate 2 (`0` GPU/CPU diffs) pass cleanly;
     2. Depth scaling holds (`hb_l8_d256` beats `hb_l4_d256` on `DEV-tune` E1 `all_answer` AND `novel_absent_answer` micro-BPB);
     3. Best `< 25 MB` arm satisfies ALL THREE: (a) beats `p4.d11` on **E2 behavioral quality** (`natural_eos > 50%` vs `0%`, `fourgram_loops < 15%` vs `95%`, `diverge_by_step3 > 75%` vs `20.3%`, `modal_first_nonspace < 250‰` with `>= 20` distinct openings); AND (b) (`beats p4.d11` on `chat-v0-p2` `< 1,467,631` OR E1 `novel_absent < 1,717,007`) OR (`achieves < 1.60 BPB` with steep monotone epoch-over-epoch descent demonstrating clear Gate 2 scaling headroom for `D=512–1024, L=12–16`); AND (c) positive prompt-shuffle `ΔBPB` (`ci95_lo > 0`).
   - **`GATE1_DIAGNOSTIC_NEEDED`** iff instrument is clean (`0` diffs) yet integer training stalls (`> 2.50` BPB across all 3 arms or `L=8` fails to beat `L=4`): triggers a separately scoped diagnostic F32 reference comparison (`CHARTER §3.7`) to distinguish integer Lion/Q15 numerics from architectural bottleneck before spending Gate 2 compute.
   - **`INSTRUMENT_FAILED`** iff any PTX/SASS audit hit or GPU-vs-CPU parity diff is non-zero.
