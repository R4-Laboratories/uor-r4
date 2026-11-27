# Gate 0 Report: Clean Protocol-2 D87 Ingest & Unwrapped Eval Suite (E1–E4)

Lab: antigravity. Item: [End-to-end integration](../../../../roadmap/items/end-to-end-integration.md). References #4064 and #2030. Source baseline `1a995ebba9cc`. Compute spend: **`$0.00` GPU / `0` cloud pods** (Wick CPU only, `--threads 10`, zero GPU touch).

## What did not work

1. **Frozen `p4.d11` (`e0ffbba1...`, `3.99 GB`) FAILS unwrapped free-run generation (E2):** Across `200` `DEV-tune` prompts (cap `128`), `191/200` (`955‰`) begin with the single modal token `" We"` (`id=799`, leaked from raw donor CoT in the legacy D87 stream), only `5` distinct first tokens appear, **`0/200` reach natural `EOS`** (`200/200` cap-truncated), `190/200` (`950‰`) enter 4-gram repetition loops, and only `13/64` (`203‰`) counterfactual prompt pairs diverge by step 3 (`6/64` at step 1).
2. **Frozen `p4.d11` FAILS unwrapped `m2-fresh256` (E3):** Removing regex span extraction and `2^48` score forcing leaves **`0/128` pairs (`0/256` complete rows)** across `active`, `no_memory`, and `role_swapped` arms, and **`0/64` abstentions**.
3. **Frozen `p4.d11` FAILS deterministic checkable tasks (E4):** **`0/100`** (`0/40` arithmetic/state tracking, `0/40` long-distance passage QA, `0/20` tiny Rust completions).
4. **High novel-absent answer BPB on E1 (`1.7170` BPB):** On `256` `DEV-tune` records (`51,490` answer targets), `p4.d11` scores `1,431,433` micro-BPB (`591‰` top-1) overall, `1,310,025` in-context, **`1,717,007` novel-absent** (`9,829` targets, `340‰` top-1), and `18/82` (`219‰`) top-1 on gold `EOS`. Counterfactual prompt-shuffle shows weak positive conditioning (`true = 1,431,433` vs `swapped = 1,516,533`, `Δ = +85,100`, 95% CI `[+67,031, +105,286]`).
5. **Leading-space opening token under `DialogueProtocol::literal_roles_v2`:** Because `literal_roles_v2` ends prompts at `"Assistant:"` (no trailing space) and encodes `" " + answer` bit-identically to `DialogueEncoder`, standalone `" "` (`id=223`) accounts for `503,634` ppm of first answer tokens in `TRAIN` (`504,768` ppm `DEV-tune`, `500,087` ppm `DEV-test`) whenever the opening word lacks a space-prefixed BPE merge. Gate 1 evals will report both token-1 and first non-space content token diversity.
6. **Host/CPU scope only:** No model training, weight updates, or GPU runs occurred in Gate 0. Sealed `DEV-test` was written and SHA-256 receipted during ingest and was never read by any evaluator.

## What changed

- Added standalone pure-integer `#![forbid(unsafe_code)]` crate [`gate0-kit/`](gate0-kit/) (`Cargo.toml`, `Cargo.lock`, `src/proto2.rs`, `src/eval.rs`, `src/main.rs`) implementing:
  - `build-proto2`: parallel multi-shard `DialogueProtocol::literal_roles_v2` answer-only D87 re-ingest with exact `claude-411/d87-split/v1` splitting, `max=16` `TRAIN` template spam capping, `train-100m` Gate 1 slice emission, and `absent.u8` (`2,048`-token context-absence) sidecar generation.
  - `eval-p4`: strictly unwrapped `E1–E4` benchmark suite over `P4Session::step` (and reusable for Gate 1 Butterfly-Rotor evaluation).
- Committed Wick ZFS SHA-256 receipts and JSON summaries in [`evidence/`](evidence/): [`p4_unwrapped_e1_e4.json`](evidence/p4_unwrapped_e1_e4.json), [`receipts.sha256`](evidence/receipts.sha256), [`train.json`](evidence/train.json), [`train-100m.json`](evidence/train-100m.json), [`dev-tune.json`](evidence/dev-tune.json), [`dev-test.json`](evidence/dev-test.json).
- Added a Gate 0 progress note to [`docs/roadmap/items/end-to-end-integration.md`](../../../../roadmap/items/end-to-end-integration.md) and appended one result row to [`docs/research/notebook/m4-serving.md`](../../../../research/notebook/m4-serving.md).

## Measurements

### Protocol-2 Clean D87 Corpus (`/mnt/plexpool/uor-r4-corpus/runs/20261127-antigravity-gate0-d87-proto2/`)

Wall `1:26.92`, peak RSS `1.35 GB`, `10` CPU threads, `262` donor shards, `tokenizer_sha256 = a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377`, `receipts.sha256` **`7/7 OK`**:

| Split | Records | Template-capped | Total tokens | Response targets (`mask=1`) | `<= 2,048` tok | Distinct 1st tokens | `" We"` (`id=799`) share | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `train` | `516,861` | `2,719` | `737,542,620` | `482,759,880` (`482,208,409` ans) | `392,464` (`75.9%`) | `699` | `2,076` ppm (`0.21%`) | Ready |
| `train-100m` | `71,876` | `0` | `100,472,502` | `65,499,244` (`65,422,623` ans) | `55,013` (`76.5%`) | `329` | `2,270` ppm (`0.23%`) | Gate 1 input |
| `dev-tune` | `2,670` | `0` | `3,805,298` | `2,522,226` (`313,771` novel-absent) | `2,042` (`76.5%`) | `104` | `2,825` ppm (`0.28%`) | Dev eval |
| `dev-test` | `5,321` | `0` | `7,695,629` | `5,098,750` (`642,082` novel-absent) | `4,069` (`76.5%`) | `142` | `1,231` ppm (`0.12%`) | **SEALED** |

Compared to the legacy reasoning-inclusive D87 stream (`896,900` ppm `" We"`, `10` distinct openings, `48.2%` `<= 2,048` tokens), `" We"` share dropped `432×` and distinct openings rose `70×` (`0` empty skipped, `0` dropped bytes).

### Frozen `p4.d11` Unwrapped Baseline (`eval-p4`, wall `1:29.10`, peak RSS `3.935 GB`)

| Suite | Metric | Frozen `p4.d11` (unwrapped) | Result |
|---|---|---:|---|
| **E1** (`256` `DEV-tune` recs) | `all_answer` micro-BPB (`51,490` targets) | `1,431,433` (`591‰` top-1) | Baseline anchor |
| **E1** (`256` `DEV-tune` recs) | `novel_absent_answer` micro-BPB (`9,829` targets) | `1,717,007` (`340‰` top-1) | Non-copy anchor |
| **E1** (`256` `DEV-tune` recs) | `in_context_answer` / `gold_eos` top-1 | `1,310,025` / `18/82` (`219‰`) | Baseline anchor |
| **E1** (`256` `DEV-tune` recs) | Prompt-shuffle `Δ` (swapped − true) | `+85,100` CI `[+67,031, +105,286]` | Weak positive |
| **E2** (`200` prompts, cap `128`) | Modal 1st tok / distinct / `EOS` / 4-gram loops | `955‰` `" We"` / `5` / `0/200` / `950‰` | **FAIL** |
| **E2** (`64` counterfactual pairs) | Diverge by step 1 / step 3 | `6/64` (`94‰`) / `13/64` (`203‰`) | **FAIL** |
| **E3** (unwrapped `m2-fresh256`) | `active` / `no_memory` / `role_swapped` pairs | `0/128` / `0/128` / `0/128` (`0/64` abst) | **FAIL** |
| **E4** (`100` checkable micro) | Total (`40` arith / `40` QA / `20` Rust) | `0/100` (`0/40`, `0/40`, `0/20`) | **FAIL** |

## Checks run at the head

- `cargo test` in `docs/loop/tasks/end-to-end-integration/antigravity-1/gate0-kit`: **`18 passed; 0 failed`** (exact `DialogueProtocol::literal_roles_v2` parity, exact `2,048`-token `absent.u8`, template cap, planted E1–E4 monitors).
- `cargo clippy --all-targets -- -D warnings` in `gate0-kit`: **0 warnings**.
- Wick ZFS `sha256sum -c receipts.sha256` in `/mnt/plexpool/uor-r4-corpus/runs/20261127-antigravity-gate0-d87-proto2/`: **`7/7 OK`**.
- `UOR_LAB=antigravity uor-harness check --only harness.native_space_contract`: **PASS (`0` failed)**.
- `UOR_LAB=antigravity uor-harness loop pr-check --local`: **PASS (`verdict: PASS`)**.

## Skills used

- `.agents/skills/native-space-cycle/SKILL.md` (verified D85 compliance: no training/serialization mutation in Gate 0).
- `.agents/skills/uor-research-architect/SKILL.md` (strictly unwrapped negative-first evaluation and sealed `DEV-test` discipline).

## Still open

- **Gate 1 PR 1 (`antigravity-2`):** Pure-integer Reversible Lifting-Givens Butterfly + Non-Abelian `S³` Quaternion Rotor + Sparse PKM + Factored `64×64` Hierarchical Head engine (`UORBFY01`) in `crates/uor-r4-native-vector`.
- **Gate 1 PR 2 (`antigravity-3`):** Pre-registered, non-author `FIT`-gated Wick RTX 4090 AOT `sm_89` CUDA training pilot on Gate 0's `train-100m` clean Protocol-2 D87 slice, scored unwrapped against this `p4.d11` `E1–E4` baseline and `chat-v0-p2`.
- **Gate 2:** Scale to `12–16` layers, integrate `Q31FirstPickRead` + compact memory, qualify on Bar A `<= 0.90` BPB, unwrapped M2 `>= 52/128`, M1 `>= 1,000 tok/s`, and Browser WASM32 Studio.

## Plain summary

Gate 0 built the clean, reasoning-free Protocol-2 D87 training and evaluation streams (`" We"` opening dominance dropped from `89.69%` to `0.21%`, `699` distinct openings, `train-100m` ready, `DEV-test` sealed) and proved with an incorruptible unwrapped test harness that frozen `p4.d11` (`3.99 GB`) collapses (`95.5%` `" We"`, `0/200` `EOS`, `0/128` unwrapped M2, `0/100` checkable micro-tasks). Everything needed to train and honestly judge Gate 1's pure-integer Butterfly-Rotor model is now in place at `$0.00` GPU spend.
