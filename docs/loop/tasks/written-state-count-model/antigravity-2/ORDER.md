# ORDER: written-state-count-model, antigravity-2

Lab: antigravity. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (#4063, M2+/P1(b)). Base: `116598811`.

## Goal

Close the sealed `DEV-test` paired `P1(b)` pruning-cost gate on Wick's RTX 4090 (`$0.00` pod spend) by:

1. Reproducing and persisting the canonical `claude-422` modified Kneser–Ney reference `R*` (`N = 10`, full `TRAIN` `2,091,560,001` tokens, sealed `DEV-test` `21,349,576` targets) raw little-endian Q48 target mass tape (`170,796,608` bytes, pinned SHA-256 `59f076093c51b15469b55b01821947b7c51d8038e68d834465e29be72d56f8de`), verified with `--check-host`.
2. Capturing and independently replaying the threshold-318 full-C compact count feature tape (`UORCBF01`) across sealed `DEV-test` in both batch alignments (`65,536` primary and `65,521` shifted repeat).
3. Evaluating the frozen `codex-37`-selected `A1s` candidate (`UORS5W02`, `3,999,237,441` deployed bytes with tokenizer `<= 4,000,000,000` B) and registered no-S5 control `A0` (`UORS5W01`) on sealed `DEV-test` (`5,321` records, `21,349,576` targets, `43,326,982` bytes), computing the exact paired integer difference and 95% record-block bootstrap (`1,000` resamples, splitmix64 seed `4390`) against canonical `R*` (`1.03 x R*` gate) and `A0` (S5 Pick contribution).

## Native-space cycle contract (D85)

1. **Native state schema & carried fields:**
   - Token stream (`UORT` v1, vocab `4096`): original `u16` token IDs (`BOS = 0`, `EOS = 1`) framed by record (`0 .. 1`), never crossing `BOS`.
   - Count context state (`S2`): up to `10`-token causal `PrimeRouteKey` prefix (`u128`), `UORKNC02` threshold-318 compact tables emitting eleven `u32` raw counts, eleven `u64` raw totals, and one `u64` Q48 modified-KN mass (`1..=2^48`) per target.
   - Transport & copy state (`S1`, `S5`): `4`-block × `8`-coordinate `i32` `IntegerDiscreteModel` trunk (`04652a3d...`), `192`-token causal `PickDevice` / `code_pick` identity-code sign window emitting causal `(position, pair_distance, tie_count)` and Pick class `{0..=4}` (`Three` for `A0`, `Five` for `A1s`).
   - Mixer & scoring state (`S6`): `UORS5W01` (`528 × 14` Q32 words) and `UORS5W02` (`880 × 14` Q32 words) lossless `FrozenPickMixture` weights, mixed strictly in integers to Q48 mass (`1..=2^48`) and scored via PRLL `bits_q12` (Q12 integer bits).
   - Reference `R*` state: adjusted `KnTables` / `KnScorer` Q48 target masses (`1..=2^48`) and PRLL `bits_q12`, floored at `1` Q48 unit matching `claude-422`.
2. **Shared typed operator paths:**
   - GPU operators: `NgramCounter` + `KnScorer` (`crates/uor-r4-discrete/src/kn_score.cu`), `d36-count-batch` (`UORKNC02` CUDA lookup), `PickDevice` (`IdentityControlV1` in `discrete.cu`), and `d37-tune-device` (`device.cu` frozen Q48 mixer + `bits_q12`).
   - CPU / serving / WASM operators: `uor_r4_count_runtime::{CompactCountLookup, FrozenPickMixture, PickClassRule, code_pick}`, `CeilingTable::from_lookups_with_pick_classes`, and `uor_r4_integer::lawful_likelihood::bits_q12`.
3. **Exact transition replay & bit-identical oracle:**
   - Phase A asserts exact `host = GPU` parity on all `21,349,576` `R*` Q48 target masses and exact `masses_sha256` equality (`59f07609...`).
   - Phase B asserts byte equality between primary (`65,536`) and shifted (`65,521`) `UORCBF01` GPU tapes (`2,988,940,752` B) plus `0` divergent words against native `CompactCountLookup` via `d37-count-replay`.
   - Phase C asserts `0` divergent raw words between GPU (`PickDevice`, `UORCBF01`, `d37-tune-device`) and native (`IntegerDiscreteModel::observe_identity_pick`, `CompactCountLookup::lookup`, `FrozenPickMixture::predict_mass`, `bits_q12`) on all predetermined probes (`<= 64` record ends/first 8 prefixes, batch seams, interior modular probes `ordinal % 8191 == 197`, final 2 targets) plus synthetic finite controls and planted fault rejections.
4. **Update rule:**
   - Zero weight updates or training runs; strictly frozen read-back and one-shot evaluation of pre-existing `codex-37` fitted artifacts.
5. **Lossless serialization:**
   - `UORKNC02` (`compact.counts`), `UORS5W01` / `UORS5W02` (`FrozenPickMixture`), `UORCBF01` (`112`-byte header + `140`-byte LE feature rows), and canonical `R*` raw little-endian `u64` mass stream (`21,349,576 × 8` bytes). Zero float master weights or export conversion.

## Cancel probe (run first)

1. Confirm `/Users/casey.allard/uor-r4` clean on `main` (`116598811`) and isolated worktree created.
2. Confirm `uor-pod status` shows `Running: 0 pod(s)` (`$0.00` pod spend).
3. Confirm Wick GPU `GPU-7c813ad0-2151-bb1d-5efb-a3340b9a6a41` is idle, lock `/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock` exists, all pinned custody inputs (`compact.counts`, `trunk.d11`, `A1s` frozen weights/manifest, `A0` weights/manifest, `uor-r4-claude:/workspace/out/c420/run1/*`) verify SHA-256, and `d37-count-replay` accepts non-`R` `DEV-test` framing/cardinalities.

## Files in scope

- `docs/labs/kn-reference/instrument/src/main.rs`
- `docs/loop/tasks/written-state-count-model/antigravity-2/**`
- `docs/research/notebook/m4-serving.md`
- `docs/roadmap/items/written-state-count-model.md`

## Done when

1. `--write-masses` in `docs/labs/kn-reference/instrument/src/main.rs` and the standalone `eval/` crate pass unit tests, `rustfmt --check`, finite Wick controls (syntax/parse, `8,193`-row CUDA-vs-native parity + planted negatives, and 1‰ `DEV-tune` mass export check), non-author `FIT.md` review, and `GO.json` admission.
2. Phases A, B, and C complete cleanly under `flock` on Wick with `0` word mismatches, exact `R*` SHA-256 `59f07609...` verified, and sealed summaries/manifests (zero bulk tapes or corpus text) archived under `evidence/`.
3. `REPORT.md`, one row in `docs/research/notebook/m4-serving.md`, and `docs/roadmap/items/written-state-count-model.md` record the exact `P1(b)` pass or negative-pruning-cost result honestly.
4. All harness checks (`uor-harness check --staged`, `harness.native_space_contract`, `loop report --check`, `loop pr-check`, `claim-wording`), `pro` exact-head adversarial review, CI checks, squash merge, and worktree/branch cleanup pass.

## Stop and ask if

- Wick's GPU lock is held continuously or host memory/disk drops below admission thresholds, any pinned Wick SHA-256 mismatches, `R*` deviates from `59f07609...`, or `uor-harness` flags an out-of-scope edit.
