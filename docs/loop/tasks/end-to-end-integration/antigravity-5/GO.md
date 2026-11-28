# Pre-Run Review (`GO.md`) — `antigravity-5` (`UORBFY03`)

- **Lab:** `antigravity`
- **Date:** 2026-11-28
- **Decision:** **GO** (logged for Casey per `CHARTER.md` §5.1 / §5.10).

## Whole-Project Chain Check

1. **Prior milestones & unlock:** Gate 0 (`antigravity-1`, PR `#4316`), Gate 1 PR 1 (`antigravity-2`, PR `#4317`), Gate 1 PR 2 (`antigravity-3`, PR `#4331`), and Gate 1 `2×2` Diagnostic (`antigravity-4`, PR `#4339`, merged on `main` `b2c05b6de`) are complete. `antigravity-4` returned `DIAG_CAPACITY_OR_LOSS_BOTTLENECK`, proving Pure-Int Q15 Lazy-Sparse AdamW matches Diagnostic F32 (`+0.0026 BPB` on `chat-v0-p2`, `-0.0016 BPB` beating F32 on `E1_all`) while isolating the `2.5342 BPB` floor to backbone capacity, non-selective `S³` decay, narrow frozen PKM keys, and `64×64` head routing.
2. **Tools & pinned inputs verified:**
   - `uor-r4-native-vector`: `60/60` tests pass (incl. central FD checks on all 7 new operator groups & `24/24` opposite-role binding), `clippy --lib -- -D warnings` clean, `wasm32-unknown-unknown` clean.
   - `butterfly-v3`: `15/15` tests pass, `clippy --all-targets -- -D warnings` clean, `0` div/float CUDA static scan, `check_no_softmax.py` / `check_no_matmul.py` / `harness.native_space_contract` clean.
   - Pinned Gate 0 corpus (`train-100m.*`, `dev-tune.*`, `id-bytes.u8`), `chat-v0-p2/heldout.u16`, `tokenizer.json`, and `m2-fresh256` verified on Wick ZFS & laptop; `dev-test.*` remains sealed.
3. **Non-author `TEST FITNESS: FIT`:** Recorded in `FIT.md` (reviewer `7124a74a-4557-425a-a38b-17d1bd8b27bd`) over exact SHA-256 digests of all 10 files.
4. **What pass and fail each unlock:**
   - `(b) GATE1_CLEARED` -> immediate Gate 2 (`737.5M` `TRAIN` + sealed `DEV-test`).
   - `(c) V3_BREAKTHROUGH_NEEDS_SFT_OR_SCALE` -> `antigravity-6` multi-epoch pretrain / SFT / scale-up to clear `< 1.60 BPB` & unwrapped `E2–E4`.
   - `(d) V3_PARTIAL_GAIN_NEEDS_LOSS_OR_HEAD_SURGERY` -> `antigravity-6` lawful non-square geometric loss (rank/margin or rational-log surrogate without `exp`).
   - `(e) V3_PLATEAU_AT_RESCALED_SQUARE` -> closes this rescaled-square `UORBFY03` configuration and redirects `antigravity-6` to loss-geometry / throughput redesign.
5. **Compute caps & `L-004` guard:** `uor-pod status` confirms `0/4` pods running (`$0/h` of `$8.00/h`). Spinning up `1` Vast.ai `2 × RTX 5090` pod (`~$1.82–$2.10/h`, session `gate1-v3-r1`, `2 h` initial lease, renewed every `<= 25 min` via `schedule`). Immediate archive to Wick ZFS + `release` + `down --discard-local` upon `GATE1_V3_COMPLETE`.
