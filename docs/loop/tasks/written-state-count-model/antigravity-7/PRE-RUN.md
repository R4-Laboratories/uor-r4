# Pre-Run Review: `written-state-count-model`, `antigravity-7` (Stage B2b `DEV-tune` `P2` Equal-Byte S3 Calibration)

- **Lab:** `antigravity`
- **Review verdict:** **GO** — Non-author `pro` Round 2 audit [`FIT.md`](FIT.md) returned **`TEST FITNESS: FIT`** after fixing all 6 Round 1 findings (`O(1)`/token incremental `RandomTransport`, unlocked CPU `verify-tape` with 100% `packed` SHA parity, `L-002` exclusive `test ! -e "$ROOT"`, `df` on `/mnt/plexpool`, expanded Gate 1 SASS float/tensor opcode check, `timeout 180` Gate 3 stop guard, strict `DevTuneData`/EOF/Pick checks, `U_count` comparison, and `tc_{3..12}` sharded backoff).
- **Host & budget:** Either free Wick RTX 4090 (`GPU-7c813ad0-2151-bb1d-5efb-a3340b9a6a41`, `sm_89`, `$0.00` pod spend, `wick-codex-gpu.lock`) **OR**, since Wick's GPU is currently running a multi-hour foreign `train-in-space` job (`PID 693149`), one dedicated single-GPU Vast.ai RTX 5090 (`sm_120`) / RTX 4090 (`sm_89`) pod (`uor-pod up --count 1`, `~$0.52–$0.87/h <= $8/h` cap, torn down immediately after archiving `evidence/` + `receipts.sha256` per `L-004`). `admit_gpu` runs before every live GPU step on both hosts.
- **Protected assets:** `DEV-test` (`dev-test.tokens.u16`, `dev-test.json`, `rstar-devtest.masses.u64`) is hard-locked untouched and refused by `dispatch.sh`, `main.rs`, and `eval.rs`.
- **Unlock rules:**
  - If `e8` achieves `<= 99%` of best equal-byte control bits on `DEV-tune` with 95% CI upper bound `< 0` (`standalone_100mb` or `combined_with_p1_a1s`), advance to pre-registered Stage B2c single-shot `DEV-test`.
  - Otherwise close `e8` `S3` on `DEV-tune` (`DEV-test` untouched) and advance to `DECISION.md` Fallback Arm 1 (`VSA SplitMix`).
