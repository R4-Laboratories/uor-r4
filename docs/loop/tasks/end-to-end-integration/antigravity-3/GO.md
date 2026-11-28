# Pre-Run GO Review (`antigravity-3`, Round 6 Full 3-Arm Relaunch `r4` — CHARTER §5.10)

- **Lab / Top Session:** `antigravity`
- **Decision:** `GO` — launch the full 3-arm Gate 1 PR 2 Pure-Integer Butterfly-Rotor CUDA pilot (`hb_l4_d256`, `hb_l8_d256`, `sp_l8_d256`) on `train-100m` into fresh `OUT_DIR` (`r4`) with exact LR-scaled Lion weight decay (`wd_shift = 17`, `lr_q8 = 2,048 -> 1,024 -> 512 -> 128`).
- **Diagnostic Provenance (`L-007`):**
  - `r2` (`vast-62893144`, `28 min`, `$0.24`): Fixed hyperbolic shears (`SO(2)` Givens), `2.86 BPB` Hui-Belkin floor (rational Bayes posterior), 14-bit NITI freezing (`24`-bit NITI), and `lr_q8 = 16,384`.
  - `r3` (`vast-62899157`, `66 min`, `$0.57`): Completed `hb_l4_d256` (`10.65 MB`, **`4,557 tok/s`** Apple M1 single-core, `22.7 MB` RSS, `0` diffs RTX 5090 = Apple M1 replay, `chat-v0-p2 = 2.709 BPB`, `prompt-shuffle ΔBPB = +3,130` CI95 > 0, `E2 natural_eos = 12.5%`); weight X-ray proved unscaled `wd_term = w * 256 / 2^12` crushed `embeddings rms 2,365 -> 797` and drove Epoch 3 loss upward. Fixed to exact LR-scaled `wd_term = (w * step_lr_q8) / 2^17` and verified `TEST FITNESS: FIT` by non-author `pro` subagent `1f5d7158-cb22-41be-be24-35cddded2504`.
- **Compute Lease Within Caps:** Provision `1 × RTX 5090` (`~$0.521/h`, `5 h` lease; total active across all labs: `2/4` pods, `~$1.77/h` of `$8.00/h` cap), run all 3 arms (`~4.3 h`, `~$2.25`), renew lease `<= 30 min`, archive all `evidence/`, `audit/`, `artifacts/*.uorbfy01`, and `receipts.sha256` to Wick ZFS + local workspace, and immediately release & tear down the pod (`L-004`).
