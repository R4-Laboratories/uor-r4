# Pre-Registration Sheet: `written-state-count-model`, `antigravity-5` (Stage B1b Sealed `DEV-test` `P1(b)` Evaluation of `UORKNC03`)

- **Lab:** `antigravity`. **Milestone / Item:** `M2+` / [`written-state-count-model`](../../../../roadmap/items/written-state-count-model.md). **Base:** `de0a94ebd`.
- **Host:** Wick workstation (`GPU-7c813ad0-2151-bb1d-5efb-a3340b9a6a41`, RTX 4090 `sm_89`, 59 GiB RAM, 40 TB `/mnt/plexpool`), `$0.00` rented pod spend. Shared lock: `/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock` held exclusively during live GPU phases (`flock -n`).

## 1. Frozen Inputs & Binaries (verified by SHA-256 before execution)

| Artifact | Identity / Role | Pinned SHA-256 / Size |
|---|---|---|
| Compact count base | `antigravity-4` `full-c.v3` (`UORKNC03`, `T3*=2, T_hi*=6`) | `75ccb234e669cb6b707ffe4091822a91a225e0875a0e99edf9f21f8f2ec3f233` (`3,804,378,885` B) |
| Deployed `UORP1A01` | `antigravity-4` `artifact.d11` (+ `265,393` B tokenizer = `3,891,281,853` B `<= 4.000 GB`; CPU RSS `3,791,956`/`3,791,064 KiB <= 3,906,250 KiB`) | `dd217c0de241f725925e62fdcecab2a78cfc4829e080200e2de708a1e5c57d05` (`3,891,016,460` B) |
| Sealed `R*` masses | `antigravity-2` / `claude-422` canonical `rstar-devtest.masses.u64` (`21,349,576 × 8` LE u64; `251,672,950,854` Q12 all, `223,758,978,608` Q12 novel, `masses_at_floor == 4`) | `59f076093c51b15469b55b01821947b7c51d8038e68d834465e29be72d56f8de` (`170,796,608` B) |
| Sealed `R*` receipt | `antigravity-2` `rstar-devtest.json` (`check_host == true`, `sealed_run == true`) | `616c8eff7b4f2fb35286f1e0f55c89c0350bcacdb3f75f5aee46f4f91c0a6dac` |
| Sealed `DEV-test` census | `claude-420` `dev-test.json` (`5,321` records, `21,354,897` tokens, `21,349,576` targets, `43,326,982` all B; `14,795,227` novel / `31,328,319` B) | `ec048217ac64f05a6519d7e38dd2fabf6e244eca8009b24476b3870cbc0f73f2` |
| Sealed `DEV-test` UORT | `dev-test.tokens.u16` (`42,709,858` B) & `mask`/`segment`/`novel`/`id-bytes` | `023784b7...` / `7e1361f4...` / `25b13499...` / `5bf3881b...` / `befb22f6...` |
| `A1s` frozen candidate | `antigravity-4` `UORS5W02` (`gate5-eval/fit/identity/a1s.{weights,manifest.json,seal.json}`) | `e2fbf41d...` / `26f8f222...` / `b6e9ac04...` |
| `A0` disabled control | `antigravity-4` `UORS5W01` (`gate5-eval/fit/disabled/a1.{weights,manifest.json,seal.json}`) | `9ec8168a...` / `95b17f7b...` / `5eddd783...` |
| Audited CUDA binaries | `antigravity-4` `cuda-v3`, `lookup_v3.sm_89.cubin`, `eval-v3-tune-device` (`0` SASS floats) | `123df397...` / `4c329f9f...` / `9d717ca3...` |
| Relation trunk | `seed1.d11` (`IntegerDiscreteModel`, context 192, vocab 4096, blocks 4) | `04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d` |

## 2. Three Admitted Sequential Phases (Private root: `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-p1b-v3-devtest/`)

1. **Phase A — Sealed `R*` & `artifact.d11` Integrity Verification (`0 s` GPU):**
   - Verify SHA-256 and exact JSON fields of existing sealed `rstar-devtest.masses.u64` (`59f07609...`), `rstar-devtest.json` (`616c8eff...`), and `artifact.d11` (`dd217c0d...`). No redundant re-run of canonical `R*`.
2. **Phase B — Sealed `DEV-test` `UORKNC03` `UORCBF01` Capture & Native Verify (`~1 min` GPU + CPU):**
   - Run audited `cuda-v3 cuda` twice on `full-c.v3` (`75ccb234...`) and `dev-test.tokens.u16` (`023784b7...`) at batch widths `65,536` (`primary`) and `65,521` (`repeat`).
   - **Required checks:** exact length `112 + 140 * 21,349,576 = 2,988,940,752` B, byte-identical `cmp 65536.features 65521.features`, and CPU `cuda-v3 verify` (`divergent_rows == 0`, `probes_checked >= 25,000`, `rows == 21,349,576`, matching `tape_sha256`).
3. **Phase C — Frozen `A1s` & `A0` Sealed `DEV-test` Scoring + Paired `R*` Bootstrap (`~6 min` GPU):**
   - Run `p1b-devtest-v3-eval score` using `GO.json`, `full-c.v3`, `seed1.d11`, `65536.features`, `rstar-devtest.masses.u64`, `eval-v3-tune-device`, `A1s`, and `A0`.
   - **Required checks:** `0` divergent words across all `13,607` predetermined probes (`frame <= 64`, batch seams, `ordinal % 8191 == 197`, final 2 targets) against native `observe_identity_pick`, `CompactCountLookup::lookup` (`UORKNC03`), `FrozenPickMixture::predict_mass`, and `bits_q12`; exact `5,321` records and `21,349,576` targets; exact Q12 reconciliation against `rstar-devtest.masses.u64`.

## 3. Decision Rule, Statistical Calibration & Branch Outcomes (`P1(b)`)

- **Registered `P1(b)` bar ([`DECISION.md §4`](../../../research/reviews/pi-review-2026-11-16/DECISION.md)):** deployable model `<= 4,000,000,000` B (`3,891,281,853` B holds, CPU `uor-chat` RSS `<= 3,906,250 KiB` proven in `antigravity-4`) **AND** `BPB(A1s) <= 1.03 × BPB(R*)` on sealed `DEV-test` all targets.
- **Exact integer criterion:** `100 * sum_all_bits_q12(A1s) <= 103 * 251_672_950_854`, paired with 1,000-resample complete-record block bootstrap (`5,321` records, splitmix64 seed `4390`, `[24]` and `[974]`) on both all (`43,326,982` B) and novel (`31,328,319` B) cohorts (`a1s_minus_rstar`, `a1s_minus_103pct_rstar`, `a1s_minus_a0`, `a0_minus_rstar`).
- **Outcomes:**
  - **PASS (`<= 1.03 × R*`):** `P1(b)` passes under `<= 4.000 GB`; unblocks Stage B2 (`P2` E8 snap + Starling miss geometric state rows) on this `full-c.v3` base.
  - **NEGATIVE (`> 1.03 × R*`):** `NEGATIVE_PRUNING_COST` closes only `(T3=2, T_hi=6)` `UORKNC03`, records the exact micro-BPB / Q12 gap vs `1.03 × R*`, and records what further lossless `S2` trie compression or `P2` residual work is needed next.

## 4. Stop Rules

- Abort immediately and write no pass receipt if: (1) Wick GPU has foreign compute or `< 8 GiB` free VRAM or `< 10 GiB` free host RAM; (2) any pinned SHA-256 mismatches; (3) `R*` deviates from `59f07609...`; (4) `65,536` vs `65,521` `UORCBF01` tapes differ or `cuda-v3 verify` fails; (5) any finite or native replay probe diverges.
