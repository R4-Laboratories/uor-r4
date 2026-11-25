# Pre-Registration Sheet: `written-state-count-model`, `antigravity-7` (Stage B2b `DEV-tune` `P2` Equal-Byte S3 Calibration)

- **Lab:** `antigravity`. **Milestone / Item:** `M2+` / [`written-state-count-model`](../../../../roadmap/items/written-state-count-model.md) (`P2`, Base: `b870e2d7f`).
- **Compute Host:** Wick workstation (RTX 4090 `sm_89`, `GPU-7c813ad0-2151-bb1d-5efb-a3340b9a6a41`, `$0.00` pod spend, lock `/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock` held with `flock -n`) **OR**, while Wick's RTX 4090 is occupied by a multi-hour foreign job (`PID 693149`, `train-in-space`), one dedicated single-GPU Vast.ai RTX 5090 (`sm_120`) / RTX 4090 (`sm_89`) pod (`uor-pod up --count 1`, `<= $1/h` inside the `$8/h` cap, torn down immediately after archiving `evidence/` + `receipts.sha256` per `L-004`). `GO.json` pins `expected_host`, `expected_compute_cap`, `device_uuid`, `gpu_lock`, `UOR_NVCC_ARCH`, and compiled `p2-s3` + `s3_kernels.sm_<cap>.cubin` SHA-256s; `admit_gpu` (`nvidia-smi` compute apps empty, `>= 8,192 MiB` free VRAM, `>= 10 GiB` free host RAM) runs before every live GPU step on all hosts.
- **HARD LOCK ON `DEV-test`:** `dev-test.tokens.u16` (`023784b7...`), `dev-test.json` (`ec048217...`), and `rstar-devtest.masses.u64` (`59f07609...`) MUST NOT be opened, read, or scored in `antigravity-7`.

## 1. Frozen Inputs (verified by SHA-256 before execution)

| Artifact | Path / Role | Pinned SHA-256 / Size |
|---|---|---|
| Full `C` stream | `/mnt/plexpool/uor-r4-corpus/runs/20261120-whole-context-36/input/c.tokens.u16` (`UORT`) | `cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a` (`4,151,216,778` B) |
| Reserved `R` stream | `/home/callard/uor-codex/20261119-rival-admission-29/count/cr-v1/r.tokens.u16` (`4,060` records, `15,947,584` targets) | `5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a` (`31,903,352` B) |
| `DEV-tune` census | `/mnt/plexpool/uor-r4-corpus/runs/20261120-pruned-r-fit-37/tune/c420/dev-tune.json` (`2,670` records, `10,807,417` targets) | `90fbc73a7b19129d39860cece7403027f71d6d952a6825e89bd1619dac3f8467` |
| `DEV-tune` UORT | `dev-tune.tokens.u16` | `75c6d1ac479bf120bac3aa60ac160536e8146f8052aa49c002209500799b7d0a` |
| `DEV-tune` cohorts/bytes | `dev-tune.novel.u8` / `dev-tune.mask.u8` / `dev-tune.segment.u8` / `id-bytes.u8` | `7df42437...` / `cf359a0e...` / `202efde4...` / `befb22f6...` |
| Relation trunk | `/home/callard/uor-codex/20261118-diagnostics/inputs/seed1.d11` (`IntegerDiscreteModel`) | `04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d` |
| Finite `C128` slice | `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/gate2-c128/finite-c1.uort.u16` | `f9a685845db9ee07c82e6ae66cdbeb2cf7747d14de41193b8e349def3ac543da` (`416` B) |
| Stage B1a `R` & `DEV-tune` `UORCBF01` | `r-65536.tape` / `tune-65536.tape` | `8b6b39d2...` / `dc0785bf...` |
| Stage B1a `R` Pick & `A1s` weights | `a1s.weights` / `a1s.manifest.json` / `a1s.seal.json` / `pick.rows` (`R`) | `e2fbf41d...` / `26f8f222...` / `b6e9ac04...` / `ba9e646a...` |

## 2. Eleven Registered Equal-Byte Arms (`max_bytes = 100,000,000` B per `UORS3T01` table)

Every arm builds its deterministic `UORS3T01` table (`<= 100,000,000` B, leaving `>= 8,718,147` B margin inside `4,000,000,000` B alongside Stage B1a `artifact.d11` + tokenizer `3,891,281,853` B) using exact ascending count cutoff pruning from the full `C` stream:

1. **`e8`**: `S3AddressArm::E8Product` (`Cl(0,6)` `F_b` / `S_b` causal transport on `seed1.d11`, `E8` 240-root product key + `prev_tok`, `4,096`-byte `token_e8_roots` table, Starling 7-neighbour miss + affine-linear Q16 output Pick).
2. **`rand`**: `S3AddressArm::RandomTransport { seed: 20261125 }` (path-dependent signed permutation transport, no E8 geometry).
3. **`tc_1`, `tc_2`, `tc_3`, `tc_4`, `tc_6`, `tc_8`, `tc_10`, `tc_12`**: `S3AddressArm::TokenContext { k }` across all 8 registered context lengths.
4. **`splitmix`**: `S3AddressArm::SplitMixPrefix { seed: 20261125 }`.

## 3. Five Sequential Execution Gates (`ROOT=/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-p2-s3-devtune`)

- **Gate 1 — PTX + SASS 0-Float / 0-Tensor-Core Static Audit (CPU):** Inspect `nvcc -ptx -arch=$UOR_NVCC_ARCH` (`0` PTX float ops) and `cuobjdump --dump-sass "$CUBIN"` (`0` FP32/FP64/FP16/MUFU/MMA ops; on Blackwell `sm_120`, CUDA 12.8 `cuobjdump` disassembles `0x...7431` `HFMA2 Rd, -RZ, RZ, imm` integer-constant moves alongside `0`-float PTX and `0`-hit `sm_89` SASS, so only exact `HFMA2 R<n>, -RZ, RZ,` aliases are excluded and counted separately).
- **Gate 2 — Finite `C128` Builder + Kernel Bit-Exactness (`flock -n`):** Run `p2-s3 finite` on `finite-c1.uort.u16`. Asserts byte-for-byte equality between disk-partitioned `build_partitioned_uors3t01` and `uor_r4_count_runtime::s3_state::build_equal_byte_s3_table`, shifted batch parity (`128 == 97`), and `0` divergent rows vs `S3StateTable::lookup_q48` across all 11 arms.
- **Gate 3 — Full-`C` `100,000,000` B Table Materialization (`<= 100,000,000` B per arm, `<= 180 s` guard):** Build all 11 `UORS3T01` tables from `c.tokens.u16` (`build/<arm>.uors3t01` + `<arm>.meta.json`), validating strict `<= 100,000,000` B and full cold `S3StateTable::from_bytes` reload on each arm under a hard `timeout 180` / `<= 180 s` stop guard.
- **Gate 4 — `DEV-tune` Pick + `R`/`DEV-tune` `UORS3F01` Capture (`flock -n`) + Unlocked Native Verify:** Run `p2-s3 devtune-pick` once (verifying `seed1.d11` SHA/shape, `readback.position`, `rec[5..]` clean, real `(pick_w, dist_w, tie_w, hit)` `R37PK001` capture, and `>= 8,000` native `observe_identity_pick` CPU probes), then for each of the 11 arms run `p2-s3 capture` under `flock -n` at shifted batches `65,536` and `65,521` (`cmp`-identical) followed outside `flock -n` by `p2-s3 verify-tape` (verifying 100% `cpu_packed == gpu_packed` SHA parity and `>= 12,500` native `S3StateTable::lookup_q48` probes with `0` divergent rows).
- **Gate 5 — Frozen `R`-Fit EM + `DEV-tune` Paired Bootstrap & `U_geo` vs `U_count` Ceiling (CPU):** Run `p2-s3 eval`: asserts `P1(A1s)` `DEV-tune` oracle `all.bits_q12 == 125_750_699_315` (`1,400,045` micro-BPB from Stage B1a `selection.json`), fits 64-pass Q32 integer EM strictly on `R` for both `standalone_100mb` and `combined_with_p1_a1s` over all 11 arms, evaluates frozen on `DEV-tune`, runs 1,000-resample complete-record block bootstrap (`2,670` records, `splitmix64` `seed = 4390`), and reports oracle `U_geo` (`DEV-tune` EM ceiling) directly against pinned `U_count` (`claude-438`, `128,614,178,256` Q12 / `1,431,925` micro-BPB on `DEV-tune`).

## 4. Branch Outcomes & Stop Rules

- **`e8` PASS (`gate_p2_standalone_1pct` or `gate_p2_combined_1pct` with 95% CI upper bound `< 0` and `<= 99%` of best equal-byte control bits):** `e8` advances to pre-registered Stage B2c single-shot sealed `DEV-test`.
- **`e8` NEGATIVE (`>= 99%` or non-negative CI vs best equal-byte control):** Close this `e8` `S3` configuration on `DEV-tune` without touching sealed `DEV-test`, record the exact `U_geo` ceiling and micro-BPB gap against each control, and advance immediately to `DECISION.md` Fallback Arm 1 (`VSA SplitMix` geometric binding).
- **Stop immediately if:** any SASS float opcode appears, `DEV-test` is touched, any `C128` or capture parity probe diverges, `p1_all.bits_q12 != 125_750_699_315`, any source or binary SHA mismatches `GO.json`, or any single arm build exceeds 3 minutes.
