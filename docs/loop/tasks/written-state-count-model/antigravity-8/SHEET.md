# Pre-Registration Sheet: `written-state-count-model`, `antigravity-8` (Stage B2-Fallback `DEV-tune` `P2` Equal-Byte S3 Calibration)

- **Lab:** `antigravity`. **Milestone / Item:** `M2+` / [`written-state-count-model`](../../../../roadmap/items/written-state-count-model.md) (`P2` Fallback Arms 1–4, Base: `6fdd03679`).
- **Compute Host:** Wick workstation (RTX 4090 `sm_89`, `GPU-7c813ad0-2151-bb1d-5efb-a3340b9a6a41`, `$0.00` pod spend, lock `/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock` held with `flock -n`) **OR**, while Wick's RTX 4090 is occupied by a foreign job, one dedicated single-GPU Vast.ai RTX 5090 (`sm_120`) / RTX 4090 (`sm_89`) pod (`uor-pod up --count 1`, `<= $1/h` inside the `$8/h` cap, torn down immediately after archiving `evidence/` + `receipts.sha256` per `L-004`). `GO.json` pins `expected_host`, `expected_compute_cap`, `device_uuid`, `gpu_lock`, `UOR_NVCC_ARCH`, and compiled `p2-fallback` + `s3_kernels.sm_<cap>.cubin` SHA-256s; `admit_gpu` (`nvidia-smi` compute apps empty, `>= 8,192 MiB` free VRAM, `>= 10 GiB` free host RAM) runs before every live GPU step.
- **HARD LOCK ON `DEV-test`:** `dev-test.tokens.u16` (`023784b7...`), `dev-test.json` (`ec048217...`), and `rstar-devtest.masses.u64` (`59f07609...`) MUST NOT be opened, read, or scored in `antigravity-8`.

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

## 2. Fourteen Registered Equal-Byte Arms (`max_bytes = 100,000,000` B per `UORS3T02` table)

Every arm builds its deterministic `UORS3T02` table (`<= 100,000,000` B, leaving `>= 8,718,147` B margin inside `4,000,000,000` B alongside Stage B1a `artifact.d11` + tokenizer `3,891,281,853` B) using exact ascending count cutoff pruning from full `C`. Priority order of candidate fallback arms follows [`DECISION.md`](../../../../research/reviews/pi-review-2026-11-16/DECISION.md#L293-L298):

1. **`vsa` (Fallback Arm 1):** `VsaSplitMix` — canonical 64-bit SplitMix64 token hypervectors (`VSA_SEED = 0x5653_415F_4255_4E44`), causal window `W = 7` XOR+rotate+vertical-majority bundling (`pk in 0..32_768`), 7-neighbour Hamming-1 $Q_{15}$ miss reads, and exact popcount Pick `(65 - dh) * 1008` (`1008..=65_520`) normalized via `VSA_DENOM_BYTE[8][256]`.
2. **`hopf` (Fallback Arm 2):** `HopfSector` — `seed1.d11` `Cl(0,6)` `F_b` / `S_b` causal transport quantized into `96` integer Hopf $S^2 \times S^1$ sectors per lane (`pk in 0..9_216`), 7-neighbour $S^2 \times S^1$ miss reads, and face/quadrant/fiber Q16 Pick `16_384 + same_face * 24_576 + same_quad * 12_288 + same_fiber * 12_288` (`16_384..=65_536`) normalized via precomputed `denoms_hopf[96]`.
3. **`h4_600cell` (Fallback Arm 3):** `H4SixHundredCell` — `seed1.d11` `Cl(0,6)` causal transport snapped via exact pure-integer `PHI_Q30` dot products to the 120 vertices of the 600-cell (`pk in 0..14_400`), 7 shell-1 neighbour miss reads (`H4_ADJ_12`), and `8_192 + (8 - H4_SHELL) * 7_168` (`8_192..=65_536`) Q16 Pick normalized via `denoms_h4[120]`.
4. **`golden_shell` (Fallback Arm 4):** `GoldenShell` — exact 4-lag 600-cell shell tuple `(s1, s2, s3, s4) in {0..8}^4` over token H4 roots (`pk in 0..6_561`), up to 7 $L_1 = 1$ shell-tuple miss reads, and `8_192 + (8 - H4_SHELL) * 7_168` (`8_192..=65_536`) Q16 Pick normalized via `denoms_h4[120]`.
5. **10 Equal-Byte Controls:** `rand` (`RandomTransport { seed: 20261125 }`), `tc_1`, `tc_2`, `tc_3`, `tc_4`, `tc_6`, `tc_8`, `tc_10`, `tc_12` (`TokenContext { k }`), and `splitmix` (`SplitMixPrefix { seed: 20261125 }`).

## 3. Five Sequential Execution Gates

- **Gate 1 — PTX + SASS 0-Float / 0-Tensor-Core Static Audit (CPU):** Inspect `nvcc -ptx -arch=$UOR_NVCC_ARCH` (`0` PTX float ops) and `cuobjdump --dump-sass "$CUBIN"` (`0` FP32/FP64/FP16/MUFU/MMA ops, excluding exact `HFMA2 R<n>, -RZ, RZ,` constant-load aliases on `sm_120`).
- **Gate 2 — Finite `C128` Builder + Kernel Bit-Exactness (`flock -n`):** Run `p2-fallback finite` on `finite-c1.uort.u16`. Asserts byte-for-byte equality between CPU and GPU builders, shifted batch parity (`128 == 97`), and `0` divergent rows across all 14 arms (`bytes <= 8192`).
- **Gate 3 — Full-`C` `100,000,000` B Table Materialization (`<= 100,000,000` B per arm, `<= 180 s` guard):** Build all 14 `UORS3T02` tables from `c.tokens.u16`, validating strict `<= 100,000,000` B and full cold `S3FallbackTable::from_bytes` reload on each arm under `timeout 180`.
- **Gate 4 — `DEV-tune` Pick + `R`/`DEV-tune` `UORS3F02` Capture (`flock -n`) + Unlocked Native Verify:** Run `p2-fallback devtune-pick` once (`>= 8,000` native `observe_identity_pick` probes, `0` diffs), then for each of the 14 arms run `p2-fallback capture` at shifted batches `65,536` and `65,521` (`cmp`-identical) followed outside `flock -n` by `p2-fallback verify-tape` (`100%` `packed_sha256` match and `>= 12,500` probes with `0` divergent rows on `mass_q48`, `hit_kind`, `pred`, `report_pk`, `pick_q16`, and `pick_denom`).
- **Gate 5 — Frozen `R`-Fit EM + `DEV-tune` Paired Bootstrap & `U_geo` vs `U_count` (CPU):** Run `p2-fallback eval`: asserts `P1(A1s)` `DEV-tune` oracle `125_750_699_315` Q12 (`1,400,045` micro-BPB), fits 64-pass Q32 integer EM strictly on `R` for `standalone_100mb` and `combined_with_p1_a1s` across all 14 arms, evaluates frozen on `DEV-tune`, runs 1,000-resample complete-record block bootstrap (`seed = 4390`), and reports oracle `U_geo` vs `U_count` (`128,614,178,256` Q12 / `1,431,925` micro-BPB) for each of the 4 fallback arms in priority order (`1. vsa -> 2. hopf -> 3. h4_600cell -> 4. golden_shell`).

## 4. Branch Outcomes & Stop Rules

- **First passing fallback arm (`gate_p2_standalone_1pct` or `gate_p2_combined_1pct` with `<= 99%` of best equal-byte control bits and 95% CI upper bound `< 0`):** That arm advances to pre-registered Stage B2c single-shot sealed `DEV-test`.
- **All four fallback arms `NEGATIVE`:** Close all four `100 MB` fallback configurations on `DEV-tune` without touching sealed `DEV-test`, record the exact `U_geo` ceiling and micro-BPB gap for every arm plus the `DECISION.md:298` §3.1 owner-queue summary in `REPORT.md`, and advance to Stage B3 (`P3`).
- **Stop immediately if:** any non-alias SASS float/MMA opcode appears, `DEV-test` is touched, any `C128` or capture parity probe diverges, `p1_all.bits_q12 != 125_750_699_315`, any source or binary SHA mismatches `GO.json`, or any single arm build exceeds 180 seconds.
