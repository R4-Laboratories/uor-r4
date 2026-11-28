# Pre-Registration Test Fitness Record (`antigravity-3`, Round 6 Amended)

- **Verdict:** `TEST FITNESS: FIT`
- **Reviewer:** Non-author fresh-context `research` (`pro`) subagent `1f5d7158-cb22-41be-be24-35cddded2504` (Round 6 amended audit following `r3` diagnostic completion of `hb_l4_d256` on `vast-62899157`, `66 min` / `$0.57`, which proved `4,557 tok/s` single-core Apple M1 decode, `0`-diff RTX 5090 = Apple M1 replay, and exposed unscaled `wd_term = w * 256 / 2^12` crushing `embeddings rms 2,365 -> 797`).
- **Prior Rounds:** Round 4 (`180ccb90`) & Round 5 (`b2674e66`, `TEST FITNESS: FIT`), plus honest `r2` and `r3` diagnostic provenance recorded in `SHEET.md`.
- **Base `main` SHA:** `f48dd8bacc83cbae907d04f1b24c1e9730414e1e`
- **Timestamp:** `2026-11-27T20:01:58Z`

## Audited Round 6 SHA-256 Digests

| File | Exact SHA-256 |
|---|---|
| `crates/uor-r4-native-vector/src/butterfly_rotor.rs` | `9438b932bf378577a9fa11c5bdd6b319085425f214c237604a841f2937831cb1` |
| `antigravity-3/butterfly-pilot/src/butterfly_kernels.cu` | `3abe68f4c5ce5e423685ce4711f2884bc2f237e83d7689b2e27c75b30f720e97` |
| `antigravity-3/butterfly-pilot/src/cuda_backend.rs` | `29b5fe12c53bcddd8b5436b59b79aba94fe90d7514626a2a2bcfbda66626cf87` |
| `antigravity-3/butterfly-pilot/src/main.rs` | `3521c18ada4dffcc4b159eca03af205bb6299f228731b4c32c4ec4c29d8830c3` |
| `antigravity-3/dispatch.sh` | `afb58f26eb0196258a07d93eb96ea4f193fc0d19a3c9fa8f0f93ccaf91fa41ca` |
| `antigravity-3/SHEET.md` | `728a3328f47de2b4d1c651df04cfb3ba8c019e53c6c6500a43fa110d04b09129` |
| `antigravity-3/butterfly-pilot/Cargo.toml` | `58a64219209e6e1652095ae8c3f1340e6dce8735407caead8d68879f0eed1740` |
| `antigravity-3/butterfly-pilot/Cargo.lock` | `95ea9925bbba5e72fb1ab7ee657032fa09f9d1f6b1eb9763073e83ab30f4b1be` |
| `antigravity-3/butterfly-pilot/build.rs` | `d3eda180776cf5fc9a85d207ca319d87d53b558ab68df14cd798bc8fc18cef09` |

## Round 6 Verification Highlights
- **Exact LR-Scaled Decoupled Lion Weight Decay (`wd_term = (w * step_lr_q8) / 2^wd_shift`):** Exhaustive `4,704`-combination parity audit confirmed `0` diffs between Rust `/` and CUDA division-free sign-symmetric shift (`eff_wd = min(wd_shift, 30)`), zero `i64` overflow, exact `0` motion at `step_lr_q8 == 0`, and scale invariance across `lr_q8 = 2,048 -> 1,024 -> 512 -> 128` (`wd_shift = 17`, `lambda = 0.25` in Q15).
- **Zero Charter §3 Regressions:** `50/50` `uor-r4-native-vector` tests, `17/17` `butterfly-pilot` bin tests, `0` clippy warnings (`-D warnings`), `wasm32` check PASS, `check_no_softmax`, `check_no_matmul`, `ratchet d23/d25/d32`, zero CUDA `/`, and `bash -n` PASS.
