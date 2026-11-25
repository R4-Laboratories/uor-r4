# REPORT: written-state-count-model, antigravity-5 (Stage B1b Sealed `DEV-test` `P1(b)` `PASS`)

Lab: antigravity. Item: [`written-state-count-model`](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P1(b)`). Order: [`ORDER.md`](ORDER.md) (`28bf0de46443e26ef164e8bbd3a8f50d7cdea2b4ffa202a4b385dcb9277fca14`). Sheet: [`SHEET.md`](SHEET.md) (`5bb90ab1276b53a3ca23f3d2f3a84d427553a87d52232124b1a49271454741d8`). Non-author FIT: [`FIT.md`](FIT.md) (`56c0e0ea1acbdfa160c073f0a6f8821670ddb21d077e1eacd90397a2f4cf15a7`, `TEST FITNESS: FIT`). Pre-run & GO: [`PRE-RUN.md`](PRE-RUN.md), [`GO.json`](GO.json). Evidence: [`evidence/receipts.sha256`](evidence/receipts.sha256).

## What did not work, first

- **Foreign-job GPU contention delayed execution until Wick became idle (no job touched):** Before finite GPU testing and Phases B–C could acquire `/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock`, Wick's RTX 4090 was running another session's `./local/probe-target/release/train-in-space` (`pid 345771`). We scheduled passive waits without touching `pid 345771`, then ran cleanly at `$0.00` rented pod spend once `nvidia-smi` reported zero active compute apps.
- **No-Pick `UORKNC03` count control `A0` (`UORS5W01`) still misses the `1.03 × R*` bar alone:** On sealed D87 `DEV-test`, pure-count `A0` (S5 Pick disabled) scored **`259,345,936,468` Q12 (`1,461,372` micro-BPB all)** vs canonical `R*` `251,672,950,854` Q12 (`1,418,136` micro-BPB) — `1.030488 × R*` (`+692` micro-BPB above `1,460,680`; `a0_minus_rstar = [+41,360, +45,051]` all / `[+36,142, +38,826]` novel micro-BPB). Clearing `1.03 × R*` (and beating canonical `R*` itself) required both `UORKNC03` high-order retention and S5 Pick session-copy mixing (`A1s − A0 = [−78,294, −70,727]` all / `[−104,592, −94,299]` novel micro-BPB).
- **What `P1(b)` `PASS` does NOT close:** Passing `P1(b)` closes the `<= 4.000 GB` compressed `S2` count base + S5 Pick `<= 1.03 × R*` gate on `DEV-test`, leaving `P2` (E8/Starling state-row geometry), `P3` (error-write updates and reply panels), `P1(d)` (`ft`/`dev` floor), and `P4`/M2–M4 dialogue capability open for their own pre-registered stages.
- **Bulk artifacts intentionally omitted from git:** Intermediate binary `per-record-q12.json` (`e62ce876e6d312307e3faa251641e00ce1fc4f306ccdfe1ab0cb80cc93c0e9d2`, recorded in [`eval.sha256`](evidence/eval/eval.sha256)) and `2.989 GB` `UORCBF01` tapes (`65536.features == 65521.features`, `f0e3432d...`) stay on Wick under `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-p1b-v3-devtest/`.

## What changed

- [`docs/loop/tasks/written-state-count-model/antigravity-5/`](.): Added standalone `#![forbid(unsafe_code)]` Rust evaluator [`eval/`](eval/Cargo.toml) (`p1b-devtest-v3-eval`), [`ORDER.md`](ORDER.md), [`SHEET.md`](SHEET.md), [`dispatch.sh`](dispatch.sh), non-author [`FIT.md`](FIT.md) (`0` post-FIT source changes across all 6 audited files), [`PRE-RUN.md`](PRE-RUN.md), [`GO.json`](GO.json), and [`evidence/receipts.sha256`](evidence/receipts.sha256).
- Reused audited Stage B1a [`antigravity-4`](../antigravity-4/REPORT.md) binaries (`cuda-v3` `123df397...`, `lookup_v3.sm_89.cubin` `4c329f9f...`, `eval-v3-tune-device` `9d717ca3...`, `0` SASS floats) and frozen artifacts (`full-c.v3` `75ccb234...`, `artifact.d11` `dd217c0d...`, `A1s` `e2fbf41d...`, `A0` `9ec8168a...`) plus sealed [`antigravity-2`](../antigravity-2/REPORT.md) canonical `rstar-devtest.masses.u64` (`59f07609...`).

## Native-space cycle verification (D85)

1. **Single persistent state schema:** Original 12-bit `UORT` token IDs (`BOS = 0`, `EOS = 1`, `V = 4096`), suffix-closed reversed trie `UORKNC03` (`(T3*=2, T_hi*=6)`, `full-c.v3` `3,804,378,885 B`), `192`-token causal `IdentityControlV1` Pick window over `seed1.d11` (`04652a3d...`), lossless `UORS5W01` (`7,392` Q32 words) and `UORS5W02` (`12,320` Q32 words) `FrozenPickMixture` weights, integer Q48 masses (`1..=2^48`), and PRLL Q12 `bits_q12`. Zero floats, zero matmul, zero softmax.
2. **Shared typed operators:** `cuda-v3 cuda` (`lookup_v3.sm_89.cubin`) + `PickDevice` + `eval-v3-tune-device` on RTX 4090 `sm_89`; `CompactCountLookup::from_bytes` / `lookup` + `IntegerDiscreteModel::observe_identity_pick` + `CeilingTable::from_lookups_with_pick_classes` + `FrozenPickMixture::predict_mass` + `lawful_likelihood::bits_q12` on CPU.
3. **Bit-identical oracle & transition replay (`0` divergent rows/words everywhere):**
   - **Phase A (Sealed `R*` & `artifact.d11`):** Verified `rstar-devtest.masses.u64` (`59f07609...`, `170,796,608 B`, `251,672,950,854` Q12 all / `223,758,978,608` Q12 novel, `masses_at_floor == 4`) and `rstar-devtest.json` (`616c8eff...`, `check_host == true`, `sealed_run == true`), plus `artifact.d11` (`dd217c0d...`, `3,891,016,460 B`).
   - **Finite control (`8,193` rows × 2 arms):** `p1b-devtest-v3-eval finite` against `eval-v3-tune-device` compared `49,158` words across the `8,192`-row batch seam with `divergent_words: 0` ([`finite-scorer.json`](evidence/finite/finite-scorer.json)), plus census `validate` ([`validate.json`](evidence/finite/validate.json)).
   - **Phase B (`UORCBF01` `DEV-test` capture & verify):** `cuda-v3 cuda` at widths `65,536` (`16.32 s`, `249,880 KiB` RSS) and `65,521` (`16.22 s`, `250,760 KiB` RSS) emitted byte-identical `2,988,940,752`-byte tapes (`cmp` clean, `tape_sha256 = f0e3432d...`, `cubin_sha256 = 4c329f9f...`); CPU `cuda-v3 verify` ([`native.json`](evidence/count/native.json), `17.69 s`, `3,766,396 KiB` RSS) checked **`27,655` native `UORKNC03` probes with `divergent_rows: 0`**.
   - **Phase C (`p1b-devtest-v3-eval score`, `6:12.36`, `4,932,912 KiB` RSS):** Replayed and verified **`13,607` predetermined probes per arm** across Pick blocks/ties/hits, `CompactCountLookup::lookup`, `FrozenPickMixture::predict_mass`, and `bits_q12` with `0` divergences.
4. **Update rule:** Zero updates or parameter fitting in Stage B1b; strict single-shot evaluation of Stage B1a's frozen fits.
5. **Lossless serialization:** Direct evaluation of deployed `artifact.d11` (`UORP1A01`, `3,891,281,853 B` with `265,393 B` tokenizer `<= 4,000,000,000 B`; CPU `uor-chat` peak RSS `3,791,956` / `3,791,064 KiB <= 3,906,250 KiB` proven in `antigravity-4`).

## Measurements (`$0.00` pod spend, Wick `GPU-7c813ad0-2151-bb1d-5efb-a3340b9a6a41` RTX 4090)

Sealed D87 `DEV-test` (`5,321` records, `21,354,897` tokens, `21,349,576` non-BOS targets, `43,326,982` all B; `14,795,227` novel targets / `31,328,319` novel B; `18,751,676` response targets / `37,995,416` response B; `14,190,847` novel-response targets / `29,964,782` novel-response B) ([`p1b-devtest.json`](evidence/eval/p1b-devtest.json)):

| Model / Arm | All Q12 | All micro-BPB | Novel Q12 | Novel micro-BPB | Response micro-BPB | Novel resp micro-BPB | vs `R*` (All / Novel) |
|---|---:|---:|---:|---:|---:|---:|---|
| **`R*` (canonical, unpruned `N=10`)** | `251,672,950,854` | `1,418,136` | `223,758,978,608` | `1,743,746` | — | — | `1.000000×` / `1.000000×` |
| `1.03 × R*` (`P1(b)` bar threshold) | `259,223,139,379` | `1,460,680` | `230,471,747,966` | `1,796,059` | — | — | `1.030000×` / `1.030000×` |
| `codex-37` `A1s` (`UORKNC02` `T=318`, `antigravity-2`) | `278,311,440,324` | `1,568,240` | `234,015,065,683` | `1,823,672` | `1,675,412` | `1,826,666` | `1.105845×` / `1.045835×` |
| `antigravity-5` `A0` (`UORS5W01`, `UORKNC03`, no Pick) | `259,345,936,468` | `1,461,372` | `228,573,306,501` | `1,781,264` | `1,577,215` | `1,783,863` | `1.030488×` / `1.021515×` |
| **`antigravity-5` `A1s` (`UORS5W02`, `UORKNC03` + Pick)** | **`246,099,676,775`** | **`1,386,732`** | **`215,795,693,587`** | **`1,681,689`** | **`1,496,526`** | **`1,684,865`** | **`0.977855×` / `0.964411×`** |

- **`P1(b)` gate verdict (`p1b-devtest.json`):** **`PASS`** (`deployed_bytes_with_tokenizer = 3,891,281,853 <= 4,000,000,000`; `passes_103pct_all = true`; `passes_103pct_novel = true`). `A1s` improves on canonical `1.03 × R*` (`1,460,680`) by `−73,948` micro-BPB all and **beats unpruned `R*` itself** (`1,418,136`) by `−31,404` micro-BPB all (`−62,057` novel micro-BPB below `R*` `1,743,746`) and `codex-37` `A1s` (`1,568,240`) by `−181,508` micro-BPB.
- **Paired record-block bootstrap (`5,321` records, `1,000` resamples, splitmix64 seed `4390`, 95% empirical CI):**
  - `a1s_minus_103pct_rstar` (`100·A1s − 103·R*`): All **`[−78,439, −69,377]` micro-BPB** (`strictly_below_zero = true`); Novel **`[−120,152, −108,093]` micro-BPB** (`strictly_below_zero = true`).
  - `a1s_minus_rstar` (`A1s − R*`): All **`[−35,024, −27,966]` micro-BPB** (`strictly_below_zero = true`); Novel **`[−66,755, −57,460]` micro-BPB** (`strictly_below_zero = true`).
  - `a1s_minus_a0` (S5 Pick out-of-sample gain): All **`[−78,294, −70,727]` micro-BPB** (`strictly_below_zero = true`); Novel **`[−104,592, −94,299]` micro-BPB** (`strictly_below_zero = true`).
  - `a0_minus_rstar`: All `[+41,360, +45,051]` micro-BPB (`strictly_below_zero = false`); Novel `[+36,142, +38,826]` micro-BPB (`strictly_below_zero = false`).

## Checks run at the head

- `rustfmt --edition 2021 --check docs/loop/tasks/written-state-count-model/antigravity-5/eval/src/*.rs` — **PASS**.
- `cargo test --locked --manifest-path docs/loop/tasks/written-state-count-model/antigravity-5/eval/Cargo.toml` — **6 passed, 0 failed**.
- `cargo clippy --locked --manifest-path docs/loop/tasks/written-state-count-model/antigravity-5/eval/Cargo.toml --bins -- -D warnings` — **PASS (`0` warnings)**.
- Pre-dispatch -> HEAD SHA-256 verification across all 6 [`FIT.md`](FIT.md) files (`ORDER.md`, `SHEET.md`, `dispatch.sh`, `eval/Cargo.toml`, `eval/src/main.rs`, `eval/src/score.rs`) — **6/6 exact match (`0` post-FIT edits)**.
- Wick finite checks (`validate.json`, `finite-scorer.json`: `49,158` words `0` diffs), Phase A–C `dispatch.sh` (`COMPLETE: P1B-V3 DEV-TEST PIPELINE SEALED`), [`evidence/receipts.sha256`](evidence/receipts.sha256) (`19/19` `OK`), and `scripts/pod/uor-pod status` (`0` running pods, `$0/h`).
- `uor-harness check --staged`, `check --only harness.native_space_contract`, `ratchet d23|d25|d32`, `claim-wording`, `loop report --check`, `loop pr-check --local` — **PASS**.

## Skills used

- `native-space-cycle`

## Still open

- **Stage B2 (`P2` geometric state rows):** Test E8 snap + Starling neighbour-miss geometric state rows against equal-byte controls on top of the verified `<= 4.000 GB` `full-c.v3` + `A1s` `P1(b)`-passing base.
- **Stage B3–B4:** Brier error-write updates & held reply panels (`P3` / `P1(d)`) and browser Studio WASM SIMD128 + `vram-wasm` serving (`B4`).

## Plain summary

On sealed D87 `DEV-test` (`5,321` records, `21.35M` targets), Stage B1a's frozen `UORKNC03` + S5 Pick candidate `A1s` (`artifact.d11`, `3.891 GB` deployed with tokenizer, previously measured at `3.79 GiB` CPU `uor-chat` RSS) scored **`1,386,732` micro-BPB all (`1,681,689` novel)** with zero divergent words across `27,655` count probes and `13,607` native mix/Pick replay probes per arm (`$0.00` pod spend). That reaches **`0.977855 × R*` overall (`0.964411 × R*` novel)** — clearing the `<= 1.03 × R*` `P1(b)` gate by `73,948` micro-BPB (`[−78,439, −69,377]` 95% CI) and outperforming canonical unpruned `R*` itself (`1,418,136` micro-BPB all / `1,743,746` novel), unlocking Stage B2 (`P2`).
