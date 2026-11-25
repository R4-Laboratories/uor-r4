# REPORT: written-state-count-model, antigravity-2

Lab: antigravity. Order: [ORDER.md](ORDER.md). Sheet: [SHEET.md](SHEET.md). FIT: [FIT.md](FIT.md).

## What did not work, first

- **P1(b) pruning-cost gate (`NEGATIVE_PRUNING_COST`):** On sealed D87 `DEV-test` (`5,321` records, `21,349,576` non-BOS targets, `43,326,982` target bytes), the frozen `codex-37`-selected five-class `A1s` candidate (`UORS5W02`, `3,999,237,441` B deployed with tokenizer `<= 4,000,000,000` B) scored **`278,311,440,324` Q12 (`1.568240` BPB)** against canonical `R*`'s **`251,672,950,854` Q12 (`1.418136` BPB)** — a ratio of **`1.105845 × R*`**, missing the registered **`1.03 × R*` (`1.460680` BPB)** P1(b) gate by **`+107,560` micro-BPB** (95% paired record-block bootstrap on `100·A1s − 103·R*`: `[+101,091, +113,939]` micro-BPB, `[+1,794,036,140,772, +2,022,034,300,407]` scaled Q12). On the novel subset (`14,795,227` targets, `31,328,319` B), `A1s` reached **`1.045835 × R*`** (`1.823672` vs `1.743746` BPB; `+21,982..+32,746` micro-BPB above `1.03 × R*`). Per CHARTER §4.1, this closes only the flat uncompressed `12 B/entry` threshold-`318` pruning configuration; it does **not** close the written-state count or geometric mechanism.
- **First Phase C launch stopped on `A0` (`disabled`) weight validation:** The first `p1b-devtest-eval score` invocation aborted cleanly before GPU scoring (`Error: "R-fit carrier has wrong class count or nonzero disabled Pick weights"`, [evidence/eval/aborted-v1.score.stderr](evidence/eval/aborted-v1.score.stderr)). Inspecting the sealed `codex-37` `A0` artifact (`disabled/a1.weights`) confirmed `0` nonzero Pick weights across all reachable `class 0` buckets (`0..176`), while unvisited buckets `176..528` (`total == 0` in `device.cu` `maximize`) retain the `initialize` `1` Q32 unit out of `2^32`. Fixed in `validate_mode_pick_weights` (`eval/src/main.rs` + unit test) matching `FrozenPickMixture::with_rule` semantics, re-audited via non-author `FIT` addendum ([FIT.md](FIT.md)), and executed cleanly in the relaunched `eval/run-v1` ([evidence/eval/run-v1.score.stderr](evidence/eval/run-v1.score.stderr)).
- **Host/scope limits:** `eval` uses synthetic `8,193`-row finite controls on GPU (`49,158` words) plus `13,607` predetermined native-vs-GPU probes per arm on real `DEV-test`. No reply-panel benchmarks (`P1(d)` / `P3`), `P2` geometric state rows (`S3`), or post-`#4237` streaming `uor-chat` RSS changes were tested in this unit.

## What changed

- [`docs/labs/kn-reference/instrument/src/main.rs`](../../../../labs/kn-reference/instrument/src/main.rs): added `--write-masses PATH` (refusing `--count-only`, `--extension-census`, `--fit-frame`, `--pick-arms`, duplicate flags, and pre-existing files via `symlink_metadata` + `OpenOptions::create_new(true)`) to persist canonical little-endian `u64` Q48 target masses after `--check-host` parity passes, plus unit test `write_masses_flag_parses_and_refuses_conflicts`.
- [`docs/loop/tasks/written-state-count-model/antigravity-2/`](.): added `ORDER.md`, `SHEET.md`, non-author `FIT.md` (with Phase C addendum), `PRE-RUN.md`, `GO.json`, `dispatch.sh`, standalone `#![forbid(unsafe_code)]` Rust crate `eval/` (`Cargo.toml`, `Cargo.lock`, `src/main.rs`, `src/score.rs`; post-Wick host-compile fix removed `#[cfg(any(test, feature = "cuda"))]` on the `bits_q12` import in `eval/src/main.rs`, moving its SHA-256 from Wick-evaluated `010a58b875fb51ff87473839345adf7b7568474a90bf837f7866cb98317a06b0` to `e354f1042d07f0e8c8dec3adb15a6b726fe43be3f0bf90fa49df21fa0cb897ad` with zero AST/binary change under `--features cuda` or `cfg(test)`), compact Wick receipts under `evidence/`, and updated [`docs/research/notebook/m4-serving.md`](../../../../research/notebook/m4-serving.md) + [`docs/roadmap/items/written-state-count-model.md`](../../../../roadmap/items/written-state-count-model.md).

## Native-space cycle verification (D85)

- **Representation & carried fields:** Original `UORT` `u16` token IDs (`BOS = 0`, `EOS = 1`), BOS-reset `10`-token `PrimeRouteKey` counts (`UORKNC02` threshold `318`, `u32`/`u64` raw counts/totals), `192`-token causal `IdentityControlV1` Pick window `(position, pair_distance, tie_count)` over `trunk.d11` (`04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d`), lossless `UORS5W01` (`7,392` Q32 words) and `UORS5W02` (`12,320` Q32 words) `FrozenPickMixture` weights, integer Q48 masses (`1..=2^48`), and PRLL Q12 `bits_q12`. Zero floats, zero matmul, zero softmax.
- **Single operator paths:** `NgramCounter` + `KnScorer` (`host = GPU`), `d36-count-batch` (`UORKNC02` CUDA) + `uor_r4_count_runtime::CompactCountLookup`, `PickDevice` + `IntegerDiscreteModel::observe_identity_pick`, `d37-tune-device` (`device.cu`) + `FrozenPickMixture::predict_mass`, and `lawful_likelihood::bits_q12`.
- **Replay & identity checks (`0` divergent words everywhere):**
  - Phase A canonical `R*`: `kn-reference --sealed-run --check-host --write-masses` matched `host == GPU` across all `21,349,576` targets (`masses_sha256 == 59f076093c51b15469b55b01821947b7c51d8038e68d834465e29be72d56f8de`, `170,796,608` B, `masses_at_floor == 4`).
  - Phase B `UORCBF01`: `d36-count-batch` at batch widths `65,536` and `65,521` produced byte-identical `2,988,940,752`-byte tapes (SHA-256 `27af4908bfa16dc28a2f8e1dcd5e7c5ff0b22161b7ec57031ce182ba7aa08d9d`); independent native `d37-count-replay` matched all `16,507` sampled probes (`23` words each).
  - Phase C `p1b-devtest-eval`: matched all `13,607` predetermined probes per arm (`frame <= 64` ends and first 8 prefix positions, `8192`/`4096`/`65536` seams, `ordinal % 8191 == 197`, final 2 targets) across Pick blocks/ties/hits, `CompactCountLookup` counts/totals/KN mass, `FrozenPickMixture::predict_mass` bucket/Q48 mass, and `bits_q12`.

## Measurements (Wick RTX 4090 `GPU-7c813ad0-2151-bb1d-5efb-a3340b9a6a41`, `$0.00` pod spend)

Sealed D87 `DEV-test` (`5,321` records, `21,354,897` tokens, `21,349,576` targets, `43,326,982` all B; `14,795,227` novel targets, `31,328,319` novel B; `18,751,676` response targets, `37,995,416` response B):

| Model / Arm | All Q12 | All micro-BPB | Novel Q12 | Novel micro-BPB | Response micro-BPB | Novel resp micro-BPB | vs `R*` (All / Novel) |
|---|---:|---:|---:|---:|---:|---:|---|
| **`R*` (canonical, unpruned `N=10`)** | `251,672,950,854` | `1,418,136` | `223,758,978,608` | `1,743,746` | — | — | `1.000000×` / `1.000000×` |
| `1.03 × R*` (`P1(b)` bar threshold) | `259,223,139,379` | `1,460,680` | `230,471,747,966` | `1,796,059` | — | — | `1.030000×` / `1.030000×` |
| `A0` (`UORS5W01`, threshold-`318`, no Pick) | `295,844,864,503` | `1,667,038` | `249,862,662,991` | `1,947,172` | `1,781,175` | `1,949,602` | `1.175513×` / `1.116659×` |
| **`A1s` (`UORS5W02`, threshold-`318` + Pick)** | `278,311,440,324` | `1,568,240` | `234,015,065,683` | `1,823,672` | `1,675,412` | `1,826,666` | **`1.105845×` / `1.045835×`** |

- **`A1s` per-class Pick firing (`DEV-test`):** Closed `10,642` (`0` hits); distance `0`: `8,114,555` (`4,440,202` hits, `54.72%`); distance `1`: `0` (`0` hits); distance `2`: `74,374` (`12,945` hits, `17.41%`); distance `>= 3`: `13,150,005` (`1,006,755` hits, `7.66%`).
- **Paired record-block bootstrap (`5,321` records, `1,000` resamples, splitmix64 seed `4390`):**
  - `a1s_minus_103pct_rstar` (scaled `100·A1s − 103·R*`): All **`[+101,091, +113,939]` micro-BPB** (`strictly_below_zero = false`), Novel `[+21,982, +32,746]` micro-BPB (`strictly_below_zero = false`).
  - `a1s_minus_rstar`: All `[+143,465, +156,994]` micro-BPB, Novel `[+74,385, +84,926]` micro-BPB.
  - `a1s_minus_a0` (**S5 Pick gain out of sample on `DEV-test`**): All **`[−103,336, −94,001]` micro-BPB** (`strictly_below_zero = true`), Novel **`[−129,287, −117,257]` micro-BPB** (`strictly_below_zero = true`).
  - `a0_minus_rstar`: All `[+240,123, +257,371]` micro-BPB, Novel `[+196,407, +210,106]` micro-BPB.
- **Wick runtime & RSS:** Phase A (`kn-reference`) `6:46.04` wall, `13,612,268 KiB` max RSS (`12.98 GiB`); Phase B (`d36-count-batch` `65536` `16.24 s` wall / `15.96 s` GPU `total_ms` / `250,516 KiB` RSS; `65521` `16.52 s` wall / `16.06 s` GPU `total_ms` / `251,232 KiB` RSS; `d37-count-replay` `9.76 s` wall / `3,906,952 KiB` RSS); Phase C (`p1b-devtest-eval`) `6:05.12` wall, `5,038,252 KiB` max RSS (`4.80 GiB`). Custody root on Wick: `/mnt/plexpool/uor-r4-corpus/runs/20261124-antigravity-p1b-devtest/`.

## Checks run at the head

- `rustfmt --check docs/loop/tasks/written-state-count-model/antigravity-2/eval/src/main.rs docs/loop/tasks/written-state-count-model/antigravity-2/eval/src/score.rs` — PASS.
- `cargo test --locked --manifest-path docs/labs/kn-reference/instrument/Cargo.toml` — PASS.
- `cargo test --locked --manifest-path docs/loop/tasks/written-state-count-model/antigravity-2/eval/Cargo.toml`, non-CUDA `cargo check --locked`, and `cargo clippy --locked --bins -- -D warnings` — **9 passed, 0 failed**.
- Wick finite controls (`validate.json`, `finite-scorer.json`: `49,158` words across `8,193` rows × 2 arms `0` divergent, `kn-1pm.json` 1‰ `--check-host --write-masses` + overwrite refusal `kn-overwrite.stderr`) — PASS.
- Wick Phases A (`rstar.sha256`), B (`count.sha256`, `cmp 65536.features 65521.features`, `d37-count-replay` `16,507` probes), and C (`eval.sha256`, `13,607` probes × 2 arms `0` divergent) — PASS.
- `local/harness-target/debug/uor-harness check --staged`, `check --only harness.native_space_contract`, `loop report --check`, `loop pr-check --base origin/main --head HEAD --local`, `cargo run -q --locked -p uor-r4-claim-wording -- .`, and `scripts/pod/uor-pod status` (`Running: 0 pod(s)`) — PASS.

## Skills used

- `native-space-cycle`

## Still open

- **P1(b) under `4 GB`:** Compression of `S2` orders `3..10` (`DECISION.md §3.5`: Elias-Fano trie + rank-quantized counts, ~`3.2 B/entry`, or entropy pruning / optional mmap order `> 6` suffix array) so `<= 4 GB` retains orders `4..10` within `<= 1.03 × R*` instead of threshold-`318` discarding `69.48%` of high-order raw support.
- **`1B` follow-up (next unit):** Streaming/mmap `UORLUT01` load in `IntegerDiscreteModel::load` to bring full-dose `uor-chat` peak RSS inside `4.00 GB` without transient `std::fs::read` overlap.

## Plain summary

On the sealed D87 `DEV-test` split (`5,321` records, `21.35M` tokens), we reproduced the exact canonical `R*` target-mass tape (`1.418136` BPB all, `1.743746` novel, matching `claude-422` byte-for-byte with `host = GPU`), captured byte-identical count features from the `3.91 GB` threshold-`318` compact table, and evaluated the frozen `A1s` candidate (`UORS5W02`) against `R*` and no-Pick control `A0` with zero mismatches across `13,607` native replay probes per arm. The S5 Pick session-copy expert gains roughly `0.10` BPB out of sample (`[−103,336, −94,001]` micro-BPB over `A0`), bringing `A1s` to `1.568240` BPB overall (`1.105845 × R*`, novel `1.045835 × R*`), which still misses the `1.03 × R*` (`1.460680` BPB) `P1(b)` bar by `107,560` micro-BPB because flat uncompressed threshold-`318` pruning discards too much order `4..10` mass under `4 GB`.
