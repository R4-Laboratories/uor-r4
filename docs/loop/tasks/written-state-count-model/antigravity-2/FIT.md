# Independent Test Fitness: `written-state-count-model`, `antigravity-2` (Sealed `DEV-test` Paired `P1(b)` Evaluation)

Reviewed cold (read-only) by an independent non-author subagent against [ORDER.md](ORDER.md), [SHEET.md](SHEET.md), [dispatch.sh](dispatch.sh), [`kn-reference`](../../../../labs/kn-reference/instrument/src/main.rs), [`eval`](eval/src/main.rs), [`DECISION.md`](../../../research/reviews/pi-review-2026-11-16/DECISION.md), [`claude-422/REPORT.md`](../claude-422/REPORT.md), and [`codex-37`](../codex-37/tune/RESULT.md).

## Audited Digests

- `ORDER.md`: `c30ae682c7d6f6eb02848f8e06707d97cd9f86de946fca4482b80e4e391ff025`
- `SHEET.md`: `8015eb924f1b65b16584d5403793adca76ba0d79b226e53bffa42c3bf0277244`
- `dispatch.sh`: `7ee82deca8fd5292e344d3fd5ef838ecc481d2bfae94f149cf479c522d565493`
- `eval/Cargo.toml`: `f2e33cd34ddfad6440e60721eeb7c06381f69cc7fde4bab90143027eb2abe1ca`
- `eval/src/main.rs`: `aafc7f453096e2626e50319000468986ca35ca91d72df2fa3e81d9f40d850779`
- `eval/src/score.rs`: `f5c0f3f558d8ff9f48fce3dbe39ad511fc0ccb60d0d11c53d3c2afff39a6544e`
- `docs/labs/kn-reference/instrument/src/main.rs`: `ba602b55154d8f7a413c5cfbbd5c204f2f0f059847176fd314db1a19e7fb9604`
- `evidence/finite/binaries.sha256` (`12faa8636fa2c826ed222aef0362971a5035a036fdb18f1d4e8f9c0344c8454d`):
  - `kn-reference`: `00cd6ef3efd003612fa74d508088a0dc0e693c97a400d62e41ff36026ba2a3f8`
  - `p1b-devtest-eval`: `ee5fe3c103f9639cf62c2df521433bddf690a1e97620548e5e7c248d8cdcb050`
  - `d37-tune-device` (`codex-37` audited scorer): `b44b430c7121624751c920f45783b4b1246cfc3e31cbed5b2c082500ca14744c`
  - `d36-count-batch` (`codex-36`/`codex-37` audited capture): `d7aeeaf1859e5853bf961ab1bde8f64ba37d61ff4e6923a241518281dc2f9fd3`
  - `d37-count-replay` (`codex-37` audited replay): `53435175c783ab644da6e2aa7e80dd0f0efdf5c5702a1b54c4d1b6875ccb4fdd`

## Verification Findings

1. **Frozen input & split identity (`p1b-devtest-eval` & `dispatch.sh`):**
   - `Split::load` verifies `claude-420` `dev-test.json` (`ec048217ac64f05a6519d7e38dd2fabf6e244eca8009b24476b3870cbc0f73f2`), `dev-test.tokens.u16` (`023784b74c98a796401a064d50bf2f83f2f8eb95ab8bbead1cbe2aec6e7a176b`, `42,709,858` B), `id-bytes.u8` (`befb22f6...`), `dev-test.novel.u8` (`5bf3881b...`), `dev-test.mask.u8` (`7e1361f4...`), and `dev-test.segment.u8` (`25b13499...`), enforcing exact BOS/EOS framing over `5,321` records, `21,354,897` tokens, `21,349,576` targets, `43,326,982` target bytes, and `31,328,319` novel bytes (`14,795,227` targets). Pre-run `evidence/finite/validate.json` passed (`c854135c...`).
   - `read_fit_files` loads `A1s` (`a1s_dir`: `fit.seal.json`, `candidate.manifest.json`, `candidate.weights` matching `codex-37`'s `frozen/` layout) and enforces `A1S_WEIGHTS_SHA` (`b6b3da9e94ac33d0d8dc6c192c3117afc4fdc1f8e9c83dbf051010f5c2f31bb0`), `A1S_MANIFEST_SHA` (`133e691cca11e6663163271c63d1f248aa8a9c575d7adf42ae4131316b74ca97`), `UORS5W02` (`12,320` Q32 words), manifest binding, and `prepared` digests.
   - `read_fit_files` loads `A0` (`a0_dir`: `a1.seal.json`, `a1.manifest.json`, `a1.weights` matching `codex-37`'s `disabled/` layout) and enforces `A0_SEAL_SHA` (`5a7d108997da3378f3bee79599b95910155e646dc81d54ae1fafe6891d06267e`), `seal["weight_sha256"]` / `seal["manifest_sha256"]`, `UORS5W01` (`7,392` Q32 words), `pick_mode == "disabled"`, and zero Pick weights (`row[13] == 0`) in every bucket.
   - `CompactCountLookup::from_bytes` validates `compact.counts` (`064b691a54895771df1a957b2f97c91163990b6c6b6e09504c43285a97c49ef8`, `3,912,334,481` B, `C_SHA = cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a`); `load_trunk` validates `trunk.d11` (`04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d`, 4 blocks, context 192).

2. **Phase A canonical `R*` verification (`kn-reference` `--write-masses` + `aggregate_rstar_masses`):**
   - `--write-masses` rejects `--count-only`, `--extension-census`, `--fit-frame`, `--pick-arms`, duplicate `--write-masses`, and pre-existing output paths (`symlink_metadata` + `create_new(true)`), writes raw LE `u64` masses only after `--check-host` passes, and records `masses_bytes` and `masses_sha256`. Finite controls passed (`evidence/finite/kn-1pm.json`: `8,311` targets, `66,488` B, `host = GPU`; `kn-overwrite.stderr` refused overwrite).
   - `dispatch.sh` enforces `--sealed-run --check-host`, `170,796,608` B, `masses_sha256 == 59f076093c51b15469b55b01821947b7c51d8038e68d834465e29be72d56f8de`, `.masses_at_floor == 4`, `.all.bits_q12 == "251672950854"`, and `.novel.bits_q12 == "223758978608"` (matching `claude-422/run1/sealed/sealed.json`). `aggregate_rstar_masses` independently re-verifies all `21,349,576` masses and Q12 sums across `5,321` records.

3. **Phase B & Phase C bit-exact integer replay, `A0` gating, and bootstrap:**
   - Phase B verifies `cmp`-equality between `65,536` and `65,521` `UORCBF01` captures (`2,988,940,752` B), releases the GPU lock, and runs `d37-count-replay` verifying `23` native words/probe across record ends, `65,536` seams, and `ordinal % 4093 == 197`.
   - `eval` enforces `#![forbid(unsafe_code)]`, zero floats, and zero tolerance. `finite-scorer.json` verified `49,158` words across `8,193` synthetic rows for both `PickClassRule::Three` and `PickClassRule::Five` (`0` divergent words) plus planted distance `2 -> 3` class separation.
   - `read_pick` and `Inputs::score` enforce exact native parity on all predetermined probes (`frame <= 64` ends and first 8 positions, `8192`/`4096`/`65536` seams, `ordinal % 8191 == 197`, final 2 targets) against `observe_identity_pick` (4 blocks × 5 words + `successor` + `hit`), `CompactCountLookup::lookup`, `FrozenPickMixture::predict_mass`, and `bits_q12`. `A0` forces `(None, None, None)` -> `class = 0`, `hit = false` on every target; `A1s` enforces causal predecessor bounds (`pred != Some(0)`, `p + 1 < prefix.len()`) and `device_hit == hit`.
   - `p1b_passes_threshold` checks exact integer `100 * bits_q12(A1s) <= 103 * 251_672_950_854`. `paired_band_scaled` computes exact `scale_x * a - scale_y * b` (`100 * a1s - 103 * rstar` with `denom = bytes * 4096 * 100`), outward `div_euclid` integer rounding, `splitmix64` seed `4390`, `1,000` complete-record resamples over `5,321` records, endpoints `[24]` and `[974]`.

4. **Host/GPU admission, memory & custody conditions (non-blocking notes):**
   - `dispatch.sh` holds exclusive `flock -n 8` on `/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock` around live GPU phases (`admit_gpu` checks zero foreign compute PIDs, `>= 8192` MiB VRAM, `>= 10 GiB` RAM) and binds `EVAL_STAGE/GO.json` (`schema == "uor-r4.antigravity-2.devtest-go/1"`, `dev_test_tape_sha256`, `sheet_sha256`, `fit_sha256`, binary digests). Peak Phase C host RSS is ~5.6 GiB on Wick's 59 GiB RAM.
   - Note: `admit_gpu` checks `>= 8192` MiB VRAM whereas Phase A peaked at `11,145` MiB in `claude-422` (Wick's idle 24 GiB 4090 provides headroom; OOM aborts cleanly prior to writing `rstar-devtest.masses.u64`). `a1s_dir/fit.seal.json` and `a0_dir/a1.weights` are verified at Phase C rather than pre-Phase-A in `dispatch.sh`, `df` checks `$stage` (co-located on `/mnt/plexpool` with `$ROOT`), and `within_4gb_cap` (`3,999,237,441` B) cites `codex-37/client/REPORT.md`.
   - Parent `GO.json` must bind `schema = "uor-r4.antigravity-2.devtest-go/1"`, exact staged `SHEET.md`/`FIT.md`/`dispatch.sh` and `binaries.sha256` digests, Wick `GPU-7c813ad0-2151-bb1d-5efb-a3340b9a6a41`, `compact.counts`, `trunk.d11`, `claude-420` split inputs, `a1s_dir` (`frozen/`), and `a0_dir` (`disabled/`).

TEST FITNESS: FIT

## Addendum (Phase C A0 disabled-weight validation & sealed Phase A/B verification)

Audited cold (read-only) after the first Phase C launch aborted cleanly before GPU scoring (`Error: "R-fit carrier has wrong class count or nonzero disabled Pick weights"`).

### Updated Audited Digests

- `dispatch.sh`: `06db09d8ef5790985cbbb09f0fd6aabf8f6c45c277f0675a0fe44252a0744730`
- `eval/src/main.rs`: `010a58b875fb51ff87473839345adf7b7568474a90bf837f7866cb98317a06b0`
- `eval/src/score.rs`: `fc229ddac366995d3af8466e2d5e119123afd2e4ddf486cdad9a4345c4843533`
- `evidence/finite/binaries.sha256` (`a17e8ecb45cf35b68a3340bc3f991f3736ddf26d4630a284feb2b0e664bb6fe6`):
  - `p1b-devtest-eval`: `f87953e5487eddf315c69106ebfaf16e8a1f86f494772f1270ba80ceb7acf19e` (`kn-reference`, `d37-tune-device`, `d36-count-batch`, `d37-count-replay` unchanged)
- `evidence/finite/finite-scorer.json` (`335786a1f4968193c21ccca08a45e807482f37d5fcd3e384d8ea4d04ee634f51`): `{"arms":2,"divergent_words":0,"raw_words_compared":49158,"synthetic_rows_per_arm":8193}`

### Verification Findings

1. **`A0` (`disabled`) Pick-weight validation & runtime gating (`eval/src/main.rs`, `eval/src/score.rs`):**
   - Root cause confirmed in `codex-37/fit/device.cu` and `codex-37/fit/evidence/controls/real-v1/disabled/a1.manifest.json`: `A0` (`disabled`, `UORS5W01`, `528` buckets) classifies all `15,947,584` R targets into `class 0` (`buckets 0..176`), leaving `176..528` unvisited (`total == 0` in `maximize`) with `device.cu` `initialize`'s initial `1` Q32 unit in `row[13]`. `codex-37/tune/score.rs` gated `all(|row| row[13] == 0)` on `mode == "masked"` (`C1`) rather than `"disabled"` (`A0`).
   - `validate_mode_pick_weights` (`eval/src/main.rs:383-399`, called from `read_fit_files` at `eval/src/score.rs:221`) enforces `row[EXPERTS - 1] == 0` across all reachable `disabled` buckets `0..176` (`take(176)`) while keeping `all` buckets zero for `"masked"`; `FrozenPickMixture::with_rule` independently rejects any `row[13] != 0` in `0..176`. Unit test `disabled_mode_allows_unvisited_init_one_and_rejects_active_pick` (`eval/src/main.rs:495-515`) verifies both behaviors.
   - In `Inputs::score` (`eval/src/score.rs:396-444`), `mode == "disabled"` still forces `(pred, distance, device_hit) = (None, None, None)`, `successor = None`, `hit = false`, and `PickTarget { class: 0, hit: false }` on every one of the `21,349,576` `DEV-test` targets, so `CeilingTable::from_lookups_with_pick_classes` and `DeviceScore` address only buckets `0..176` (`bucket == table.buckets()[i]` checked on every row) with zero Pick mass and zero Pick weight. `A0_SEAL_SHA` (`5a7d1089...`) and `manifest_sha` binding remain enforced.

2. **Sealed Phase A & Phase B reuse verification (`dispatch.sh`):**
   - **Phase A (`lines 70-101`):** When `$RSTAR_OUT/rstar.sha256` exists, `dispatch.sh` skips re-running `kn-reference` and unconditionally runs `sha256sum --strict -c rstar.sha256`, `stat -c %s` (`170796608`), SHA-256 `59f076093c51b15469b55b01821947b7c51d8038e68d834465e29be72d56f8de` on `rstar-devtest.masses.u64`, and `jq -e` asserting `.check_host == true`, `.sealed_run == true`, `.masses_sha256`, `.masses_at_floor == 4`, `.all.bits_q12 == "251672950854"`, and `.novel.bits_q12 == "223758978608"`. `p1b-devtest-eval` re-hashes and re-aggregates `rstar-devtest.masses.u64` in Phase C.
   - **Phase B (`lines 109-155`):** When `$COUNT_OUT/count.sha256` exists, `dispatch.sh` skips re-running `d36-count-batch` / `d37-count-replay` and unconditionally runs `sha256sum --strict -c count.sha256`, `cmp "$COUNT_OUT/65536.features" "$COUNT_OUT/65521.features"`, recomputes `TAPE_SHA`, and asserts `native.json` fields (`.sha256 == $sha`, `.words_per_native_row == 23`, `.rows == 21349576`, `.records == 5321`, `.width == 65536`, `.bytes == 2988940752`). `p1b-devtest-eval` streams full SHA-256 verification of `65536.features` against `EVAL_STAGE/GO.json`'s `dev_test_tape_sha256` and replays all kept probes via `CompactCountLookup::lookup`.
   - **Phase C relaunch pre-conditions (operational):** `dispatch.sh:160` refuses existing `$EVAL_STAGE` or `$EVAL_OUT`, so the aborted Phase C `$ROOT/eval/stage-v1` (and failed `score.*` logs) on Wick must be moved/archived aside, and `GO.json` must be updated to bind the updated `dispatch_sha256`, `eval_sha256`, and `fit_sha256` (post-append) before re-dispatch.

TEST FITNESS: FIT
