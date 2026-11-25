# Independent Test Fitness (Round 2): `written-state-count-model`, `antigravity-4` (Stage B1a `UORKNC03`)

Reviewed cold (read-only) by an independent non-author subagent against [CHARTER.md](../../../../CHARTER.md), D85 ([native-space.json](../../../harness/native-space.json)), [ORDER.md](ORDER.md), [SHEET.md](SHEET.md), and all source modules under `antigravity-4/` and `crates/uor-r4-discrete/src/kn_ceiling/`.

## Audited Digests

- `docs/loop/tasks/written-state-count-model/antigravity-4/ORDER.md`: `0324c40601c10f64fb615f950f0d3c67d1cd65ceab610a90db2505f4e3133598`
- `docs/loop/tasks/written-state-count-model/antigravity-4/SHEET.md`: `e48b3ba64c1be2429b1d657fdf5993e7bd6ebbfedc71adfee2f3733266d31a9e`
- `docs/loop/tasks/written-state-count-model/antigravity-4/compress-v3/Cargo.toml`: `49f8ba6ccaadc98bd81c0cdfb0bcd4a7be712e69eed224087cef99c36b713a7d`
- `docs/loop/tasks/written-state-count-model/antigravity-4/compress-v3/src/main.rs`: `9fb66f771ced10fc1ca264d6f784e6bb3a8e45f6246c74545c37358fd2d658e5`
- `docs/loop/tasks/written-state-count-model/antigravity-4/cuda-v3/Cargo.toml`: `15fa05506ba5a1bf6ccebe55b866c61d6767082db473f50fcaff2f443d4fcd45`
- `docs/loop/tasks/written-state-count-model/antigravity-4/cuda-v3/build.rs`: `f78aaf6ee4aac5a6496423e15c610192348825ba1f7b991b836e8aeb01f3a169`
- `docs/loop/tasks/written-state-count-model/antigravity-4/cuda-v3/src/lookup_v3.cuh`: `1e668bc2c6de34f38ad20a11358d90f4ac35a38d64201d89c7229b5c32008ad3`
- `docs/loop/tasks/written-state-count-model/antigravity-4/cuda-v3/src/cuda.rs`: `ebbaf7fb15bcb8a0d06aff52887204fe73c65dd8b060d207868900f091a59f7a`
- `docs/loop/tasks/written-state-count-model/antigravity-4/cuda-v3/src/main.rs`: `6166dbd077115d5c9247e122ee6520b42aaff7c0e590ed510bd976993d59c8a2`
- `docs/loop/tasks/written-state-count-model/antigravity-4/eval-v3/Cargo.toml`: `18c593b2db4464cc83b2f6b46cadc6f557715f2bcaea11f8e501c7f5d8f51247`
- `docs/loop/tasks/written-state-count-model/antigravity-4/eval-v3/src/main.rs`: `b80de4565e056b5522892d329d76e22328ef76f5175f9854243e9fd4bc93f2ce`
- `docs/loop/tasks/written-state-count-model/antigravity-4/eval-v3/src/fit.rs`: `78f79c3971dde2ec6211daa475c1501cbef10f6ec313f8d4c8e90cd66b0d322f`
- `docs/loop/tasks/written-state-count-model/antigravity-4/eval-v3/src/tune.rs`: `ea6832af7238514beef45d3cc7062580ad0b080b4d29a5d167c0929c40ed4a92`
- `docs/loop/tasks/written-state-count-model/antigravity-4/eval-v3/src/fit_device.cu`: `49efc0ccfa0dc5d547b6edb98c482cea8f9a884b3c8b2267a43dfbd64e651450`
- `docs/loop/tasks/written-state-count-model/antigravity-4/eval-v3/src/tune_device.cu`: `42faeb9c25a5171f075cef403f0b394b042a3ae04c1a825476273c59dcf85fd1`
- `docs/loop/tasks/written-state-count-model/antigravity-4/pack-v3/Cargo.toml`: `ac1ca2c175a3ac7fb5aa206bf1db31e0fc95f0255f6fde42d30db2ff5ef688c1`
- `docs/loop/tasks/written-state-count-model/antigravity-4/pack-v3/src/main.rs`: `7ced200a41ae9d688b6e5b69fe4a18d21a0891f17b6ea6383f47a90f48035aa0`
- `crates/uor-r4-discrete/src/kn_ceiling/compact_lookup.rs`: `d855664545dd0636eea5bdfb1a071f819b100d9ba8231086c1811f872626bdaf`
- `crates/uor-r4-discrete/src/kn_ceiling/runtime/compact.rs`: `fe2d1f9e0793d7e475c6e83de822809fe68c769ab27e2faa4bef7f74e8afb7e4`
- `crates/uor-r4-discrete/src/kn_ceiling/runtime/rank.rs`: `ecba4e7abfe957a5ae23fcc0fe526e6eef304abb379f043e9c3441563087f020`
- `crates/uor-r4-discrete/src/kn_score_division.cuh`: `5b2eaa7f1e3844f2412674d15327174f064bb60b86883cd13a5b4133667a2d95`
- `crates/uor-r4-discrete/src/kn_ceiling.rs`: `04701f6124609b2a75b777390c9b0cd6d425ee80d6ff9471a43a85213b1d95e1`
- `crates/uor-r4-discrete/src/kn_ceiling/compact_lookup/replay.rs`: `0d746d386bdb6de9c7cfc7fcaa60c44e45ea1fe240f0086578081ea444e0bf1`
- `crates/uor-r4-discrete/src/kn_ceiling/compact_lookup/ranking_tests.rs`: `2960914026441d761f54edbb1b01f2664e4c81590654b0267d85b35128ef64bc`

## Verification Findings

### 1. Resolution of All Prior Round 1 Findings

1. **`cuda-v3` native probe coverage and explicit `12,500` floor (`cuda-v3/src/main.rs`):**
   - `keep_probe_ordinals` (`cuda-v3/src/main.rs:367-390`) selects `frame <= 64` boundaries (`pos <= 8 || pos == 192 || id == 1`), batch seams `[65_536, 65_521, 8_192]` (`ord % w == 0 || ord % w == w - 1`), periodic interior samples `ord % 1009 == 197`, and the final two targets (`ord + 2 >= total_targets`).
   - Mathematical probe guarantees:
     - On `R` (`15,947,584` targets): `ord % 1009 == 197` alone selects `15,805` probes; combined with `8_192` seams (`~3,892`), `65_521` seams (`~486`), and `frame <= 64` (`~600`), total coverage exceeds `20,600` native probes (`>= 12,500`).
     - On `DEV-tune` (`10,807,417` targets): `ord % 1009 == 197` alone selects `10,711` probes; `8_192` seams add `2,639` probes; `65_521` seams add `329` probes; `frame <= 64` adds `~600` probes. Accounting for minor set intersections (~3-4 overlaps), total coverage is `~14,270` native probes (`>= 12,500`).
   - `verify_tape` (`cuda-v3/src/main.rs:427-439`) explicitly sets `min_probes = 12_500` whenever `!all_rows && input.targets > 1_000_000` (covering both `R` and `DEV-tune`) and returns an `Err` if `kept.len() < min_probes`.

2. **Dual tokenizers pinned, `pack-v3` deployment packer added, and Gate 6 parameters locked (`SHEET.md`, `eval-v3`, `pack-v3`):**
   - **Dual tokenizers separated and pinned:** `SHEET.md` §2 explicitly pins both the `DEV-tune` census tokenizer (`/home/callard/uor-codex/20261118-diagnostics/inputs/c420/tokenizer.json`, `265,393` B, SHA-256 `d36d3e8700a123e620012df77de195f244fbdb4d05d9e1aa7e77fa9407590f89`, enforced by `eval-v3/src/main.rs:48-49` and `eval-v3/src/tune.rs:50`) and the `seed1.d11` serving tokenizer (`/home/callard/uor-codex/20261118-diagnostics/inputs/tokenizer.json`, `265,393` B, SHA-256 `a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377`, bound in `seed1.d11` `04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d` and checked at runtime by `uor-chat` at `crates/uor-r4-integer/src/bin/uor-chat.rs:2285`).
   - **`pack-v3` deployment packer (`antigravity-4/pack-v3/src/main.rs`):** Enforces `#![forbid(unsafe_code)]`, pins `C_SHA`, `R_SHA`, `TRUNK_SHA`, `COMPACT_CAP_BYTES = 3_912_936_331`, and `MAX_DEPLOY_BYTES = 4_000_000_000`; validates fit manifest (`uor-r4.d37.r-fit/1`, `pick_mode == "identity"`) and `FrozenPickMixture::from_bytes`; builds `UORP1A01` state; writes `uor-r4.lut-discrete-count/1` via `write_deployment` (refusing overwrite); enforces `deploy_len <= 4_000_000_000`; and re-verifies streaming readback via `load_client_artifact(out_d11, false)` and `DiscreteCountModel::bind_embedded` (`count_sha256 == compact_sha`, `trunk == TRUNK_SHA`, `state_sha256`).
   - **Gate 6 prompt and decode parameters pinned (`SHEET.md` §4 item 6):** Fixes exact `uor-chat` command (`--discrete artifact.d11 --tokenizer /home/callard/uor-codex/20261118-diagnostics/inputs/tokenizer.json --say "Write a clear one-paragraph explanation of discrete geometry." --max-tokens 128 --temperature 0 --p1-trace`), two fresh `/usr/bin/time -v` runs, `cmp` byte-identity between runs, and cold-load + 128-token warm tok/s capture.

3. **`compress-v3 verify-v2` row-by-row parity and deterministic `c1.tokens.u16` finite slice (`compress-v3/src/main.rs`, `SHEET.md`):**
   - `verify_v2_parity` (`compress-v3/src/main.rs:855-903`) verifies `v2` and `v3` SHA-256 digests, constructs `CompactCountLookup::from_bytes` on both artifacts under `train_pin`, collects all `(order, key, next, raw, adjusted)` tuples via `visit_counts`, enforces exact length and row-by-row equality (`divergent_rows == 0`), and is unit-tested for both lossless acceptance and pruned-table refusal (`compress-v3/src/main.rs:1187-1190`).
   - `slice_uort` (`compress-v3/src/main.rs:905-972`) verifies `c1.tokens.u16` (`8d94e88ac31305ad6973734181a9727f520da46a5ecc263b944247ceedd07030`), slices deterministically to `<= 256` tokens cleanly terminated at EOS (`1`) with valid BOS/EOS framing (`write_atomic` overwrite guard), pinned in `SHEET.md` §2 & Gate 2 (`finite-c1.uort.u16`).

4. **Hygiene items resolved:**
   - **Strict decimal 4.000 GB RSS bar:** `SHEET.md` Gate 6 & Gate 7 enforce `Maximum resident set size <= 3,906,250 KiB` (`3,906,250 * 1024 = 4,000,000,000 B`).
   - **`TMPDIR` on `/mnt/plexpool`:** `SHEET.md` §2 pins `TMPDIR=/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/tmp` matching `std::env::temp_dir()` usage in `compress-v3` (`histogram` & `write_v3`).
   - **`deg` bounds check:** `compress-v3/src/main.rs:270, 617` enforce `deg == 0 || deg >= NGRAM_VOCABULARY as usize` (`1..=4095`), matching `compact.rs:230`.
   - **`eval-v3/src/main.rs` digest binding:** `eval-v3/src/fit.rs:512` records `"eval_main_source_sha256": sha(include_bytes!("main.rs"))` in `prepare.json`, and `seal` (`fit.rs:550`) verifies it alongside `fit.rs` and `fit_device.cu`.
   - **`// SAFETY:` comments:** Both `unsafe` blocks in `cuda-v3/src/cuda.rs` (`lines 46-47` and `94-95`) carry accurate `// SAFETY:` invariants covering chunk-by-chunk device initialization and bounded kernel launch dimensions; all portable Rust crates and binaries enforce `#![forbid(unsafe_code)]`.

### 2. CHARTER §3, §4.5, and D85 Compliance Audit

1. **Zero floats, zero matmul, zero softmax, no NVRTC (CHARTER §3.1–§3.4):**
   - Full source scan across `compress-v3`, `cuda-v3`, `eval-v3`, `pack-v3`, `uor-r4-count-runtime`, and `crates/uor-r4-discrete/src/kn_ceiling/**` confirms zero floating-point types/operations, zero dense matmul/GEMM, and zero softmax/squashing functions.
   - `cuda-v3/build.rs` AOT-compiles `lookup_v3.cuh` with `nvcc -cubin -arch=sm_89 -O3` using `kn_score_division::divide` restoring integer division (`u128`/`u64`). `cuda-v3/Cargo.toml` and `crates/uor-r4-discrete/Cargo.toml` pull `cudarc` with `default-features = false, features = ["std", "driver", "cuda-12080", "dynamic-linking"]` (no `nvrtc`).
   - `eval-v3/src/{fit_device.cu,tune_device.cu}` match the audited `codex-37` CUDA EM/forward kernels (restoring integer division `fraction()` and integer `bits_q12()`). Gate 1 enforces `cuobjdump --dump-sass` `0` float opcodes across all three CUDA binaries prior to GPU runs.

2. **One persistent native state & bit-identical replay (CHARTER §3.7, D85):**
   - Schema `S0`–`S6` uses 12-bit `UORT` token IDs (`BOS = 0`, `EOS = 1`, `V = 4096`), `PrimeRouteKey` suffix-closed reversed-trie sections `0..=10` in `UORKNC03`, `KnDiscounts`, `IdentityControlV1` Pick codes, Q32 `FrozenPickMixture` (`UORS5W01`/`UORS5W02`), Q48 masses, and Q12 `bits_q12`.
   - Verified cross-path structural parity across `compress-v3` (`write_v3`, `section_bytes`), `crates/uor-r4-discrete/src/kn_ceiling/runtime/{compact.rs,rank.rs}` (`CompactCountLookup::from_bytes`, `visit_counts`, `lookup`, `visit_next_masses`, `device_replay_tables`), and `cuda-v3/src/{lookup_v3.cuh,cuda.rs,main.rs}` (`OrderV3` 144 B / `Probe` 24 B / `OrderResult` 32 B alignment, 12-bit `child_tok` packing, `span_from_sb`, `decode_inline`, `esc_rank`, `esc16`/`esc32`, `step()` Q48 interpolation and `max(mass, 1)` floor). Because `ngram_admitted` forbids `BOS = 0` in non-oldest context slots `1..=k-1`, `V3` trie traversal in `CompactCountLookup::lookup` and `visit_next_masses` terminates cleanly at `BOS` identically to `V2` and `Rows::fill`.
   - `pack-v3` embeds `UORKNC03` + `a1`/`a1s` `FrozenPickMixture` + manifest losslessly inside `uor-r4.lut-discrete-count/1` (`UORP1A01`) and verifies cold streaming load via `load_client_artifact`.

3. **Custody, split lock, & pre-registration gates (CHARTER §4.5, §5.4):**
   - `DEV-test` isolation is enforced in `SHEET.md` §1 and hardcoded in `eval-v3/src/tune.rs:42` (`sha(&census) == CENSUS_SHA`, `"wrong DEV-tune census (DEV-test refused)"`, covered by unit test `tune_load_refuses_non_dev_tune_census`).
   - Deterministic `(T3*, T_hi*)` selection rule (`SHEET.md` §3: minimal `t_hi` fitting `<= 3,912,936,331` B, then minimal `t3` within `T_hi*`) and deterministic `choose` rule (`eval-v3/src/tune.rs:750-756`, `a1s_minus_a1.all.q12[1] < 0`) leave zero post-hoc tuning degrees of freedom.
   - Fresh run root `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/` satisfies `L-002`; Wick local RTX 4090 under `flock` on `wick-codex-gpu.lock` during active GPU steps incurs `$0.00` pod spend.

### 3. Operational Pre-Run Notes (Non-Blocking)

1. Export `TMPDIR=/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/tmp` in the Wick execution shell prior to running `compress-v3` on `148 GB` `full-c` so intermediate per-order temp files (`k*.tot` / `keys_k*.bin`) do not hit `/tmp`.
2. Build `uor-chat` on Wick with `--features p1-count-serving` (required by `crates/uor-r4-integer/Cargo.toml` for `uor-r4.lut-discrete-count/1` and `--p1-trace`), run Gate 6 with `CUDA_VISIBLE_DEVICES=""` under `/usr/bin/time -v`, and set `UOR_P1_TIMING=1` when capturing the `uor-r4.p1-client-timing/1` cold-load and 128-token decode timing breakdown alongside RSS.
3. Run `compress-v3 write` without `--no-verify` (default) so cold `CompactCountLookup::from_bytes` validation runs inline on both `C128` and `full-c`.

TEST FITNESS: FIT
