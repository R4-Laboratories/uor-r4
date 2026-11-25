# ORDER: written-state-count-model, antigravity-4

Lab: antigravity. Item: [`written-state-count-model`](../../../../roadmap/items/written-state-count-model.md). Base: `7d52854e7`.

## Goal (Stage B1a: `UORKNC03` Reversed-Trie Compression, `R`-Fit & `DEV-tune` Selection)

Replace flat `12 B/entry` `UORKNC02` threshold-`318` pruning with the pure-integer lossless reversed-suffix trie codec `UORKNC03`, compress full `C` (`148,096,362,864 B`) inside `3,912,936,331 B` on Wick (`<= 4.000 GB` deployed + client RSS), refit S5 `A0`/`A1`/`A1s` on reserved `R`, and select the frozen candidate on `DEV-tune` strictly without touching sealed `DEV-test`. `$0.00` pod spend.

## Native-space cycle contract (D85)

1. **Native state schema (`S0`–`S6`):** Original 12-bit `UORT` token IDs (`BOS = 0`, `EOS = 1`, `V = 4096`), BOS-reset `10`-token `PrimeRouteKey` ordered keys packed into `UORKNC03` sections (`0..=10`: `parent_sb`, `parent_deg`/`parent_deg_esc`, 12-bit `child_toks`, `fol_sb`, `fol_deg`/`fol_deg_esc`, `fol_words`, `esc_sb`, `esc16`, `esc32`, exact `raw`/`kn` continuation spectra), `KnDiscounts`, `IdentityControlV1` Pick codes, and Q32 `FrozenPickMixture` (`UORS5W01`/`UORS5W02`) weights, producing integer Q48 masses and PRLL Q12 `bits_q12`. Zero floats, zero matmul, zero softmax.
2. **Shared typed operator paths:** `CompactCountLookup` (`crates/uor-r4-discrete/src/kn_ceiling/runtime/{compact.rs,rank.rs}`, re-exported through `uor-r4-count-runtime`) on CPU/WASM and AOT `nvcc` `lookup_v3.cuh` on CUDA (`cuda-v3/`, audited float-free via `cuobjdump --dump-sass`).
3. **Exact transition replay & bit-identical oracle:** `UORKNC03` decodes bit-identically to `UORKNC02` on identical retained contexts (`visit_counts`, `canonical_digests`, `lookup`, `visit_next_masses`, `device_replay_tables`); `cuda-v3` matches CPU `CompactCountLookup` word-for-word (`0` divergent probes) on `C128` (`T=1`) and `12,500+` real-stream `R` / `DEV-tune` probes.
4. **Update rule:** Deterministic commutative integer whole-context counting (suffix-closed non-decreasing thresholds `T_3 <= ... <= T_10`, orders `0..=2` complete `T=1`) plus integer EM S5 mixture fitting (`d37-r-fit-production`) on reserved `R` only. Zero float master.
5. **Lossless serialization:** Canonical `UORKNC03` byte layout embedded losslessly inside `uor-r4.lut-discrete-count/1` (`DiscreteCountModel::load_client_artifact`), with `encode_compact_lookup_v3` roundtrip & planted-fault refusals.

## Scope & Custody Boundaries

- Writes in `crates/uor-r4-discrete/src/kn_ceiling/**`, `crates/uor-r4-integer/**` (tests/wiring only), `docs/loop/tasks/written-state-count-model/antigravity-4/**`, `docs/research/notebook/m4-serving.md`, and `docs/roadmap/items/written-state-count-model.md`.
- Sealed `DEV-test` is **strictly untouched** in Stage B1a. Zero writes to sealed Wick custody dirs (`/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/` only).

## Done when

1. `UORKNC03` runtime/encoder + parity/refusal tests in `uor-r4-discrete`, `uor-r4-count-runtime`, `uor-r4-integer` (incl. `wasm32` check & `audit_zero_matmul_{serving,wasm}.py` `float=0`) pass.
2. Pre-registered `SHEET.md` + fresh-context non-author `FIT.md` (`TEST FITNESS: FIT`) + `PRE-RUN.md` + `GO.json` pass before Wick GPU runs.
3. On Wick (`$0.00` pod spend): finite `C128` (`UORKNC03 == UORKNC02 == cuda-v3`, SASS float=0), full-`C` `UORKNC03` compression `<= 3,912,936,331 B`, `uor-chat` Wick peak RSS `<= 4.000 GB` (`cmp`-identical decode), reserved-`R` `cuda-v3` (`65536 == 65521`, `>= 12,500` native probes) + S5 EM refit + `DEV-tune` evaluation vs `codex-37` (`1,581,444` micro-BPB) and `R*` (`DEV-tune`).
4. `REPORT.md`, harness checks (`harness.native_space_contract`, `check --staged`, `ratchet`, `claim-wording`, `loop report --check`), fresh `pro` adversarial review, CI, and merge pass.
