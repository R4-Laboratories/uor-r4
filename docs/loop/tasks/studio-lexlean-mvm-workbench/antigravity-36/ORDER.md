# ORDER: studio-lexlean-mvm-workbench, antigravity-36

Lab: antigravity. Item: [`studio-lexlean-mvm-workbench`](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (Issue `#3939`, M7).

## Task (Stage A1: `Casey-allard/lexlean-r4` Sync & Core Gap 4 Port)

Execute Stage A1 of the approved 5-repo integration plan (`A1 -> A2 -> A3 -> A4 -> A5`, followed by Phase B `1 -> 2 -> 3 -> 4`) at `$0.00` GPU spend:

1. **Core `crates/uor-r4-integer` Gap 4 Paraphrase Continuation Handshake (`R4` `/8 -> /7`):**
   - Extend [`StackPointKey`](../../../../../crates/uor-r4-integer/src/stack/format.rs) with optional `#[serde(default, skip_serializing_if = "Option::is_none")] anchor_len: Option<usize>` and optional `memory.point.anchors` table in [`point_key.rs`](../../../../../crates/uor-r4-integer/src/stack/point_key.rs) & [`session.rs`](../../../../../crates/uor-r4-integer/src/stack/session.rs) so when `ServedPointKey` matches on a paraphrased prompt opening (`point_miss`), `session.memory.anchors.reply` updates to the winning row's canonical reply anchor while keeping byte-identical behavior for existing `/8`/`/10` artifacts without anchors.
2. **SpiralCore Stateless Edge Relay (`R4-39`) & Browser Tool Integration:**
   - Import MIT/Apache-2.0 zero-dependency relay files from `Casey-allard/lexlean-r4` (`relay/{worker.mjs,local-relay.mjs,relay.test.mjs,README.md}`) into `web/studio/relay/`, add opt-in [`spiralcore-relay.mjs`](../../../../../web/studio/foundry/studio/shell/spiralcore-relay.mjs) (zero network calls on boot; strict `wss://`/`https://` + loopback-only `ws://`/`http://`), and expose opt-in `web_search` in [`r4-agent-loop.mjs`](../../../../../web/studio/foundry/studio/shell/r4-agent-loop.mjs) & [`host.js`](../../../../../web/studio/foundry/studio/shell/host.js).
3. **`R4-54` `.holo` BLAKE3 Footer Audit & `R4-60` Off-Origin Hydration Consent Gate:**
   - Audit [`holo.js`](../../../../../web/studio/foundry/studio/shell/holo.js) BLAKE3 footer verification against `core.wasm`.
   - Enforce opt-in off-origin weight download consent (`ConsentRequiredForWeightDownload`) and hydration provenance reporting in [`uor-r4-runtime.mjs`](../../../../../web/studio/foundry/studio/shell/uor-r4-runtime.mjs) while keeping same-origin, loopback mirrors, OPFS/IDB cached loads, and explicit user-selected models seamless.
4. **Cross-Repo Engine Audit & Tests:**
   - Document `wasm-v13-foundry` (`0b5b6d30a`), `native_q31_lane.rs`, and `sparse_basin_v15.rs` audit findings in `REPORT.md`; add unit & integration tests across [`stack/tests.rs`](../../../../../crates/uor-r4-integer/src/stack/tests.rs) and [`studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs).

## Cancel probe

- Any edit touches `CHARTER.md`, `DECISIONS.md`, `docs/harness/native-space.json`, `Cargo.lock`, or `src-tauri/**`.
- Any D23/D25/D32 ratchet (`4451`/`93`/`31`), `audit_zero_matmul_wasm.py` (`float=0`), or Chrome CDP Studio verification regresses.

## Scope

Writes inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-36/**`, `crates/uor-r4-integer/src/stack/**`, `crates/uor-r4-integer/src/bin/fbs-export.rs`, `web/studio/relay/**`, `web/studio/foundry/studio/shell/**`, `web/studio/foundry/studio/tools/**`, `scripts/studio/stage-studio-site.sh`, and `scripts/studio/runtime-files.sh`.

## Done check

`cargo test --locked -p uor-r4-integer --lib`, `node --test web/studio/relay/relay.test.mjs`, `bash web/studio/build_foundry_studio.sh`, `hologram.typecheck.mjs`, `host_bundle.mjs --check`, `shell_parse.mjs`, 9 Foundry suites, `command_ownership_registry_browser.mjs`, `verify_studio_worker.mjs`, `runtime-files.sh --verify web/studio/dist --typed`, `audit_zero_matmul_wasm.py`, Chrome CDP `verify_foundry_studio.mjs`, `uor-harness` checks/ratchets/`loop pr-check`, fresh `pro` exact-head review, merge, and live `gh-pages` Studio deploy PASS at `$0.00` GPU spend.
