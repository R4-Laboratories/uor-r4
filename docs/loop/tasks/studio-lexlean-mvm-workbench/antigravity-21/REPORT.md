# REPORT for Decision D91 & Phase 5 Qwen3-1.7B v3 Pre-Registration (`antigravity-21`)

Lab: `antigravity`. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939). Base: `f9bd3b425` (PR #4222, [`antigravity-20/REPORT.md`](../antigravity-20/REPORT.md); `ORDER.md` committed alone first at `4e65d7d51`).

## What did not work

- **D90 `Qwen3-0.6B` hit a parametric generalization ceiling across Runs 8–10 (`$2.7427 / $3.00`, closed at PR #4222):** In Run 10 (`v2.3`), `HoloCycleGuardLogitsProcessor` cut decoding loops (`low_unique_ratio: 8 -> 1`, `repeated_4gram: 3 -> 0`) and tool/build held at `317/340` (`93.24%`), while held-out Chat plateaued at `78/100` (`78.0%`) with `val_loss_chat` rising monotonically (`3.18798 -> 3.57613 -> 3.64605`) across `15..16×` canned target repeats and unseen Foundry essays confabulating details. Because continuing `0.6B` could not overcome capacity/memorization limits cleanly inside D90's `$3.00` cap, owner [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43) (`2026-11-24T04:32:05Z`) extends the temporary Studio-only (`web/studio/`) exemption to **`Qwen/Qwen3-1.7B`** (`+$3.00` fresh Vast.ai cap), followed by returning to core native UOR-R4 M2+ milestones.
- **Host unit-test environment caveat & pre-run guardrails (`$0.00` offline checks):**
  - Laptop system `python3` has no `torch` installed, so `HoloCycleGuardLogitsProcessor` parity (`7/7` directed assertions plus independent `5,000`-trial + `1,000`-stress-trial differential fuzzing inside `35f1e7c6-fc5d-46c3-be08-7b27cfcd5abd` against `holo_geometric_runtime.rs:6762-6827`) used lightweight in-memory `torch`/`transformers`/`peft` stubs; live `torch 2.8.0+cu128` training with gradient checkpointing executes on the GPU pod after merge.
  - Non-blocking operational notes identified during FIT review (`35f1e7c6-fc5d-46c3-be08-7b27cfcd5abd`): Vast's cheapest verified `5090` offer fluctuated to `$0.908/h` (so `UOR_POD_MAX_RATE=0.65` enforces waiting for a `<= $0.65/h` host), and `foundry_v13_eval_pack_v2 --mode pack` must pass `--mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry` explicitly since `DEFAULT_MIRROR_URL` points to `weights-v13-qwen3-0.6b-foundry`.
- **No GPU pod launched in this PR:** `uor-pod status` confirms `Running: 0 pod(s) ($0/h of $8.00/h)`; `d91-foundry-17b-v1` awaits merge of `antigravity-21` and final pre-run launch execution.

## What changed

- **Recorded Owner Decision [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43) (`2026-11-24T04:32:05Z`):**
  - [`docs/integration/DECISIONS.md`](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43): Added `D91` row extending D90's temporary Studio-only (`web/studio/`) exemption to `Qwen/Qwen3-1.7B` (`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, `UORR4V13`, `weights-v13-qwen3-1.7b-foundry`) with a fresh `+$3.00` Vast.ai cap, keeping fast `0.6B` default and `1.7B` opt-in, leaving `crates/**` and M1–M8 strictly §3-quarantined, and returning to native M2+ milestones after Phase 5 completes.
  - [`docs/rules/changes.md`](../../../../rules/changes.md#24-november-2026--d91-studio-only-qwen3-17b-phase-5-extension-and-return-to-m2): Recorded Entry `#22` with Casey's verbatim approval quote and updated `Entries in this log` (`21 -> 22`).
  - [`PLAN.md`](../../../../../PLAN.md): Updated the Studio owner-direction row with D90 closeout (`$2.7427 / $3.00`, PR #4222) and D91 Phase 5 scope (`Qwen3-1.7B` + return to native M2+).
- **Charter §4.5 Pre-Registration (`docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-21/`):**
  - [`ORDER.md`](ORDER.md): Recorded verbatim owner approval, ISO UTC `2026-11-24T04:32:05Z`, and both owner paths (`3,830 B < 4,096 B`, committed alone first at `4e65d7d51`).
  - [`PREP.md`](PREP.md): Pinned `d91-foundry-17b-v1` (`v3`) with all 10 SHA-256 digests (`DATA_SEED = 20261127`), `2,400`-row split cutting Chat repeats to `7..8×` across `164` unique targets (`104` at `8×`, `60` at `7×`), memory-safe `1.7B` LoRA (`r=16, alpha=32, lr=8e-5, ∫lr=0.006000`, gradient checkpointing), unchanged Bars #1–#5 (`--max-new-tokens 448`), hard pod cap (`UOR_POD_MAX_RATE=0.65`, `1.5h / <= $1.00`), and `TEST FITNESS: FIT` signoff (`02bb18487` by `35f1e7c6-fc5d-46c3-be08-7b27cfcd5abd`).
  - Untracked local toolchain built & verified offline at `$0.00` under `local/antigravity/d90-foundry-lora/v3/` (zero repo code, scripts, weights, or binaries committed).

## Measurements

- **Pinned local script SHA-256 digests (`local/antigravity/d90-foundry-lora/`):**
  - `v3/foundry_schemas_v2.json`: `56fe4459d45f0e29dd1ffceba4e34b704e7c9e3fafab0e76f3f67177ea52cc9a` (`system_prompt` sha256 `758c77e7b2be4006a2418f4bc87cdcceca8f864b1fa15b3e309a5e07489967f2`, `18` tools byte-identical to [`r4-agent-loop.mjs`](../../../../../web/studio/foundry/studio/shell/r4-agent-loop.mjs))
  - `v3/generate_foundry_sft_v3.py`: `b4222f26b82e6169a522a2e1585e91f13eb8ca1fe6c40754eddad28401c4a132`
  - `v3/train_foundry_lora_v3.py`: `52158a3fc0ac2ef971e2d31902cbf18ed5f1b0997fb00fbd3b6c15baf59dba0e`
  - `v3/eval_foundry_v13_v3.py`: `a6e9eba7fd0d09535c132b3fa060911cbfbdf538f91da22cf1a006527b78c47d`
  - `v3/foundry_v13_eval_pack_v2.rs`: `0feb9b07d821b3a816700757f6c63b32b0e7ae835b501f2d01bca004c474c1b5`
  - `patch_recompile.py`: `bb5da659f37d7590791c723a230e21bc4dc6a4752003a94297ac0ccf775460de`
- **Deterministic dataset verification (`DATA_SEED = 20261127`):**
  - `train.jsonl` (`2,400` rows: `540` `tool_call`, `320` `prism_write`, `288` `multi_turn`, `1,252` `chat`; `164` unique Chat targets repeated `7..8×` [`104` at `8×`, `60` at `7×`]; `1,252/1,252` unique Chat prompts; `96` default + `2,304` `FOUNDRY_SYSTEM_PROMPT`; `0` `.lex.tex` / `ood_01..12` leakage; `0` repeated 4-grams): `a2e44fa69496e4fb99802e40be1fce051bd5ccc66c74bdd3d227a2c54ff6ccef`
  - `val.jsonl` (`440` rows: `288` `tool_call`, `52` `prism_write`, `100` `chat`; `440/440` `FOUNDRY_SYSTEM_PROMPT`; `train_val_overlap_hashes: 0`): `06814bd9dcfece630998cc33699150da46afa7fae2102c046491e1300a8289a1`
  - `calib_foundry.txt` (`96` user-only rows sampled from `train.jsonl`): `8524a836d5a44a6c240929defe876ffa588fcdaf2451f1d81a2ccef199c652a6`
  - `parity_probe_prompts.json` (`16` stratified `val.jsonl` episodes -> `576` positions): `56500d4b5deffed61a5f4840987a158af0d92d7fabb4cdb765318fffa99e8be9`
- **Pod ledger & balance:** `uor-pod status` confirms `Running: 0 pod(s) ($0/h of $8.00/h)`; D90 `0.6B` closed at `$2.7427 / $3.00`, and D91 begins at `$0.0000` spent of its fresh `+$3.00` cap (`<= $1.00 / 1.5h` allocated for `d91-foundry-17b-v1`).

## Checks run at the head

- `python3 -B -m py_compile local/antigravity/d90-foundry-lora/v3/*.py` — PASS (`0` errors).
- `HoloCycleGuardLogitsProcessor` host unit tests & differential fuzzing (`7/7` directed checks + `5,000` random + `1,000` stress trials by `35f1e7c6-fc5d-46c3-be08-7b27cfcd5abd` against `holo_geometric_runtime.rs:6762-6827` using in-memory `torch` stub) — PASS (`0` mismatches).
- `python3 -B local/antigravity/d90-foundry-lora/v3/eval_foundry_v13_v3.py --self-test` — PASS (`self_test: "PASS"`, `train_count: 2400`, `val_count: 440`, `train_val_overlap_hashes: 0`, `gold_eval: 340/340` tool + `100/100` Chat = `100.0%`, `bars_passed: true`).
- `eval_foundry_v13_v3.py --wasm` local WAT audit on `web/studio/foundry/studio/shell/v13/uor_foundry_v13_wasm_bg.wasm` (`307` funcs) and local `lexlean_r4_wasm.wasm` (`527` funcs) — PASS (`compute_float_ops: 0`, `passed: true`).
- `35f1e7c6-fc5d-46c3-be08-7b27cfcd5abd` (`pro` non-author FIT review at `02bb184873b98fc655a351aeae627b7a208f6d82`) — `TEST FITNESS: FIT`.
- `uor-harness check --staged`, `uor-harness check --only charter,docs,loop,ratchet,claim,hygiene,prep`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, and `uor-harness loop pr-check --base origin/main --head HEAD --local` (`PASS owner-paths`, `PASS scope`, `verdict: PASS`) verified cleanly inside `.worktrees/antigravity-studio-phase5-d91-prep`.

## Skills used

- Standard git, `python3 -B`, `wasm-tools print`, `uor-pod`, and `uor-harness` loop/ratchet verification.

## Still open

- Push `antigravity/studio-phase5-d91-prep`, open PR, pass fresh-context `pro` exact-head adversarial review (`VERDICT: APPROVE`) + compliant `[antigravity] uor-review v1` comment, confirm `uor-harness loop pr-check --pr <N>` & GitHub CI green, merge into `main`, clean worktree/branch, and hand back via `enter_plan_mode` for `d91-foundry-17b-v1` GPU launch approval.
- After `Qwen3-1.7B` Phase 5 completes and Studio ships the opt-in `1.7B` selector alongside default `0.6B`, transition focus back to core native UOR-R4 M2+ milestones per [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43).

## Plain summary

Recorded Owner Decision **D91** (`2026-11-24T04:32:05Z`: extend temporary Studio-only D90 exemption to pure-integer `UORR4V13` `Qwen/Qwen3-1.7B` with a fresh `+$3.00` Vast.ai cap and opt-in selector alongside default `0.6B`, then return to core native UOR-R4 M2+ milestones) in `DECISIONS.md`, `changes.md` (`#22`), and `PLAN.md`; built and offline-verified the `v3` `1.7B` local toolchain (`2,400` train / `440` held-out val cutting Chat repeats from `15..16×` down to `7..8×`, `r=16/alpha=32`, `lr=8e-5`, `∫lr=0.006000`, gradient checkpointing, and `5,000`-trial cycle-guard parity); earned non-author `TEST FITNESS: FIT` (`35f1e7c6-fc5d-46c3-be08-7b27cfcd5abd`); and pre-registered `d91-foundry-17b-v1` with `0` running GPU pods.
