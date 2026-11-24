# REPORT for D91 Run 9 Closeout, Charter §5.1 / D68 Decision Review, & Return to M2+ (`antigravity-30`)

Lab: `antigravity`. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939). Base: `22245bb8a` (PR #4235, [`antigravity-29/REPORT.md`](../antigravity-29/REPORT.md); `ORDER.md` committed alone first at `c4ddfec9a`).

## What did not work

- **Final D91 Run 9 (`d91-foundry-17b-v9`, `v3.5`, `vast-62353488`, `uptime_s: 1411` @ `$0.508/h` = `$0.1991`) passed Bar #1 WAT (`compute_float_ops: 0`) and BF16 Bar #2 (`97.65%` = `332/340`), then halted cleanly at Stop Gate #1 on BF16 Bar #3 (`92.0%` = `92/100 < 95.0%`), triggering D91 finality:**
  - Running Run 6's strong KL/regularization config (`--kl-anchor-weight 1.0 --weight-decay 0.015 --label-smoothing 0.02`) for `2.5` epochs at `--lr 1.4e-4` (`opt_step=375/375`) drove `train_loss` down to `0.60128` (`anchor_kl_train=0.011514`) and achieved the strongest OOD first-token anchoring of all anchored runs (`anchor_val_kl_first=0.019485` vs `0.021992` in R6, `0.040735` in R7, `0.055465` in R8).
  - However, extended training over `v3`'s `7–8×` repeated Chat templates pushed held-out validation loss to a Run 1–9 high (`val_loss=2.16749`, `val_loss_tool=0.58664`, `val_loss_chat=3.78158`), widened adapter weight drift (`delta_w_rel_fro=0.012503`), spiked bilingual Chinese character leakage to **`30/440` rows** (including `val_pw_010` `prism/spec潭.tex` inside `<prism-write>` and `val_ch_085` `\ufffd` + `常規規則`), leaked `1` `<tool_call>` inside Chat (`val_ch_080`: `逐步 Action Turns`), hit `1` short reply (`val_ch_054`, `too_few_words:46`), and missed `5` expected Chat keywords (`val_ch_022, 027, 069, 078, 093`) for `total_failures: 16`.
  - Combined Runs 1–9 evidence (`results_v3_run1..9/d91-out/`) demonstrates an irreducible frontier trade-off inside `v3`'s `2,400`-row bilingual fine-tune (`kl=0` fails OOD Bar #4a `0.191..0.243 > 0.08`; `kl=0.25..1.0` fails BF16 Bar #3 `< 95%`), so Run 9 halted cleanly before `recompile_holo_v13` (`run9_meta.json`: `stop_gate: "gate1_bf16"`, `active_mode: "none"`), archived diagnostics locally, tore down (`Running: 0 pod(s)`), and closes the D91 `1.7B` LoRA sweep under Charter §5.1 / D68 ([`DECISION.md`](DECISION.md)).
- **Host environment caveat (`$0.00` spent in `antigravity-30`):**
  - Laptop system `python3` has no GPU `torch`/`transformers`/`peft`; verification here audits all locally archived Run 1–9 receipts (`results_v3_run1..9`), cumulative `~/.local/share/uor-r4/compute/ledger.jsonl`, pod state (`Running: 0 pod(s)`), and repo harness/ratchet/claim-wording gates.

## What changed

- **Zero code, weight, or owner-path edits (`$0.00`):**
  - `crates/**`, `src-tauri/**`, `.github/**`, `web/studio/**`, `local/**`, `CHARTER.md`, `DECISIONS.md`, `PLAN.md`, and live `gh-pages` Studio (`PR #4213`, verified pure-integer `Qwen3-0.6B`) remain completely untouched. `0` unverified `1.7B` weights were published.
- **D91 Run 9 Closeout & Charter §5.1 / D68 Decision Review (`docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-30/`):**
  - [`ORDER.md`](ORDER.md) (`3,819 B <= 4,096 B`, committed alone first at `c4ddfec9a`).
  - [`DECISION.md`](DECISION.md): Synthesizes complete Runs 1–9 (`v1..v9`) telemetry and receipts (`$2.0104 / +$3.00`, `$0.9896` unspent), preserves live `0.6B` Studio (`PR #4213`), closes D91's `v3`–`v3.5` `1.7B` LoRA configuration sweep (§4.1–§4.2) with zero unverified publication, transitions lab focus immediately back to core native UOR-R4 M2+ milestones per [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43), and carries `D68 DECISION REVIEW: APPROVE` (`d24e93157` by `deaae69f-a90c-43bf-b19d-c202a9ad8f8e`).

## Measurements

- **Archived Run 9 receipts (`local/antigravity/d90-foundry-lora/results_v3_run9/d91-out/`, `results_v3_run9.tar.gz` `sha256: e2ecfe5900c4851c912c173a1d41a10b90a666ac31561904dc640bcc329349c0`):**
  - `wasm_float_audit.json` (`462 B`): `wasm_sha256: a6f57c99f62957e09d651679a0a406b76662803bfffe03af68bb104a711fc4b6`, `holo_inference_functions_scanned: 248`, `compute_float_ops: 0`, `float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_src_sha256: 6b84de0364efe6dd80fe8146c9bf1cc08631e3069e13bb6414c358a0c00907ae`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`. `pod_build.log` (`10,310 B`, recording `AMD Ryzen 7 7800X3D 8-Core Processor`, `16` cores, `30 GB` RAM / `28 GB` avail, `foundry_v13_eval_pack_v2_native` `5.2M`, `recompile_holo_v13` `4.6M`, and unchanged `.wasm` digest) and `job_gpu0.log` (`10,857 B`) are archived locally alongside `run9_meta.json` (`1,216 B`, `stop_gate: "gate1_bf16"`, `skip_baseline: true`, `native_cpu_build: true`).
  - `lora_eval.json` (`2,611 B`, `pred_sha256: 957ef1a60f46083aa46cd124c2bc35e93ce2f688466fdc488f4c9c4ebc320538`): `tool_pass: 332/340` (`97.65%`: `284/288` `tool_call`, `48/52` `prism_write`), `chat_pass: 92/100` (`92.0%`), `total_failures: 16`, `bars_passed: false` (Stop Gate #1 FAIL).
  - `train.log` (`4,950 B`, `34,865,152 / 1,755,440,128 = 1.9861%`): `0.50ep` (`train=2.81850, val=1.49568, tool=0.40950, chat=2.60469, anc_tr=0.116440, anc_val_resp=0.135711, anc_val_first=0.038780`) -> `0.75ep` (`1.77841, 1.50338, 0.43804, 2.59112, 0.130276, 0.130895, 0.033112`) -> `1.00ep` (`1.49456, 1.60316, 0.47232, 2.75777, 0.071219, 0.108705, 0.031254`) -> `1.25ep` (`1.02484, 1.77598, 0.51273, 3.06579, 0.039636, 0.089903, 0.030200`) -> `2.50ep` (`batch=1500/1500, opt_step=375/375, lr=0.0, train=0.60128, val=2.16749, tool=0.58664, chat=3.78158, anc_tr=0.011514, anc_val_resp=0.081873, anc_val_first=0.019485`, restored); `[delta_w_rel_fro]`: `overall_mean=0.012503` (`down=0.006808, gate=0.021695, up=0.017161, q=0.010688, k=0.01032, v=0.009541, o=0.011306`).
  - `parity_probe_tokens.json` (`102,176 B`): `total_positions: 576`, `vocab_size: 151936`, `logits_sha256: 54aea66b389d35381d8d022b090129efcdf2984552a3bea189bd99afe2ceaf9a`.
- **Cumulative D91 ledger (`~/.local/share/uor-r4/compute/ledger.jsonl`):**
  - Runs 1–8 (`$1.8113`) + Run 9 `vast-62353488` (`$0.508/h`, `uptime_s: 1411`): **`$0.1991`** = **`$2.0104` cumulative across `14,247s` (`$0.9896` remaining of D91's `+$3.00` cap)**; `/Users/casey.allard/uor-r4/scripts/pod/uor-pod status`: `Running: 0 pod(s) ($0/h of $8.00/h)`.

## Checks run at the head

- Verified unpacked `results_v3_run1..9/d91-out/` metrics, `lora_predictions.jsonl` `\ufffd`/CJK counts, `results_v3_run9.tar.gz` digest (`e2ecfe59...`), and cumulative `ledger.jsonl` totals (`$2.0104 / +$3.00`) directly on disk — PASS.
- Fresh-context `pro` D68 decision review (`deaae69f-a90c-43bf-b19d-c202a9ad8f8e`) at `d24e93157e9980fad3a62fc241fd273a119666c2` — `D68 DECISION REVIEW: APPROVE` (PASS).
- `/Users/casey.allard/uor-r4/scripts/pod/uor-pod status` — `Running: 0 pod(s) ($0/h of $8.00/h)` (PASS).
- `uor-harness check --staged` and `uor-harness check --only prep arena briefing budget harness mistakes prlint receipts` — PASS.
- `uor-harness ratchet d23`, `uor-harness ratchet d25`, `uor-harness ratchet d32` — PASS.
- `uor-harness loop report --check --order docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-30/ORDER.md` and `uor-harness loop pr-check --base origin/main --head HEAD --local` — PASS.
- `cargo run -q --locked -p uor-r4-claim-wording -- .` — PASS.

## Skills used

- Standard git, `python3 -B`, `uor-pod`, and `uor-harness` loop/ratchet verification.

## Still open

- Open PR, pass fresh-context `pro` exact-head adversarial review (`VERDICT: APPROVE`) + `[antigravity] uor-review v1`, confirm `uor-harness loop pr-check --pr <N>` & CI green, merge via squash queue, pull fresh `main`, clean worktree/branch, post `[antigravity] owner log:` closeout on `#3939`, and hand back via `enter_plan_mode` for core native UOR-R4 M2+ work.

## Plain summary

Closed out D91 Run 9 (`d91-foundry-17b-v9`, `vast-62353488`, `$0.1991`; cumulative D91 `$2.0104 / +$3.00`, `$0.9896` unspent, `0` pods active), where `--kl-anchor-weight 1.0` at `--epochs 2.5 --lr 1.4e-4` achieved Bar #1 (`compute_float_ops: 0`), BF16 Bar #2 (`97.65%` = `332/340`), and lowest anchored OOD drift (`anchor_val_kl_first=0.019485`) while halting at Stop Gate #1 on BF16 Bar #3 (`92.0%` = `92/100`, `val_loss_chat=3.78158`, `30/440` CJK rows, `1` `\ufffd`). Executed the Charter §5.1 / D68 closeout decision (`DECISION.md`) closing D91's `v3`–`v3.5` `1.7B` LoRA sweep with zero unverified weights published, keeping live `gh-pages` Studio untouched on verified pure-integer `Qwen3-0.6B` (`PR #4213`), and returning immediately to core native UOR-R4 M2+ milestones.
