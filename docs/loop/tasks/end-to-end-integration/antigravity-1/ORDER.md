# Gate 0: Clean Protocol-2 D87 Ingest & Unwrapped Eval Suite (E1–E4)

Lab: antigravity. Item: [End-to-end integration](../../../../roadmap/items/end-to-end-integration.md). References #4064 and #2030. Source baseline `1a995ebba9cc`.

## Goal

Build two pure-integer (`#![forbid(unsafe_code)]`, zero float, zero softmax, zero matmul) host instruments inside this task directory before Gate 1 (the native-integer Butterfly-Rotor pilot) starts:

1. **`build-proto2`**: Re-ingest all `262` sealed D87 donor shards under `DialogueProtocol::literal_roles_v2` framing (`BOS` + `"User: {q}\nAssistant: {a}"` + `EOS` per turn, bit-identical to `uor_r4_tokenizer::dialogue::DialogueEncoder`), preserving exact `claude-411/d87-split/v1` record splits (`TRAIN` / `DEV-tune` / sealed `DEV-test`), capping `TRAIN` template spam (`--max-template-repeats 16`), writing a deterministic `100M`-token `train-100m` slice, preserving `mask.u8` & `segment.u8`, and adding `absent.u8` (targets absent from preceding `2,048` record tokens).
2. **`eval-p4` (`E1–E4`)**: Evaluate frozen `p4.d11` (`e0ffbba1...`) strictly unwrapped (zero regex span extraction, zero `2^48` score forcing) across:
   - **E1**: Answer-only (`mask == 1`) stratified PRLL BPB (`all_answer`, `novel_absent_answer`, `in_context_answer`, `gold_eos`) + counterfactual prompt-shuffle (`ΔBPB` with integer paired bootstrap 95% CI).
   - **E2**: Unwrapped free-run collapse (`N = 200`, cap `128`) & counterfactual swap divergence (`64` pairs).
   - **E3**: Unwrapped `m2-fresh256` (`128` pairs / `256` rows) + `64` abstention rows vs `NoMemory` and `RoleSwapped`.
   - **E4**: `100` deterministic checkable items (`40` arithmetic/state, `40` long-distance QA, `20` tiny Rust).

## Scope

- `docs/loop/tasks/end-to-end-integration/antigravity-1/**`
- `docs/roadmap/items/end-to-end-integration.md` (progress note)
- `docs/research/notebook/m4-serving.md` (append-only row)

## Native-Space Note (D85)

No model training, weight update, GPU execution, or artifact format change (`$0.00` GPU spend). Measures existing frozen `p4.d11` strictly via `P4Session::step` on Wick CPU.

## Cancel Probe

Confirm all `262` D87 donor shards, `tokenizer.json` (`a9fb13f7...`), and `p4.d11` (`e0ffbba1...`) are readable without modifying existing `/mnt/plexpool` inputs or disturbing active Wick GPU jobs.

## Done Check

1. `gate0-kit` passes `cargo test` and `cargo clippy --all-targets -- -D warnings` with zero floats/softmax/matmul.
2. Sealed Protocol-2 D87 streams land on Wick ZFS (`/mnt/plexpool/uor-r4-corpus/runs/20261127-antigravity-gate0-d87-proto2/`) with verified `receipts.sha256` and `DEV-test` untouched by evaluation.
3. `eval-p4` records unwrapped `E1–E4` receipts for frozen `p4.d11` in `evidence/`.
4. `uor-harness loop pr-check --local`, `uor-harness check --only harness.native_space_contract`, fresh-context adversarial review (`uor-review v1`), and CI pass.
