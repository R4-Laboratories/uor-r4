# Whole-turn fit and ordinary serving panel

The panel reopens the sealed #4329 SQuAD 2.0 and authored inputs and the #4328 protected ordinary cases. It independently serializes every exact original user message through the artifact-bound protocol-2 encoder. Its half-open `user_bounds` are original token positions in the complete prefix. It checks the model's token-class table for every target and same-ID rival. Each split is sorted by stable case ID. [Study sheet](../SHEET.md)

| Input | Rows | Positive | NoRead or unsupported | Positive delayed by one authored answer space |
|---|---:|---:|---:|---:|
| SQuAD fit pairs | 228 | 114 | 114 | 0 |
| Authored fit | 240 | 144 | 96 | 0 |
| SQuAD article reserve | 400 | 200 | 200 | 30 |
| SQuAD published DEV | 50 | 25 | 25 | 2 |
| Authored DEV | 120 | 72 | 48 | 0 |
| Protected #4328 | 48 | 36 | 12 | 0 |

The sealed corpus-only panel is `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-whole-turn-fit/panel-v1`. Only `train.jsonl` adds the supervised `target` field, an exact original position or explicit null. `eval.jsonl` has 1,086 unlabeled prefixes, 468 of which also occur in TRAIN. `prompts.jsonl` carries only case IDs and ordinary raw user messages. `gold.jsonl` carries source, original ID, causal source delay, pair membership, expected reply IDs where defined, and protected source addresses. Case IDs are for joins. The runtime receives only the user message. A natural impossible question has no reference reply. NoRead is assessed at the first actual transition. [Panel file hashes](../evidence/panel.sha256)

Build on Wick from this directory using the locked Rust packages. The source permits rebuild to a fresh private directory.

```sh
cargo build --release --offline --locked
target/release/whole-turn-fit-panel "$WICK_RUNS" "$PANEL_REBUILD"
target/release/free-chat "$WICK_RUNS" "$FREE_REBUILD"
```

## Descriptive real chats

The eight additional single-turn requests are unchanged roots of distinct, reviewed English OASST2 conversations with a recorded human assistant reply. The selection excludes the #4312 open-panel tree IDs and exact D87 prompt-bank matches, screens for ordinary protocol fit, and ranks deterministically within four opening request forms. The 13,854 source trees yielded 51 eligible roots. Novelty applies to the inspected prior panel and prompt bank. The artifacts' wider historical exposure is unresolved. Only the root user request enters the ordinary client. There is no gold reply or source label and these eight requests do not enter a training or qualification denominator. Raw prompts and source identities remain on Wick under `free-chat-v1`. [Published OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2) · [Private file fingerprints](../evidence/free-chat.sha256)

## Ordinary replay and grading

The ordinary path uses the product tokenizer, the exact user prompt, greedy decoding, authenticated copy and the native occurrence trace through EOS or the 16-ID cap. The fresh process runner is `ordinary.sh`. Each arm's artifact and prompt bytes are checked before and after collection. The process count is explicit. The batch dispatcher compiles [a study-local patch](batch.patch) against the product source in a separate checkout. It loads the artifact once and calls the unchanged post-load generation body with fresh per-row encode, generation, context, and native state. The same model object exposes immutable access. The 103-call control selects one case per split/article family, every authored statement type and protected family, the named c021–c024 cases, delayed examples and all eight unlabeled requests. Each arm's batches are compared field for field with independent ordinary client processes before processing the full panel.

```sh
bash ordinary.sh "$CLIENT" "$TOKENIZER" "$CONTROL_PROMPTS" "$CONTROL_SHA" \
  "$ARM" "$ARTIFACT" "$ARTIFACT_SHA" "$FRESH_COLD" 1
bash batch.sh "$BATCH_CLIENT" "$TOKENIZER" "$CONTROL_PROMPTS" "$CONTROL_SHA" \
  "$ARM" "$ARTIFACT" "$ARTIFACT_SHA" "$FRESH_WARM"
cmp <(jq -cS . "$FRESH_COLD/ordinary.jsonl") <(jq -cS . "$FRESH_WARM/ordinary.jsonl")
bash batch.sh "$BATCH_CLIENT" "$TOKENIZER" "$PANEL/prompts.jsonl" "$PROMPTS_SHA" \
  "$ARM" "$ARTIFACT" "$ARTIFACT_SHA" "$FRESH_ARM"
jq -n --arg arm "$ARM" --slurpfile eval "$PANEL/eval.jsonl" --slurpfile gold "$PANEL/gold.jsonl" \
  --slurpfile ordinary "$FRESH_ARM/ordinary.jsonl" -f grade.jq > "$FRESH_GRADE"
```

The scorer authenticates the causal prefix, every actual committed ID, exact original read and claim positions, and the native proposal on every non-copy. The supported rows with separately encoded leading space require the actual first output to be ID 223 before their source position is scored as reached. Both matched pair cohorts and copy/claim failures are split by this phase. Exact reference termination requires the protocol's actual EOS ID 1. EOS-like decoded text has its own field. Expected IDs beyond 16 are reported separately from the exact-through-EOS denominator. The published protected scores preserve each of c021–c024. The separate same-source content comparison and free-chat description use `comparison.jq` and `free-grade.jq`.

The cold and shared-load records matched in all 309 comparisons. Wick kept the actual client traces at `ordinary-controls-v1` and `ordinary-v1` beneath the study root. Each complete 1,086-row labeled batch took 10.04–10.70 seconds including the initial artifact load. The complete arm and real-chat run can be replayed after the cold controls produce their manifest with `compare-all.sh WICK_RUNS BATCH_CLIENT VERIFIED_CONTROLS NEW_OUTPUT_DIR`. [Cold replay evidence](../evidence/ordinary-controls.json) · [Labeled comparison](../evidence/ordinary-comparison.json) · [Descriptive real-chat output](../evidence/ordinary-free-summary.json) · [Ordinary run manifest](../evidence/ordinary.sha256)
