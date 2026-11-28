# Grounded conversation cohort for the reset trial

The complete-reservation selector yields 1,096 independent source components. The selected TRAIN has 256 passages, development 108 and held 222. The held opening target always belongs to the third original assistant answer. Each tape has numeric IDs unique within its split. The complete protocol document preserves two prior assistant EOS closures. The prefix ends after the real final assistant marker. Only slots in the separate `expected` stream, including the final response EOS, are supervised. The preceding input ID, all expected IDs, and EOS fit one actual BOS-relative 257-ID chunk. The full conversation stays available to both arms as their identical input stream.

| Split | Rows | Near | Clean far | Intermediate | Later literal mention in the target chunk |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 256 | 6 | 234 | 16 | 5 |
| Development | 108 | 5 | 95 | 8 | 2 |
| Held | 222 | 3 | 209 | 10 | 4 |

Later mention counts include the initial question after its passage and both subsequent authentic questions and historical assistant answers. Near uses a complete normalized literal target inside the original annotated passage interval, wholly contained in the current chunk. Far requires absence of the complete normalized literal value from every actual causal message in the chunk. Far rows can still contain the first target ID or a paraphrase. No selected partition supplied a same-question pair with distinct first original IDs. The small near counts support a descriptive contrast only. Three separate old-stream token overlap checks compare exact 32-ID spans of the complete canonical record. Shorter or paraphrased overlap and unrelated pretraining remain outside that screen. CoQA exposes no worker identity.

All 151 distinct #4360 article IDs from V1 and V2 are reserved. The first data build reserved V2 only, produced a different development and held tape, and was superseded once the omitted V1 article keys were found. The preregistered seed, algorithm, quotas and thresholds were unchanged. The corrected TRAIN tape is byte-identical to the earlier TRAIN tape. The original frozen selector and recorded correction are preserved alongside the evidence.

Raw text and reversible IDs remain private on Wick. The authoritative directory is `/mnt/plexpool/uor-r4-corpus/runs/20261128-codex-grounded-reset/cohort/complete-reservation/final`. Its tape members are `train.jsonl`, `development.jsonl`, `held.jsonl`, and `held-messages.jsonl`. The separate held sidecar contains only the five causal role messages per held record, including the first-user passage, and never contains its target assistant message. Only the selected ordinary-client calibration rows may be used where the evaluation order authorizes them. No model ran in this unit.

## Replay on Wick

Use a checkout containing this bundle and the pinned repository tokenizer and donor bank. The fixed corpus and preview dependencies are pinned in `src/inputs.rs` and the aggregate. The standalone Cargo lock binds the package. Use a new private output root on each replay.

```sh
unit=docs/loop/tasks/end-to-end-integration/codex-4361/cohort
runs=/mnt/plexpool/uor-r4-corpus/runs
prior="$runs/20261128-codex-compact-carry-preflight/cohort"
out=$(mktemp -d "$runs/grounded-reset-cohort-replay.XXXXXXXX")
CARGO_TARGET_DIR="$out/target" cargo build --release --locked --manifest-path "$unit/Cargo.toml"
"$out/target/release/held-coqa-dialogue" "$runs" \
  "$runs/20261128-codex-held-coqa.9eVz1QPo/previews" \
  "$prior/v2/final-2048/selected-index.jsonl" \
  "$prior/final-2048/selected-index.jsonl" \
  "$out/replayed"
(cd "$out/replayed" && sha256sum -c "$OLDPWD/$unit/evidence/private-members.sha256")
```
