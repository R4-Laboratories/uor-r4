# Frozen request and occurrence capture

Build `finite-native-recurrence` on the Wick Ryzen 9 9950X with the study's locked Rust dependencies. This finite host probe observes the original integer artifact and the frozen reader. It consumes blind case IDs, serialized prefixes, user bounds and raw prompts. Source labels enter only after capture.

```sh
unit=docs/loop/tasks/written-state-count-model/codex-4332/probe
runs=/mnt/plexpool/uor-r4-corpus/runs
old="$runs/20261127-codex-whole-turn-fit/archive-readback-v1"
balanced="$runs/20261127-codex-period-margin/cold-v1/balanced/artifact.d11"
trunk="$runs/20261127-codex-marker-question/base-v1/trunk"
cargo build --locked --offline --release --manifest-path "$unit/Cargo.toml"
"$unit/target/release/finite-native-recurrence" \
  "$balanced" "$old/reconstructed/frozen.d11" "$trunk" "$old/tools/tokenizer.json" \
  "$PANEL/input.jsonl" "$PANEL/prompts.jsonl" > "$OUTPUT/capture.jsonl"
```

The loaded authentic writer uses the exact balanced #4322 artifact and the #4330 whole-turn envelope with the same 513 signed words. The envelope makes the original broad-admission comparison available for natural passages. Capture retains the original framed and expanded-frozen admissions, all score words and addresses, both authenticated first writer events, and the frozen native opening head's independent request-state observation. A separate sum over the actual 16 indexed words checks every candidate score. Candidate codes include tokens through the candidate's original position. The real query strips the exact six-ID assistant marker, including its leading newline. The no-question control ends at the verified tokenized passage separator, where a panel recorded this boundary. The fixed control uses code zero. The paired gate addresses swapped-question control by using the other question's code against the same site.

The recurrence oracle pins the #4314 flat-trunk SHA-256 and verifies all 4,096 packed token-root words against the loaded product. It applies the public signed-axis actions directly to integer root coordinates, replays all step transports from the stored gates, snaps by distance and Hamming tie-break, and applies the stored per-block alignment. The product uses composed group states and inverses. Compare the 16-bit recurrence word from the oracle and the ordinary authenticated writer before recording any case. Also require the actual #4314 marker head on the same prompt to give the same pre-marker word.

The archived #4330 tape can substitute its `panel-v1/eval.jsonl` and `panel-v1/prompts.jsonl` for capture. Its projection onto `case_id`, `old_framed` and `expanded_frozen` can be compared byte for byte after sorting JSON object keys against corresponding fields in `reconstructed/cold-candidates.jsonl`. This is raw artifact parity. The older tape supplies no separately sealed no-question boundary.

For the static original-address audit of previously authored TRAIN support reversals, use the unchanged archived labels with the captured raw integer values.

```sh
jq -cn --slurpfile gold "$old/panel-v1/gold.jsonl" \
  --slurpfile captures "$OUTPUT/capture.jsonl" -f "$unit/train-signatures.jq" \
  > "$OUTPUT/train-signatures.jsonl"
```

The signed inequalities require the supported occurrence to exceed the fixed NoRead word by the requested margin and the same occurrence on the unsupported side to fall below it by that margin. Each added signed slot is bounded to ±4,096. Equal slot indices require a single common integer to meet both bounds.
