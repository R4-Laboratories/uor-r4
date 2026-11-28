# Scoring the registered reciprocal transfer study

The scorer handles the frozen #4340 sample of 67 pairs and the #4341 development extension of 46 pairs. The expanded sample retains 23 reciprocal and 90 answerability pairs. The #4341 score reports the original title-specific selected and qualified denominators alongside the expanded ones. Original dispositions join by complete source and case identity because the original selector and source compiler used different family orders. The added 46 must match in their exact registered order.

The source gate evaluates the first ordinary client decision. It credits a positive only when the exact original half-open address defeats every interval and NoRead by at least one signed unit. A NoRead decision must strictly defeat every interval. Both decisions must win on the same pair. Title averages and registered contrasts use exact fractions. The source report retains exclusions as unsolved in the full selected denominator, exact mate replays, all four protected originals and the 12 fixed model variants. The writer adjudication is a separate receipt. All commands operate only on the registered development and TRAIN paths.

## Build and prefit custody

```sh
export CARGO_TARGET_DIR="$SCRATCH/score-target"
cargo test --locked --offline --manifest-path "$SCORE/Cargo.toml"
cargo build --locked --offline --manifest-path "$SCORE/Cargo.toml"
tool="$CARGO_TARGET_DIR/debug/validation-transfer-score"

bash "$SCORE/stage-controls.sh" "$MATCHED" "$HISTORICAL_RUN" "$NATIVE_V2" \
  "$BINDER" "$TOKENIZER" "$CONTROLS"
bash "$SCORE/assemble-roster.sh" "$tool" 4341 "$EXTRACTION" "$EXTRACTION_PIN" \
  "$QUALIFICATION_ROOT" "$QUALIFICATION_PIN" "$CONTROLS" "$CONTROLS_PIN" "$ROSTER"
bash "$SCORE/stage-eval.sh" "$INTERVAL_FIT" "$FIT_PIN" "$ROSTER" "$ROSTER_SEAL_PIN" "$EVAL"
```

- `MATCHED` is #4338's matched-v1 seal. `HISTORICAL_RUN` is the restored #4337 root with `data/fit-v1`, `data/gate-v2` and `fit-device-v1`. `NATIVE_V2` is the restored #4338 natural TRAIN panel. The controls command authenticates the exact registered manifests and stages just the selected TRAIN and protected members. The tokenized rebound full prompts come from the pinned #4338 opening binder. Anchors retain their historical expected replies and original same-token duplicate addresses.
- `EXTRACTION` is #4341's authorized expanded extraction, and `QUALIFICATION_ROOT` is the outer `repair/qualify.sh` output. Its inner qualification contains `pairs.jsonl`, consensus, blind openings, both gold-blind key streams, and original endpoint truth. The assembler cross-checks qualified pairs against their exact retokenized bounds, ordered candidates, answer bytes and token IDs. It verifies the tracked expansion identity digests. For the original panel pass `4340` and the original extraction and qualification directories instead. Obtain the manifest hashes from the custody receipts before running the next stage.
- `ROSTER/ROSTER.sha256` authenticates `roster.json`, `coverage.json`, blind role manifests, the exact mate-request ledger and the source-backed writer context projection. Normal scoring uses the digest of `roster.json` itself. Blind client captures use `ROSTER/blind/validation/input.jsonl`, `ROSTER/blind/protected/input.jsonl`, and the separate `ROSTER/blind/validation/swaps.jsonl`. The mate input contains the authentic partner ID and verbatim client prompt. The request ledger associates each partner replay with the original case. Only give `writer-contexts.jsonl` to the writer adjudication process.
- The eval assembler refuses a failed coverage floor. It writes four independently sealed #4337 AOT score-only tapes for validation/protected and count/geometry. `prepare-eval` produces zero actual and partner gold fields. Stage these tapes apart from the existing TRAIN capsule and use them only after all fitted words have been frozen. The 12 conditions each replay two panels using 24 CUDA invocations. The exact ordinary mate capture is compared with the mate's already verified primary score and uses no extra CUDA invocation.

## Cold evidence and source decision

Run #4340's `fit/cold-train.sh` on all 12 states and #4341's `device/replay.sh` for post-freeze validation and protected scores. The return layout contains `<mixture>.<key>.<fit>.<validation|protected>.scores`, the frozen weight manifest digest, the eval manifest digest, and `OUTPUTS.sha256`. After verifying retrieval and independently recording the return manifests:

```sh
bash "$SCORE/cold-evidence.sh" "$tool" "$INTERVAL_FIT" "$CLIENT" "$TOKENIZER" \
  "$ROSTER" "$ROSTER_SEAL_PIN" "$EVAL" "$EVAL_PIN" "$TRAIN_COLD" "$TRAIN_COLD_PIN" \
  "$TRAIN_RETURN" "$TRAIN_RETURN_PIN" "$TRAIN_CAPSULE" "$MATCHED" \
  "$EVAL_RETURN" "$EVAL_RETURN_PIN" "$EVIDENCE"

"$tool" source "$ROSTER/roster.json" "$ROSTER_JSON_PIN" \
  "$EVIDENCE/evidence.json" "$EVIDENCE_JSON_PIN" > source-report.json
```

The builder verifies each frozen artifact against its weight and pack receipt, runs the pinned `verify-device` scorer against the complete candidate keys and signed scores, and checks every ordinary selected address against the independent CUDA binary stream. It holds a separate exact mate capture and writes per-variant receipts before sealing `EVIDENCE.sha256`. The independent verifier must issue the final evidence JSON pin. A digest recomputed from an alternative file supplies no independent authority. The standalone source scorer refuses a missing model, qualified case, explicit NoRead, protected original, mate replay or independent reference.

## Writer evidence

Use the ordinary trace verifier under [client](../client/) and its receipt to classify emitted tokens, source commits, visible claims and actual EOS. The optional scorer `writer` mode consumes the corresponding pinned writer projection for natural geometry actual, schema `uor-r4.reciprocal-transfer-validation-writer-evidence/1`. Its model, roster, and scores must be from the same source run. The gate distinguishes a false source commit, a visible false claim, unresolved visibility and incomplete natural output. The client-side supplied interval and supplied-EOS diagnostics have separate receipts. Combined evidence can establish eligibility to register a separate locked study. The scorer has no locked input mode.
