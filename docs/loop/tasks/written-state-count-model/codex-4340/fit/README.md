# Matched natural supervision GPU recipe

This private draft consumes only the four #4338 matched TRAIN tapes and label-blind features derived from the same openings. Each mixture has 264 rows with the identical ordered 204 source and 60 NoRead targets. The historical files use the `original` name from the sealed packet. The four tape digests in `tapes.sha256` are individually pinned to the #4338 manifest at revision `0bc863b1414a6cbe7aeaf9a7253e9431132a3aa3`. The original #4337 compiler and image scanner prefix is byte-identical to the prefix of the pinned launch script at `92b919f41ff19fd7d954b789e35e6e9560e26192`. The pinned CUDA runner builds one native image set with `nvcc` on the assigned GPU and scans it before a fit. Only the scanner prefix is invoked because the full #4337 driver accepts its own additional panels.

| Matched TRAIN mixture | Rows per pass | Partner label changes | Candidate scores per pass | Candidate scores per fit | Rival scores per fit |
|---|---:|---:|---:|---:|---:|
| Historical | 264 | 168 | 25,141 | 804,512 | 796,064 |
| Natural | 264 | 264 | 42,776 | 1,368,832 | 1,360,384 |

Geometry and token-count context each run actual and partner-permuted controls. Each process starts at zero and uses 32 passes, 8,448 update opportunities, one signed 4,096-word table, 16 keys per candidate and one shared unit-margin update rule. A second identical process must return the same 8,192 payload bytes and identical stats. A zero payload supplies a score-only control for each cell. The 168 historical paired rows change labels in the partner control and the 96 historical unpaired rows keep their label. All 264 natural rows change labels. Replay score tapes contain zeroed target fields and no labels. Candidate counts include NoRead. Exactly one score per row is the target, with all remaining scores used for rival selection.

## Stage in Wick custody

Provide the authenticated #4338 matched directory and rebuilt binary. The stage verifies every member of the matched manifest, copies exactly four original training tapes into the pod bundle, generates the four blind score tapes with the pinned Rust fit tool, and emits the exact manifest digest for the FIT and lease plan.

```sh
r=/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-natural-reciprocal
bash "$TOOLS/stage.sh" "$r/matched-v1" \
  "$r/native-build-v2/fit/target/release/interval-fit" "$CAPSULE"
```

Register the emitted digest separately as `CAPSULE_PIN`. Send the capsule and its registered launch and job drivers into a fresh private path on the owned pod. Pass the real leased device architecture, such as 89 for the recorded RTX 4090 or 120 for a compatible RTX 5090 toolchain.

## Dispatch, collect and verify

Use the project pod manager after the independent FIT and pre-run review. Use the owned session and GPU from its lease. Save the lease and creation timestamp, GPU index and UUID, hourly quote, transport and startup times. Arm the existing owned-session watcher before dispatch and monitor its exit marker. Renew the lease while an active process is still making progress. Preserve the diagnostics from a failed attempt in its original directory and use a fresh job root for any approved retry.

```sh
"$MANAGER" run "$POD" --lab codex --session "$SESSION" --gpu "$GPU" -- \
  bash "$REMOTE/capsule/job.sh" "$REMOTE/capsule/launch.sh" "$ARCH" "$REMOTE/capsule" "$CAPSULE_PIN" "$REMOTE/job"
"$MANAGER" ssh "$POD" "cd '$REMOTE' && tar -cf - job" > "$RETURN_TAR"
```

Collect after the job reaches `job.exit`. Compute the archive SHA-256 on return, verify that digest again in Wick custody, then extract into a fresh directory. Require exit zero and run the verifier before release. Preserve the archive, job diagnostics, elapsed time, manifests, images, all 16 weight and statistics outputs and all 12 raw score files. After verified readback, record lease end and actual provider billing duration. Report the GPU process wall time separately from capsule transport and lease lifetime. The per-process timers include CUDA initialization, synchronization, host/device transfer and filesystem output. The job timer also includes AOT build and scan. These are wall clocks around device-backed calls.

```sh
test "$(cat "$RETURN/job/job.exit")" = 0
bash "$TOOLS/check-return.sh" "$RETURN/job/measured" "$CAPSULE_PIN" > "$RETURN/device-summary.jsonl"
```

## Reopen on Wick

`cold-train.sh` consumes the verified device return, its capsule and registered digest, the matched custody path, pinned `interval-fit`, `uor-chat`, `pack-native-interval`, balanced artifact, tokenizer, two original label-blind TRAIN opening files and a fresh output directory. The prior historical openings are in #4337 `data/fit-v1/train-input.jsonl`; the natural openings are in #4338 `native-v2/train-input.jsonl`. Verify at least 16 GB free for twelve 1,202,185,782-byte artifacts and their receipts. Use the digest of the frozen-weight manifest as the common fit identity. The cold verifier reopens all twelve artifact/control combinations and compares all raw scores, selections, original-candidate endpoints and keys on the registered openings. The TRAIN grades then count correct originals, NoRead errors and strict paired selection from the same payload. The protected duplicate-address panel is a separate regression with its own capture and label custody.

The experiment changes supervision mix, including provenance and passage distribution, at fixed optimizer budget and state size. Natural rows present more candidates per pass. The candidate-cost and timing receipts keep that difference measurable. Matched TRAIN parity checks this interval reader and payload. Qualification of unseen prompts, complete replies, useful chat and target-device generation needs separate evidence.
