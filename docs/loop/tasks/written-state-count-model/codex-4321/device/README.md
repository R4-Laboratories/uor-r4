# Verified-address choice CUDA driver

The CUDA trainer updates three independent, zero-initialized stage segments in panel order. Each segment has 4,096 signed i16 weights followed by one signed i16 bias. Each of the eight epochs first visits individual reachable release states in input order, then visits common-prefix contrasts in side-0 input order. A violated individual margin of eight updates each distinct bucket and the bias by one unit in the target direction. A violated stop-to-copy score gap of sixteen adjusts only buckets exclusive to each wording by one unit in its target direction. A signed word overflow refuses the fit. The trainer serializes these same native words as little-endian bytes.

The AOT build uses the exact instruction bucket enumerator extracted from the digest-pinned [#4319 device source](../../codex-4319/device/src/ending.cu). Scoring uses that enumerator and the stage-specific bias. The frozen [#4318 GPU writer runner](../../codex-copy-id/device/src/gpu.rs) supplies the authenticated opening and a fresh native writer event at every later emission. The action kernel supplies the first period and checks each requested original source against the exact ID at the original position behind the fact fence. Every committed word becomes input for the following writer call. Selection expires after abstention or refusal, and release advances only on a verified copy. Decode stops upon EOS or after five committed tokens.

The trace keeps the writer fields even when an external release action selects the commit. `selected_source` identifies the selection that committed. `copied` is the writer's own authenticated opening flag. `requested_pos` and `requested_id` carry the release-copy attempt. `refusal` is zero for ordinary commits and three when a source fails verification. `stage` is the pre-action release stage and is null outside the active release policy. `action_score` is the consulted own or paired score on learned arms. The independent `common` probe reads both wording scores at the shared fact/release stage even if either self-fed trajectory departed earlier.

```sh
cargo build --locked --release --manifest-path device/Cargo.toml
device/build-image.sh NEW_IMAGE_DIR
choice-device check-rows PANEL
choice-device finite IMAGE SHA256
choice-device train IMAGE SHA256 BASE_DIR TRAIN_JSONL NEW_FIT_DIR
choice-device score IMAGE SHA256 COPY OCCURRENCE BASELINE BASE_DIR CLASSES READER_WEIGHTS \
  MARKER PANEL CHOICE_WEIGHTS NEW_REPLAY_DIR
```

Use nvcc and ptxas on the registered sm_89 GPU pod for the AOT build. Linux host builds do not compile or invoke CUDA. The finite command uses planted geometric roots and rows to exercise signed release scores, stage transitions, abstention and original-address refusals. Train and score use the digest-pinned frozen #4318 roots and native writer inputs. The replay command checks exact forward versus reverse GPU traces before writing its receipt. Compare its trace independently with the cold Rust replay.
