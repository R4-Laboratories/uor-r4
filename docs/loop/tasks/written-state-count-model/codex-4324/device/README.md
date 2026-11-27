# Frozen GPU release factorial

Build the diagnostic driver on Wick using the archived CUDA 12.8 driver ABI. Keep the original cubins from #4321 and #4323. The device runner accepts their pinned SHA-256 digests and checks the image bytes before loading. It also checks both frozen weight files, the #4318 copy cubin, the writer's other cubins and the byte-verified writer state.

```sh
CARGO_TARGET_DIR=LOCAL_BUILD_DIR cargo build --locked --release --manifest-path device/Cargo.toml
CARGO_TARGET_DIR=LOCAL_BUILD_DIR cargo test --locked --release --manifest-path device/Cargo.toml
release-factorial-device check-factorial-rows PANEL
CUDA_DISABLE_PTX_JIT=1 release-factorial-device finite SCOPE_CUBIN SCOPE_SHA CHOICE_CUBIN CHOICE_SHA
CUDA_DISABLE_PTX_JIT=1 release-factorial-device score SCOPE_CUBIN SCOPE_SHA CHOICE_CUBIN CHOICE_SHA \
  COPY OCCURRENCE BASELINE BASE_DIR CLASSES READER_WEIGHTS MARKER PANEL \
  LINK_WEIGHTS LINK_SHA BALANCED_WEIGHTS BALANCED_SHA NEW_OUT
```

| Frozen input | SHA-256 |
|---|---|
| #4323 scope cubin | `e69ffddfd7682113109fba31a43ba644d833cf90e3cc69830e1edf85d56171e7` |
| #4321 choice cubin | `e6981880ef98f5e3ea5327e12e84ca3f184acbfa7c742525af31e31b86721402` |
| #4323 link weights | `bf13f776dc40af2be2ef1323fd9aacbaec4486ff0d74dedf0762a610b9cbbd91` |
| #4322 balanced weights | `45d517f31074aa8061307a36f5d522e3af4bb464b434e5e6ff1779f4dd82c05e` |

The diagnostic driver reuses the #4321 writer and the frozen score and commit ABI through a compile-time hook. It supplies the first release only following an authenticated A and the supplied period. It supplies the post-B release only when the actual prefix ends with B copied at its original address and the request includes B. Host-side address validation binds both exact position and original token ID before the archived CUDA commit verifies the token in the causal prefix. The writer receives each arm's actual committed tokens, and each arm stops at EOS or the historical five-token cap.

The `finite` command runs planted signed commit cases on both cubins, including wrong-token and same-token wrong-address refusals, stage expiry and learned release after C. The host tests check conditional gate reach and metadata round-trip without a CUDA context.

The score command writes `candidate.jsonl` and `balanced.jsonl`, each with a separately generated reverse-order trace. It requires exact equality after reversing that trace and binds each output by digest in `replay.json`. Every row retains its panel metadata. Every trajectory records its actual writer events, committed IDs, exact EOS status and one-based first mismatch. A release record identifies the stage's original address, the eligible and applied intervention, learned and effective actions, copy authentication and the next stage.
