# Matched release fitting and GPU replay

Both arms store the original 40,966 linked-checkpoint bytes followed by 4,096 signed little-endian i16 words. The driver verifies the linked prefix by SHA-256 at every load. CUDA updates only the appended words. Stage zero reads these words at the first-occurrence distant-root addresses from the frozen linked scorer. Stage one reads the same table with each address XORed by 2,048. Stage two reads only its frozen score. Each release adds its signed residual to its frozen score and uses the existing sign and original-address copy verifier.

The sealed four-epoch schedule visits all TRAIN rows at stage zero in tape order, followed by the same number of additional slots alternating post-B stop and post-B continue examples. Each pool cycles in tape order and resumes where the previous epoch left off. Both fits receive the same row at each slot. The first-period arm keeps stage zero throughout. The joint arm switches to stage one on the additional slots. Before either fit, the driver runs the frozen GPU writer and commit kernel to authenticate A and the supplied period on all rows, and a supervised copy from the original B address on the eligible TRAIN rows. The post-B slots must all have an actual successful GPU commit.

Each fit worker reads every distinct address before applying a signed margin of eight. Below the margin it updates at most 16 distinct addresses by one unit in the target direction. For support size `s`, select `min(16, s)` evenly spaced first-visit ranks starting at `17 * epoch modulo s`. Use a right shift when the stride denominator is sixteen and bounded subtraction for the phase reduction. Each output includes the per-slot support, raw visits, before and after scores, and actual updated addresses. Both arm receipts report the same slot and coordinate opportunity budgets, and separately report actual coordinate writes, revisits and differences from the original bytes. The driver fits each arm twice from untouched initial bytes and requires identical words and GPU audits. It then reconstructs the words from the device's recorded updates and verifies exact GPU scores after loading the serialized state.

```sh
CARGO_TARGET_DIR=BUILD_DIR cargo build --locked --release --manifest-path device/Cargo.toml
CARGO_TARGET_DIR=BUILD_DIR cargo test --locked --release --manifest-path device/Cargo.toml
device/build-image.sh NEW_IMAGE_DIR
joint-release-device check-rows PANEL
joint-release-device schedule TRAIN NEW_SCHEDULE_JSONL
CUDA_DISABLE_PTX_JIT=1 joint-release-device finite JOINT_CUBIN JOINT_SHA SCOPE_CUBIN SCOPE_SHA
CUDA_DISABLE_PTX_JIT=1 joint-release-device fit JOINT_CUBIN JOINT_SHA COPY OCCURRENCE BASELINE \
  BASE_DIR CLASSES READER_WEIGHTS MARKER TRAIN LINK_WEIGHTS NEW_FIT_DIR
CUDA_DISABLE_PTX_JIT=1 joint-release-device score JOINT_CUBIN JOINT_SHA COPY OCCURRENCE BASELINE \
  BASE_DIR CLASSES READER_WEIGHTS MARKER PANEL FIRST_WEIGHTS FIRST_SHA JOINT_WEIGHTS JOINT_WEIGHTS_SHA NEW_OUT
```

Build the host binary on Wick and the sm_89 AOT image on the registered GPU pod. The build verifies the predecessor CUDA sources and checks both PTX and SASS. The finite command compares the zero-state scores to the historical linked cubin and tests the address permutation, shared-address aliasing, deduplication, signed and no-op updates, correct and refused copies, and the frozen stage-two score.

Fit output includes `first-period.i16le`, `joint.i16le`, `fit.json`, `schedule.jsonl`, `authenticated.jsonl`, per-arm slot audits and per-row fitted score triples. Compare the generated schedule digest to the pre-registered one. The replay subcommand writes `first-period.jsonl`, `joint.jsonl` and their reverse-order equivalents in the [prior factorial wire format](../../codex-4324/device/README.md), and requires exact forward/reverse equality. The learned arm's commit receives the actual self-fed prefix and declared original source, with a zero target field. The other three arms supply releases only for the named diagnostic gates.
