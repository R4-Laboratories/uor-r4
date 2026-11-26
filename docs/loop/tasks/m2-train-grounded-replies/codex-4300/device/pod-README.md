# Rented SM89 device replay

The finite and research payloads carry the original Wick-built SM89 binary. They contain
no CUDA source compilation step. The binary and standalone finite control use a Linux
x86-64 runtime with glibc 2.34 or newer. The device probe verifies the leased UUID, an RTX
4090 with compute capability 8.9, free memory and process occupancy, and records the driver,
host and executable dependencies. Keep the finite and research phases in distinct directories
on the same leased device. Research requires the Wick release and its separately sealed
portable authorization.

## Prepare on Wick

Stage `pod-stage.sh` and `pod-run.sh` together in a new `$STUDY/device/pod-recipe-vN` directory.
Use fresh version suffixes on retries.

```sh
bash "$RECIPE/pod-stage.sh" package v1 "$STUDY/panel/panel-v1" \
  "$STUDY/device/build-v1" "$STUDY/device/host-finite-v1"
```

The output is `$STUDY/device/pod-payloads-v1`. Transfer `finite.tar.gz` and verify both the
archive hash and the extracted `finite/SHA256SUMS` hash from the package receipt. On the leased
instance, use `/workspace/uor-r4/jobs/codex/switch-source-pod` as the working directory.

```sh
tar -xzf finite.tar.gz
bash finite/pod-run.sh probe finite "$FINITE_INPUT_SHA" probe-v1 "$GPU_UUID"
bash finite/pod-run.sh finite finite "$FINITE_INPUT_SHA" finite-v1 "$GPU_UUID"
tar -czf finite-result-v1.tar.gz finite-v1
```

Return the archive, hardware and process receipts, launcher exit code and exact remote
`finite-v1/SHA256SUMS` digest to Wick. Transfer the raw trace with the original manifest.

```sh
bash "$RECIPE/pod-stage.sh" import-finite v1 "$FINITE_RETURN_TAR" \
  "$FINITE_OUTPUT_SHA" "$FINITE_INPUT_SHA"
```

The import verifies the raw manifest and checks native replay against the standalone reader,
deterministic rerun, and all 14 finite assertions. It preserves the raw `device/finite-v1`
manifest and seals the Wick checks in `device/pod-import-finite-v1`. The independent Rust
readback consumes `device/finite-v1/first` and its adjacent lease. The controller seals the
provider lease receipt, raw manifest, launcher log, GPU identity and exit code in the adjacent
`device/finite-v1-lease`.

## Authorize and capture

After exact independent fixture parity and non-author FIT, pass the sealed Wick release hash
to issue a small authorization archive.

```sh
bash "$RECIPE/pod-stage.sh" authorize v1 "$STUDY/device/pod-payloads-v1" \
  "$STUDY/release-v1" "$RELEASE_SHA"
```

Transfer `research.tar.gz` and `pod-authorization-v1/authorization.tar.gz` and verify both
archive hashes. The capture verifies the research payload, authorization hash, Wick release
manifest hash and equality of the released unpacked full state. It records every command and
its timing, device stderr, expected input and output counts, and the complete raw replay.

```sh
tar -xzf research.tar.gz
tar -xzf authorization.tar.gz
bash research/pod-run.sh capture research "$RESEARCH_INPUT_SHA" capture-v1 \
  "$GPU_UUID" auth "$AUTH_SHA" "$RELEASE_SHA"
tar -czf capture-result-v1.tar.gz capture-v1
```

```sh
bash "$RECIPE/pod-stage.sh" import-capture v1 "$CAPTURE_RETURN_TAR" \
  "$CAPTURE_OUTPUT_SHA" "$RESEARCH_INPUT_SHA"
```

Wick import verifies the sealed absolute release, methods, panel, binary, bank, state and
ancestor trace paths. It also independently byte-compares the unrestricted historical and
mixed TRAIN outputs to all four prior physical captures before moving the raw tree into
`$STUDY/capture-v1`. The controller seals the adjacent `capture-v1-lease`. The existing
independent replay and grader then read the same paths as for a local Wick capture.
