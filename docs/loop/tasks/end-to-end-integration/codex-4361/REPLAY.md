# Reproduce the matched training and free-running comparison

The [pre-registration](SHEET.md) freezes the contrast, references and decision rule. Use the [cohort recipe](cohort/README.md) and [training instructions](train/README.md) to rebuild. The model-side executable sources, including the ordinary text client and its byte checker, are at `f67507ba0d7de15ca5d395c916f6060db389ec09`. The [training intake](evidence/training.sha256) pins the real starting weights, target tape, binary, kernel and historical fixture. Revision `c70a76f7e18c622001353c6708fc1b369b7ba678` adds the [count correction](COUNT-CORRECTION.md), its integer histogram check and the separate [control dispatcher](correct-count.sh).

## Device and client

The optional `butterfly-v3-client` feature exposes a research replay through the ordinary CLI. It is off by default and uses the allocating reference transition. Its per-token forward cache allocates vectors for intermediate features. The charter requires an allocation-free steady state for the frozen product runtime. The qualified result here is original-ID, text and stop parity through the ordinary input path. M4 serving requires a separate preallocated implementation, zero-allocation instrumentation, exact transition and decode parity, and the registered target-device performance evidence.

From the repository root, set `unit=docs/loop/tasks/end-to-end-integration/codex-4361`. The source for the historical and diagnostic calibration kernel is in [the preceding experiment](../codex-4360/REPLAY.md). Build the additional histogram and CPU programs with the following commands in an isolated checkout.

```sh
nvcc -cubin -arch=sm_89 -O3 "$unit/eval/counts.cu" -o counts-int.sm89.cubin
cargo build --release --locked --manifest-path "$unit/eval/Cargo.toml"
cargo build --release --locked -p uor-r4-integer --no-default-features \
  --features butterfly-v3-client --bin uor-chat
cargo test --release --locked -p uor-r4-integer --no-default-features \
  --features butterfly-v3-client --bin uor-chat
```

The [finite](finite.sh), [TRAIN](training.sh), [calibration](calibrate-client.sh), [original held](held.sh) and [count correction](correct-count.sh) dispatchers document the historical pod layout, including the deliberately rejected count attempt. For an independent physical replay of the three saved checkpoints, restore the private archive at the revision in the [restore receipt](evidence/restore.json) to an empty directory. Verify the archive before extracting it. Run these commands with the repository script and the restored packet on an owned sm_89 lease. The input check can also run on a host without a GPU.

```sh
printf '%s  %s\n' 6c74690edfbf5666fc92f715c46d393255f02b787f47065babbff39353a973a1 \
  custody.tar.zst | sha256sum --strict -c -
mkdir -m700 restored
tar --zstd -xf custody.tar.zst -C restored
bash "$unit/replay-physical.sh" "$PWD/restored" --check-inputs
bash "$unit/replay-physical.sh" "$PWD/restored" "$PWD/new-physical-replay"
```

The [replay](replay-physical.sh) verifies all packet members, binds the three model roles to the original training and ordinary-client receipts, reruns the three-article calibrations, and compares them with archived trajectories before the 222-article stage begins. It then runs the corrected count binary and checks all four output streams. The original cubin is never loaded by this replay. The input-check branch was verified from the restored packet. A second post-publication GPU execution of the combined replay script is unmeasured.

A cold client takes an ordinary JSON message array, an independently trusted manifest SHA-256, and the two assets identified by that manifest. It prints decoded answer bytes on stdout and the termination reason on stderr.

```sh
uor-chat --butterfly-v3 manifest.json "$trusted_manifest_sha256" \
  model.uorbfy03 tokenizer.json messages.json 32 --trace new-trace.json
```

The [cold qualification script](ordinary/qualify.sh) binds each model identity from the TRAIN receipt, checks each byte of its output against the GPU trajectory on the predeclared text inputs, and writes `ordinary-gate.json`. The [client replay](ordinary/replay-held.sh) also checks each encoded prefix, little-endian digest, original output IDs and stop. Release the 222-article device panel only when all three arms pass all three cold conversations. Every device history intervention also reruns with placeholder diagnostic labels. A mismatch aborts scoring.

## Recount the captures on Wick

The private run root is `/mnt/plexpool/uor-r4-corpus/runs/20261128-codex-grounded-reset`. Run the following from that root or from the restored packet. It reads recorded trajectories and original references without executing the model. Its new output directory must not already exist.

```sh
umask 077
a=received/panel
messages=cohort/complete-reservation/final/held-messages.jsonl
if ! test -e "$messages"; then messages=cohort/final/held-messages.jsonl; fi
(cd "$a" && sha256sum --strict -c members.sha256)
bin/grade inputs/tokenizer.json inputs/held.jsonl \
  "$messages" \
  "$a/frozen.rows.jsonl" "$a/reset.rows.jsonl" "$a/full.rows.jsonl" \
  "$a/counts.rows.jsonl" received/attempt02/checkpoints/COMPLETE.json graded-replay
```

The summary is deterministic. Each grading run randomly permutes the three candidate replies separately for each conversation and writes a private key. Use only the blind packet and rubric with graders. Once actual ratings have been collected, retain the particular key generated with the distributed packet. The model weights, reversible token sequences, complete original conversations and per-article key belong in the private archive.
