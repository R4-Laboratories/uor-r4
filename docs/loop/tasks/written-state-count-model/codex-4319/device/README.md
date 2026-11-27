# CUDA word-ending experiment

The Linux driver reuses the #4318 copy instrument's frozen state loader, launch interfaces, and native-proposal checks. The original copy, occurrence, and marker cubins are hash-pinned. The new ahead-of-time image contains only instruction fitting, checked packing, scoring, and the post-copy ending choice.

The driver isolates the instruction after the final question mark in the question region and before the exact assistant marker. The CUDA fit and score kernels enumerate the same occupied unigram and ordered-pair addresses from the frozen packed R4 roots. Fitting increments each distinct address once per TRAIN prompt with the signed target ending. The device refuses an accumulator outside signed i16 range and returns all 4096 exact words in little-endian order. The score kernel reads the instruction and weights. Only the oracle arms use the target ending during dispatch.

Each pair contains two rows with identical facts, question, target, gold position, and family and distinct instructions. The driver joins them by pair ID and side, even across nonadjacent rows. It checks the supplied copy at the opening and self-feeds its actual token. The next step records the ordinary proposal and the final commit for each arm. The swapped arm uses the actual paired prompt's signed score. A third step uses the untouched frozen writer on the emitted prefix. The trajectory terminates on actual EOS or reaches the three-step cap. Each arm records its one-based first mismatch and terminal flag.

```sh
cargo build --locked --release --target x86_64-unknown-linux-gnu --manifest-path device/Cargo.toml
bash device/build-image.sh NEW_IMAGE_DIR
ending-device check-rows PANEL.jsonl
ending-device finite NEW_IMAGE_DIR/ending.cubin
ending-device train NEW_IMAGE_DIR/ending.cubin BASE_FULL_DIR TRAIN.jsonl NEW_TRAIN_DIR
ending-device score NEW_IMAGE_DIR/ending.cubin COPY.cubin OCCURRENCE.cubin BASELINE.cubin \
  BASE_FULL_DIR CLASSES.u8 READER_WEIGHTS.i16le MARKER.tsv NEW_TRAIN_DIR/weights.i16le \
  PANEL.jsonl NEW_SCORE_DIR
```

Run the AOT build and device checks on the leased GPU. The finite command checks planted address presence, cancellation, order, signed scores, the matched swap, native fallback with zero weights or an unverified copy, and both i16 overflow boundaries. The scoring command writes `observations.jsonl` and `reverse.jsonl`. The reverse traversal loads the state and code images afresh. `replay.json` records exact comparison across all rows and arms.
