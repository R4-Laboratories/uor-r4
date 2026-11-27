# CUDA copy-ID instrument

This instrument loads the frozen #4315 occurrence cubin and all 513 native reader words, the #4314 marker cubin and its frozen signed state, and the complete integer D11 buffers used by #4315. It reads the marker head on every trailing colon and the occurrence head on every prefix. The new AOT writer uses the same audited D11 core, with a narrow build-time patch at the vocabulary writer. The build checks the SHA-256 of the original core and its generated tables before patching a copy.

The frozen occurrence kernel supplies the eligible factual positions and learned read. The device keeps the selected read and asserted write position independently. It re-reads the original token at an eligible claim and accepts the copy only for the selected address when that ID is admitted and its distance equals the minimum in the full admitted candidate set. It commits an accepted copy before ranking the ordinary top eight. In each arm, the native proposal uses that arm's selected read and remains available as the causal fallback. The marker arm explicitly removes the occurrence read.

The first output is the actual raw committed ID, including a space if generated. The driver appends it to the prefix exactly once for the successor, using the ordinary learned reader and no copy claim. EOS stops immediately and gets no successor. An already full context slides by one token, following ordinary generation. The output records the checked address, exact ID, distance, native proposal, committed ID, integer read and recurrence state, causal pointer, and first admitted cell for both steps. The ordinary proposal is also compared against the unchanged #4315 cubin's writer for every prefix.

Use the Wick release driver or build the detached host manifest in this directory with a Linux CUDA driver available. Build the AOT image on a GPU pod with CUDA tools on `PATH`.

```sh
cargo build --locked --release --target x86_64-unknown-linux-gnu --manifest-path device/Cargo.toml
bash device/build-image.sh FROZEN_HEADERS NEW_IMAGE_DIR
copy-id-device check-rows PANEL.jsonl
copy-id-device score NEW_IMAGE_DIR/copy.cubin OCCURRENCE.cubin BASELINE.cubin \
  BASE_FULL_DIR TOKEN_CLASSES.u8 WEIGHTS.i16le MARKER.tsv PANEL.jsonl NEW_OUTPUT_DIR
```

Run the sealed finite, held and unsupported panels into independent output directories. Each contains `observations.jsonl` in input order and `reverse.jsonl` in reverse traversal order. The reverse pass opens all images and native state afresh. A `replay.json` receipt is written only after exact agreement on every scored event. Input rows and outputs bind the raw prefix and pair ID.
