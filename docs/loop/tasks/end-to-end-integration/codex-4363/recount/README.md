# Recount the captured opening scores

Build with Rust and run against the verified private capture. The checker reads originals without changing them and prints aggregate JSON with no article IDs or text. It consumes already captured GPU traces. The fixed-point head-dot recalculation uses retained 512-word hidden vectors and the three original UORBFY03 checkpoints.

```sh
cargo build --release --locked --offline
prior=/mnt/plexpool/uor-r4-corpus/runs/20261128-codex-grounded-reset
capture=/mnt/plexpool/uor-r4-corpus/runs/20261128-codex-grounded-margin/received/attempt01
test -f "$capture/COMPLETE"
(cd "$capture" && sha256sum --strict -c members.sha256)
target/release/grounded-margin-recount \
  "$prior/inputs/train.jsonl" "$prior/bin/probe-final.sm89.cubin" \
  "$prior/inputs/historical.json" "$prior/inputs/base.uorbfy03" \
  "$prior/received/attempt02/checkpoints/reset.uorbfy03" \
  "$prior/received/attempt02/checkpoints/full.uorbfy03" \
  "$capture/rows.jsonl" "$capture/instrument.jsonl"
```

A positive signed quote gap means ID 1228 beats every eligible competitor in cell 3. Zero is an exact tie, and a negative value means another eligible token scores higher. The capture only includes the elected and reference cells, so compare signed gaps on paired rows where both captures contain cell 3. Also report the narrower subset where both ordinary elections admitted cell 3. Integer medians report both central order statistics.

Raw score gradients are exact local derivatives of the checkpoint's 64-way squared loss for the coarse and reference fine cells. Their training eligibility differs from the deployed BOS emission mask. A missing elected-cell derivative is a different supervised cell, and an exact clamp endpoint alone does not establish saturation. Recount saturation checks whether the unclamped derivative actually exceeds the Q15 activation clamp.
