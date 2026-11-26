# Capture the eight-sweep signed transport fit

The Wick build pins the released [#4289 fitter](../../codex-4289/fit/README.md), #4288 reader and physical Wick lease helper. The patch adds an observation call after the original eight updates and before elected-candidate readback reuses a device measurement buffer. Enrollment, scoring, tie handling, nearest-center assignment, strongest-rival mining, integer update and the training objective use the released CUDA code. The build compares the complete disassembly of the enrollment, measurement and update kernels with the released binary. The observation kernel receives constant references to the completed states and their measurements. Distance labels feed only this kernel and the reporting code.

Under the inherited physical Wick lock, call the SM89 binary with the TRAIN geometry and signed labels, initial state, distance labels and a fresh output directory. The distance file has one `0` for far or `1` for near per geometry row. Both supervised signs in the row use that distance.

```sh
transport-mixed-sm89 TEXT LABELS PARENT_STATE DISTANCE_TXT NEW_OUTPUT_DIR
```

| Output | Contents |
|---|---|
| `step-0.state` to `step-8.state` | The unconverted UORSTT01 bytes at parent and successive updates. Frozen signed lexical bytes, transport counts and inactive slots are verified at every checkpoint. |
| `trace.jsonl` | The unchanged #4289 per-edge training trace with initial, elected and terminal reads and all nine per-state read histories. |
| `quality.jsonl` | `step`, total `hinge`, `strict_wins`, `strict_losses`, then the same fields and eligible edge count for `far` and `near`. |
| `votes.jsonl` | `step`, `sign`, `slot`, `axis`, and exact signed `near` and `far` vote sums for every active coordinate on steps 0–7, before the update to the following state. |
| `vote-summary.jsonl` | Counts per pre-update step where both strata have nonzero opposing signs (`opposed`), the opposing sums cancel exactly (`cancelled`), both strata have nonzero agreeing signs (`agree`), just one stratum has a nonzero sum (`near_only` or `far_only`), or both sums are zero on an active coordinate (`zero`). Cancellation is a subset of opposition. |

The unit-margin hinge assigns zero to every eligible edge that already strictly wins on that state. Each such edge contributes zero votes to both its gold and rival centers. A row can become active again as other rows move a shared center. The observer replays every coordinate transition from the net sum, including attempted moves outside the i16 interval. The unchanged blind joint ranker provides the separate TRAIN retention measurement.

On Wick, stage this directory to a fresh path, then run `bash run-wick.sh build v1` and `bash run-wick.sh finite v1 v1`. Use distinct versions for subsequent builds and finite runs. The finite control checks exact opposed votes with a residual, exact cancellation, agreement, each one-sided vote, a dormant near win and both i16 boundaries. The released #4289 binary supplies the exact trace, elected and terminal checkpoints, and cold signed readback. All nine raw states agree across mixed, all-far labels and a duplicate mixed invocation. The unmodified #4288 binary reads each serialized checkpoint and gives identical original-address readout on the duplicate checkpoints. Only the synthetic fixture runs from `run-wick.sh`.
