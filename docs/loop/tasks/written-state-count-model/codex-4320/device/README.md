# Post-period CUDA diagnostic

The Linux driver includes the #4318 driver source and loads its frozen copy,
occurrence and marker cubins and eight exact D11 buffers. The #4319 ending
weights belong to the frozen parent artifact. The supplied period action makes
no inference from those weights. The registered approval binds the driver, AOT
image, panel and frozen dependencies before scoring.

Build the host driver on Linux. Build the AOT image on the leased CUDA device
using the exact geometry tables shared with #4318. The image build checks those
tables and the core source, scans PTX and SASS, and requires exactly the two
diagnostic entry points.

```sh
CARGO_TARGET_DIR=NEW_TARGET_DIR cargo build --locked --release --manifest-path device/Cargo.toml
bash device/build-image.sh FROZEN_TABLE_DIR NEW_IMAGE_DIR
postperiod-device check-rows PANEL.jsonl
postperiod-device finite NEW_IMAGE_DIR/completion.cubin COMPLETION_SHA256
postperiod-device score NEW_IMAGE_DIR/completion.cubin COMPLETION_SHA256 \
  COPY.cubin OCCURRENCE.cubin BASELINE.cubin BASE_FULL_DIR \
  CLASSES.u8 READER_WEIGHTS.i16le MARKER.tsv PANEL.jsonl NEW_SCORE_DIR
```

The row check requires a complete opposite pair, equal facts and question,
different suffixes, and two distinct original factual IDs at their bound raw
positions. The finite CUDA check plants verified and refused period selections,
opposite completion states, a native EOS retained by continue, and a separate
supplied original word. It exercises missing, mismatched and fenced-off
addresses and a bad peer. The score command writes the original-width native
writer events to `observations.jsonl` and `reverse.jsonl`. Reverse traversal
opens the frozen images and state afresh. Exact agreement produces
`replay.json`.

Each row contains its input fields, `third_trace` and `arms`. All four arms
share the actual authenticated copy and supplied output period. At token three,
continue writes the ordinary proposal, including EOS if proposed. Stop selects
EOS. The swapped arm reads the actual opposite row's completion state. The
`supplied_next` intervention additionally verifies the factual address
against the original prompt and reads its exact ID when continuing. Every EOS
terminates the trajectory. A fourth token uses the native writer on the actual
self-fed prefix with the copy claim absent.

The trace records the native read before writer rotation, recurrence state and
sums, the recurrent frames, the four blocks' routed node spans and actions,
pointer predecessor and successor, the gate's mixed token and the actual gate
ID after any occurrence override. The separate `pair_successor` and
`pair_distance` report the nearest ordered-pair calculation, which the
ordinary gate leaves dormant. The diagnostic checks its codes, selected gate,
pointer, final 32-word rotation, selected writer ID, first admitted cell and
minimum distance against the frozen writer. The final writer's top eight are
ordered by distance then ID.

Candidate scores use the ordered terms `[c_q,c_qr,c_r,c_qc,c_rr,pointer]` with
integer weights `[1,64,2,32,8,1]`. Candidate IDs include EOS, the requested
continuation, 73, the mixed winner, actual gate, final writer and pointer
successor where present. `recurrence_count` records the raw recurrence bucket
used to populate the pool. `writer_rank` is one-based within admitted cells.
`distance_ties` counts all vocabulary IDs at that distance and
`admitted_ties` counts the admitted IDs. Both counts include the candidate
itself when applicable.

The `sensitivity` array gives three gate winners with the actual candidate
pool held fixed. Index 0 removes the pointer bonus. Index 1 removes the
read-address terms `c_qr,c_r,c_rr`. Index 2 removes the recurrence-address terms
`c_qc,c_rr`. These checks reuse the frozen positive-score unigram/ID tie order
and the frozen prior fallback for a zero maximum.
