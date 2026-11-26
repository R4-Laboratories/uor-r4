# Fixed-assignment CUDA median refit

The fitter uses the #4288 typed `Geometry::transport`, `closest` and `l1` operators sealed by #4297. It reads both matched `UORTXT01` geometries and one `UORLNK01` label packet. It verifies the immutable lexical points, exact addresses, tuple lists, first 20 transport axes and all axes of the short-gap path. On the device it gathers both signed gold attachments for every row and runs `closest` against the protected endpoint state. Equal distances choose the first (lowest) active slot. The GPU reuses this assignment buffer for both arms and all candidate scores.

For each signed slot and each axis from 20 through 31 the GPU binary-searches signed i16 values for the first coordinate whose cumulative count reaches rank `(support + 1) / 2`. Empty active slots and inactive slots keep their complete state. Each arm uses the same sequence of kernels and 18 native `UORSTT01` snapshots. The GPU scores full 32-axis L1 to the fixed assignment and separately reports the 12-axis component. The 20-axis difference between these two scores is constant across arms and candidates. The host writes the state bytes downloaded from the GPU and verifies their round trip through the native state parser.

```sh
median-fit-sm89 ENDPOINT_TEXT ORDERED_TEXT LABELS PROTECTED_STATE NEW_OUTPUT_DIR
```

- `assignments.jsonl` records each event `2*row+sign`, the canonical slot ID `8*sign+slot`, original addresses and endpoint L1 cost.
- `{endpoint,ordered}/coordinates.jsonl` records the 32 signed coordinates of each event and that shared slot ID.
- `{endpoint,ordered}/medians.jsonl` records all 16 slots in ID order, with active status, support and the 12 lower medians, or null for zero support.
- `{endpoint,ordered}/scores.jsonl` has `base`, `full`, then `slot-00` through `slot-15` in election order. Each row records `candidate`, `assigned_l1`, `assigned_tail_l1` and `changed`. The matching `.state` file uses the same basename.
- `complete.json` records the counts and state size after both arms finish.

Stage the files in a new subdirectory of the owned Wick `fit/` root. Run `bash run-wick.sh build v1` to precompile SM89 SASS and verify the shared archived device kernels, then `bash run-wick.sh finite v1 v1` under the inherited exclusive lease. Use fresh versions for repeat builds and tests. The synthetic fixture checks the fixed endpoint assignment tie order, gold-address fidelity against the archived scorer, signed even medians, axis 20 through 31 behavior, an inactive slot and an empty active slot, two actual different refits, rejected short-gap drift, deterministic repeat, and fractional-free byte-exact `UORSTT01` serialization. The independent numerical replay uses a full sort over the recorded support and compares every scored candidate. Research TRAIN launch belongs to the parent study release.
