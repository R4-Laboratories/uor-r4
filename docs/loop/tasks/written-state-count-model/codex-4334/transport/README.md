# Finite boundary-relative feature replay

For each original site, the probe reads the four packed native root indices from the state through that site, inclusive. It obtains the query word immediately before the exact assistant marker. The two relative cuts are the recorded passage separator and `max(user.start, user.end - 32)`. Cut (B) means the recurrence of `prefix[..B]`. The relative anchor excludes the token at (B). The probe uses `cl06_bfs(query_root, anchor_root)` and applies that canonical action to the site's root with `cl06_action`. Dividing the raw coordinates by two gives exactly two signed unit axes in each block. All arms have four blocks and use the same 32 i16 score slots.

Each relative frame holds the current case's anchor and original occurrence fixed for all four feature arms. The actual arm takes the current query word, the fixed arm takes zero, and the swapped arm takes the opposing case's query word. The removed arm sets the query to the current anchor. The exhaustive raw check verifies that this acts as the identity on every site root. In the absolute-zero reproduction, the removed field retains its historical #4333 definition and transports the passage-only query to zero.

The extractor compares the product recurrence against the #4332 independent coordinate replay at the query cut, both boundary cuts, the complete opening and **every** captured broad-admission candidate site, including sites outside the gate's labeled set. It checks candidate positions and token IDs against the original user interval and checks coordinate, offset and membership identity against the archived frozen assessment. The original #4332 capture tool independently authenticates the serialized user opening, exact separator, frozen candidate scores and NoRead score. Both instruments consume blind files only. The diagnostic opposing map is reused from the sealed #4333 record, where it contains only paired case IDs.

## Wick replay

Set `RUNS` to the Wick corpus runs directory, `COLD` and `PRIOR` to completed #4333 and #4332 cold restores, and `OUT` to a fresh path. The replay builds both tools in the disposable original checkout with the visibility-only patch previously used by #4333. It compares original captures and the zero-anchor files byte for byte against their sealed predecessors. Reversing both the input and prompt streams must reproduce each of the five outputs for each panel after undoing the record reversal. It writes a local checksum manifest only after those comparisons succeed.

```sh
bash "$TRANSPORT/run.sh" "$RUNS" "$COLD" "$PRIOR" "$OUT"
for panel in diagnostic fresh; do
    jq -n --slurpfile cuts "$OUT/$panel-cuts.jsonl" \
        --slurpfile zero "$OUT/$panel-zero.jsonl" \
        --slurpfile separator "$OUT/$panel-separator.jsonl" \
        --slurpfile tail32 "$OUT/$panel-tail32.jsonl" \
        -f "$TRANSPORT/summary.jq"
done
```

The working capture and evidence files contain private rows. The summary also contains the case ID and original position of the first raw feature change for each reported contrast. Keep the source rows with the private packet and publish only the intended aggregate evidence.
