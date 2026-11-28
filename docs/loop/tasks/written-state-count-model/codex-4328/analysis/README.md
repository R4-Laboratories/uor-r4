# Grade the ordinary client from separate truth and trace tapes

The grader consumes the sealed truth, traced runtime rows and uninstrumented runtime rows after capture. It binds the original prefix IDs, tokenizer and all three artifacts. It checks every transition against the actual prior committed IDs, enforces one-shot copy semantics, and requires matching uninstrumented output for all nine arms and 48 cases per arm.

The opening is generation step 0. A selected source is correct only if the absolute original prompt position and ID both match the expected source. A correct authenticated opening also requires a copy commit from that same position and the expected first emitted ID. Same-ID wrong occurrences are scored as wrong reads and, when committed, wrong copies. An unsupported read is false whenever it selects an occurrence. An unsupported copy is false whenever it commits a copy, independent of its token text. Parsed abstention, framing rejection and the diagnostic disabled arm are separate. Ordinary no-read counts include all first-step absences, with their cause retained.

Strict reply exactness compares original IDs through EOS. The first divergence includes its zero-based step, expected ID and actual committed ID, including an early EOS or missing ID. The actual next self-fed ID is always retained when emitted. An unforced period requires the emitted prefix through the expected period to match. A release-boundary witness additionally requires the correct authenticated opening. For matched stop requests, the expected postperiod ID is EOS. For matched continuation, it is the next requested source ID. Source-level interpretation uses the occurrence trace. No text-based heuristics can mark a source correct.

```sh
cargo run --locked --release --manifest-path "$UNIT/analysis/Cargo.toml" -- \
  "$PANEL/truth.jsonl" "$RUN/traces.jsonl" "$RUN/uninstrumented.jsonl" "$NEW_GRADE"
```

`rows.jsonl` records each verdict, both tapes' IDs, source provenance, next IDs, exactness, first divergence and EOS/cap. `summary.json` carries totals by artifact, arm, group and wording family, including emitted successor IDs and uninstrumented parity. It also compares entire transition records and output IDs between the three artifacts for each mode and case.
