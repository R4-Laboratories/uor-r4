# Replay the frozen ordinary client on the exposed source probes

This replay runs the unmodified archived #4330 Linux `uor-chat` binary on Wick and verifies its exact bytes before invocation. The commands use `--discrete --tokenizer --protocol 2 --say`, greedy decoding at seed zero and a 16-token cap. The balanced and frozen whole-turn artifacts use authenticated copy, occurrence tracing and the existing period-release operator. The historical D26 contextual artifact uses its ordinary read-enabled contextual path and emits its native copy-step words. No positions or grades are passed to the client.

The separate Rust grader re-encodes protocol 2, verifies each raw prompt ID and checks the committed stream against the trace and decoded text. For the 13 supported questions it reads the sealed original UTF-8 span, tokenizes it with the canonical tokenizer, authenticates every span ID and its exact first source address in the prompt, and appends the canonical EOS ID to form the extractive probe target. It checks first emitted ID, first committed source address, the first different token index and exactness through EOS. For the five unspecified questions it records the chosen and committed addresses, emitted tokens and stopping behavior, without assigning response targets. Counts describe this exposed finite panel.

On Wick, use the pinned #4333, #4332 and #4330 cold restores, the verified historical D26 contextual D11, and a fresh output root.

```sh
bash "$CLIENT/run.sh" "$RUNS" "$COLD_4333" "$COLD_4332" "$COLD_4330" "$D26_ARTIFACT" "$NEW_OUTPUT"
(cd "$NEW_OUTPUT" && sha256sum --strict -c SHA256SUMS)
```

Raw client tapes are under `client/`, with scores written beside the aggregate summary. The raw JSON retains every client field except the relocated artifact pathname. The artifact digest stays in each record. The capture can run separately from the grade join through `capture.sh` and `grade.sh`. Source spans, prompts, raw traces and per-case scores stay in private custody.
