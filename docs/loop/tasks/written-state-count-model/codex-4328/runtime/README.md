# Replay the ordinary frozen client on Wick

The runner sends the sealed user texts to the ordinary `uor-chat --discrete --say` command with the artifact-bound tokenizer, literal dialogue protocol 2, greedy decoding and a 12-token cap. Its only input from the panel is `prompts.jsonl`, which holds case IDs and exact user text. The shell preserves trailing newlines in the text. The deterministic grader and supplied-address controls consume the trace separately.

Build the trace CLI from the [frozen source revision](../evidence/runtime-source-revision.txt) in an isolated checkout. The [runtime source manifest](../evidence/runtime-sources.sha256) binds the reader, writer, CLI and runner from that revision. It also matches the isolated Wick build files. The later release comparison uses a different executable. Use the existing #4327 generic client as the original executable and verify its digest from the run's input manifest. The unchanged executable provides the uninstrumented replay for default and authenticated copy.

```sh
frozen_revision=$(cat "$UNIT/evidence/runtime-source-revision.txt")
while read -r digest source_path; do
  git show "$frozen_revision:$source_path" |
    sha256sum | awk -v expected="$digest" '$1 != expected { exit 1 }' || exit 1
done < "$UNIT/evidence/runtime-sources.sha256"
# From the isolated checkout at "$frozen_revision".
cargo build --locked --release -p uor-r4-integer \
  --features p1-count-serving,release-factorial-diagnostic \
  --bin uor-chat --bin verify-native-copy --bin verify-native-occurrence
bash "$UNIT/runtime/run.sh" "$WICK_RUNS" "$FRESH_OUTPUT" \
  "$TRACE_CHAT" "$ORIGINAL_CHAT" "$SEALED_PROMPTS" "$PROMPTS_SHA256"
(cd "$FRESH_OUTPUT" && sha256sum --strict -c SHA256SUMS)
```

`WICK_RUNS` is `/mnt/plexpool/uor-r4-corpus/runs`. Create the new output as a child of `$WICK_RUNS/20261127-codex-ordinary-source`. The driver verifies the base, balanced and paired artifact SHA-256, the tokenizer, both client executables, the panel and the runner around collection. The manifests name the original Wick paths. The per-process `.time` rows record case ID, execution mode, wall seconds, user seconds, system seconds and peak RSS in KiB. They include process start, model loading, generation and serialization.

The driver emits a traced and uninstrumented row for every case, artifact and mode. The plain records agree field for field after removing the trace field. Authenticated copy consumes its copy request on the first emitted token. Every following token is a native prediction from the actual self-fed prefix. The default trace checks its observed writer ID against the original production prediction at each step. The diagnostic `--no-native-occurrence` arm runs the existing marker fallback on the same prefix and has no occurrence or copy action at any step. Its uninstrumented replay uses the same diagnostic flag with tracing omitted. The ordinary `--read-mode no_read` option is tested separately on the first four sealed cases and compared to the complete default traced records for all three artifacts.

Each trace records the retained window offset and length, last causal token, occurrence framing and score decision, separately read and claimed original positions and IDs, the native proposal and committed token, whether the claim won the write, and the frozen trace words. `prefix_start + source_pos` is the original token address when the reader selects a source. The runner checks that the exact token at that address matches the native gate token, every commit matches the generated IDs, and the next step sees the actual preceding commit. Source-read framing rejection, exhausted token cap and emitted EOS remain distinguishable in the raw records.

The focused finite tests hit an existing test-build error in the stack's pair rows. The isolated Wick copy passed all three occurrence feature tests after the [one-line test-only correction](../evidence/runtime-test-compat.patch). The Wick copy's original source was restored after the check. The [initial log](../evidence/runtime-finite-tests.log) and [passing excerpt](../evidence/runtime-finite-tests-corrected.log) retain both outcomes. The original logs and contextual diff are in the Wick study's `staging/` directory.
