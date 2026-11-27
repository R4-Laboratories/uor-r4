# Pack and check the two native fits on Wick

Build these binaries from the integer runtime at `6a1d84dcdd31f3ac42c67b159ac46a6310c8b578` or a checkout with the same serving sources. The generic CLI and grounded server are unchanged.

```sh
cargo build --locked --release -p uor-r4-integer \
  --features p1-count-serving,release-factorial-diagnostic \
  --bin pack-native-source-choice --bin serve-native-source-choice \
  --bin verify-native-source-choice --bin uor-chat
cargo build --locked --release --manifest-path "$UNIT/product/ordinary/Cargo.toml"
export PAIRED_PRODUCT_BIN="$PWD/target/release"
PROBE="$UNIT/product/ordinary/target/release/paired-ordinary-dialogue-probe"
```

`UNIT` points at this study directory in the checkout. `BASE` points at the pinned #4319 D11 writer. `ROOT` is the private Wick study root and `TOKENIZER` the artifact's bound tokenizer JSON. Verify the returned measurements before running the three product checks, each of which creates a fresh output directory.

```sh
sha256sum --strict -c "$UNIT/product/runtime-sources.sha256"
(cd "$ROOT/measured-v1" && sha256sum --strict --quiet -c OUTPUTS.sha256)
bash "$UNIT/product/pack.sh" "$ROOT" "$ROOT/measured-v1/raw/fit" "$BASE"
bash "$UNIT/product/replay.sh" "$ROOT" "$ROOT/panel-v1/held/rows.jsonl" \
  "$ROOT/measured-v1/raw/held/trained"
bash "$UNIT/product/ordinary.sh" "$ROOT" "$PROBE" "$TOKENIZER" "$BASE"
```

The packer reopens the artifact and binds the 49,158-byte state and fit receipt to the frozen base. The cold verifier checks each grounded serving request against the source-choice writer. Those requests include the token prefix and four physical positions. The server opens with an authenticated copy from the first supplied position and commits a supplied period before the learned release decision. For each artifact the verifier also independently reloads the file and compares every field of each of the four factorial modes against forward and reverse GPU observations. The input and product manifests bind artifacts, panels, serving binaries and observations. File stems `first-period` and `joint` in the GPU observations are aliases of `rowwise` and `paired`, respectively, from the reused scorer binary.

The separate prompt-only script uses the generic `uor-chat --discrete --say` entry and product `--prompt-ids` path for follow-ups. The grounded server refuses JSON without positions. The CLI's `--stack` path uses a different artifact schema. `--discrete` loads the source-choice envelope and its frozen base but the generic decode loop calls the base predictor without calling the learned release selector. The [sealed rubric](ordinary/RUBRIC.md) specifies inputs, exactness and EOS. Timing files cover process overhead, artifact load and decode together. A generic probe arm starts a fresh CLI process for each turn and makes two extra refusal calls for each fitted artifact. The timings describe that workload.
