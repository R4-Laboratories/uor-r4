# Directed interval operator replay

Run the finite Rust tests and build the ordinary client and interval packer from the same source.

```sh
cargo test --locked -p uor-r4-integer --features serving-wasm --lib discrete_d11::native_interval::tests
cargo build --locked --release -p uor-r4-integer --features serving-wasm \
  --bin uor-chat --bin pack-native-interval
```

The new capture uses the existing protocol-2 `{case_id,text}` input.

```sh
uor-chat --discrete "$BASE" --tokenizer "$TOKENIZER" \
  --interval-relation-features "$OPENINGS"
```

For each opening the client emits `geometry`, `count`, `directed_geometry`, `directed_count`, and `swapped_geometry`. The existing `--interval-features` emits only the first two. The three directed rows carry a `relation_endpoints` array in candidate order and `relation_tokens` in original protocol-2 position order. The query endpoints surround the selected lexical anchor, and the source endpoints surround the candidate within its original clause. Each endpoint is the original subtoken facing the anchor or candidate. The arrays keep original order in every arm.

Bank 10 uses block 2 of the frozen codebook. Each complete neighborhood defines the canonical left-to-right Clifford displacement. The relative action is the composition of the inverse query displacement on the left and source displacement on the right. The swapped arm recomputes the source displacement from right to left. The count arm applies FNV-1a to the four little-endian token IDs in query-left, query-right, source-left, source-right order. Each arm has 64 reachable complete slots and the same missing slot. Each directed arm inherits the other fifteen banks and every NoRead key from the existing geometry arm.

The packer accepts the five capture arm names. The first two retain the version-1 container and the three directed arms use the version-2 container. The ordinary loader binds the version and arm tag to the persisted words. A logged score comes only from the interval state whose arm matches the captured row.

Run `replay.sh` on Wick with the pinned prior client, the two new binaries, the balanced writer, tokenizer and previous `cold-train-v1` directory. The final argument is the source root used for the new build. Allow roughly 7 GB for temporary repacked containers and captures.

```sh
bash "$SOURCE/docs/loop/tasks/written-state-count-model/codex-4342/operator/replay.sh" \
  "$OLD_CLIENT" "$CLIENT" "$PACKER" "$BASE" "$TOKENIZER" "$PRIOR_COLD_TRAIN" \
  "$UNUSED_OUTPUT" "$SOURCE"
(cd "$UNUSED_OUTPUT/evidence" && sha256sum -c REPLAY.sha256)
for arm in geometry count; do
  cmp "$PRIOR_COLD_TRAIN/natural.$arm.actual.cold.jsonl" \
    "$UNUSED_OUTPUT/$arm.train.after.jsonl"
done
```

The replay checks the 264 previously registered TRAIN openings for both version-1 arms. It compares normal, self-fed `--say` traces on one of those openings and three fabricated prompts, and repacks both old containers for byte comparison. Each version-2 container uses a fabricated signed state that gives one selected candidate a weight of 100. The replay checks its cold scores and then compares the writer trace to the original writer with the identical source span supplied. The compact evidence contains synthetic inputs and traces and hashes of the historical TRAIN traces and artifacts.

## Query-window boundary

For an unseparated question the historical query window takes the last 32 tokens. Its first token can occur inside a word. The directed path identifies complete lexical words from the original user bytes, restricts source and query neighbors to the original bounds, and uses a query anchor only when its entire original word matches. Other banks and candidate enumeration retain the original anchor. The [ordinary-client witness](check-window.sh) compares the first v2 implementation from the [initial packet](../evidence/archive-initial.json) against the corrected executable on an exact split-word fixture. Its left query site becomes missing and the candidate key moves from 2606 to the reserved value 2560. Run it as `bash check-window.sh INITIAL_V2 FIXED_V2 BASE TOKENIZER FRESH_OUTPUT`.
