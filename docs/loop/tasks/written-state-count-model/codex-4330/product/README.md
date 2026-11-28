# Whole-turn occurrence carrier and cold comparison

The envelope `uor-r4.lut-discrete-native-whole-turn/1` stores 513 little-endian signed 16-bit words, followed by the complete balanced checkpoint at SHA-256 `5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa`. The 513th word is the NoRead score. The header binds the raw state hash, the fit receipt hash and the balanced artifact hash and length. The loader recognizes the exact nested balanced header before parsing it and verifies the entire nested hash. Token classes, roots, the prior framed weights, the trunk and the writer remain in the nested artifact.

With the new artifact, `uor-chat --say` plus `--authenticated-copy` uses protocol 2 and the current user range emitted by the dialogue serializer. The existing occurrence operator receives the exact original IDs in that range, with the final at most 32 in its query range. Typed serializer provenance is required for the expanded opening. The reader gets one decision at the initial marker. A selected address reaches the old authenticated writer, where its ID still has to belong to an admitted cell and reach the exact minimum distance. A tie, threshold refusal or later step uses the marker and trunk fallback. Literal marker text inside the message remains in the user content range.

The packer extracts the original 513 words from the exact balanced checkpoint. It verifies that each packed envelope reloads with exactly the input words before it publishes the path. The cold verifier restores all three artifacts and accepts JSONL with a `case_id` and a raw `text` or `prompt` field. It serializes each request itself and returns the exact source candidate positions, token IDs, sixteen feature slots and signed score, plus the two ranges, winner, best score, ties and NoRead threshold for each arm. Gold labels, source addresses and target IDs are unnecessary for the verifier.

```sh
cargo build --release --locked --offline -p uor-r4-integer \
  --features p1-count-serving --bin uor-chat --bin pack-whole-turn-occurrence \
  --bin verify-whole-turn-occurrence

pack-whole-turn-occurrence extract "$BALANCED" "$INITIAL_WORDS"
pack-whole-turn-occurrence pack "$BALANCED" "$INITIAL_WORDS" "$FROZEN" "$INITIAL_WORDS_SHA256"
pack-whole-turn-occurrence pack "$BALANCED" "$FIT_WORDS" "$FITTED" "$FIT_RECEIPT_SHA256"
verify-whole-turn-occurrence "$BALANCED" "$FROZEN" "$FITTED" "$TOKENIZER" "$RAW_PROMPTS"
uor-chat --discrete "$FITTED" --tokenizer "$TOKENIZER" --protocol 2 \
  --say "$TEXT" --authenticated-copy --native-occurrence-trace --max-tokens 16
```

On Wick, the initial occurrence words hashed to `c4c2cb4139f1af761da5c5d6dcc656dab81ad9a813646c3efef7be672ca5a08a`. Packing that exact state with its digest as the frozen provenance field yielded `6415f6cb9fbf4694b449c08170d17c7e821c168d28e811cfa4e7a3448a333422`. The finite check and transport records are under `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-whole-turn-fit/product-v1`.
