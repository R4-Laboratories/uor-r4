# Cold integer product replay

The candidate feature and native state contract is in [FEATURES.md](FEATURES.md).

Build the product tools with the locked workspace.

```sh
cargo build --release --locked -p uor-r4-integer --features p1-count-serving \
  --bin pack-native-occurrence --bin verify-native-occurrence --bin uor-chat
```

Derive the frozen token classes from the canonical tokenizer. The 4,096 class bytes have SHA-256 `a0afae48560e5a8112d682fb067f24c57a2f432935b2405acfcc8b7dd7d31a23`. The packer requires the CUDA-produced 1,026-byte file of 513 signed little-endian words, including NoRead last, and the pinned #4314 marker artifact. `FIT_SHA256` binds the sealed fit evidence. The new carrier keeps the complete marker and D11 writer bytes intact.

```sh
pack-native-occurrence classes "$TOKENIZER" token-classes.u8
pack-native-occurrence pack "$MARKER" weights.i16le token-classes.u8 artifact.d11 "$FIT_SHA256"
verify-native-occurrence artifact.d11 "$HELD_PANEL" "$DEVICE_OBSERVATIONS" > cold.jsonl
```

The verifier cold-loads the supplied file in a new process. It compares raw prompt and annotation identity, parsed bounds, exact source geometry, eligibility, each active feature and signed score, rank, ties, abstention and writer observations across the two producers. It reports the first mismatching field and rejects missing or extra rows. It records keys and source positions on every scoped row, including positions that failed admission. A parsed span can differ from the annotated passage span. Unsupported prompts carry null parsed bounds and invoke the frozen fallback.

Only the verifier supplies source addresses to the diagnostic controls. On a reference that begins with a standalone space, correct and swapped controls run the production fallback at colon and supply the address at the following phase only when the writer emitted that actual space. If an offset-zero control emits a standalone space after its one supplied event, its subsequent event uses the original fallback. The ordinary learned path derives both phases from the actual token prefix. The reader's NoRead and the explicit absent-source diagnostic share the original marker fallback at colon and bare trunk after a real emitted space.

Exercise the same packed artifact through the raw and text entry points. Use a short generation and compare both prompt IDs and the actual emitted first token or first lexical step with the cold observation.

```sh
uor-chat --discrete artifact.d11 --prompt-ids "$PREFIX_IDS" --max-tokens 2
uor-chat --discrete artifact.d11 --tokenizer "$TOKENIZER" --protocol 2 \
  --say "$ORIGINAL_USER_TEXT" --max-tokens 2
```

The finite cold replay matched 5 of 5 CUDA rows using the original marker fallback. The trained artifact was reopened from the saved archive, verified against the member manifest, and matched 1,461 of 1,461 CUDA rows. Both CLI entry points then reproduced the reopened artifact's causal emissions on three post-fit prompts. [Finite receipt](FINITE.json) · [Fitted receipt](FITTED.json)
