# Cold integer product replay

The packed artifact uses the `UORLUT01` container and the
`uor-r4.lut-discrete-native-word-ending/1` schema. Its prefix holds the header
and the 8,192 raw little-endian bytes of the 4,096 signed weights. The remainder
is the complete, byte-identical [#4315 occurrence D11](../../codex-4315/RESTORE.md).
The header records SHA-256 digests for the frozen D11, raw learned weights and
fit receipt. The loader verifies the D11 and weight bytes. The recorded fit
digest allows independent verification against the original training receipt.

Build the integer client, packer and verifier from the locked workspace.

```sh
cargo build --locked --release -p uor-r4-integer --features p1-count-serving \
  --bin pack-native-word-ending --bin verify-native-word-ending --bin uor-chat

pack-native-word-ending "$BASE_D11" "$WEIGHTS" artifact.d11 "$FIT_SHA256"
verify-native-word-ending artifact.d11 "$PANEL_ROWS" "$DEVICE_ROWS" > cold.jsonl
```

The verifier cold-loads the artifact and independently scores each prompt and
its matched opposite using their actual instruction tokens. Every arm supplies
the same annotated factual position on the opening step. The replay compares
the exact reader and writer fields, signed scores, intervention sources,
committed IDs, first mismatch and terminal flag. It stops on the first differing
field and rejects incomplete or extra CUDA rows. When it reports a matched row,
all five trajectories matched, including the integer read vectors, recurrence
codes and causal successor positions. The optional device input can be omitted
to capture ordinary Rust output for a finite fixture.

Use `--word-ending` to select the learned ending in ordinary text or raw-ID
mode. Without a supplied address, the artifact's occurrence reader selects the
source. For controlled runs, `--word-ending-source N` supplies the zero-based
factual position for the first transition. Both entry points write the actual
tokens and structured `word_ending` events.

```sh
uor-chat --discrete artifact.d11 --word-ending --word-ending-source "$GOLD_POS" \
  --prompt-ids "$PROMPT_CSV" --max-tokens 3
uor-chat --discrete artifact.d11 --word-ending --word-ending-source "$GOLD_POS" \
  --tokenizer "$TOKENIZER" --protocol 2 --say "$(cat prompt.txt)" --max-tokens 3
```

An authenticated copy consumes the opening action. The ending decision applies
at the next emission, once. A copy refusal or a zero score takes the native
writer. Each subsequent prediction takes the actual preceding token, and an EOS
commit stops the answer immediately.

In the finite Wick fixture, weights 289 and 395 were planted with values −7 and
+9. The 32 rows exercise negative, positive, zero and matched swapped scores.
The supplied copy opened all 160 trajectories with the same exact factual ID
within each row. Raw and text CLI events independently matched the cold replay
for three prompts, including a negative score ending at EOS and a positive
score followed by non-EOS token 73 on the third step. Supplying an invalid
address left the ending unused despite a negative score. The default learned
source path copied an eligible different word on that example. The cold product
and CUDA observations agreed on every field for all 32 finite rows with each
of the planted and GPU-fitted states. [Finite replay receipt](FINITE.json)

The packed qualified state matched the CUDA run on all 400 held prompts, all
2,000 trajectories and all 5,598 writer transitions. The cold verifier also
produced byte-identical output to both CUDA passes after sorting JSON fields
and restoring the reverse pass to panel order.

Both ordinary CLI modes reproduced four held prompts from matched lake and
anchor pairs. The learned scores remained positive for both instructions in
each pair. Both sides of the lake pair emitted `[1808, 16, 73]` (` lake.g`),
and both sides of the anchor pair emitted `[3984, 16, 73]` (` anchor.g`).
The extra period caused the first mismatch on the bare instructions. For the
period instructions, the first mismatch was token three, where the native
writer emitted `73` after consuming the actual period. All four reached the
three-token cap without EOS. [Qualified replay and CLI receipt](QUALIFIED.json)
