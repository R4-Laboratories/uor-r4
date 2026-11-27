# Ordinary native occurrence copy

`uor-chat --authenticated-copy` opts one answer into the `uor-r4.native-authenticated-copy/1` action. The existing native occurrence D11 supplies the original reader and writer. The flag is a session choice. Its pending state is consumed on the first raw emission, including a standalone space or a refused copy. The next step appends the actual emitted ID to the ordinary native context. EOS ends the answer immediately, including in raw-ID mode.

The first transition reads the selected occurrence and re-reads its original ID at the same eligible factual position. The writer scores that ID before assembling its normal top-eight ranking and commits it only at the admitted minimum distance. A refused copy commits the native proposal for the same reader selection. The ordinary CLI resolves the address from the learned reader. The verifier supplies the correct and rival positions separately, and can attach an inconsistent, out-of-window or missing claim to a selected read. These supplied positions are instrument inputs.

Build the ordinary client and detector from the workspace root. Restore the existing D11 and tokenizer with the [#4315 instructions](../../codex-4315/RESTORE.md). The tested artifact SHA-256 is `591ea4a21e39942ac86625a8903ff90e2e6055b382966e740787211edcfe73c8`.

```sh
cargo build --release --locked -p uor-r4-integer --features p1-count-serving \
  --bin uor-chat --bin verify-native-copy

uor-chat --discrete "$ARTIFACT" --authenticated-copy --prompt-ids "$PROMPT_CSV" --max-tokens 2
uor-chat --discrete "$ARTIFACT" --authenticated-copy --tokenizer "$TOKENIZER" \
  --protocol 2 --say "$(cat sample.txt)" --max-tokens 2

verify-native-copy "$ARTIFACT" "$SEALED_ROWS" "$GPU_ROWS" > cold.jsonl
```

The CLI outputs `generated_ids` and `authenticated_copy_steps`. Each step gives the selected factual position, claimed position, claim ID, native gate and proposal, exact distances and commit. The two input modes require identical raw prompt IDs for replay. The verifier fails on a differing field, missing or extra device row, or a differing prompt. It compares every arm, including the post-gate integer vector, causal successor position and ID, and the ordinary transition after the actual commit. The device file is optional when capturing an instrument observation.

The [sample text](sample.txt) reproduced the same prompt and events in both CLI modes. The legacy writer emitted EOS `1`. The opt-in writer emitted original `1808` and its native successor `16`, decoded as ` lake.`. A separate finite arm self-fed standalone space `223` and emitted EOS. An inconsistent factual position with the same original ID and zero distance was rejected, while the authentic rival resolved to `3984`. The [CLI receipt](CLI.json) records these finite observations. Full raw records and the [checksum manifest](finite-cold.sha256) are in Wick ZFS at `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-copy-id/product-finite-v1`. The isolated Rust `discrete_d11` suite passed 36 tests, including an authenticated ID that falls outside all eight ranked IDs.

The sealed panel's GPU and cold Rust records matched on all 1,416 prompts, 11,328 arm trajectories and 21,976 writer transitions. The comparison includes EOS and the absence of a successor after first-step EOS. The freshly reopened verifier compared each JSON field. Canonicalizing the device's JSON field order also produced the same byte stream as the two independent Rust passes. Full raw records and the [checksum manifest](qualified-cold.sha256) are in Wick ZFS at `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-copy-id/fitted-v1`. [Exact replay receipt](qualified-exact.json)

The [fresh sample](qualified-sample.txt) was selected from the cold replay to demonstrate a successful collision case. Raw and text modes committed `[1808, 1]` and ended at EOS. The same stock text invocation emitted `[1]` and an empty reply. The valid rival emitted `[3984, 1]`. All eight alternate same-ID occurrences were refused at the same minimum distance. On unsupported prompts the learned reader still authorized 124 of 200 unbound-subject copies and all 400 omitted-value copies. [Fresh CLI receipt](CLI-qualified.json) · [Control observations](qualified-controls.json)
