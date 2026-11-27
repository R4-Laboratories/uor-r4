# Independently authored held prompts

Both fits consume the original #4325 TRAIN tape, including all 256 rehearsal rows. The held panel assembles only fresh rows. Keep its rows and keys on Wick until both fits and repeats seal. No fitted state or score is an input to the panel builder.

Two authors contributed four scoped and two direct families apiece. The authors wrote their contrasts independently. The panel author supplied the common facts, question and release format. The independent author's bell directive names slot order so the physical rotation of fact clauses cannot change the requested second entry. The receipt contrast uses “Sealed” and “Unsealed” so the marks are mutually exclusive. A pre-score token budget check rejected the first draft at 189 tokens. The approved common fact and format framing is shorter, with the selector and count contrasts unchanged. Both preparation attempts are retained on Wick.

The 64 scoped contexts cross the governing clause ordinal and the position of the one-entry clause. The facts, original source addresses, question and release format remain fixed across each quartet. Each scoped family contributes eight contexts that rotate the labels across numbered and physical positions. The independent selector axis changes the governing clause while preserving count-clause order. The clause-order axis swaps the count clauses under a fixed selector. The remaining rows cover 64 direct one/two pairs and 64 direct two/three pairs. Every expected reply contains the leading source token, a period, any permitted following source tokens in ordinal order, and EOS. The controlled scorer supplies and authenticates the opening source and period. The learned release is self-fed afterward. Supplying the correct decisions provides a positive control, and the grade requires it to reach EOS with the full reference token sequence.

The held builder reuses the #4325 native tokenizer, token-position checks, label selection and census. It verifies ancestor manifests and excludes exact archived overlaps in normalized prompt, context, wording templates, encoded prefix, source token ID and pair ID. Ordinary lexical tokens can recur. These are new wording families within a fixed scoped release task and share a formatting contract.

## Replay on Wick

The sealed tape directory is `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-paired-release/panel-v1`. Set `RUNS` to its containing runs directory and `NEW_PANEL` to an unused path. Run from this repository directory.

```bash
cargo build --locked --offline --release
target/release/paired-release-held prepare "$RUNS" "$NEW_PANEL"
(cd "$NEW_PANEL" && sha256sum --strict -c SHA256SUMS)
(cd "$NEW_PANEL/held" && sha256sum --strict -c SHA256SUMS)
```
