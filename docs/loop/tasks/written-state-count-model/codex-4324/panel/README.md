# Fresh release diagnostic panel

The 320 requests comprise 64 scoped quartets and 32 direct-count pairs. Each scoped wording family contributes eight contexts. Within a quartet, the facts, physical source addresses, question, formatting, punctuation outside the directive, and dialogue framing stay fixed. The only changed text is within the directive. Its two clauses request one and two words. Crossing which clause appears first with which clause the user selects produces four requests.

| Selected clause | One-word clause first | One-word clause second |
|---|---|---|
| First | A . EOS | A . B EOS |
| Second | A . B EOS | A . EOS |

The four direct wording families contribute eight pairs each. Sixteen pairs contrast one word with two, and sixteen contrast two with three. The latter exercise both exact outcomes at the first release after B. Four original source occurrences remain available in every row. Answer words appear in role order whatever their physical order in the prompt. The builder checks the literal-role dialogue encoding and authenticates each supplied address against the original occurrence's byte offset and token ID. Prompt lengths are 130–178 tokens against a 192-token writer limit and a five-token reply cap.

The scientific rows use the [#4323 wire format](../../codex-4323/panel/README.md) with an additional `row_id` field. In every canonical pair the short reply has side 0 and the long reply has side 1. The number of expected words after the first period is `side + contrast`. A scoped row's `pair_id` holds its context and selected clause ordinal fixed. Its side then varies which clause appears first. The keys join on `row_id` or `(pair_id, side)`, and expose `context_id` for full quartets and `selector_swap_id` for pairs that hold clause order fixed and vary the selection. The one-based clause order and selection also appear separately in `one_word_clause_ordinal` and `operative_clause_ordinal`. Direct controls leave these three selector fields null.

The sealed receipt records the complete census and shared directive vocabulary. All 128 clause-order edges and 96 of 128 selector edges have equal encoded lengths. All four prompts are equal length in 48 of 64 quartets. The agent-authored template instances use eight source token IDs absent from the four verified prior panels. Their 320 unique normalized texts and encoded prompts have zero exact overlap with the 4,868 checked archived rows. The 87 new template strings were checked against 409 distinct normalized archive signatures covering known source-choice, post-period, period-margin and scoped authoring. The authored families retain ordinary lexical and semantic overlap. Each family uses every source label in every role and balances the physical fact order, whether A is before B, and fact punctuation.

## Reproduce and verify

Wick stores the read-only panel under `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-release-factorial/panel-v1`. [SEALED.sha256](SEALED.sha256) contains file identities relative to that panel root. To reproduce from this directory, set `RUNS` to the Wick runs directory and choose an unused output directory on the same filesystem.

```sh
cargo build --locked --offline --release
target/release/release-factorial-panel prepare "$RUNS" "$NEW_PANEL"
(cd "$NEW_PANEL" && sha256sum --strict -c SHA256SUMS)
(cd "$NEW_PANEL/fresh" && sha256sum --strict -c SHA256SUMS)
```

Two independent builds of the panel files produced identical bytes. A structural JSONL check independently recovered the 64 complete quartets, 32 direct pairs, both matched scope axes, and every authenticated source, row identity, and EOS reference. This preparation executes no model.
