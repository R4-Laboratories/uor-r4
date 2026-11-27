# Natural continuation panel

The builder selects one recorded assistant reply per OASST2 tree and one continuation per Wikipedia article. It takes the pinned P4 tokenizer, the admitted raw sources, and a list of D87 bank opening digests. Run it before fitting. The panel, generation logs and decoded grades contain corpus text or token IDs and stay in the private run root.

| Input | Identity |
|---|---|
| P4 tokenizer | Qualified Wick `tokenizer.json`, SHA-256 `a9fb13f7…` |
| OASST2 | `2023-11-05_oasst2_ready.trees.jsonl.gz`, revision `179dd21fc55192153d94adb0e0ce8f69e222bf75`, Apache-2.0 |
| English Wikipedia | `20231101.en`, shard 00020, revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, CC BY-SA 3.0 and GFDL |
| D87 bank metadata | Sorted unique 16-hex opening digests extracted from original IDs in both prompt banks, without fetching either D87 development split |

The pinned raw files are under `results/claude/real-text-corpus-2026-11-05/raw/` in the project store. [Source admission and attribution](../../../../../roadmap/items/real-text-corpus.md)

The tree scan keeps ready, reviewed, human-authored English messages and follows alternating roles. It searches branches in rank order, with message ID as the tie-break, up to the first eligible full assistant reply. A prompt has 2–3 actual user turns in the resulting panel, at least one earlier recorded assistant turn and its EOS, and an open assistant marker. The literal-role v2 encoder renders the original messages, and the complete document must have exactly that token prefix. The gold suffix includes the final assistant EOS. The selection does not truncate prompts or replies. Short replies have 8–48 tokens including EOS, and long replies have 49–128. Teacher forcing scores all selected targets. Free generation stops at its first emitted EOS or at 128 tokens.

The builder omits all OASST2 tree IDs in the historical held-out SHA bucket. It groups remaining rows by normalized opening before sampling fit and held. A strict row has no user-turn digest matching a first prompt in either D87 bank. The supplemental rows have a matching opening and omit openings assigned to D87 DEV-tune and DEV-test via the published split hash. Later user turns in both groups must also differ from the fixed D87 follow-up templates. These checks exclude known record and opening overlap as stated. Historical training exposure and approximate or verbatim reply overlap with D87 remain unmeasured. [D87 split rules](../../claude-411/REPORT.md)

The prose scan starts at zero-based shard row 16,384, after the prior admitted build's first 8,402 considered rows in that shard, and ends before row 32,768. It keeps one unmarked excerpt per article, with 128 prompt tokens after BOS and 64 immediately following target tokens. There is no EOS inserted at the boundary. Prose generation stops at its first emitted EOS or at 64 tokens. Each row includes at least one target token ID absent from the starting prefix. The teacher novelty mask uses the growing original prefix, including preceding gold targets, and excludes EOS from novelty.

Use the exact `bank.jsonl.zst` and `bank-a.jsonl.zst` from the D87 `inputs/` directory. Derive only their existing opaque ID digests. The builder verifies the resulting list and all other input checksums before it scans sources.

```sh
zstdcat bank.jsonl.zst bank-a.jsonl.zst |
  jq -r '.id | capture("^[^-]+-(?<digest>[0-9a-f]{16})(-t2)?$") | .digest' |
  LC_ALL=C sort -u > d87-bank-prompt-digests.txt

cargo run --release --locked --bin s4-source-panel -- TOKENIZER OASST2_GZ WIKI_00020_PARQUET BANK_DIGESTS NEW_OUTPUT_DIR
cargo run --release --locked --bin s4-source-panel -- grade TOKENIZER HELD_JSONL GENERATION_JSONL NEW_GRADE_JSON
```

`fit.jsonl` and `held.jsonl` carry original token-ID prefix and gold arrays, source IDs and positions, split, domain, stop cap, D87-opening exposure and per-target novelty. The builder orders rows within each split and length band by exposure then by a fixed SHA-256 rank. The first eight short and 16 long held chats and the first 24 held articles have `free_run` set. In the frozen draw all 48 free-generation rows are strict. `receipt.json` binds the exact output and gives row counts and exclusions.

The independent grade confirms that every arm generated each marked prompt through EOS or the fixed cap. It requires a trace for every emission, checks the full active context length at each step, and verifies that the disabled and frozen arms emit identical tokens. It records exact replies through EOS and exact prose continuations through cap separately. It reports one-based first wrong and first different-from-frozen token numbers, including the expected and emitted original token IDs. Premature EOS is a conversation EOS before the gold EOS position or any EOS before completion of the 64-token prose span. Conversation output reaching the cap without EOS counts as censored. The grader verifies each admitted source ID against its actual position in the model's causal context. Admission and active mass transfer are counted separately, and admitted sources are distinguished by whether the source came from the original prompt or from the emitted reply. Exact matching measures agreement with one recorded continuation.
