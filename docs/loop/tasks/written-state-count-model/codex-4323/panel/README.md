# Crossed first-period instruction panel

TRAIN, DEV, and HELD each contain 256 first-period pairs in eight split-specific wording families. Four original factual occurrences, the question, formatting instruction, and supplied prefix `A .` stay fixed within each pair. Side 0 requests one word and has the full reference `A . EOS`. Side 1 requests two words and has the full reference `A . B EOS`. C and D remain eligible source occurrences. Positions point to exact whole-word tokens in the original prompt. The verified factual addresses are the only supplied read inputs.

Every split has 224 opposed-clause pairs and 32 direct unquoted count controls. Half of the opposed pairs hold the selector fixed and exchange the clauses. The other half hold the clause sequence fixed and exchange which clause applies. The two selectors and the two clause orders populate all four cells. Each family repeats the four pair configurations across eight physical fact arrangements, two forms of fact punctuation, and both A-before-B and B-before-A. The unquoted opposed cases and duplicate-word directives are retained as distinct families. The plain controls have no operative-clause ordinal.

| Split | Pairs | Same directive word multiset | Equal encoded prefix lengths | Stop prefix longer | Copy prefix longer | Prefix tokens |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 256 | 176 | 208 | 8 | 40 | 116–146 |
| DEV | 256 | 208 | 224 | 16 | 16 | 131–168 |
| HELD | 256 | 176 | 192 | 16 | 48 | 116–141 |

The Codex agent authored the directive, fact and question templates. The Rust builder deterministically crosses templates with factual values. Each family is repeated across layouts and factual assignments, with wording families held out by split. This is an experiment on generated template instances and makes no claim of externally authored, human-authored or conversation data. The three splits use 24 distinct factual token IDs, eight per split. The builder verifies the #4320, #4321 and #4322 manifests, excludes their factual IDs, and checks exact prompt and template overlap. Split receipts give the remaining lexical overlap. Prefix length and common directive words can still be cues in individual wording families.

The scientific `rows.jsonl` files retain the #4321 schema. In `keys.jsonl`, `pair_axis` records `clause-order-swap`, `selector-swap` or `direct-count-control`. `quoted` and `duplicate_directive_words` identify the surface controls. `operative_clause_ordinal` and `one_word_clause_ordinal` use one-based clause order and are null for direct controls. Keys also include `expected_first_period_action`, `fact_punctuation`, and the zero-based fact roles in physical order. Only TRAIN and DEV enter phase one. HELD remains reserved for phase two.

## Reproduction and verification

The read-only seal is on Wick at `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-scope-trace/panel-v1`. Each split has its own checksum manifest, receipt, rows and keys. [SEALED.sha256](SEALED.sha256) records the root manifest, the split manifests and the preparation identity. `PREPARATION.sha256` records the exact sources and executable relative to the Wick `panel-build-v1` staging root. Two completed preparations were byte-identical before sealing. The first attempt failed the per-role label-coverage check because the assignment reused label positions in some roles. That unsealed attempt was retained on Wick. The final assignment rotates values across all roles.

```sh
cargo test --locked --offline --release
cargo build --locked --offline --release
target/release/scope-trace-panel prepare "$RUNS" "$NEW_PANEL"
(cd "$NEW_PANEL" && sha256sum -c SHA256SUMS)
for split in train dev held; do
  (cd "$NEW_PANEL/$split" && sha256sum -c SHA256SUMS)
done
```
