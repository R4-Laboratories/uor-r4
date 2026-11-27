# First-period release panels

DEV and HELD each contain 200 pairs at the first period and 100 pairs after B. Each decision has four directive families and two opposed wording variants per family. The fact position order spans all 24 permutations in every family. Both sides keep the same four original source occurrences, question and formatting instruction. A and the first period are supplied. D remains an eligible source after C. The reference includes emitted EOS and the replay cap is five committed tokens.

| Contrast | Stop side | Copy side | Shared committed prefix |
|---|---|---|---|
| 0 | A . EOS | A . B EOS | A . |
| 1 | A . B EOS | A . B C EOS | A . B |

Each first-period split has 50 pairs per family and each post-B split has 25. The post-B families cover before/after, keep/omit, total counts and reversed scope. Both releases include 2 same-word scope variants per split. In those variants each side has the same word multiset and the instruction order changes which part applies. These variants cover 50 first-period pairs and 25 post-B pairs per split. Raw prompt lengths are equal on 213 DEV pairs and 163 HELD pairs. The stop prompt is longer on 62 DEV pairs and 25 HELD pairs. The copy prompt is longer on 25 DEV pairs and 112 HELD pairs. Length remains a potential cue, particularly within wording families.

Fresh source-value IDs are disjoint from the four sealed #4321 splits, the #4320 finite and held inputs screened by #4321, and the other new split. The preparation also checks exact raw prefixes, case and whitespace normalized full prompts, contexts, directive templates, fact templates, query templates, and original source positions. Each factual label occurs once in the entire encoded prompt and before the facts delimiter. The labels participate in all four roles. The protocol-2 tokenizer identity, assistant marker, response mask and full references are checked before a row can seal. Template checks establish separation at this exact panel boundary. Ordinary words, the formatting instruction and semantic cues recur.

| Directive vocabulary | Distinct word types | Word occurrences | Shared with #4321 TRAIN | Shared with DEV |
|---|---:|---:|---:|---:|
| DEV | 77 | 6,096 | 23 types, 3,731 occurrences | |
| HELD | 84 | 6,424 | 29 types, 3,821 occurrences | 37 types, 4,684 occurrences |

The per-split receipts count lexical overlap by prior #4321 split, each shared directive word, common cue counts and role assignments. Words are lowercased ASCII alphanumeric runs. Counts include repeated prompts across facts. The vocabulary census concerns the directive, excluding the shared format instruction and facts.

PROTECT consists of the exact 50 original held-primary ceiling pairs that passed #4321. Extraction preserves every selected row and key line in its original order. The generator tests reject substituted action targets, a missing EOS and an attempted copy of D. The original score instrument remains the authority for model traces.

The panel is sealed on Wick at `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-period-margin/panel-v1`. Two successful preparations produced identical files. HELD is separately manifested and enters model evaluation only in phase two. All splits use the #4321 row and key schemas.

| Manifest | SHA-256 |
|---|---|
| Root | `2d10cc6d64a4197385ca83482f4e93b45a888d3ed10dcf7e18f892d7091c240c` |
| DEV | `20370c2e4b69e5702f304c745822c5a633326280d23cc74dadd1c25dc2cb9069` |
| HELD | `5ad7d39cd377d1f8ac4a45e6c0bb866571f1d3e7dea7df4f011b35b39273624d` |
| PROTECT | `7d1fb00473004baee193f027bcc7a5c3f27ae8966db23873e866312f570b05a9` |

```sh
cargo test --locked --release
cargo build --locked --release
target/release/period-margin-panel prepare "$RUNS" "$NEW_PANEL"
```

The first preparation encountered a held prefix beyond the fixed protocol budget. The revised held fact and question templates bring the sealed prefixes to 129–164 tokens, with room for the current reply cap.
