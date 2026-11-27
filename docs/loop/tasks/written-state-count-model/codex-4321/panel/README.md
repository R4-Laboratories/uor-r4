# Paired source-choice panel

Each user prompt supplies four original source occurrences A, B, C and D. A and the first period are supplied at replay. Both sides of each pair have identical facts, question, original IDs and raw source positions. The wording changes after the question mark. D is an explicitly identified fourth word on both sides, is never required by the reference, and remains available after a verified C. The panel tests release and copy continuation with supplied source identities.

| Contrast | Side 0 | Side 1 | Matched decision |
|---|---|---|---|
| 1, primary | A . B EOS | A . B C EOS | After B, EOS or verified C |
| 0, secondary | A . EOS | A . B EOS | After the period, EOS or verified B |

The row format is `{pair_id, side, family, contrast, prefix, positions, ids, expected}`. Positions are zero-based in the canonical protocol-2 token prefix. Each value is a single leading-space piece before the fact delimiter and occurs exactly once in the full prefix. Preparation checks raw occurrence offsets in decoded source text, the complete response mask and raw reference IDs through EOS. The prefixes use the pinned 4,096-token vocabulary, period 16 and EOS 1. They are 132–162 tokens across the four splits.

| Split | Pairs | Wording families | Equal raw lengths | Stop wording longer | Continue wording longer |
|---|---:|---|---:|---:|---:|
| TRAIN | 384 | orders, inclusion, quota, movement | 288 | 72 | 24 |
| Finite | 16 | finite-controls | 12 | 4 | 0 |
| Held primary | 200 | transcript, editor, handoff, ceiling | 150 | 25 | 25 |
| Held secondary | 200 | printing, line, deadline, remaining | 150 | 25 | 25 |

TRAIN has 48 pairs per family at each contrast. Each held family has 50 pairs, balanced across the two directive variants and the two final-punctuation conditions. The factual value order varies, with 24 factual permutations in each TRAIN and held family. Finite fixtures use a smaller set of orders. Each split has eight distinct value IDs with no overlap between splits or with any of the original factual IDs screened from the pinned #4320 panel. Role assignment rotates within each split. The deterministic preparation screens exact raw prefixes and normalized prompts, contexts, directive templates, fact templates and query templates against the pinned #4320 finite and held splits and across the new splits. Family, source words, and fact and query templates are held out by split. The panels share the format instruction and ordinary vocabulary. They contain a deliberately bounded set of word orders and directive templates, and exact-template comparisons give no guarantee about broad-corpus lexical exposure.

The [directive definitions](src/design.rs) contain both side wordings for each variant. The primary handoff variant with “stays on the page” and “goes out” uses a publication metaphor to distinguish retained source text from emitted text. That phrasing is more ambiguous than the other families and covers 25 of the 200 primary pairs. Family results retain that subset.


The private panel is sealed on Wick at `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-source-choice/panel-v1`. The SHA-256 of the [root manifest](SEALED.sha256) is `2b6abf25514e9a744a447659388003994ea5a0d1e92c50dd7934337da0968b7c`. Two independent preparations on Wick produced identical rows, keys, receipts and manifests. Each split stores its own nested digest manifest.

## Independent scoring

The scorer joins every serialized observation to its full sealed row and the actual other side. It checks authenticated A, the supplied period, fresh native proposals at equal committed prefixes, stage changes and request identity for verified copies. It independently derives the first mismatch and checks observed EOS and the five-token cap. A conditional flip at the common prefix requires a negative score on the stop side and a positive score on the continue side. Full exact pairs require each self-fed trajectory to match every reference ID through emitted EOS. The native arm retains supplied A and the first period and uses the fresh native proposal for subsequent tokens. The report counts selected EOS, native EOS and open replies; the private rollouts include raw and decoded outputs.

The Wick finite checks used three planted states and the measured GPU traces. On each state the correct oracle completed all 32 rows and the opposite oracle completed zero exact rows. The constant zero and copy learned arms had zero exact rows and 32 cap openings. The constant stop learned arm had eight exact rows and no cap opening. All three states had zero directed conditional flips. Three Rust unit tests also exercised a conditional flip with an early self-fed error, exact/counterfactual source actions, EOS provenance and a wrong open reply at the cap.

```sh
cargo build --locked --release
cargo test --locked --release
target/release/source-choice-panel prepare "$RUNS" "$NEW_PANEL"
target/release/source-choice-panel score "$TOKENIZER" "$SEALED_PANEL" finite "$OBSERVATIONS" "$NEW_ANALYSIS"
```
