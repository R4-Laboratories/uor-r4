# Canonical D87 answer-first donor

The #4312 tape moved already encoded final-answer IDs from behind the D87 reasoning separator to behind the assistant marker. The #4313 boundary audit found that none of those 4,345 immediate IDs matched the ordinary protocol-2 opening for the same text. The repair rebuilds the original structured D87 TRAIN records with the deployed tokenizer's ordinary protocol-2 dialogue encoder. This unit runs offline CPU corpus preparation on frozen donor data.

The build verifies all 262 raw D87 shards against the original split, reconstructs all 519,580 TRAIN memberships in order, and takes every 128th group from ordinal zero. It checks this selection against both the full canonical TRAIN UORT and the byte-for-byte historical reconstruction of R. The rebuild also checks that all extracted JSONL records match the sealed #4312 source rows, in order. It directly uses the #4312 input pin and extraction implementation.

Each visible group starts at BOS and takes all users and original final answers through the literal-role protocol-2 renderer. The renderer normalizes CRLF and CR to LF, trims outer whitespace, and encodes each content span with its leading delimiter space. The complete marker is `[201,35,560,652,714,28]`. Protocol 2 inserts EOS after each completed assistant reply. At the 285 earlier assistant closures in two-turn groups, that ID is visible causal context with response mask 0. The historical D87 terminal EOS stays supervised with mask 1. The first user's marker follows BOS directly. Each detached reasoning turn retains the historical BOS, reasoning IDs and separator, target masks, and lack of EOS. Each detached document starts its own causal history.

The verifier reads the persisted data in a separate invocation. It reconstructs each exact prefix with `encode_open_history` and `encode_user_prefix`, the methods used by the ordinary conversation path. It independently normalizes and encodes every original answer and compares all IDs, decoded source bytes, span offsets, immediate and first non-whitespace IDs, and target masks. It compares every detached token and mask against the sealed #4312 answer-first tape. An independent full scan of the saved tape checks the entire colon target set, all windows of up to 192 IDs and every original target ID. For all 4,345 genuine openings it additionally compares each persisted JSONL prefix with the last up to 192 IDs of the independently assembled serving prefix. [Tokenizer protocol](../../../../../../crates/uor-r4-tokenizer/src/dialogue.rs) · [Conversation path](../../../../../../crates/uor-r4-integer/src/generation/conversation.rs)

The structural opening flag derives from source turns. The fitting inputs are `prefix` and `target`. The fitting implementation decides marker admission from the supplied causal IDs. The private offsets and source opening flag support readback and auditing.

The [verified receipt](RECEIPT.json) counts 4,060 groups, 4,345 full prefixes, 4,345 full answer spans and 3,763,868 answer IDs. It records 118,915 eligible colon targets, including 4,345 genuine openings, 28,671 other visible targets and 85,899 detached reasoning targets. Among the genuine openings, 2,196 immediately target standalone space ID 223. That space is followed by the separate first non-whitespace target. The other 2,149 openings have the same immediate and first non-whitespace offset. Every eligible complete marker belonged to a genuine opening.

The provisional `donor-v1` is retained for diagnosis. Its short JSONL windows omitted BOS. `donor-v2` includes it whenever it falls within the 192-ID device window, matching the prior fit and the held panel. As a negative control, the verifier refused a private copy with BOS removed from one genuine row at tape offset 1,635. The v2 source rebuild took 14.39 seconds on Wick's AMD Ryzen 9 9950X. Independent readback took 1.02 seconds. [Code SHA-256](CODE.sha256)

## Private output formats

| File | Contents |
| --- | --- |
| `answer-first.tokens.u16` | UORT v1, 64-byte header and little-endian original IDs |
| `answer-first.mask.u8` | One binary response-mask byte per payload token |
| `groups.jsonl` | Group order, original TRAIN ordinal, payload start, visible end and payload end |
| `openings.jsonl` | One source turn per row. Full prefix is `[prefix_start, answer_start)` and full answer is `[answer_start, answer_end)` in the same UORT payload. Includes immediate and first non-whitespace IDs and the absolute first non-whitespace offset `lexical_offset` and its answer-relative offset `lexical_answer_offset`. |
| `colon-targets.jsonl` | One row per mask-1 target whose immediately preceding ID is 28. Fields are `prefix`, `target`, `offset`, and `genuine_assistant`. The prefix contains the last at most 192 original IDs starting at the current BOS when it fits in the window, and ends in the colon ID. Rows follow increasing physical offset. |
| `source.json` | Pins on the raw input selection, old receipt and deployed tokenizer protocol |
| `receipt.json` | Independent readback summary and SHA-256 for every data output |

Offsets exclude the 64-byte UORT header. Span starts are inclusive and ends are exclusive. Genuine openings with a true standalone space target keep original ID 223 in the colon rows. Their lexical IDs can be read separately from `openings.jsonl`. The first non-whitespace token can include whitespace attached to the subword. A non-whitespace token may be punctuation.

## Reproduce on Wick

Run from the repository root. Use a new private output directory for each build. The verified primary data are under `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-marker-question/donor-v2`.

```sh
unit=docs/loop/tasks/written-state-count-model/codex-4314/donor
cargo build --offline --release --locked --manifest-path "$unit/Cargo.toml"
umask 077
stored=/mnt/plexpool/uor-r4-corpus/runs
prior=$stored/20261127-codex-visible-answer/donor/pair-01
probe=/home/callard/uor-codex/20261118-diagnostics/inputs/c420
tokenizer=$stored/20261126-codex-s4-output/portable/readback/panel/tokenizer.json
train_r=/home/callard/uor-codex/20261119-rival-admission-29/count/cr-v1/r.tokens.u16
exe=$unit/target/release/uor-canonical-donor
"$exe" build /mnt/plexpool/uor-r4-corpus/m2-donor-corpus-2026-11-16 \
  docs/loop/tasks/written-state-count-model/claude-411/split-summary.json \
  "$probe/tokenizer.json" "$tokenizer" "$probe/train.tokens.u16" "$train_r" \
  "$prior" "$NEW_PRIVATE_OUT" 4
"$exe" verify "$tokenizer" "$prior" "$NEW_PRIVATE_OUT" > "$PRIVATE_READBACK_JSON"
```
