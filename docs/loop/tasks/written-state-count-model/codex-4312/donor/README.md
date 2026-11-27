# Paired D87 training inputs

The builder reads the 262 pinned raw D87 shards and reconstructs the original prompt-group split and TRAIN membership with [`d87-split-audit`](../../../../../labs/d87-split-audit/instrument). It checks the full canonical TRAIN UORT digest, extracts every 128th group from index zero, and compares that subsequence byte for byte with the saved R. It encodes the original structured records with [`student-ingest`](../../../../../labs/m2-student-corpus-ingest/instrument) and compares the reconstructed legacy content with R again. Both tokenizer files are pinned and their vocabulary and merge lists must be identical. The builder runs no model.

Both arms retain 4,060 groups, 4,345 assistant turns and 15,955,989 physical tokens. Each has 13,983,847 masked targets, 8,405 BOS tokens and 4,060 EOS tokens. The geometry fitting sample contains the same 259,604 original target events in both arms. It includes 64 events from each of 4,040 groups and every event from each of the 20 shorter groups. The [aggregate receipt](RECEIPT.json) binds all input and output files.

In reasoning-first, each ordinary document contains its original user/assistant framing, reasoning, separator, final answer and terminal EOS. Each reasoning turn contributes an auxiliary singleton BOS with its target mask clear. In answer-first, the visible document contains the original role markers, user prompts, final answers and terminal EOS. After that document, each reasoning turn occupies its own auxiliary document. A BOS opens the auxiliary document, followed by the exact original reasoning IDs and separator. The auxiliary documents have no EOS targets. BOS-aware contexts exclude the detached reasoning and separator from the visible turn and all subsequent visible context. Detachment also removes the user prompt from the reasoning sequence's causal context. The original D87 policy supplies EOS only at donor-group end. The original answer IDs following a reasoning separator carry their original whitespace, so the ablation does not independently test changes to leading-space or mid-group EOS supervision.

Verification reads the persisted masks, tags and index files. It checks that every original target event maps to a token with the same ID, segment and mask in both arms. It derives the expected answer-first visible and auxiliary spans independently from the serialized reasoning-first tape, checks every boundary, and recomputes the geometry selection in canonical target order. It also verifies every private output file's byte length and SHA-256.

## Restore and interfaces

Run from the repository root with an output path that does not exist. Store the outputs under a private Wick directory. The pinned split summary is [in the repository](../../claude-411/split-summary.json).

```sh
unit=docs/loop/tasks/written-state-count-model/codex-4312/donor
cargo build --release --locked --manifest-path "$unit/Cargo.toml"
umask 077
"$unit/target/release/uor-visible-answer-donor" \
  "$RAW_SHARD_ROOT" "$SPLIT_SUMMARY" "$D87_TOKENIZER" "$P4_TOKENIZER" \
  "$FULL_TRAIN_UORT" "$ORIGINAL_R_UORT" "$NEW_PRIVATE_OUT" 8
"$unit/target/release/uor-visible-answer-donor" --verify "$NEW_PRIVATE_OUT"
```

| File for each arm | Contents |
|---|---|
| `*.tokens.u16` | Canonical UORT v1, little-endian token IDs, BOS 0, EOS 1 |
| `*.mask.u8` | One binary byte per payload token for full output-fit supervision |
| `*.segment.u8` | One byte per payload token, with frame 0, prompt 1, reasoning 2 and answer 3 |
| `*.targets.u32` | Physical payload offset for each full supervision event in original event order |
| `*.geometry-mask.u8` | One binary byte per payload token for the common geometry selection |
| `*.geometry-targets.u32` | Physical payload offset for each geometry event in original event order |

Offset files contain zero-based little-endian `u32` values without a header. Order-sensitive matched updates use `*.geometry-targets.u32`. Enumerating either binary mask follows that arm's physical token order. The geometry sample ranks every original mask-selected event by SHA-256 of the byte string `uor-r4/4312/matched-geometry/1`, a zero byte, and the seed `4312`, original TRAIN group ordinal and within-group original mask-selected event ordinal as three consecutive little-endian `u64` values. It retains up to 64 events by least digest, breaking ties by event ordinal, and lists them in original event order.

The private `groups.jsonl` records original TRAIN ordinals, payload offsets and per-arm visible lengths, paired event offsets, and turn counts. Both `groups.jsonl` and `records.jsonl` contain private corpus identity material. Keep those files, the tape and the position maps outside Git. Only the aggregate receipt belongs here. Wick keeps the verified primary pair under `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-visible-answer/donor/pair-01`.
