# Compact V3 held dialogue cohort

FIT found that the archived version-one tape admitted 17 rows with a standalone first space, including five pairs that shared the same first emitted ID. That tape remains preserved in the original Wick directory under SHA-256 `71183364bdbe2eccef140fbe75b5561d68fb0fa70a6a0748b991446512ffc12e` as rejected input. Version two excludes all 660 structurally eligible standalone-space turns before matching. The 512-ID-only pilot supplied 2 pairs. The frozen up-to-2,048-ID selection supplies 64 pairs from 128 source groups and 40 distinct literal final questions. It selects the 2 short pairs first and 62 pairs with at least one longer history. The actual longest history is 1,045 IDs. Both sides of each pair have an identical complete final user question and assistant-start marker. The common compiled suffix spans 13–23 IDs. Their first emitted IDs equal their first content IDs and differ between sides. Each reference has a distinct literal source in its own passage. The same question string can still depend on different antecedents in the preceding conversations. Alternate human-valid replies were not annotated. The single original reference supports exact-token and natural-EOS measurements.

The extraction screen admitted 6,103 turns from 3,131 conversations before matching by full literal question. There were 662 candidates in 215 recurring question buckets. One candidate shared an exact 32-token span with the pinned training stream. Neither prior-selection stream excluded an additional candidate. The 128 selected source IDs, domain/filenames and normalized passage hashes are unique within the corresponding identity columns. The historical source manifest and D87 bank cover exact prior identities and known questions. A 32-token exact screen cannot establish novelty of paraphrased or related prose. CoQA has no worker IDs available for this claim.

The token builder reruns the original `literal_roles_v2` document framing and independently checks each reference ID before EOS, including a genuine standalone space when one exists. It verifies one BOS at the start, the original number of earlier assistant EOS closures, one final reference EOS, and response-mask parity. The target assistant turn is excluded from the prefix. The cohort builder runs no model. D87 DEV-test was left sealed.

## Rebuild on Wick

Use a checkout containing this unit and the #4352 source dependencies. `src/dialogue.rs` and `src/inputs.rs` come from the #4352 held builder. Cargo uses the repository tokenizer and includes the repository's original D87 donor bank. The lockfile pins the standalone package. The public aggregate pins the complete CoQA source, earlier supplemental selection, exact model training and prior-selection token streams, historical preview rosters and tokenizer. Dry runs and replay write to new private directories.

```sh
unit=docs/loop/tasks/end-to-end-integration/codex-4360/cohort
runs=/mnt/plexpool/uor-r4-corpus/runs
previews="$runs/20261128-codex-held-coqa.9eVz1QPo/previews"
scratch=$(mktemp -d "$runs/compact-cohort-replay.XXXXXXXX")
CARGO_TARGET_DIR="$scratch/target" cargo build --release --locked --manifest-path "$unit/Cargo.toml"
"$scratch/target/release/held-coqa-dialogue" "$runs" "$previews" "$scratch/selected"
"$scratch/target/release/held-coqa-dialogue" "$runs" "$previews" "$scratch/pilot" --512-only
(cd "$scratch/selected" && sha256sum -c "$OLDPWD/$unit/evidence/private-members.sha256")
(cd "$scratch/pilot" && sha256sum -c "$OLDPWD/$unit/evidence/pilot-private-members.sha256")
```

Pinned CoQA and the prior supplement live under `$runs/20261128-codex-lexical-transfer/input/coqa-train-v1.0.json` and `$runs/20261128-codex-lexical-transfer/supplement-1/truth.jsonl`. The historical previews are the retained #4352 Wick replay input. The old D87 source and preview paths are in the aggregate and `src/inputs.rs`. The train and tune streams live under `$runs/20261127-antigravity-gate0-d87-proto2/corpus/`. The earlier selection chat stream is `$runs/../inputs/chat-v0-p2/heldout.u16`.

The frozen private directory is `$runs/20261128-codex-compact-carry-preflight/cohort/v2/final-2048`. `pairs.jsonl` has one numeric pair ID per line and two complete prefix/reference ID streams. `selected-index.jsonl` contains the original source conversation/filename/digest keys and turn indices. Both files stay private. The byte-identical replay verified the file digests and directory/file permissions 0700/0600. The version-two pilot is the separate private `v2/pilot-512` directory. Its pair and index checksums are in `evidence/pilot-private-members.sha256`. Independent readback verified zero first-ID disagreements with the first content ID, zero pairs sharing a first ID, and 128 unique values in each of the three source identity columns.

The frozen cohort contains 84,843 prefill IDs across 128 histories. A linear estimate from the one-run Wick Arm 2 anchor of 3.565 seconds per 2,048 IDs gives approximately 148 seconds of prefill for one pass, or 443 seconds if all three arms repeat it. These estimates exclude continuation, diagnostics and process overhead.
