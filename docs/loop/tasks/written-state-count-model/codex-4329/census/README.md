# Donor source census

This is a Rust corpus instrument for frozen D87 TRAIN data on Wick. It joins the corrected #4314 donor-v2 tape to the original #4312 records. It checks each source group ID, raw record hash, D87 TRAIN ordinal and corrected opening address. The instrument uses the canonical tokenizer and verifies the protocol identity, directly re-encodes the current user segment and original answer, and rebuilds the exact prompt with the ordinary protocol-2 tokenizer calls. The matching rows retain absolute UORT token addresses, the exact raw record and turn identity, and byte spans. [Corrected donor receipt](../../codex-4314/donor/RECEIPT.json) · [Historical framed census](../../codex-4315/evidence/donor-admission.json) · [Census receipt](../evidence/census.json)

## Target and candidate definitions

- **Immediate** is the original ID immediately after the assistant marker. **First non-whitespace** is the original ID covering the first non-whitespace byte of the normalized protocol answer. These are the same phase when the ID already covers that byte. A separate after-space phase uses the original standalone space ID 223 as the intervening causal input.
- **Full current user** scans the user-content range delimited by protocol structure. The range includes the leading content space. It excludes role markers and any earlier turn. Role text embedded in user content remains user text. Earlier user and assistant occurrences are recorded separately.
- **Ordinary window** requires a complete initial prompt of at most 192 IDs and applies the discrete client's sliding window at the later phase. The one-turn subset has exact `uor-chat --discrete --say` encoding. Second turns use the original protocol history, including the preceding assistant EOS, as an exact prompt-ID replay. Using the second user alone changes the context.
- **Framed window** applies the existing single-turn native occurrence fence within the same phase and context. It requires BOS, the outer role markers and exactly one two-LF split. Just the initial fact paragraph is eligible. The original reader strips one standalone space at the end of the phase prefix before applying this fence. The separate historical scan reproduces the #4315 receipt on the stored 192-ID immediate tape and checks all saved colon rows against the UORT tokens and response mask.
- **Unique** means one occurrence of an original target ID in a specific admitted span. Classes 0 and 9 are separately counted as nonlexical under the original reader's canonical class table. Other character classes retain their historical treatment. A semantic source address or a confirmed NoRead label requires reading the source and the request.
- **Complete word** requires that the target ID bytes and unique occurrence each cover a complete alphanumeric run in their own normalized text. The boundary test treats letters, digits, underscore, apostrophe and hyphen as word continuations. This deliberately strict byte test can reject valid lexical surfaces with punctuation or combining marks. The finite controls cover duplicates outside the fact fence, nested marker IDs, loss of BOS in a sliding window, and a target that is only a BPE piece.

The private `rows.jsonl` has one line per genuine opening. `group` and `turn` are zero-based. `source_record_sha256` identifies the exact original JSONL line without the terminating LF. `source_id` is the original record identifier and `train_ordinal` is the canonical D87 TRAIN ordinal. Each token offset is an absolute index into the corrected UORT payload after its 64-byte header. Ranges are half-open. `prompt_index` is relative to the whole prompt including BOS, and `user_index` is relative to the current user role's content. Byte ranges address the normalized protocol content after CRLF/CR conversion, trim and one leading space. The line retains the original user and answer strings for adjudication.

## Wick reproduction

Build against the checked-out tokenizer with a locked dependency resolution. Use a fresh private output directory on Wick ZFS. The audit opens no model artifact.

```sh
umask 077
runs=/mnt/plexpool/uor-r4-corpus/runs
root=$runs/20261127-codex-whole-turn-census
unit=docs/loop/tasks/written-state-count-model/codex-4329/census
historical=docs/loop/tasks/written-state-count-model/codex-4315/evidence/donor-admission.json
cargo test --offline --locked --release --manifest-path "$unit/Cargo.toml" --target-dir "$root/target"
cargo build --offline --locked --release --manifest-path "$unit/Cargo.toml" --target-dir "$root/target"
"$root/target/release/audit-whole-turn" \
  "$runs/20261127-codex-marker-question/donor-v2" \
  "$runs/20261127-codex-visible-answer/donor/pair-01" \
  "$runs/20261127-codex-ordinary-source/readback-v1/payload/tokenizer.json" \
  "$runs/20261127-codex-shared-source/product-inputs-v1/token-classes.u8" \
  "$historical" "$NEW_PRIVATE_OUT"
```

The final private readback is under `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-whole-turn-census/census-v2`. The retained first pass under `census-v1` supplied the prompt adjudication. The shared fields across the two passes, including all current-user match addresses and classifications, produced the same canonical JSONL SHA-256 `5638189d605a1071f1eaeb8c4762e14ad6da4d178300b89dd78c6272f5fe923d`. A separate full rerun reproduced the final summary and private rows byte for byte. The original #4315 executable also reproduced its committed receipt byte for byte. A wrong class-table input failed its hash check before output creation. The final audit took 1.271 seconds wall time on Wick's AMD Ryzen 9 9950X. All three finite tests and Clippy passed. [Wick output hashes](../evidence/census-wick.sha256) · [Source hashes](../evidence/census-sources.sha256)
