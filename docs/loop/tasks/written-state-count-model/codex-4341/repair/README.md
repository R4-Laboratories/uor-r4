# #4340 validation instrument repair

The repair adds every previously unreviewed validation passage in 1973 oil crisis (10), Harvard University (13) and Islamism (23). The original 67 rows, order, pair IDs, both signed review ledgers and all original qualifications remain fixed. The 113-row roster contains 23 reciprocal source pairs and 90 source/NoRead pairs across the same 18 titles. Each repaired title remains exposed validation. The locked partitions are out of scope.

## Register the identities

On Wick the only input to the selector is the restored #4340 `validation-selection-v1` directory. It reads the original extraction seal and two authenticated selection metadata files. It does not require a copy of the official dataset or #4338 semantic roster.

```bash
bash "$REPAIR/select.sh" "$PRIOR_4340_EXTRACTION" "$FROZEN_SELECTION"
```

Record the two resulting digests in the separate registration's committed JSON, as `additional_identities_sha256` and `expanded_identities_sha256`. The selector uses the original passage rank and pair ID definitions. The additional identities are exactly the out-of-quota candidates in the three registered titles. The expanded metadata is the unchanged 67 followed by those 46. Cold replay from only the same metadata inputs must be byte-identical.

## Materialize after registration

Once the 46- and 113-row digests are committed in the separate registration, use the committed JSON as `REGISTRATION`. Pass the #4338 `roster/sealed` directory and its pinned author-published DEV JSON. The `PRIOR_VALIDATION_CODE` path is the #4340 `validation` directory. Any locked rows in the mixed parent roster stay in custody. Only validation records matching the 46 frozen passage IDs enter the new source projection.

```bash
bash "$REPAIR/materialize.sh" "$PRIOR_4340_EXTRACTION" "$FROZEN_SELECTION" \
  "$REGISTRATION" "$SEALED_4338" "$OFFICIAL_DEV" "$PRIOR_VALIDATION_CODE" "$EXTRACTION"
```

The original #4340 extractor checks verbatim questions, the client-visible source window, published polarity and candidates, UTF-8 coordinates, positive annotated span consensus, prompt bytes, original token positions, and canonical reply IDs. The wrapper also checks the full original paragraph SHA-256 from author-published bytes in custody. Each source row carries the exact ordinary prompt, case/published IDs, original paragraph hash and index, original and window coordinates, supplied annotation, and native opening labels. The reviewer packet omits token arrays and user token bounds. Provide each of two independent source reviewers only `selected-semantic/additional-rows.jsonl` and the frozen #4340 semantic rubric. Retain both authenticated reviewer returns and their independent reviewer attributions, including reasons, disagreements and ambiguous rows. Each new return uses the original `consensus.jq` decision schema and accounts for all 46 additional IDs. The 67 old reviews come unchanged from the original #4340 packet.

## Qualify

The registered shared qualifier provides a registered 113-row invocation to the existing qualifier. The 67-row invocation retains its original checks and outcome schema. The optional mode checks the exact order and metadata and keeps the existing semantic rules, ordinary label-blind input capture, rebind, candidate, and title-floor logic.

Create private `REPAIR_PINS` with these SHA-256 fields using the recorded custody and reviewer receipts:

| Field | Pinned bytes |
|---|---|
| `prior_packet_members` | Original #4340 archive `MEMBERS.sha256` from its preservation receipt |
| `extraction` | Expanded `EXTRACTION.sha256` |
| `additional_review_a` | Independently returned first 46-row review |
| `additional_review_b` | Independently returned second 46-row review |

The original ordinary client, binder, base and tokenizer pins are embedded from #4340. `PRIOR_4340_PACKET` is its restored archive root. Use the committed #4340 validation code at the #4341 registration revision for `PRIOR_VALIDATION_CODE`.

```bash
bash "$REPAIR/qualify.sh" "$EXTRACTION" "$PRIOR_4340_PACKET" "$NEW_REVIEW_A" \
  "$NEW_REVIEW_B" "$REPAIR_PINS" "$PRIOR_VALIDATION_CODE" "$CLIENT" "$BINDER" \
  "$BASE" "$TOKENIZER" "$QUALIFICATION"
```

The wrapper requires the original 67-row source and reviewer packet prefix byte for byte. It authenticates the old review/qualification files, appends the two matched 46-row reviews independently, and verifies the original semantic, native and final decisions byte for byte against the archived qualifier. `coverage.json` records both denominators, per-title mechanical, selected, semantic and native-qualified counts, and the answerability negative groups. The detailed paired ledgers preserve all rejection reasons. The original floors govern `coverage_pass`. With any floor still deficient, declare exhaustion of the present registered validation supply for that title. A pass moves to the separately registered frozen-weight study. Both outcomes have `model_fit: "NOT_RUN"`.

## Cold replay and errors

Run the stages again in distinct, fresh directories. Reopen the same sealed prior files and reviewer returns and pass the independent replay inputs through subsequent phases. For each phase the replay takes the first output and fresh replay output, followed by the original phase arguments except the output argument.

```bash
bash "$REPAIR/replay.sh" select "$FROZEN_SELECTION" "$REPLAY_SELECTION" "$PRIOR_4340_EXTRACTION"
bash "$REPAIR/replay.sh" materialize "$EXTRACTION" "$REPLAY_EXTRACTION" "$PRIOR_4340_EXTRACTION" \
  "$REPLAY_SELECTION" "$REGISTRATION" "$SEALED_4338" "$OFFICIAL_DEV" "$PRIOR_VALIDATION_CODE"
bash "$REPAIR/replay.sh" qualify "$QUALIFICATION" "$REPLAY_QUALIFICATION" "$REPLAY_EXTRACTION" \
  "$PRIOR_4340_PACKET" "$NEW_REVIEW_A" "$NEW_REVIEW_B" "$REPAIR_PINS" "$PRIOR_VALIDATION_CODE" \
  "$CLIENT" "$BINDER" "$BASE" "$TOKENIZER"
```

A nonzero stage or replay exit is an integrity, extraction, or instrumentation failure. Preserve its private output and log. Retry only in a fresh directory using the registered identities. A completed qualification with `coverage_pass:false` is a measured coverage shortfall. The substantive roster cannot change during either recovery.

Synthetic-only checks can run before any registration, using `bash "$REPAIR/check-synthetic.sh" "$REPO" "$FRESH_TEST_DIRECTORY"`. They cover the exact roster counts, deterministic selection, legacy consensus path, two-review agreement and disagreement, absent/corrupted IDs, out-of-split rejection, original-coordinate drift and denominator reporting. They do not open the real dataset, run the native client, fit model weights, or score model output.
