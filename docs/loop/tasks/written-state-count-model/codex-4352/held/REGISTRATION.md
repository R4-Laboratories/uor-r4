# Held human dialogue supply from complete CoQA passages

## Frozen inputs

Use the Wick copy of CoQA dev v1.0 and the protocol-2 tokenizer, bound by the [#4315 input receipt](../../codex-4315/panel/RECEIPT.json). The paths below are relative to the Wick runs directory.

| Input | SHA-256 |
|---|---|
| `20261127-codex-visible-answer/panel/inputs/coqa-dev-v1.0.json` | `dfa367a9733ce53222918d0231d9b3bedc2b8ee831a2845f62dfc70701f2540a` |
| `20261127-codex-visible-answer/panel/inputs/tokenizer.json` | `a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377` |
| `20261127-codex-visible-answer/panel/inputs/d87-bank-prompt-digests.txt` | `d95090332157731b3d02bace4bd3e3467a3a6ee80e5934a6fed3cf89cb3e1541` |
| `20261127-codex-visible-answer/panel/frozen-v1/judgments.jsonl` | `bea3926ba99a205a2035ae81c8aaa78cc864f6cccf408a071bcccca4553f86aa` |
| `20261127-codex-question-swap/panel/frozen-v2/keys.jsonl` | `866a01878f6c2890a4dbe715a45808df171afe47f7bbc466bf74f6cf578815bd` |
| `20261127-codex-marker-question/panel-v2/train/keys.jsonl` | `91a7f5c743dd5ede11e5ce6876229f5692a3cb82cf770ae7d0882a8634d2165d` |
| `20261127-codex-marker-question/panel-v2/held/keys.jsonl` | `c7e67e6c96f297bd1e17614354f9525325895872b82d872fea6999974310acc5` |
| `20261127-codex-shared-source/panel-v1/human/keys.jsonl` | `fcd87c28afde5a9e1dd9284d31e4630fe28535c8cde475e153b669cfc6f9f231` |

Exclude the union of original CoQA conversation IDs from #4312–#4315, the same source and filename, and the SHA-256 of the complete original passage under the historical prompt normalization. Resolve every previously enrolled CoQA ID against the frozen complete dataset. Verify the known panel sizes and the 210-source union before #4315. The later #4329, #4337 and #4338 units consume CoQA DEV as a denylist and supply other datasets. The repository inventory through the #4350 merge identifies #4312–#4315 as the panels with CoQA rows. Reuses of these frozen panels remain covered by their source IDs. Screen every earlier and current user question, and the full passage plus first question, against the exact normalized original D87 prompt-bank digests and all 40 original normalized D87 continuation templates. Screen source questions against normalized CoQA panel questions. This assesses exact known overlap, with no access to sealed D87 DEV. Unrecorded historical enrollment, near duplicates and exposure in unrelated pretraining are outside the screen.

## Sampling and token identity

Use CoQA source domains `gutenberg`, `cnn` and `wikipedia`, the same custody scope as #4312. Keep the complete source story as provided, followed by two line feeds and the first source question in one user message. Then preserve the recorded role-alternating questions and original assistant answers in order. Use targets 2 through 5, each with one through four earlier consecutive human exchanges and their ordinary EOS closures. Scan target turns in ascending order and take the first admissible target per original conversation and normalized passage. Traverse whole conversations in ascending lexicographic order of SHA-256 of `uor-r4/4352/coqa/source/v1\0` followed by original CoQA ID bytes, breaking ties by original ID. Continue through the complete input to measure supply, and retain at most the first 200 admissible, independently sourced conversations. No passage excerpts, gold containment checks, answer agreement filters or answer-dependent placement. The only target-answer filters are corpus well-formedness, ability to serialize the original human reference and exactly one trailing EOS without embedded special IDs.

Use the production literal-role protocol-2 encoder, including its documented whitespace normalization. Keep only complete prefixes of at most 2,048 original IDs. Compare one-shot prefix and full-document encoding against incremental role closure and re-encoding from the stored originals. Reject empty content, unknown or out-of-range IDs, inner BOS/EOS/UNK, role or turn count discrepancies, malformed canonical reply, and repeated normalized passages. Preserve every refusal in the census. Validate planted prior-ID, passage-digest and D87 overlaps, changed historical EOS, context overflow, malformed turn order and embedded special token controls.

Write sealed private prompt-only and truth streams keyed by unrelated opaque row IDs. Truth retains originals, source attribution, provenance, history, original target response and gold IDs through EOS. Public output gives only input receipts, counts, source/turn summaries, controls and output hashes. This unit prepares a held supply only. Keep the observed shortfall if the strict CoQA source scope supplies fewer than 200. OASST2 is excluded from this registration.
