# First-token question-swap panel

The controlled panel has 200 independently drawn groups, 50 per wording template. Each group has two assignments of the same two words to the same two subjects. The second assignment exchanges the words. Each assignment has two openings with the same fact text. The question changes only the subject name. Both openings have the same token count, the same source text in original token IDs, and the same ordinary protocol-2 assistant suffix. This gives 400 question pairs and 800 openings. The first fact answers side 0 in 200 pairs and side 1 in 200. Each answer ID appears once in each prompt, in its assigned fact. Trials that share a group should be analyzed together.

The literal replies are drawn from tokenizer-compatible English words. For both answers in every pair, the original first token contains the answer word and its attached protocol space. These sequences therefore give a lexical boundary target at writer step zero. The full marker suffix is `[201,35,560,652,714,28]` and the standalone ASCII space ID is `223`. The Rust builder uses the repository's literal-roles-v2 `DialogueEncoder` for both the ordinary opening and full dialogue. It independently checks the fixed suffix, shared context IDs, complete response mask and the removal of the open prefix from the closed dialogue.

The descriptive transfer cohort has five pairs from distinct CoQA dev v1.0 passages. The builder excludes all 200 passages selected in codex-4312, including any alternate source ID with the same normalized passage. It also checks exact D87 prompt digests and normalized follow-up templates. Four human annotations must agree on a short answer that occurs inside the original annotated evidence, and one complete shared excerpt must contain the evidence for both questions. Both original replies must begin with different lexical IDs at step zero. Five candidate pairs had a standalone space before at least one lexical answer and were excluded. Fourteen candidate pairs from 12 passages survived the programmatic screens. A text-only passage review accepted five with independently answerable questions and explicit support in the excerpt. Earlier dialogue is omitted. The natural openings preserve original casing, wording and answer style. One natural pair has equal-length openings. Evidence-conditioned excerpt admission, exact overlap checks, and five reviewed passages limit transfer interpretation. [CoQA v1.0 dataset, attribution and reuse information](https://stanfordnlp.github.io/coqa/)

## Reproduction

Run the builder against the pinned files from codex-4312 and an unused output directory.

```sh
cargo run --locked --manifest-path docs/loop/tasks/written-state-count-model/codex-4313/panel/Cargo.toml -- \
  TOKENIZER COQA PREVIOUS_JUDGMENTS D87_DIGESTS \
  docs/loop/tasks/written-state-count-model/codex-4313/panel/coqa-review.txt NEW_OUTPUT
```

For a controlled-only build, pass `TOKENIZER NEW_OUTPUT`. For text-only natural eligibility review, pass `--preview` in place of the review file and redirect stdout to a private file.

- `rows.jsonl` contains `pair_id`, `group_id`, `side` (0 or 1), `cohort`, the original `prefix`, the first lexical ID `target`, boundary ID `raw_target`, `lexical_offset`, `leading_whitespace`, `space_token`, and `equal_length`. Scoring can obtain the alternate ID from the other row in the pair. All frozen offsets are zero.
- `keys.jsonl` keeps the ordinary user text, question, reply, reply IDs before EOS, fact assignments or source provenance, and evidence boundaries. The panel manifests and [receipt](RECEIPT.json) bind the private row and key tapes. The review file contains five opaque pair IDs.
- Wick retains the sealed output at `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-question-swap/panel/frozen-v2`. The manifest SHA-256 is `51cd6366ad937d81328f7a06c6c14eb26b4a85465ac0af5d69cd09ecb5a5da2d`. A fresh build reproduced the row tape, keys and receipt byte for byte. The shorter controlled prefixes span 47–73 IDs and the longest natural opening spans 182 IDs, below the 192-ID window.
- The initial host-only candidate remains in private custody. It had eight controlled assignments whose template boilerplate repeated an answer ID. The final builder requires unique original-ID support in the common fact text and admitted replacement groups before inference.
