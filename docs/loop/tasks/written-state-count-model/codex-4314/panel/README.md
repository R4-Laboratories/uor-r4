# Marker-and-question fitting panel

The controlled TRAIN and HELD splits each contain 200 groups and 800 assistant openings. Each group has the same two subjects and the same two labels under both value assignments. Each assignment supplies two equal-length ordinary protocol-2 prompts that vary only in the subject named in the question. Swapping the value assignment changes exactly the two fact label IDs. Within every opening, each candidate answer occurs once in the shared facts and nowhere else in the encoded prefix. Across the 400 pairs in each split, the requested subject occupies the first fact in 200 pairs on each question side.

TRAIN uses four wording families about lockers, auditions, notebooks and orchards. HELD uses four different families about studios, trails, ferries and exhibits. Each family supplies 50 groups. A new seed drives the draws. Unordered subject pairs are unique across TRAIN and HELD and excluded if they occurred in the 4313 controlled panel. Every prefix is unique across the new splits and the 4312–4313 encoded panels. The builder also rejects exact normalized user or question matches to the D87 prompt bank.

HELD contains 100 groups whose two label IDs appear as targets in the TRAIN contrast tape, and 100 groups with one seen label and one held-only label. Across the 800 controlled openings, that is 600 seen and 200 held-only target occurrences, with the same count for rival occurrences. Label novelty refers to the **contrast tape**. Among the controlled HELD rows, 758/800 target IDs occur somewhere in the D87 canonical colon stream and 0/800 occur as immediate targets at genuine D87 assistant openings. The union of the TRAIN contrasts and D87 colon stream covers all 800 controlled and all ten natural held target and rival occurrences. The independent exposure audit compared all 118,915 canonical donor colon prefixes to all primary TRAIN and HELD prefixes and found zero exact matches. Canonical donor data share the tokenizer and full protocol alignment with the panels. [Donor exposure receipt](DONOR-EXPOSURE.json)

## Natural passage transfer

Five distinct CoQA dev v1.0 passages contribute two independently answerable questions about each shared excerpt. The questions use original CoQA wording, four annotations agree after normalization, and the two questions have distinct lexical target IDs. A text-only review admitted passages with explicit support for both answers. It preceded fitting and used no model outputs. The builder excludes all 205 earlier CoQA source IDs and the full normalized story text of each. It rejects exact historical prefixes, normalized questions from the 4313 natural panel, and recognized D87 follow-up templates. These first-token observations are descriptive. The natural pairs did not use fact swaps and the two context lengths differ in all five. Source counts and exclusions are in the [panel receipt](RECEIPT.json).

Three pairs start with lexical IDs on both sides. Two pairs use one standalone space at the immediate position and first reach the lexical token at the next step. Their immediate reference and opposite-side immediate reference remain explicit in the colon sheet. A separate four-row tape provides the two later lexical prefixes with the observed separator added and the two step-zero counterpart prefixes unchanged. Only the primary held tape belongs in the fitted post-colon readout. [CoQA dataset and attribution](https://stanfordnlp.github.io/coqa/)

## Custody and reproduction

The definitive private panel on Wick is `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-marker-question/panel-v2`. The [sealed file manifest](SEALED.sha256) has SHA-256 `e2375bc0c445295a4804f83608c7fe4e0963598c8dabea09cac50d08ac01d8fe`. A fresh build reproduced the manifest and every bound file byte for byte. The first build remains in private custody. Its nominal shared label pool included two IDs absent from the actual TRAIN target set, which gave 590 rather than 600 contrast-seen target occurrences. The definitive build filters the HELD shared pool by actual TRAIN exposure and reproduces the original TRAIN tape.

Run the Rust preprocessor with the pinned inputs and an unused directory.

```sh
marker-question-panel TOKENIZER COQA 4312_JUDGMENTS D87_DIGESTS \
  4312_PROMPTS 4313_ROWS 4313_KEYS coqa-review.txt NEW_OUTPUT
marker-question-panel --verify TOKENIZER 4312_JUDGMENTS D87_DIGESTS \
  4312_PROMPTS 4313_ROWS 4313_KEYS SEALED_OUTPUT
marker-question-panel --exposure SEALED_OUTPUT DONOR_COLON_TARGETS
```

Build via `cargo build --locked --manifest-path Cargo.toml` from this directory. For text-only eligibility review, use `--preview` in place of `coqa-review.txt` and redirect stdout to private storage. The [static verifier](src/verify.rs) independently reconstructs the literal prefix from its segments, checks candidate directions against the private fact slots, verifies the two-ID fact swap and checks freshness on persisted rows. [Audit receipt](AUDIT.json)

- `train/rows.jsonl` carries 800 TRAIN contrast openings. `held/rows.jsonl` carries 810 held colon openings. The last six causal IDs are `[201,35,560,652,714,28]`. The primary rows contain no leading answer whitespace after the marker. Labels and identifiers are available to update or evaluate the model. Causal feature construction consumes the prefix alone.
- `target` and `rival` identify the two first lexical IDs, ordered for the current question. `raw_target` and `raw_rival` identify the immediate tokens under ordinary answer rendering. The standalone space is ID 223. `lexical_offset` tells how many whitespace IDs the true lexical step requires. Every controlled opening has offset zero. `held/lexical-steps.jsonl` is for secondary evaluation of the four rows from mixed natural pairs.
- Each split's private `keys.jsonl` carries the original context, question, reference, encoded answer, and source details. Model inputs only require row files. Pair and group IDs support postprocessing. A controlled group contains two pairs, and a natural group contains one pair with assignment index zero.

## Score analysis

```sh
marker-question-panel --summarize SEALED_OUTPUT COLON_TRACE MARKER_TRACE NEW_OUTPUT
```

Pass the two device trace tapes from the same HELD order. The analyzer binds each result to its panel side, checks both candidate identities and confirms the direction of each swap. It produces a numeric `summary.json` and private spreadsheets. `openings.csv` has both arms per row. `controlled-matrix.csv` has both arms and all four assignment/question cells per group. Each group has permanent label A/B columns tied to the assignment-zero question order, so the same label's scores stay on the same columns even after the fact swap. `natural-openings.csv` puts the lexical and immediate positions side by side. A separate private example is drawn by the lowest controlled group ID. The chosen key or readout address, support, question state, selection and writer output accompany each score and rank.

The primary group rate requires strict correct-over-rival scores on both questions under both assignments. The exact paired discordance test uses the two-sided equal-tail binomial test over discordant groups. The analysis reports 95% Clopper–Pearson intervals and a one-sided 95% lower bound for all four correct lexical targets beating both the rival and space. Equal scores stay ties even when the prior yields different ranks. Rank summaries use minimum, the lower and upper middle ranks, and maximum. Mixed-whitespace CoQA comparisons preserve the actual first token and treat later lexical-token scores at the colon as probes.

The offline analyzer passed a synthetic two-arm fixture over the frozen row schema and rejected an altered candidate direction before writing results. The fixture checks aggregation and confidence calculations without supplying model evidence.
