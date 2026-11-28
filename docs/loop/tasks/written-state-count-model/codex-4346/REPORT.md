# A 512-token opening restores 34 original gold spans

Lab: codex. [Order](ORDER.md) · [Registration](REGISTRATION.md)

## What did not work

[codex] References #4063. The only true reversal still has **0/2 openings with both annotated addresses available**. The larger diagnostic limit admits both full openings. One target is 12 tokens wide and remains outside the eight-token copy cap in both directions. This cohort still supplies zero complete four-wording strict reversals. The result does not reopen the directed fit. [Original source qualification](../codex-4342/REPORT.md)

[codex] The first instrument attempt failed its synthetic equal-suffix check because the two names produced different source token widths. Two subsequent intake checks exposed redundant legacy arm rows in the corrected archive and different wording-family labels within a shared target group. The final instrument checks the two original arm captures plus the five corrected ones, and joins grouped records by original source bytes. The original truth retains the wording-family identities. None of the failed attempts produced a grade.

[codex] The retained precursor branch briefly published two public-dataset TRAIN question IDs in its aggregate. The current aggregate omits the IDs and the complete private audit is byte-identical. No protected holdout was opened. [Custody and checksums](REPLAY.md)

## What changed

[codex] The bounded census and the ordinary interval feature path share the same pre-arm Rust span iterator. The tokenizer, complete untouched opening, fact and question rules, original coordinates, lexical starts and ends, copy stops and eight-token cap are common. The census stops before key construction. It does not load an artifact. The 192-token ordinary limit, runtime and writer are outside this change. [Shared Rust iterator](../../../../../crates/uor-r4-integer/src/discrete_d11/native_interval.rs) · [Replay and custody](REPLAY.md)

## Measurements

The fixed TRAIN roster contains **73 openings in 29 groups**, ranging from 86 to 491 full protocol tokens. All 73 fit the diagnostic limit of 512, including the 53 rejected by the balanced 192-token artifact. Gold reachability is exact original byte identity, permitting only the adjacent ASCII space that BPE may attach to a source token. A named rival is the other annotated answer address in the same group. There are four questions in one-target groups.

| Structural measure | Admitted at 192 | Additional openings admitted at 512 | All admitted at 512 |
|---|---:|---:|---:|
| Openings | 20/73 | 53/53 | 73/73 |
| Exact gold span available | 10/20 | 34/53 | 44/73 |
| Gold span and at least one distinct source rival available | 10/20 | 34/53 | 44/73 |
| Named rival available where one was specified | 10/20 | 34/49 | 44/69 |
| Gold span and named rival both available | 2/20 | 23/49 | 25/69 |
| Enumerated source spans | 2,790 | 21,851 | 24,641 |
| Source spans per opening, min–max | 62–194 | 170–598 | 62–598 |
| NoRead choices | 20 | 53 | 73 |

All 73 admitted openings have a distinct source rival under the structural definition. At 512, **29/73** gold spans are still missing. Twenty-three are 9–30 tokens wide, one is outside the fact role, and five have exact token edges but fail the candidate edge rules. Among the 53 additional openings, **15** fail the copy width and **4** fail the candidate edge rules. The aggregate records the full over-width histogram. Candidate counts quantify the number of spans a reader would receive. They do not measure runtime or memory cost. [Aggregate](evidence/aggregate.json)

| Groups | Denominator | All openings fit, 192 → 512 | All own spans available, 192 → 512 | Both named spans available under every wording, 192 → 512 |
|---|---:|---:|---:|---:|
| Four-wording switches | 4 | 2 → 4 | 0 → 1 | 0 → 1 |
| Three-wording switches | 7 | 0 → 7 | 0 → 5 | 0 → 5 |
| Two-wording switches including the true reversal | 16 | 6 → 16 | 1 → 3 | 1 → 3 |
| Single-target paraphrases | 2 | 0 → 2 | 0 → 1 | Inapplicable |
| Total | 29 | 8 → 29 | 1 → 10 | 1 → 9 of 27 |

The two reversal questions each offer 340 source spans and one NoRead choice at 512. The coordinates below are half-open. Byte offsets are within the original passage. Token offsets are absolute within the full protocol-2 opening.

| Opening length | Gold annotation | Named opposing annotation |
|---|---|---|
| 349 tokens | Original bytes `[4,35)` map to tokens `[5,17)`. All 12 IDs would be required. The span is excluded. | Original bytes `[110,120)` map to tokens `[48,53)`. The five-ID span is available. |
| 345 tokens | Original bytes `[110,120)` map to tokens `[48,53)`. The five-ID span is available. | Original bytes `[4,35)` map to tokens `[5,17)`. The 12-ID span is excluded. |

## Checks run at the head

[codex] The Rust census reproduced all **73/73** archived ordinary lengths and admission decisions. Its original role ranges and ordered source candidates matched **140/140** archived rows covering **100** distinct case/arm pairs. The registered finite pair used 400 full protocol IDs in each opening. Its final 192 IDs matched exactly and the original source at token interval `[4,5)` differed as ID 439 versus 376. Two separately verified restores of the same private #4342 revision reproduced the full audit and published aggregate byte for byte. All four targeted native interval finite checks passed with Rust 1.97.1, exercising directed and ordered features and the original lexical boundaries. [Replay](REPLAY.md) · [SHA-256 receipts](evidence/OUTPUTS.sha256)

## Skills used

[codex] The native-space cycle guided reuse of the original token addresses and native interval operator. Raw token IDs and complete ordered spans are the replay units.

## Still open

[codex] First wrong token, ordinary client output at 512, source selection, natural EOS, and equal-total-deployed-byte geometry versus ordered count are **NOT_RUN**. The registered work used TRAIN data only and no model, fit or GPU. [Next experiment](PICKUP.md)

## Plain summary

The longer opening brings 53 complete questions into the span reader and makes 34 of their exact answers available. In the only true reversal, one answer still needs 12 tokens and the candidate limit is eight.
