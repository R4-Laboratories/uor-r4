# Can a directed neighborhood identify the original answer under reversal?

Lab: codex. [Order](ORDER.md) · [Registered screen](SHEET.md)

## What did not work, first

[codex] The sample supplied **zero** of the required two complete reversal groups across two articles. A complete group needs two distinct original questions in each direction of one factual relation. Of 24 passages mechanically selected for having two questions at each of two answer spans, four survived source review. All four switch between different facts and all come from the same article. The one agreed reversal in the sampled two-question groups concerns a film and the themes it influenced. It has only one question in each direction. The strict four-wording denominator is empty before native tokenization. Its accuracy is undefined. [Selection and adjudication](supply/REGISTRATION.md)

[codex] This result closes the registered sampling scheme for a fit decision. The pretrained D11 weights retain their published capability result from [the prior run](../codex-4341/REPORT.md). No optimizer ran, no GPU was rented, and no new useful-reply, exact-EOS, speed or browser result was measured in this unit.

[codex] Exact-head review found a real implementation error outside the 20 admitted source prompts. The 32-token fallback for a question without a sentence separator cut through the word “cyan”. Both the first v2 reader and the initial verifier treated its suffix as a word. The old candidate received key 2606. The corrected ordinary client recovers whole words from the original text and gives the missing-neighbor key 2560. The initial client, capture and structural report remain pinned for comparison. [Ordinary-client witness](operator/check-window.sh) · [Initial packet](evidence/archive-initial.json)

## What changed

[codex] One opt-in version of bank 10 of the native interval reader computes the relative rotation between the question neighborhood and the source neighborhood. Each uses the nearest complete lexical neighbor on each side, represented by its original token and the existing block-2 root. Missing neighbors get a reserved key. The source-swapped arm recomputes the canonical rotation after reversing the source neighbors. The ordered-token control uses the same four token sites and the same range of 64 nonmissing keys. The arm is serialized with the 4,096 signed 16-bit words. A stored arm determines how those words are used on reopen. The existing format retains its two original arm tags. [Native implementation](../../../../../crates/uor-r4-integer/src/discrete_d11/native_interval.rs) · [Finite tests](algebra/README.md)

## Measurements

[codex] The eight reserved TRAIN articles contained 513 paragraphs and 2,422 eligible official questions. Each question had one listed answer annotation. Source annotator identities were unavailable. The historical comparison includes six previously fitted questions among the 222 sampled, of which one survived review. The prospective split identifies 286 construction questions sharing 78 wording skeletons with the reserved articles. This sample can inform instrument design. Any estimate of generalization requires separate articles. [Reproduction](supply/REPLAY.md)

| Source grouping | Sampled groups | Groups accepted by both reviewers | Interpretation |
|---|---:|---:|---|
| Two wordings at each of two addresses | 24 | 4 | Four changes of requested fact |
| Two wordings at one address, one at the other | 24 | 7 | Seven changes of requested fact |
| One wording at each address | 24 | 16 | Fifteen changes of fact and one reversal of the same fact |
| Two wordings at one address | 3 | 2 | Two equivalent requests |
| Total | 75 | 29 | 73 of 222 questions, across seven articles |

[codex] Two agents independently curated the source without model outputs, using the registered endpoint, equivalence and reversal definitions. They agreed on the across-target classification for all 75 groups. Endpoint labels differed in 27 groups and within-target equivalence labels differed in eight. The deterministic join retains the full selection and both judgments, including exclusions. This is source curation. The native measurements below use a deterministic grader.

[codex] Wick's balanced D11 has a 192-token opening limit, including the protocol markers. The 73 qualified prompts ranged from 86 to 491 tokens. The ordinary client refused 53 for length, including the sole true reversal, at 349 and 345. The initial batch stopped on the third prompt. We retained the raw refusal, preflighted every original prompt and ran the admitted rows unchanged. Of 20 admitted prompts, ten had no candidate matching the exact annotated original byte interval. Eight answer spans exceeded the eight-token candidate cap, one fell outside the fact role and one lacked a permitted lexical edge despite having exact token boundaries. Every refusal remains in the original denominator. The corrected client and grader reproduced the original admission, every captured feature row and the structural grade byte for byte. [Structural totals](evidence/structural.json) · [Structural grader](probe/src/main.rs)

| Source qualification | Groups | All openings admitted | All intended source addresses available under every wording |
|---|---:|---:|---:|
| Four-wording groups | 4 | 2 | 0 |
| Three-wording groups | 7 | 0 | 0 |
| Two-target switches including one strict reversal | 16 | 6 | 1 |
| Single-target paraphrase groups | 2 | 0 | 0 |

[codex] Across the 20 ordinary openings the verifier checked 2,790 source candidates and 20 NoRead choices. Among candidate rows, 919 lacked a query neighbor, 282 lacked a source neighbor and 1,084 lacked one or both. The two legacy output streams matched bit for bit. Independent reconstruction checked the directed features at original token positions, the ordered count, the source swap, and equality of the other fifteen banks across the three experimental arms and original geometry.

[codex] One ordinary two-fact switch supported the complete comparison. Its new bank tied the intended answer to the opposing answer on one opening, so no possible integer weights for that bank alone could satisfy both directions. Reversing the source neighborhood changed one of the two contrasts, which left the tie. The previous geometric and count banks could each satisfy both. Counting exact third-source key ties across both openings gave 11 for previous geometry, four for previous count and 231 for each of the new bank variants. Over all sixteen banks the comparison is jointly feasible. Feasibility says weights exist for these own-versus-mate inequalities. A selected address, a win against all rivals, and an emitted reply require separate measurements.

## Checks run at the head

[codex] The exhaustive [finite oracle](evidence/finite.json) evaluated 65,536 combinations of four endpoint roots. All 64 nonmissing addresses occurred. Reversing the source changed 57,344 addresses and changing the multiplication order changed 26,624. The reverse canonical transport equals the algebraic inverse for only 64 of 256 ordered root pairs. This is why the source-swap control computes the canonical transport afresh.

[codex] On Wick the previous and updated ordinary clients reproduced all 1,056 archived TRAIN feature rows from the two previous fitted states, eight self-fed legacy traces and both old artifact bytes and packing receipts. Three fabricated states packed and reopened in the new schema and used the selected original span in the unmodified writer. Swapping the arm tag or schema independently caused the loader to refuse the altered envelope. The fabricated-state self-fed trace is bounded to three tokens. [Reproduction](operator/README.md)

[codex] The private 4,520,703-byte packet was force-downloaded at its immutable revision. All 105 members verified. Reopening the balanced artifact and regrading the restored raw captures reproduced both the admission report and every structural decision byte for byte. Nine Rust verifier tests also passed. The Wick finite fixtures included two authentic context refusals, an explicit NoRead target and fourteen rejected mutations. [Archive receipt](evidence/archive.json) · [Regrade](archive/regrade.sh)

## Skills used

[codex] The native-space cycle specifies shared integer state and exact replay of the artifact at its ordinary entry point. The source inventory and structural audit apply that contract to the proposed one-bank change.

## Still open

[codex] The model still needs learned source selection and learned release under natural wording, whole-reply evidence through EOS, multi-turn useful conversation and the equal-total-byte longer-context comparison. The [next experiment](PICKUP.md) specifies a bounded source check for the independent writer problem. [Archive and reconstruction](RESTORE.md)

## Plain summary

[codex] Most of the text never reached the reader. For half of the questions it could read, the right answer was missing from its choices. Training different weights cannot select a choice the reader never receives. On the one ordinary pair with both choices, the new feature lost a distinction the previous feature could make. The writer verifiably preserves an address handed to it, but this experiment produced no new conversational checkpoint.
