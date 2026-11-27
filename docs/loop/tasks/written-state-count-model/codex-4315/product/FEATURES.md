# Shared-source occurrence score version 1

The integer state has 512 signed `i16` weights and one `i16` NoRead score. Every word lies in `[-4096, 4096]`. Native forward scoring adds the 16 active weights exactly in `i32`. Each feature ID is `32 * slot + category`. Candidate IDs and positions address the original prompt, its exact identities, and the frozen four-block root table. There is no weight dimension for an original token ID, a person, a reply label, a data family or a fact slot. This version uses no persistent carry change. The original writer receives a selected original ID, and its own existing integer recurrence, four-block read, address admission and vocabulary ranking determine the emission.

## Causal input

The raw protocol-2 IDs have length at most 192. Vocabulary is 4096. The prefix starts `[0,55,2728,28]` and ends `[201,35,560,652,714,28]`, optionally followed by exactly one real emitted space ID `223`. The body between markers contains exactly one `[201,201]`, with nonempty fact and question spans. Reject IDs outside `0..4095`, IDs below two inside the body, and a complete in-body user marker `[201,55,2728,28]` or assistant marker. The facts start at position four and end just before the blank line. The question starts just after the blank line and ends before the suffix. A malformed prompt has no learned source. All eligible factual positions, in ascending original order, enter the score.

Classify the single-ID bytes from the canonical tokenizer `a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377`. Trim ASCII whitespace for class inspection. Specials `0` and `1`, empty and ASCII-whitespace-only tokens are class 0. A trimmed single-byte member of `.,;:!?` has class 9. All other tokens get a class from their first non-whitespace byte and whether they had a leading ASCII whitespace byte. Lowercase continuation and onset are 1 and 2, uppercase are 3 and 4, digit are 5 and 6, other bytes are 7 and 8. UTF-8 bytes outside ASCII fall in the other class. Candidate admission excludes classes 0 and 9. Punctuation remains visible in contextual features. `Upper(p)` is class 4 or class 3 at span start or immediately after ID 223. `WordStart(p)` is class 2, 4, 6 or 8, or any eligible token at span start or immediately after 223.

`F(t)` and `Q(t)` count exact ID appearances within the fact and question spans. Eligible factual positions with `Q(t)>0` offer exact anchor identities; the matched question anchor is the first question position with that ID. Anchor quality is 4 for `Upper` with `F=Q=1`, 3 for other `Upper`, 2 for non-`Upper` with `F=Q=1`, and 1 otherwise. Choose each candidate's anchor by highest quality, then least absolute token distance to candidate, then lowest fact position. A missing anchor has quality 0. `eq(x,y)` counts equal corresponding nibbles in the four-block frozen roots. It is zero if either operand is absent; token ID equality remains separate. All lookups use original causal IDs.

## Ordered slots

| Slot | Category |
|---|---|
| 0 | Candidate class. |
| 1 | Signed token displacement `c-a` clipped to `[-14,14]`, plus 14, or 29 for no anchor. |
| 2 | Signed word displacement clipped to `[-14,14]`, plus 14, or 29 for no anchor. Count `WordStart` on `(min(a,c),max(a,c)]` and sign by `c-a`. |
| 3 | Anchor quality. |
| 4 | `5 * min(F(c),4) + min(Q(c),4)`, with `c` interpreted as the candidate's ID here. |
| 5 | Class directly left in the facts, or zero at the start. |
| 6 | Class directly right in the facts, or zero at the end. |
| 7 | Token distance to the nearest strictly left class-9 position or virtual boundary `fact_start-1`, capped at 15. |
| 8 | Token distance to the nearest strictly right class-9 position or virtual boundary `fact_end`, capped at 15. |
| 9 | `eq(candidate, fact_anchor)`, or 5 for no anchor. |
| 10 | `5 * eq(candidate, prev) + eq(candidate, next)`, using adjacent factual positions. |
| 11 | `5 * eq(prev, query_anchor_prev) + eq(next, query_anchor_next)`, with each neighbor limited to its own span. |
| 12 | `5 * min(F(prev),4) + min(F(next),4)` for adjacent factual positions, with count zero for absent neighbors. |
| 13 | Signed displacement `c-u` clipped to `[-14,14]`, plus 14, to the nearest other factual `Upper` position `u != c`. Ties choose the smaller position. Missing is 29. |
| 14 | `6 * min(abs(word displacement),4) + anchor quality`, or zero for no anchor. |
| 15 | `5 * min(right punctuation or boundary distance,5) + min(candidate class,4)`. |

## Selection and custody

The number of tied top candidate scores and best score are observable. A candidate is selected only when its score uniquely beats every other candidate and strictly exceeds the learned NoRead score. A tie or a score at or below NoRead uses the original marker fallback. Every ordinary read uses prompt-derived candidates. Diagnostic supplied position interventions are separate and check the same eligibility before taking the original ID. The absent and marker diagnostics both use the actual NoRead writer path. They use the marker fallback at colon and the bare trunk at the actual one-space position. All invoke the same original geometric writer. If a writer actually emits ID 223, replay its first lexical event at the resulting prefix plus exactly that ID, provided the window still fits.

The independent replay compares the full ascending candidate provenance, all feature IDs, signed scores, top ties, NoRead, chosen source position, gate ID and emitted ID. Device computes features from raw prompt IDs and frozen roots. The cold Rust artifact decoder uses the same contract and compares exact integer words, including the emitted opening and first lexical event. The FIT binds the signed update rule and dose.

The `uor-r4.lut-discrete-native-occurrence/1` D11 envelope nests exactly the `d782af466c2ebdfba1304158a3f995b50937c434de6ee5a505f8c49cc944f143` marker artifact. It binds its byte length and SHA-256, the fit manifest, and the state length and SHA-256. The state is eight bytes `UOROCC01`, 32 raw bytes of the SHA-256 of this specification, 513 signed little-endian `i16` words with NoRead last, and the 4096 original-ID-indexed byte classes. The canonical tokenizer determines those classes, which carry no parameters from training. The format preserves all fitted words byte for byte. SHA-256 hashes use lower-case hexadecimal strings in the JSON header. Older D11 and marker envelopes retain their exact decoding paths.
