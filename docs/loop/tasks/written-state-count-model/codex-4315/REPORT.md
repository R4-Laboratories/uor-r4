# Shared source weights found the right address in 66 of 200 complete groups

The frozen D11 writer emitted a correct first token on **14 of 800** fresh grounded questions when it received the correct original token ID at the correct address on every question. The learned reader emitted six. No arm answered both questions in even one assignment with the correct first token. The 113-of-200 four-way gate for scoring entire replies through EOS was missed. Whole-reply, general reasoning and browser performance remain unmeasured in this unit. [Registration](SHEET.md) · [Held outcomes](evidence/results.json)

| Controlled endpoint | Individual questions (800) | Both questions with one fact assignment (400) | Both questions with both assignments (200) |
|---|---:|---:|---:|
| Requested source scores strictly above the named rival | 578 | 196 | 96 |
| Learned reader picks the correct position among all factual positions | 379 | 132 | 66 |
| Learned source and first emitted ID both correct | 6 | 0 | 0 |
| Supplied correct source, first emitted ID correct | 14 | 0 | 0 |
| Swapped source, forced missing or frozen marker, first emitted ID correct | 0 each | 0 each | 0 each |

## Where the first token goes wrong

All 800 gold occurrences were eligible. The reader selected 379 correct positions, 199 opposite values and 214 other positions. It abstained eight times. The gold score strictly exceeded the rival on 578 rows, tied twice and lost 220 times. A third occurrence scored strictly above gold on 274 rows. The reader selected the correct address on only six of the 14 rows on which the supplied-correct writer could realize the answer. Of the 379 correct learned reads, the writer changed the ID on 373. [Per-cohort and per-arm counts](evidence/results.json) · [Verified raw output manifest](evidence/returned.sha256)

The supplied-correct writer received the original ID in all 800 cases, and its first admitted vocabulary cell contained that ID in all 800. Its distance ordering emitted the smallest vocabulary ID among the tokens with the same four packed roots. All 786 wrong emissions went to a smaller ID with **the same roots and the same cell**. Seventy-four resulted in EOS at the opening. This identifies the first loss of token identity in the frozen writer under this intervention. A source mistake remains the first discrepancy on the other learned reads. [Post-run writer audit](evidence/writer-identity.json) · [Ordinary writer](../../../../../crates/uor-r4-integer/src/discrete_d11.rs)

For every one of those 786 misses, between 210 and 236 vocabulary IDs had zero distance to the supplied roots. The requested ID ranked between 42nd and 227th in that tie, below all eight output candidates. The 14 correct outputs came from two IDs with unique codes. Across this frozen vocabulary, 4,096 IDs occupy 388 distinct four-root codes. Since equal codes also have equal cells and tie order, at most 388 distinct IDs can win this ordinary greedy geometric ranking for any read vector. This bound applies to the measured checkpoint and scoring path. A different representation or an exact-address action has a different output space. [Code-level census](evidence/writer-identity.json)

In the greenhouse group selected by lowest group ID before examining outcomes, Berta and Abram exchange “mist” and “fork.” Every requested address wins, under both questions and both assignments. The leading space is part of both reference tokens. Emissions below are actual output tokens at the first generated position. [Word report manifest](evidence/word-report.sha256)

| Person asked about | Facts, Berta / Abram | Requested ID and source position | Requested / rival score | Learned and correct-source writers both emit |
|---|---|---|---:|---|
| Berta | mist / fork | ` mist` 2095 at 29 | 6,509 / 1,469 | `+` 13 |
| Abram | mist / fork | ` fork` 3210 at 44 | 6,509 / 3,592 | `h` 74 |
| Berta | fork / mist | ` fork` 3210 at 29 | 6,509 / 1,469 | `h` 74 |
| Abram | fork / mist | ` mist` 2095 at 44 | 6,509 / 3,592 | `+` 13 |

## Scope, absence and transfer

The source weights varied substantially across the four unseen wording families. Correct addresses were 2/200 for archive, 130/200 for courtyard, 99/200 for greenhouse and 148/200 for regatta. The gold beat the opposite value on 140/200 archive questions, illustrating how restricting the competition to two values would overstate performance. Moving the value before the subject gave 236/400 correct addresses. Placing the subject first gave 143/400. The two fact-order halves gave 190/400 and 189/400. These are descriptive strata from fixed families. [Subgroup counts](evidence/subgroups.json)

On genuine negative questions the reader took an unsupported source in 130/200 cases about an unasserted subject and 398/400 cases with the requested value missing. Removing the paragraph break on 32 matched fresh openings made the reader abstain on all 32. All five arms used the fallback under that unsupported scope. Across 29 human CoQA passages, 22 had a unique eligible address, three had ambiguous support and four lacked the exact source. The learned arm found zero of the 22 unique addresses and matched zero of 29 first lexical IDs. Supplied evidence yielded two correct lexical IDs. It also matched both legitimate initial spaces, then missed the following lexical token twice. Grounded use on natural conversation remains open. [Human offset results](evidence/subgroups.json) · [Eligibility and freshness](panel/RECEIPT.json)

## What was trained and verified

The updated state comprises 513 signed integer words, 1,026 bytes. The 512 shared feature weights and one abstention score consume original positions, repeated question anchors, relative word positions and joint frozen root geometry. The 20 batches consumed the entire 118,915-row corrected donor tape, 800 controlled positive rows and 200 missing-subject contrasts. Of 4,345 genuine donor openings, 188 met the single-turn paragraph contract and five supplied a unique lexical copy update, across four distinct IDs. All five matched the independent pre-fit census. Most source supervision came from the constructed contrasts. All 76 controlled held answer IDs were unseen as supervised targets in both contrast fits and at genuine donor openings. They occurred elsewhere in the underlying token corpus. [Input audit](evidence/donor-admission.json) · [Device fit](evidence/fit.json) · [Signed update rule](SHEET.md)

**Correction to the frozen FIT exposure wording.** One distinct held answer ID, 2057, appeared incidentally as input in positive contrast training. It is the target on 12 of the 800 held questions. The pre-run FIT described 12 as though it counted distinct IDs. The sealed field counts held rows. Zero of the 76 held IDs were supervised answer targets in either contrast fit, and the fitted state, split, scores and advance decision are unaffected. [Fresh join of sealed inputs](evidence/incidental-training-input.json)

An RTX 4090 ran the registered fit in 113 ms inside the driver. The all-arm held test, including reverse-order replay and writing the observations, took 3.289 s inside the scorer. From fit process start through the final held receipt the shell measured 7.305 s, including input checks, process loads and computing the parent fallback. The rental ledger recorded 569 s at $0.869/h, approximately **$0.14** of compute at the listed rate. The instance was deleted after verified transfer to Wick ZFS. The deciding fit and scoring ran off the laptop. [Rental](evidence/rental.json) · [Device replay](evidence/replay.json)

The restored 1,202,143,657-byte D11 artifact reproduced all 1,461 device observations, including candidate features, ranks, addresses, abstention, gate IDs and actual emissions. Three prompts were also exercised through separate ordinary raw-ID and text client processes. The lexical positive control emitted original ID 261. The validator rejected a changed feature score and a wrong source position. The saved native delta and previous parent reconstructed the complete artifact byte for byte after download. [Cold product receipt](product/FITTED.json) · [Finite tests](product/FINITE.json) · [Reproduction](RESTORE.md)

**Decision.** The result supports testing preservation of authenticated source identity inside a tied writer cell. Keep the trained reader fixed during that comparison, and score wrong-source and unsupported reads explicitly. Further reader training and accurate abstention remain prerequisites for general grounded use. The registered run advances no model milestone. [Next bounded comparison](PICKUP.md)
