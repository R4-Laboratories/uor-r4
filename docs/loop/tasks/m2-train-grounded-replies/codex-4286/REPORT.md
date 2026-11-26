# Which information locates the governed noun?

## What did not work

With the two signed verb addresses alone, choosing the closest proposed source tuple failed on all 256 conversations. Each winner had a smaller summed BPE gap to the verbs than the authored nouns. On 140 prompts the reader actually selected a governing verb as one of its two nouns. For example, one theatre exclusion required the original `score / script / script` triple. Proximity selected `is / reject / is`. The strict pair score for the native D11 was also zero. Knowing the two original noun occurrences recovered 16 near pairs and zero distant pairs. [Source trace](evidence/source-trace.json)

## What changed

The study tests source information on the same Wick RTX 4090 and pooled integer D11. It compares native choice, signed verb positions alone, an unordered pair of noun positions alone, and two supplied governor-to-noun links in correct or exchanged order. A link read joins the exact original BPE span of each signed governor to an unsigned edge, then admits the noun at the other end. No weights were fitted. The same D11, ordinary client, copy writer and retry policy serve every comparison. [Registered gate](PRE-RUN.md) · [Independent review](FIT.md)

Each of 128 matched pairs exchanges the two governor words while preserving the other user bytes and token IDs. The 256 new prompts cross four families, two distances, whether the governor precedes the noun, first fact requested, and the order of the two same-spelling occurrences. A differently spelled second noun must follow both equal spellings for the fixed proposer to retain both the real and neutral rival. Horizon varies with the fact set. The exact neutral competitor and its exchanged orientation occurred once on every prompt. The 256 second turns were distinct and had zero exact-text overlap against the previous 512 training, 256 ordinary and 256 reserve cases. [Panel construction and source identities](panel/README.md) · [Enrollment](evidence/enrollment.json)

## Measurements

The source score requires the unique correct original byte spans and absolute BPE spans for the affirmed noun, excluded noun and later reference, in that order. A pair requires both the TAKE and DEFER member. The local control gaps were 1–4 intervening BPE tokens. The distant gaps were 10–19. [Device and integer oracle](device/README.md) · [All scores, ties and eligibility counts](evidence/source-gate.json)

| Source input | Exact source triples / 256 | Near source pairs / 64 | Distant source pairs / 64 | All source pairs / 128 | Abstentions / 256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native | 24 | 0 | 0 | 0 | 44 |
| Governor sites only | 0 | 0 | 0 | 0 | 0 |
| Unordered nouns only | 116 | 16 | 0 | 16 | 36 |
| Correct links | 256 | 64 | 64 | 128 | 0 |
| Exchanged links | 0 | 0 | 0 | 0 | 0 |

The full proposer offered 52–384 tuples per prompt. Supplying the two correct nouns left both orientations, sharing a single later reference. Resolving either set of supplied links left exactly one tuple per prompt. The link controls therefore establish that the device can use the supplied attachment. They supply its orientation directly. The 16 nouns-only wins split into eight theatre, four ceremony and four kitchen pairs. Across the full pool, native chose the neutral same-spelling occurrence on 24 prompts and the governor-only reader on 32. [Full arm and cell counts](evidence/source-gate.json)

### What reached the ordinary reply

Every fresh second turn contains the genuine 13-ID first assistant reply. The ordinary client autonomously selects scope in the native arm. The four intervention arms replay the physical source winner, with the requested first role and ending supplied on every arm. The target for a grounded request is the literal requested reply from the original fact addresses. The exclusion target is the separate signed stock response to that very prompt. An unresolved read can match stock and counts as a visible match. Explicit exclusions and authenticated copies have separate counts. [Ordinary-client evidence](evidence/ordinary.json) · [Conditional review](CLIENT-FIT.md)

| Source input | First wrong at token 1 / 256 | First wrong at token 8 / 256 | Exact replies through EOS / 256 | Both exact through EOS / 128 | Correct prefixes at the cap / 256 | Replies retried / 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 64 | 64 | 94 | 10 | 34 | 160 |
| Governor sites only | 52 | 52 | 120 | 36 | 32 | 144 |
| Unordered nouns only | 60 | 60 | 94 | 12 | 42 | 192 |
| Correct links | 0 | 0 | 204 | 76 | 52 | 128 |
| Exchanged links | 128 | 128 | 0 | 0 | 0 | 128 |

Correct links removed the native first-token difference on 64 replies and the eighth-token difference on another 64. None appeared elsewhere. Native and exchanged positions that disagreed at token 1 chose between ` You` (617) and ` Your` (2997). At token 8 the wrong branch exchanged ` You` with ` I` (340) or EOS (1). The first attempt and final returned reply have identical first differences on all 1,280 intervention runs. One theatre request makes the distinction concrete. It asked to open with the user's home, then alternate with job. Native returned ` Your` at token 1 instead of ` You`. The proximity read picked the wrong `is / reject / is` source tuple yet happened to give the expected reply. The correct link picked the original `script / score / script` and gave that reply from the correct binding. [Independent raw token audit](evidence/token-audit.json) · [Original words and addresses](evidence/source-trace.json)

The correct links grounded all 128 affirmative replies from the original job or home facts through EOS. All 128 exclusions explicitly deferred and reproduced the same-prompt stock text and retry trace, including 76 EOS and 52 exact 48-token prefixes at the cap. Those 52 cap stops set the observed ceiling at 204 replies and 76 complete pairs through EOS on this panel. Each distance accounted for 38 of those pairs. All 256 exchanged links made the opposite admission decision. The three partially assisted or native rates should be read against this control. Native achieved 48 authenticated affirmative replies through EOS and 60 explicitly deferred stock matches. Governor-only achieved 68 and 84 respectively. Unordered nouns achieved 36 and 80. All 44 native unresolved cases and 36 nouns-only no-winner calls entered the signed stock path without emitting a bound-writer token.

## Checks run at the head

The device matched the independent Rust oracle on 220,320 point-to-bank costs, 276,080 proposal-cost and eligibility comparisons, and 1,280 arm reductions. Each arm saw 55,216 native proposals. Repeating device execution and permuting the supplied noun and edge lists reproduced the exact reported addresses. The finite wrong-address control replaced the end address of one governor while keeping both signed queries and noun targets fixed. It yielded zero eligible tuples on all three finite cases. Swapping the signed keys with both edges fixed reproduced the exchanged-target arm byte for byte. The measured source collection took 2.69 seconds wall time under the Wick GPU lease. [Device receipt](evidence/device-parity.json)

There were zero selected-address, token, or source-binding disagreements across the 1,280 ordinary intervention runs. A second grader read raw integer emissions from the client records and agreed for each case on the exact first and returned wrong tokens, returned EOS, correct prefix at the cap and retry. Capture took 350.14 seconds wall time including the 256 stock comparators and per-case client launches. The separate physical SM89 replay verified 17,160 original writer steps and 528 authenticated third releases. It accounted separately for 1,200 source-bound turns, 80 unresolved turns that emitted no bound token, and all 1,280 first-history turns. Writer verification took 6.76 seconds wall time. These timings include the instrumentation. [Writer replay](evidence/writer.json) · [Restoration and raw identities](RESTORE.md)

## Skills used

The native-space cycle guided the typed original-address and bit-identical device comparisons. The research-architect skill guided the frozen artifact, opposed transfer and ordinary-client controls.

## Still open

The missing learned operation is selecting which original noun occurrence a governing verb acts on. Governor proximity selected a strictly closer wrong tuple on all 256 prompts. Removing distractors by supplying the two nouns helped at short range, while their native orientation still failed every complete pair at long range. The next comparison should present the original competing noun occurrences and use a geometric edge score to select their attachment from the user's text. Hold out whole wording families and keep the neutral copy of the same noun. Grade exact edges before the ordinary writer, using the supplied and exchanged links here as controls. Locating the governor without signed assistance and selecting the remaining action fields are separate open requirements for [M2](https://github.com/UOR-Foundation/uor-r4/issues/2030).

## Plain summary

The same writer copied every requested fact correctly when it received the right verb-to-noun link. Reversing only that link made every reply diverge. Giving the reader the verb positions alone sent it to closer, incorrect words. The experiment isolates an original-address relation that can now be a training target.
