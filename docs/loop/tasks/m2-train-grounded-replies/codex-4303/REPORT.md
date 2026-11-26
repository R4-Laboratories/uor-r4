# Protected source selection on a rented RTX 4090

Lab codex · 26 November 2026 · [M2 grounded replies](https://github.com/UOR-Foundation/uor-r4/issues/2030)

## What did not work

**All four arms retained the four signed occurrence-switch pairs and the 24 full near pairs. All four scored 0/128 exact signed source pairs on fresh wording.** Each also scored 0/256 individual signed sources and 0/128 complete tuples. A signed source requires the unique native read to recover both the intended governor and the exact original noun occurrence. A pair must pass on both the first and later instructions. The exact writer, D11 and tokenization were fixed. The source gate stayed closed. First wrong emitted token and exact replies through EOS are NOT_RUN. [Fresh scores](evidence/fresh.jsonl) · [Guard identities](evidence/train-and-guards.jsonl)

## What changed

We started from the [protected slot-3 state](../codex-4301/REPORT.md) and fit on 64 new paired instructions from four TRAIN wording families. In each prompt the requested noun had a same-spelling rival and additional candidate sites. A third site excludes both original same-spelling occurrences. For the margin comparison, each candidate uses the same correct signed governor while the other signed link and heading can vary. The margin trained both first/later instructions against the cheapest legally completed third site. The elected edit shifted nine integer coordinates of one excluded-role transport center. In the eligibility arm, 64 original affirmative TRAIN occurrences of “retain” supplied a coordinatewise lower median. The elected edit placed one affirmative center halfway from its nearest existing center to that median. The ordinary signed reader still uses strict nearest-center distance on unlabelled text. The two edits changed disjoint bytes, 14 for transport and 60 for eligibility, in the same 13,336-byte native state. [Fit](fit/README.md) · [Eligibility operator](eligibility/README.md) · [Frozen states](evidence/states.tsv)

The 4090 measured 295 candidate states. Of 232 crossed grids with both factors edited, 120 preserved every guard and 66 also met both registered TRAIN improvement definitions. The optional backoff was unnecessary. On TRAIN the selected margin reduced the sum of each eligible pair’s worse one-unit transport deficit from 1,898 to 1,762. Its strict pinned-third wins stayed at 0/64 pairs and 2/128 rows. Eligibility admitted 64/64 TRAIN “retain” words, up from zero. The stronger unprotected `m47-e4` reached 12/64 pinned-third pairs on TRAIN, but lost all four signed guards and kept 16 of the 24 full near pairs. All four chosen arms retained the identical 28 identities. We froze the election before transferring the fresh packet to the rental. [Pre-registration](PRE-RUN.md) · [Election](evidence/election.json) · [Selected and diagnostic TRAIN records](evidence/train-and-guards.jsonl) · [Held FIT](audit/HELD-FIT.md)

## Measurements

The fresh panel contains 128 first/later pairs, 32 in each of four unseen wording families, across eight new fact sets. Within each family the design crosses “retain” and “accept”, requested sign, and governor order. “Retain” occurs as the affirmative governor on 128 of the 256 prompts. On 64 of those it governs the contested noun. The table distinguishes a word passing the sign gate, a legally complete intended source, a win against the strongest third at the correct governor, and the ordinary native read. [Panel design](panel/README.md) · [Per-arm and per-family counts](evidence/fresh.jsonl)

| Fresh measurement | Slot 3 | Margin | Eligibility | Both | Wrong-sign control |
| --- | ---: | ---: | ---: | ---: | ---: |
| Affirmative “retain” admissions / 128 words | 0 | 0 | 128 | 128 | 0 |
| Legally completable intended signed link / 256 prompts | 128 | 128 | 192 | 192 | 128 |
| Target beats the same-spelling rival locally / 128 pairs | 12 | 12 | 16 | 16 | 12 |
| Target beats strongest third at the correct governor / 128 pairs | 7 | 7 | 7 | 7 | 7 |
| Correctly governed target beats strongest third with any governor / 128 pairs | 0 | 0 | 0 | 0 | 0 |
| Exact native signed source / 128 pairs | 0 | 0 | 0 | 0 | 0 |
| Exact native full tuple / 128 pairs | 0 | 0 | 0 | 0 | 0 |

**Eligibility.** The affirmative and opposing distances for “retain” changed from 84/56 to 42/56. The word itself appeared in TRAIN. The fresh test measures its admission across unseen wording families and facts. Every one of the 64 newly eligible contested targets lost at transport to a third site under the correct governor. The 64 remaining absent contested targets used “bypass” or “ignore”, 32 each. Other annotated governors kept their sign. Across nine other word surfaces, 576 occurrences became affirmative and 192 moved from excluded to a tie. The wrong-sign control admitted zero “retain” words. It added the full TRAIN median to the opposing bank. The elected affirmative edit used the half step, so the control tests the direction of membership and has a different displacement. [Independent lexical and source grades](evidence/independent-fresh.json)

**Transport.** Each primary state won exactly the same 22 individual pinned-third comparisons and seven paired comparisons. All seven pairs used the excluded role in the salvage-tool-invoice family. Margin alone reduced the sum of paired worst deficits from 2,587 to 2,571. Adding margin after eligibility reduced it from 4,467 to 4,403 on that arm’s larger eligible population. Neither margin contrast changed a selected source on any of the 256 prompts. When the competing third site could choose its own governor, it beat all 192 legally complete intended sources in the combined arm. The lexical cost decided 88, transport 102, and the frozen tuple bank two. The combined native read chose a third site under the correct governor 96 times. On the other 160 prompts it chose a different governor, landing on the requested noun 34 times, its same-spelling rival 34 times, and a third site 92 times. [Row analysis](panel/READOUT.md) · [Fresh scores](evidence/fresh.jsonl)

**Original addresses.** On a band-room pair the requested first and later “drum” lay at BPE [65,66) and [70,71). Under the admitted “retain”, their completed transport costs were 98 and 93. “One” at [89,90) cost 40 with the same governor. Once all governors competed, both reads instead used “to” at [90,91) to govern “retain” itself at [91,93). In a salvage pair both intended “hammer” sites beat the pinned third, yet the native reader routed both prompts through “followed” to the later “hammer”. Supplying the correct signed links recovered 128/128 original pairs. Exchanging the contested noun for the legal same-spelling rival made the device follow that rival on 128/128 pairs and recover 0/128 original pairs. [Four original-address traces](evidence/original-addresses.jsonl) · [Controls](evidence/controls.json) · [Independent grades against the original](evidence/controls-against-original.json)

## Checks run at the head

The protected starting state reproduced the historical regression, near and previous fresh GPU captures. Physical finite fixtures demonstrated a reversible transport win and a strict lexical tie, sign admission and wrong-sign rejection. Independent integer replay reproduced 37,837,362 fields across the complete 295-state TRAIN grid and 1,585,749 native, competition and word fields across the five held states. All five ordered artifacts unpacked byte-identically to their frozen inputs. The checks include both native read paths, source-address ties, every guard identity, single-factor byte ownership, and both supplied controls. [Non-author TRAIN FIT](audit/FIT.md) · [Independent checks](audit/README.md)

One rented RTX 4090 ran the fit and TRAIN/guard reads in 19 seconds and held scoring in four seconds, including job overhead. Provider-reported uptime was 1,012 seconds at $1.256/hour, approximately $0.35 at that rate. Wick ZFS verified all 2,196 extracted files of the complete rental-job archive before release. The provider confirmed deletion. The first build needed ripgrep to complete its source scan. The subsequent build and both physical scoring jobs completed successfully. [Compute receipt](evidence/compute.json) · [Cold recovery](RESTORE.md)

## Skills used

The native-space-cycle skill supplied the integer state and train/serve replay checks. The UOR research architect skill supplied the original-address success criterion and the separate pinned- and free-governor comparisons.

## Still open

Keep protected slot 3 as the reference. The measurements identify two sources of false positives. With the governor fixed, a word such as “one” can outrank the argument noun. With the governor free, ordinary words such as “to” can govern a winning link. The next bounded test is a crossed diagnostic on new pairs. Supply only the signed action governor in one arm, only the set of genuine argument noun sites in another, and both restrictions in a third. Keep both same-spelling occurrences and all legitimate rival arguments in the noun set so the read must still resolve first/later from wording. Swap the annotations as controls, measure both signed occurrence and full tuple, and retain the 28 guards. A gain confined to supplied constraints would identify which relation to learn from native text before evaluating the writer. [Source witnesses](panel/READOUT.md)

## Plain summary

The reader learned to admit “retain” on every unseen example. It still pointed to the wrong place in every fresh pair. We can now separate a word’s eligibility from the harder decision of which word it governs and which original noun occurrence should reach the writer.
