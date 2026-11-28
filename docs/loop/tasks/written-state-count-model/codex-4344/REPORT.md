# No matched grounded release supervision in the pinned TRAIN supply

Lab: codex. [Order](ORDER.md)

## What did not work, first

[codex] References #4063. The #4314 corrected donor, #4336 SQuAD TRAIN custody and #4336 DROP TRAIN custody yielded zero matched stop/continue pairs at the same original source boundary within the registered copy and context limits. DROP supplied one original two-source continuation under annotation order. Its corresponding natural stop request was absent. The donor supplied no exact second original-source suffix through EOS. The request-dependent release fit remains unrun for this supply. [Registration](REGISTRATION.md) · [Next unit](PICKUP.md)

## What changed

[codex] The source inventory reconstructs the corrected native dialogue tokens and checks the requested action against original user byte and token addresses. Its source reviews and candidate joins are separate from ordinary model evaluation.

## Measurements

[codex] The three TRAIN sources provide separate denominators for conversational and extractive supervision. Reported addresses are original user positions.


### Corrected D87 donor

Replayed all **4,345 visible openings and 3,763,868 answer IDs** in the fixed 4,060 TRAIN groups against the corrected UORT tape and masks. The exact full protocol-2 opening fit in **2,262/4,345** cases, including **25** with actual earlier assistant history. Of **4,060** supervised terminal EOS targets, **441** fit through EOS and **134** also met the eight-source-ID limit before matching. Of **285** earlier assistant EOS closures, **30** fit through EOS. Their masks make them visible history. This is a replay of the pinned subset, which was selected every 128th ordered TRAIN record starting at ordinal zero. [D87 counts](results/d87.json) · [Corrected donor](../codex-4314/donor/README.md)

| Event within the complete 192-ID opening and observed action | Turns | What the count establishes |
|---|---:|---|
| Exact opening ID overlap of 1–8 copied IDs, before exclusions | 787 | Liberal overlap screen. The first ID also matched in all 787. |
| Exact opening overlap immediately before supervised EOS | 50 | Raw terminal candidate before uniqueness, word-boundary and writer-delimiter checks |
| Unique original whole-content span immediately before supervised EOS | 15 | Mechanically possible exact source stop |
| Unique original span immediately before earlier masked EOS | 1 | Visible closure in causally retained history |
| Any internal content with room for a preceding copy | 2,236 | Upper bound without source matching |
| Unique original span followed by internal content | 265 | Mechanical continuation in 265 turns across 491 prospective ends. Of the ends, 185 first expose a complete alphanumeric character in the next token ID and 306 first expose one in a later ID. |
| Next ID also matches the adjacent original source ID | 186 | Contiguity signal across 253 ends. It does not supply a distinct second source. |
| Exact unique, disjoint second copy through EOS | 0 | No admitted donor source-switch suffix in this configuration |

One of the 306 later observations starts an alphanumeric UTF-8 character in the immediate token and completes it in a later token. The raw field named `unique_internal_content_after_punctuation_or_spacing` counts the token that completes the first alphanumeric character. Its name does not describe every preceding byte.

The same-answer endpoint and whole-user lexical uniqueness were insufficient to establish authority. Two independent reviewers agreed on **7 supported, 7 rejected, and 1 ambiguous** among the 15 terminal candidates. All of the supported rows extract a filename field under two exact instruction wordings. Six of the seven rejects reproduce formatting examples in the request itself. In two of those six, the actual source uses an unpadded numeral and the padded answer agrees only with the example. Original byte slices reproduced all 15 answer surfaces, including the six instructional examples. Agreement of the remaining seven with the source establishes a supervised copy *target*. A model did not emit any tokens in this unit. The donor has no separately annotated interval authority. [Source-review counts](results/source-review.json) · [Checksums and replay](REPLAY.md)

### Human TRAIN

| Custody | Stop denominator | Strict mechanical stops | Continue supply | Matched pairs |
|---|---|---:|---|---:|
| SQuAD #4329 TRAIN tape and #4336 same-start search | 130 published positive extracts encode exact replies through protocol EOS. The tape also has 130 published impossible rows. | 106 | Two through-budget same-start contiguous annotation variants across the 25-title search. Neither supplies a distinct second source. | 0 |
| DROP #4336 official TRAIN member | 24,497 single-span annotations. Of the 276 pinned structural candidates, 264 fit the actual protocol reply through EOS, including any standalone leading space. | 244 | Two of 3,213 two-span annotations met the prior span and budget screen. One exact disjoint continuation survived the full strict recheck through EOS. | 0 |

The SQuAD tape stops lost 21 over-width spans, two one-ID common words and one writer delimiter. The DROP structural starts lost 12 rows when actual reply IDs consumed the remaining window. Among the 264, the strict screen excluded 17 repeated intervals, two partial words and one writer delimiter. The one DROP continuation uses the annotation order from the earlier diagnostic. The task's official grader accepts an unordered set, so the original question does not establish a supervised order by itself. The broader DROP links to one three-span question failed exact copy concatenation and did not fit the two-source action. Published extractive stops carry the task's protocol EOS, while D87 stops carry the visible donor reply's EOS. [Unchanged custody replay](results/human-v1/SCOPE.md) · [Strict recheck](results/human.json) · [Earlier continuation audit](../codex-4336/supply/drop/README.md)

## Checks run at the head

The original registration was public at `b33aac40bf002c7f2d353eb597df0b6ec43fd5fe` before any new row was inspected. An ORDER-only first commit and a second commit with the identical registration tree brought the branch into the loop gate's expected order. The original registration remains archived in the public Git ref `evidence/codex-4344-registration-before-intake`.

Four focused Rust checks on Wick cover original byte offsets after CRLF and Unicode, partial word rejection, repeated and common-word controls, and a finite stop/continue fixture with the bound tokenizer. The fixture reached the same first original address and observed EOS on the stop reply and a disjoint exact second copy through EOS on continuation. The mechanical join produced one pair, and both the shifted original address and duplicate request controls produced zero. The fixture stays inside the instrument tests. The final driver reproduced the same donor aggregate and private packet in two fresh Wick runs. The source grader rejected a reordered verdict file. The unchanged human custody replay matched all original receipts and the private canonical DROP export. The supplementary human audit additionally rejected the actual writer delimiters and checked repeated whole-word surfaces across the full user role. It reproduced its receipt in two fresh runs. The original human replay remains separately sealed. Compile-only retries fixed a lifetime bound, an unused import and a method unavailable in the pinned tokenizer library. These produced no count. [Replay and provenance](REPLAY.md)

The source census used the ordinary protocol encoder, exact original IDs and the eight-ID copy limit. No model, DEV or locked partition was opened. This is a bound on the registered two-copy-through-EOS supervision search in these pinned TRAIN sources. A longer donor reply with ordinary synthesis or a different multi-source format remains outside that admitted action. Generic EOS and internal-content targets exist as unpaired data. Zero paired examples leaves no eligible partition for a request-dependent fit. A generic stop model or a new natural source design would require a separate registration.

## Skills used

[codex] The native-space cycle grounds the capacity and token-identity checks in the ordinary protocol and artifact contract.

## Still open

[codex] Learning when to finish a natural reply remains open. The requested-action fit needs genuine continuations that share the verified first boundary. [Next decision](PICKUP.md)

## Plain summary

[codex] These records contain answers that should stop and many answers that continue. We found no matched example in which changing the request makes the model copy a second distinct source after the same first answer. The seven donor stop examples verified by both source reviewers all extract fields from filenames.
