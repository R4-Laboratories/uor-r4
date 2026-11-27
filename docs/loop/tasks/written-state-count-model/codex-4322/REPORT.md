# Balancing the first-period updates recovers two fresh forms of a stop request

## What did not work

The balanced model still missed at least one reply on 150/200 first-period held pairs. It could follow two newly worded instructions across new facts. It never completed both replies when the instructions had the same words and the deciding clause changed places. The later release after word B stayed at 13/100 exact held pairs. [Held scores](evidence/held.json) · [Every wording](evidence/held-wordings.json)

## What changed

The prior baseline reproduced all 24,582 native weight bytes. We varied the first-period learning rule in a two-by-two experiment and verified that the other two release tables stayed byte-identical. Each fit used the same 384 original TRAIN pairs, eight epochs, 768 individual opportunities per epoch and the same paired opportunities. Balancing changed first-period exposure from 192 STOP and 576 COPY to 384 of each per epoch. The separate weaker-side rule updated the member furthest below its own signed target margin. All four fits used the original writer, device scoring image, tokenizer, factual addresses and copy path. [Registered comparison](SHEET.md) · [State and fit receipts](evidence/fits.json)

| First-period fit | TRAIN correct flips / 192 | TRAIN exact pairs / 192 | Fresh DEV correct flips / 200 | Fresh DEV exact pairs / 200 | Post-B DEV exact / 100 | Protected exact / 50 |
|---|---:|---:|---:|---:|---:|---:|
| Reproduced baseline | 60 | 48 | 0 | 0 | 50 | 50 |
| Balanced exposure | 192 | 132 | 75 | 75 | 50 | 50 |
| Weaker-side update | 48 | 0 | 50 | 0 | 0 | 0 |
| Both changes | 192 | 132 | 63 | 63 | 38 | 0 |

A flip requires a negative score for STOP and a positive score for COPY on the two matched instructions. An exact pair requires both self-fed replies to match all original token IDs through emitted EOS. A score at a later stage can be correct on an externally supplied prefix even when generation stops before that stage. With the weaker-side update alone, all 100 protected replies emitted EOS at token 3 instead of the required second factual word. Their unchanged post-B scores still separated all 50 pairs when evaluated at B. On DEV the weaker-side arm correctly stopped all 200 shorter first-period replies. The longer side failed at token 3 on 150 replies and at token 4 on the 50 that made the right first decision. [All splits, errors, signed scores and parity](evidence/phase1.json)

Balanced exposure recovered both output-length directives and one dictation directive on DEV. Those three wording pairs across 75 fact contexts accounted for the entire development gain. It alone retained the original 50 explicit-count pairs and qualified for HELD. The combination still lost every protected pair despite fitting all 192 first-period TRAIN flips. [DEV wording breakdown](evidence/dev-wordings.json) · [Sealed selection](evidence/selection.json)

## Measurements

On the separate held split, facts, question, source positions and opening prefix were identical within each pair. The first decision was whether to stop after A and its period or copy B. The second contrast tested stopping after B or copying C. A protected fit changed only the first-period weights. The next action always used the actual emitted prefix. Positions below are one-based. [Panel design and split checks](panel/README.md)

| Fresh held contrast | Fit | Correct paired flips | Exact shorter replies | Exact longer replies | Both exact through EOS |
|---|---|---:|---:|---:|---:|
| After A and its period, 200 pairs | Baseline | 0 | 0 | 125 | 0 |
| After A and its period, 200 pairs | Balanced | 50 | 113 | 87 | 50 |
| After B, 100 pairs | Baseline | 13 | 38 | 75 | 13 |
| After B, 100 pairs | Balanced | 13 | 38 | 75 | 13 |

On the first-period held test the baseline copied B at token 3 on all 200 shorter replies. The balanced fit recovered 113 of those replies and still copied B on 87. For the longer replies, the baseline made its first error at token 4 on 75 and finished 125 exactly. With balancing, 38 first failed at token 3 by emitting EOS instead of B, 25 first failed there by falling back to native `g` (ID 73), and 50 first failed at token 4 by continuing past B. The remaining 87 were exact. Twenty-five first-period held replies reached the five-token cap without EOS in the balanced arm. Only 25/200 pairs met both signed margins of eight even though 50 chose both correct actions. At the later release the first error remained token 4 on 62 shorter and 25 longer replies in both fits. The native, swapped-instruction and opposite-action controls each completed zero matched held pairs. The correct supplied actions completed all 300. [Committed-ID and score counts](evidence/held.json)

“Leave only the opening word visible” versus “Leave only the first two words visible” gave scores −47 and +15. On one pair the resulting replies were `cat.` and `cat. floor`, each followed by EOS. The other successful wording asked for a total word count of one or two. Each wording passed on 25/25 factual contexts. On “follow the quoted instruction that comes first,” swapping the two quoted orders gave +28 on both sides of one pair and `floor. crown king` twice. The first wrong tokens were `crown` instead of EOS at position 3 and `king` instead of EOS at position 4. At the post-B release the gain over native came entirely from “a total of two words” versus “a total of three words,” 13/13 contexts in both fits. [Four held examples with addresses and token IDs](evidence/held-examples.json)

Each contrast had four fresh wording families and eight distinct matched formulations before punctuation. The 50/200 first-period successes represent two of those formulations. Wording families, source-value IDs and factual prompts were held out, with overlap in ordinary words. Raw prompt lengths also differed on 137/300 held pairs, with the COPY side longer on 112. The same-word reorder failures are an informative control for that possible cue. The scientific comparison supplies A, the first period and four verified factual positions. Source discovery, unconstrained prose and independent sentence formation remain outside this measurement.

## Checks run at the head

The first CUDA build emitted floating-point reciprocal/conversion instructions in an indexing operation. It was rejected before fitting and preserved. Replacing the modulo operation with circular class cursors produced integer-only images. All four finite updates then matched their planted native state, scores and counters. The four registered fits took 257, 229, 229 and 231 ms respectively on one rented RTX 4090. Both replay directions used the frozen original scoring image. The independent pre-run review bound the scientific source and input manifests. [Device finite check](device/README.md) · [FIT](FIT.md)

Phase one replayed 1,468 rows per configuration. Separate held replay covered 600 additional rows for both the baseline and the sole eligible finalist. The cold D11 matched 148,642 device transitions across all five controls and the two phases, including committed token IDs and addresses. The ordinary client received only the prompt and four positions and matched the learned trace on every row, including all 2,511 held transitions for the balanced fit. The checked archive restored and replayed all four checkpoints. [Transfer hashes](evidence/device-output.sha256) · [Restore](RESTORE.md)

## Skills used

The Codex lab used the native-space-cycle skill for the shared integer operators, lossless weights and cold serving replay. No serving mechanism changed.

## Still open

Six of eight first-period wording pairs missed. All 50 first-period same-word scope pairs had equal scores across the two requests even though their requested actions differed. The trace establishes a lost distinction at the score. The feature addresses and their learned weights can distinguish whether the loss occurs during feature construction or scoring. The 13/100 post-B held result also calls for more varied training language. [Next experiment](PICKUP.md)

## Plain summary

Changing training exposure made the existing geometric writer follow two new ways of saying how many words to return. It also made some longer replies stop too early. The next step is to identify which part of an instruction the trained weights can distinguish, then test that distinction across naturally written requests.
