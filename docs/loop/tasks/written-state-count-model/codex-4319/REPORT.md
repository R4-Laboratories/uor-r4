# The trained choice emitted a period on every held wording

## What did not work

The learned and actual swapped instruction states produced zero complete replies through EOS on 400 held prompts, the same result as native. All 400 learned scores were positive. The held instructions came in eight distinct pairs across four families, repeated on 25 facts per pair. None flipped between the two output actions. Correctly supplying the requested action made 200 bare replies exact. On all 200 period replies that same control produced the correct word and period, then `g` (ID 73) instead of EOS. The observed ceiling for correctly completing **both** sides of a pair with this one-use action was therefore zero. [Registered decision and controls](SHEET.md) · [Independent audit](evidence/held.json)

## What changed

The product can load a single D11 containing the frozen #4315 base, its original reader, the #4318 copy action and the losslessly stored learned ending state. The GPU fitted 4,096 signed i16 weights in one pass on 192 matched TRAIN pairs (six training families); 122 weights were nonzero. Per prompt, the model scores distinct unigram and ordered adjacent-pair addresses formed from the instruction tokens’ four native root coordinates. The state acts once, immediately after a verified copy. A positive score commits a period, a negative score commits EOS. Zero retains the native proposal. The copy and all prior model weights stayed fixed. [Checkpoint identity](evidence/artifact.json) · [Training receipt](evidence/train.json) · [Ordinary client](product/README.md)

## Measurements

Every arm copied the supplied original token correctly on all 400 prompts. Each of 200 matched pairs used identical facts, person, question and source address. Only the final instruction differed. Each reference is either word/EOS or word/period/EOS. Every emission is self-fed. A cap without EOS fails.

| State at token two | Correct token two out of 400 | Pairs with both token-two decisions correct out of 200 | First wrong at token two | First wrong at token three | Exact bare / 200 | Exact period / 200 | Exact pairs / 200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native | 69 | 0 | 331 | 69 | 0 | 0 | 0 |
| Learned from own wording | 200 | 0 | 200 | 200 | 0 | 0 | 0 |
| Learned state from opposite wording | 200 | 0 | 200 | 200 | 0 | 0 | 0 |
| Supplied correct instruction | 400 | 200 | 0 | 200 | 200 | 0 | 0 |
| Supplied opposite instruction | 0 | 0 | 400 | 0 | 0 | 0 | 0 |

The 131 additional correct second tokens in the learned arm all came from period requests. Their first mismatch moved from token two to token three. The other 69 period replies already reached token three in the native arm. All 200 learned bare replies emitted an extra period at token two. There were no native exact replies to lose or retain.

| Held family (50 pairs each) | Bare / period score on one wording pair | Bare / period score on the other | Pairs with learned action flip |
|---|---:|---:|---:|
| Contrastive correction | +608 / +640 | +592 / +592 | 0 |
| Final character | +48 / +688 | +240 / +704 | 0 |
| Reference to a stated rule | +784 / +736 | +544 / +832 | 0 |
| Scoped exception | +528 / +496 | +656 / +480 | 0 |

Changing only the instruction changed the score on seven of the eight distinct wording pairs. The period score exceeded the bare score on four, tied on one, and was lower on three. Moving a single monotone threshold cannot correct all eight. The learned and real swapped arms emitted identical sequences across all 400 held prompts. Both collision words (` lake` and ` anchor`) and the other values showed the same failure. A representative ordinary pair with answer `lake` produced the observable sequence ` lake.g` under both instructions. In the bare case the first wrong ID was period 16 in place of EOS 1. In the period case it was `g` 73 in place of EOS 1. [Wording scores](evidence/wordings.json) · [Panel and decoded outcomes](RESTORE.md)

## Checks run at the head

The RTX 4090 independently reloaded and ran all five arms in reverse input order. The cold packed product reproduced all 2,000 arm trajectories and 5,598 transitions. The finite tests included both signs, a real matched swap, all-zero fallback, an unverified copy and a scorer fixture where word/period/EOS succeeds. Natural third-token failure was preserved in device observations and client replay. A second Rust tabulation reproduced the CSV and summaries byte for byte. [Finite FIT](FIT.md) · [Device receipt](evidence/device.json) · [Cold differential](evidence/qualified-parity.json) · [Matched analysis manifest](evidence/analysis.sha256)

Training took 199 ms in the driver. The two held device traversals took 18.731 seconds. The rental ledger records 282 seconds on the measured 4090 at $1.245/hour (about $0.10 at the quoted rate). An earlier allocation stopped before SSH. Including its 407 seconds at the quoted compute rate gives a combined estimate of $0.24; the ledger does not itemize provider charges for the stopped interval. Both allocations were deleted. [Lease record](evidence/compute.json)

## Skills used

The native-space-cycle skill guided integer state definition, lossless artifact packing and device/client replay. The fit used native integer updates and the ordinary client read the resulting words directly.

## Still open

The native decoder’s post-period decision remains unresolved. On wording, the fixed zero threshold and signed address votes failed every paired decision. The score ordering diagnostic isolates families requiring a change beyond a scalar threshold. The held panel uses explicit final instructions after a question mark and a supplied source position. Source discovery and open-ended chat are outside this measurement. [Next experiment](PICKUP.md)

## Plain summary

The fixed copy delivered the right word every time. Wording changed most numerical scores but none of the actual decisions. Even an ideal decision at token two could finish only the bare reply; the next native step after a period produced an extra letter on every held period request.
