# Content links separate conflicting instructions on five fresh wording pairs

## What did not work

The candidate still missed one or both replies on **184/224 scoped held pairs**. It also lost every one of the balanced baseline's 75 prior first-period DEV successes and all 32 direct-count successes on the new DEV set. The 50 protected explicit-count pairs all passed. That narrow protection set failed to detect the loss on other previously working wording. Keep the balanced model as the integration baseline. [Prior DEV](evidence/prior-dev.json) · [Fresh DEV](evidence/dev.json) · [Held](evidence/held.json)

## What changed

The first mismatch in the original scope contrast occurs before any decision by the writer. All 50 previously tied pairs supplied different causal token sequences but visited exactly the same set of original single-root and adjacent-root addresses. With a signed sum that counts each address once, the two scores must agree for any weights at those addresses. For example, reversing the two quoted alternatives under “use the former” gave −1 on both requests. The shorter target was `queen.` followed by EOS, and the longer target was `queen. fish` followed by EOS. The trace recovered both −1 scores from the original signed weight words. Each side had zero exclusive original addresses.

| Source of equal scores | Consumed scope / 50 | Fresh TRAIN / 256 | Fresh DEV / 256 |
|---|---:|---:|---:|
| Identical causal token sequences | 0 | 0 | 0 |
| Distinct sequences, identical original addresses | 50 | 96 | 96 |
| Distinct addresses, every differing weight zero | 0 | 64 | 32 |
| Distinct addresses, nonzero differences cancel | 0 | 32 | 0 |

The other 64 TRAIN and 128 DEV pairs had unequal scores. On TRAIN, 96/112 clause-order reversals collided, and none of the 112 selector reversals did. On DEV, the same failure appeared on 64/112 clause reversals and 32/112 selector reversals. All 32 direct-count TRAIN controls had nonzero terms that cancelled. For example, the STOP side of “Output just the opening word” contributed +3 at address 628 and −3 at address 2829 relative to its rival. Raising all STOP scores or shifting the decision threshold cannot separate equal sums. [Consumed trace](evidence/trace-consumed.json) · [TRAIN trace](evidence/trace-train.json) · [DEV trace](evidence/trace-dev.json)

The registered TRAIN-only decision selected a new content-link table. It binds both orders of native roots at distances 2–16 tokens inside the actual instruction. Each link includes its distance and is counted once. The positional control instead keys individual and adjacent roots by their distance from the right boundary. Both added paths gave the two requests distinct feature addresses on every measured pair in the consumed, TRAIN and DEV traces. This establishes that either representation can distinguish these strings. Following the requested instruction requires training and independent reply scoring.

Both fits used the same 256 matched TRAIN pairs, eight passes, signed unit updates and 4,096 trainable i16 words. The position-only fit serialized 32,774 state bytes. The content-link fit serialized 40,966 bytes including an unused zero positional row. The original 24,582 bytes were compared against the balanced checkpoint. The linked scan visits up to 15 earlier tokens per token in each direction. Equal trainable dimensions do not imply equal scoring cost. [Registered fit and branch](SHEET.md) · [Finite instrument and signed numerical controls](device/README.md)

## Measurements

| Fit | TRAIN scoped flips / 224 | TRAIN scoped exact pairs / 224 | Fresh DEV scoped flips / 224 | Fresh DEV scoped exact pairs / 224 | DEV direct exact / 32 | Protected count exact / 50 |
|---|---:|---:|---:|---:|---:|---:|
| Balanced baseline | 0 | 0 | 0 | 0 | 32 | 50 |
| Position only | 144 | 80 | 0 | 0 | 32 | 50 |
| Content links | 224 | 160 | 24 | 8 | 0 | 50 |

The fresh scoped panel contains 28 distinct paired formulations from seven families, each repeated over eight fact layouts. The 32 direct controls use one additional paired formulation. Link training changed 1,686 of 4,096 signed coordinates. The positional update changed 483. The linked model followed both sides of 24 scoped DEV instructions at the first period. Just eight pairs remained exact through EOS, all repetitions of one selector reversal in the veto family. Sixteen pairs in the unquoted replacement family had the correct first action and copied another word after B where the longer reply needed EOS. Those after-B scores were +1. The two later tables were frozen. Across 56 family and fact-layout groups, none of the three arms chose the correct direction on all four crossed comparisons. The eight rows in a group span separate matched fact contexts. [Scoped and direct DEV scores](evidence/dev.json)

The selection rule admitted the content-link arm to the blind test because it improved scoped DEV flips, exact pairs and distinct passing formulations while preserving 50/50 original protected pairs. The fresh panel as a whole **lost** 24 exact pairs, from 32/256 to 8/256, because all 32 direct controls failed. On the earlier #4322 development panel the first-period arm also lost all 75/200 exact pairs. Post-B exact remained 50/100 for the content-link and balanced arms. [Selection receipt](evidence/selection.json) · [Earlier wording regressions](evidence/prior-dev.json)

| Separate held panel | Baseline flips | Link flips | Baseline exact pairs | Link exact pairs |
|---|---:|---:|---:|---:|
| Clause order reverses, 112 pairs | 0 | 24 | 0 | 16 |
| Which clause governs reverses, 112 pairs | 0 | 32 | 0 | 24 |
| Direct count, 32 pairs | 0 | 0 | 0 | 0 |

Forty previously failing scoped pairs became exact through emitted EOS, with no lost scoped exact pairs. These 40 cover five of 28 distinct paired formulations, across three of seven scoped families. All 16 pairs that made the right first-period flip but missed the complete answer erred later. Across all 448 scoped held replies, 192 first failed at token 3, 32 at token 4, and 224 completed exactly. At token 3 the candidate copied B on 112 shorter requests that needed EOS and emitted EOS on 80 longer requests that needed B. At token 4 the remaining 32 errors copied C instead of emitting EOS after B. The baseline had 192 shorter and 32 longer errors at token 3, and 64 longer errors at token 4. It finished 160 individual replies exactly. On the 32 direct pairs both models got just one side right, with the candidate switching which side. The ordinary native writer and the opposite-action control completed zero held pairs, the correct supplied actions completed 256/256, and the actual swapped wording completed eight matched pairs for the candidate. [Held breakdown](evidence/held.json) · [Individual switches and traces](evidence/held-examples.json)

For a held prompt that moved “use … for this request” between two instructions, baseline scores +4/+11 chose COPY twice. The learned content links gave −23/+2, producing `train.` then EOS or `train. bridge` then EOS. For “final memo … superseding” versus “initial memo … superseded by,” the candidate also correctly split the first decision, with −30/+94. On the longer reply it reached `flower. train`, then the frozen post-B head scored +9 and copied `bridge` instead of EOS at token 4. A later diagnostic should distinguish these downstream misses from cases that never reached B. [Matched original IDs and native scores](evidence/held-examples.json)

Each split contains its own eight single-token factual words, 224 scoped pairs and 32 direct pairs. These are Codex-authored, templated requests. Factual order, punctuation, governing clause and requested count were crossed, with 112 reversals on each scoped axis. Repetition across eight factual layouts per wording measures consistency. It provides five successful held wording pairs, not forty independent phrasings. Neither source discovery nor opening generation was tested. [Panel provenance, controls and splits](panel/README.md)

## Checks run at the head

All 40 exact held gains occurred among the 160 scoped pairs with equal tokenized prompt lengths, with 40 correct first-period flips on those pairs. All 40 also occurred among the 176 scoped pairs whose directive words had identical multisets after lowercasing and splitting on non-alphanumerics, with 48 correct flips on that subset. The baseline completed zero exact pairs in both subsets. Prompt length alone cannot explain those 40 recoveries.

Of the 56 held groups with a matched wording family, fact order and punctuation, zero chose both actions correctly on every crossed comparison. The best two held families covered three of four selector-by-clause settings. Correcting the STOP side accounted for the gains. Across the scoped held requests the link fit rescued 80 STOP replies, and the total number of exact longer replies fell from 128 to 112. This is selective transfer. The positional control was evaluated on TRAIN and DEV only, as registered.

The GPU trace extracted the causal slice from the full protocol prefix, checked every visited address and signed term, reconstructed each score, compared all three stage scores to the frozen scorer, and repeated in reverse order. The planted finite cases distinguished coincident support, all-zero differences, nonzero cancellation and correct signs. A deliberately substituted zero-weight bucket was rejected even though the numerical sum stayed constant. The distant-link fixture produced opposing signs on all four selector/argument combinations. Signed fit words and counters agreed with the independent finite oracle. The non-author pre-run review bound the exact source, binaries, tapes, cold-product checks and lease. [FIT](FIT.md) · [Native operators](device/README.md)

One rented RTX 4090 performed all scientific fitting and scoring. Each fit was repeated from the same starting bytes and reproduced the fitted state. The phase-one client and D11 verifier checked 1,724 rows per arm across the fresh TRAIN and DEV, original protected and earlier DEV sets. The held split checked 512 more rows each for baseline and candidate. The candidate's ordinary prompt-plus-four-address client matched all 1,856 held transitions, including which source was copied and terminal EOS. The result packet was checksum-verified on Wick before deleting the pod. The private-store readback rebuilt the trained artifacts and replayed their device records. [Fit receipts](evidence/fits.json) · [Transfer hashes](evidence/fit-output.sha256) · [Restore](RESTORE.md)

## Skills used

The Codex lab used the native-space-cycle skill. Each stage preserves its signed native weight words in the D11. The frozen writer commits and advances the actual chosen action, including on incorrect replies.

## Still open

The representation collision is established for the measured pairs. Binding nearby root content at relative distances repaired some novel phrasings, but measured gains depend on the wording and come with regressions. Further work must retain ordinary direct instructions and separate first-period mistakes from mistakes after the copied word. A useful open-ended conversation remains an independent milestone. [Next experiment](PICKUP.md)

## Plain summary

Two differently ordered requests could previously become the same model input at the decision point. The new weights used content order to give different answers on five fresh formulations. The held evidence warrants retaining the mechanism for comparison, while the prior-coverage losses preclude adopting this fit as the default.
