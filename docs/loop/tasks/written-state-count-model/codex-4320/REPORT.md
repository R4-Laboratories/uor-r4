# Why the native writer emits “g” after a verified word and period

## What did not work

All 400 native replies first failed at the third emission with `g` (73). A supplied correct completion action recovered all 200 stops. Each continuing reply still needed its own next word, and zero finished exactly. Supplying that second factual word separately made token three correct on all 400 prompts. All 200 continuations then failed at token four, where EOS was required. No arm produced an ordinary EOS. [Exact counts](evidence/arms.json)

## What changed

The Codex lab held the #4319 D11 and all weights fixed. The first emission used the verified original source address and the second was a supplied period. At token three the native arm ran the ordinary writer. In the correct and swapped arms a supplied STOP selected EOS and CONTINUE passed the native proposal through. The swap used the opposite member of each real pair. Only the separate supplied-next arm took the continuing word from its verified address. The one-use action expired on commit. Token four, whenever needed, came from the native writer on its actual prefix. No weights were fitted. [Pre-registration](SHEET.md) · [Cold client](product/README.md)

## Measurements

Two hundred fresh contexts each had a matched stop and continue request with identical facts, question and factual addresses. Four held wording families provided 50 pairs each, using eight directive pairs across the panel before terminal punctuation variants. Both original word and supplied period were correct on 400/400 prompts in every arm. Token positions below start at one. An exact pair requires both replies to emit their full references through EOS.

| Post-period action | First wrong on stop | First wrong on continue | Exact stops / 200 | Exact continuations / 200 | Exact pairs / 200 |
|---|---|---|---:|---:|---:|
| Native | 200 at 3, `g` instead of EOS | 200 at 3, `g` instead of factual word | 0 | 0 | 0 |
| Correct completion state | All exact | 200 at 3, `g` | 200 | 0 | 0 |
| Swapped completion state | 200 at 3, `g` | 200 at 3, premature EOS | 0 | 0 | 0 |
| Correct state plus supplied next word | All exact | 200 at 4, extra native token instead of EOS | 200 | 0 | 0 |

The correct state gained 200 exact replies over native, all by explicitly selecting EOS. The swapped arm emitted 200 premature EOS tokens. The lexical control produced both correct third tokens on 200/200 pairs, and zero complete pairs. Its fourth emission was a period in 80 continuations, ID 161 in 16, ID 265 in 15, ID 28 in 12 and another non-EOS ID in the remaining 77. Both punctuation strata yielded the same exact and first-error counts. A fourth-token period occurred on 43/100 continuations with input-final punctuation and 37/100 without it. [Punctuation strata](evidence/strata.json)

**The read changes, then the gate collapses the distinction.** Swapping wording changed the packed read code on 167/200 pairs and the packed recurrence code on 200/200. The 400 prompts produced 194 read codes, 361 recurrence codes and 400 distinct joint states. All 400 chose line feed (201) at the sparse gate. With a final period in the instruction the pointer selected that period's line-feed successor in the assistant marker on 200/200 prompts. Without the final period it selected a successor inside the directive on 188/200, and a line feed at the fact/question boundary on 12/200. Thus the gate still chose line feed on 188 rows where the active pointer proposed another token. The 201 query count was 84,915 on every row. Re-ranking the native candidate pool using the query term alone, or removing the pointer bonus, read-dependent terms or recurrence-dependent terms separately, left 201 first in every case. These are fixed-pool score diagnostics. [Native trace](evidence/trace.json) · [Rank and sensitivity measurements](evidence/attribution.json)

**The writer resolves a geometric tie by token ID.** At the gate, 73 ranked 41st–58th among positive-scoring candidates, and EOS ranked 28th–65th. After the gate chose 201, `g` and line feed occupied cell 172 and tied at L1 distance zero. They were the only tokens at the minimum on every row. The writer's ascending-ID tie break ranked 73 first and 201 second. EOS was distance 20 and outside the admitted cells on 400/400. The designated follow-up word was outside the sparse candidate pool on 398/400 prompts and outside the writer's admitted cells on all 400. Preserving the gate winner's exact ID would emit a line feed on these rows, still incorrect for either reference. [Writer trace and shortlist](evidence/trace.json) · [Candidate measurements](evidence/attribution.json)

On an input-selected pair, Devin Briggs's opening word was `chess` and his follow-up was `feather`. Both native trajectories emitted `chess.gth` through the cap. Correct STOP produced `chess.` and EOS. Correct CONTINUE still produced `chess.gth`. With the supplied follow-up address it produced `chess. feather.` — the extra period was the first error. The private decoded archive preserves text, original IDs, original positions, native proposals and full trajectories for every row. [Restore and inspect](RESTORE.md)

## Checks run at the head

An independent non-author checked the 16 separate calibration pairs, calibrated natural and supplied EOS, false termination, wrong source addresses and capped output, and signed FIT before opening held output. A leased RTX 4090 ran the precompiled SM89 kernel, including a reload in reverse order, in 22.292 seconds for the scored forward/reverse operation. The Wick cold client agreed exactly on 400 rows, 1,600 trajectories, 5,800 writer transitions and every common third-token trace field. Four ordinary text/raw client comparisons on held inputs checked where the learned period reproduced the supplied prefix. The archive returned from the pod before deletion, and the restored private packet replayed through the same cold verifier. [FIT](FIT.md) · [GPU](evidence/device.json) · [Cold parity](evidence/qualified-parity.json) · [Storage](RESTORE.md)

## Skills used

Codex applied the repository native-space cycle to the frozen integer state, arithmetic and one-use actions. The GPU and Rust histories include the verified source, committed IDs, exact prompt addresses and resulting recurrent transitions. [Native-state contract](../../../../harness/native-space.json)

## Still open

The experiment supplied the first source, output period and both desired state controls. It establishes conditional failures in content selection and termination. It gives no evidence of a learned finish action, general conversational quality or a milestone advance. The next experiment should train an action that selects EOS or an authenticated factual source from the prompt, then re-evaluate completion after a verified final copy. [Next experiment](PICKUP.md)

## Plain summary

The requested continuations changed the model's internal state. The gate still chose line feed, then the writer emitted `g` from a two-token tie. An explicit stop fixed every stop. An explicit second word moved every continuation's error one token later. The remaining target is to learn what is still owed after each verified emission.
