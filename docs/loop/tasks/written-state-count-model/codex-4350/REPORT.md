# Verified endpoints carry the long source in the genuine reversal

Lab: codex. [Order](ORDER.md) · [Registration](REGISTRATION.md) · [Replay and custody](REPLAY.md)

## What did not work, first

[codex] References #4063. The fixed writer finished **0/23** over-width gold intervals. Twenty-two reached the eight-token cap and one met a source delimiter first. The variable endpoint finished **21/23**. The two failures had 11- and 12-token targets, with copy delimiters after 2 and 11 emitted IDs respectively. Neither triggered supplied STOP. [Aggregate](evidence/aggregate.json)

[codex] A structurally legal endpoint supplied in both authority and claim can still be wrong for the question. In the finite control, an authority changed to include one extra source ID made the cursor emit that thirteenth ID and hand off there. When the original authority stayed fixed, the same overshooting claim was rejected before copy. The positive replays supply the registered gold span to both roles. Only the disagreement controls vary the claim against fixed authority. A learned source selector must determine the requested address and endpoint independently.

## What changed

[codex] `VerifiedEndpointCopy` binds the pinned tokenizer's complete protocol-2 opening, an independent prompt identity, and matching source authority and claim. The source remains an absolute `u32` address with a separately bound exclusive end. It uses the same `ExactCopyState` write transition as the current integer writer. The standalone finite driver supplies ordinary token IDs, observes release at the exclusive address before the next read, and applies the paired external STOP only at a verified endpoint. [Shared operator](../../../../../crates/uor-r4-integer/src/discrete_d11.rs) · [Finite driver](replay/src/main.rs)

The execution change spans 85 lines in the serving crate. The finite replay is 566 lines of Rust, with a pinned dependency lock, shell receipt validation and a JSON reduction.

## Measurements

| Opening and target | Original bytes | Absolute token range | Fixed writer | Variable writer |
|---|---:|---:|---|---|
| 349 tokens, own | [4,35) | [5,17) | Cap after 8 | 12 exact IDs, verified release at 17 |
| 349 tokens, opposing | [110,120) | [48,53) | Verified end | 5 exact IDs, verified release at 53 |
| 345 tokens, own | [110,120) | [48,53) | Verified end | 5 exact IDs, verified release at 53 |
| 345 tokens, opposing | [4,35) | [5,17) | Cap after 8 | 12 exact IDs, verified release at 17 |

[codex] All coordinates are half-open and index the untouched opening or original passage. Both addresses yield identical IDs across wordings. The 23 over-width rows emitted **298 source IDs** before release. The variable path handed off the ordinary proposal at every verified end and accepted the independent supplied STOP at those same endpoints. Calls after release passed ordinary IDs without another source read. The 349-token own interval is counted in both the 23-row cohort and the four reversal combinations. [Aggregate](evidence/aggregate.json)

[codex] Frozen controls passed for mismatched original prompts, shifted same-ID addresses, early and late claims, wrong token IDs, empty or reversed spans, both user framing edges, the original prompt fence, context above 512, and a copy delimiter before the authorized endpoint. Tested the wrong endpoint with a matching claim and authority as a separate negative control. The private trace preserves each copied ID and original cursor address. Independent Wick restores reproduced the complete trace and aggregate byte for byte. [Receipts](evidence/OUTPUTS.sha256) · [Replay](REPLAY.md)

## Checks run at the head

[codex] Standalone Rust replay compiled on macOS and ran on Wick. Seven legacy span-release tests, three exact-copy tests, the WASM serving library build check and the native space contract check passed. The finite replay checked the four exact state words and the emitted token on every variable transition. [Reproduction commands](REPLAY.md)

## Skills used

[codex] Native-space cycle. Raw integer IDs and cursor words are the replay units.

## Still open

[codex] Training a selector or an endpoint predictor, running a 512-token opening through the ordinary artifact client, and observing natural EOS are **NOT_RUN**. This result tests the integer cursor given a supplied exact span.

## Plain summary

Given the original source address and the last token boundary, the existing cursor can copy the 12-token target under both reversal questions. The only two failures among the 23 previously over-width gold intervals occur at existing copy delimiters.
