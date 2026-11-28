# A passage changes the compact model's carry, but rarely its opening choice

## What did not work, first

[codex] Zero of 128 held replies began with the reference token. Zero emitted EOS within 32 greedy tokens. This was true with native carry, donor carry and zero carry. With the trained model intact, 123 replies opened with original ID 223 (one ASCII space); the other five opened with ID 287 (space plus “The”). The first wrong token was offset zero in all 384 trajectories. Only 1/64 pairs met the registered test that both passages prefer their own first token and both reverse those preferences with exchanged carry. There were no score ties. Decoding the saved IDs and trimming ASCII whitespace also gave zero matching first reference words in each arm; the word comparison preserves case and punctuation. [Byte-level recount](evidence/decoded.json)

[codex] No weight update or milestone promotion follows from this measurement. [Independent count](evidence/independent.json)

## What changed

[codex] Evaluated the saved 30,270,020-byte UORBFY03 Arm 2 on its native recurrent transition. Each of 64 pairs uses the same complete, natural final question after two different authentic conversations with distinct first reference IDs. We retained the full passage and history, yielding 128 distinct article groups, 40 distinct final questions, 84,843 input IDs and a longest input of 1,045. Immediately before the last common input ID we exchanged or zeroed just the six layers of carry. Every arm kept the recipient's shift history, token count, input, weights and greedy writer. Recipient and donor shift histories differed in all 64 pairs. The experiment identifies the contribution of this particular carry intervention. Other state can also carry information about the passage. [Cohort](cohort/README.md) · [Frozen design](SHEET.md)

## Measurements

[codex] Original first-reference token and exact single-reference results. “Admitted” means the real model chose the group containing that token. Diagnostic ranks ask where the token scores within its correct group, even when the actual route excludes it. The artifact elects one of 64 groups, then one of the 64 original-token slots in that group.

| Carry at final common input | Reference group admitted | Both groups admitted in pair | Reference ranked first / top eight in correct group | Actual first token correct | First wrong at offset 0 | Natural EOS / exact through EOS |
|---|---:|---:|---:|---:|---:|---:|
| Recipient's own | 22/128 | 1/64 | 19 / 73 of 128 | 0/128 | 128/128 | 0 / 0 of 128 |
| Other passage's | 21/128 | 1/64 | 19 / 73 of 128 | 0/128 | 128/128 | 0 / 0 of 128 |
| All zeros | 22/128 | 1/64 | 17 / 72 of 128 | 0/128 | 128/128 | 0 / 0 of 128 |

[codex] The native model elected only two groups across references spanning 37 groups. Of the 22 admitted references, seven ranked in the top eight and zero ranked first. All 19 native diagnostic within-group winners were excluded by the actual group choice. Perfect group selection could therefore recover at most 19 first tokens with this checkpoint's frozen fine scores. A perfect within-group read with the actual group choices could recover at most 22. The median native diagnostic rank was 6 of 64. The zero-carry median was also 6. [Ranks and controls](evidence/recount.json)

[codex] Swapping carry changed the first hidden vector and coarse scores on 128/128 recipients, PKM read addresses on 126/128, the first elected group and first token on 6/128, and some self-fed token on 126/128. Across replies the swapped distribution was still 123 spaces and five “ The” openings. Zeroed carry elected ID 201 (newline) on all 128. Native target-vs-rival Q48 preferences across individual replies were 65 own, 63 other; swapped were 63 own, 65 other; zeroed were 64 and 64. Summing the two recipient masses preferred the correct assignment on 38/64 native pairs; the same sum favored the donor assignment on 36/64 swapped pairs. There were 36 native-to-swapped sum reversals but just one strict two-recipient reversal. The sum permits one recipient's change to outweigh the other's. [Independent original-ID recount](evidence/independent.json)

[codex] An exploratory calculation after scoring compared each recipient's own-to-rival mass ratio with and without its own passage carry. Both recipients' ratios increased on 37/64 pairs and fell on 27/64; no pair split and no mass was zero. With native versus zero carry the corresponding counts were one pair up, three down and 60 split. This secondary comparison uses exact integer cross-products. The two rows of a reciprocal swap are coupled. Its direction alone gives no credit toward the registered four strict preferences or a correctly emitted token. [Secondary recount](evidence/odds.json)

## Checks run at the head

[codex] Independent review caught a candidate-selection bug before model scoring. The rejected 64-pair tape contained 17 answers starting with a standalone space and five pairs sharing the same first emitted ID. The corrected builder filtered those starts before matching and regenerated the panel twice. The compact training-stream screen used exact 32-ID spans. It cannot rule out paraphrase overlap. D87 DEV-test remained sealed. [Admission and overlap record](cohort/evidence/aggregate.json) · [Non-author FIT](FIT.md)

[codex] A leased physical RTX 4090 passed the two-cell planted vocabulary, sham, label independence, EOS and cap controls. The 2,048-input historical fixture matched all 128 archived greedy IDs and integer Q48 masses. On the six predeclared pairs, device and ordinary Rust matched 12 prefill states and all 1,152 consumed single-step transitions across 36 arms. Batched GPU decoding matched all emitted IDs and stops. An independent cold ordinary Rust replay on Wick matched the first diagnostic transition and every emitted ID and slot-derived route on those 36 arms. Seven host fixture and corruption tests passed. The full GPU job took 190 seconds. Provider uptime was 266 seconds at $0.869/hour, about $0.064 before billing adjustments. The leased pod was deleted after verified transfer to Wick ZFS. A forced restore of the 47-file custody packet reproduced its SHA-256. [Calibration](evidence/finite.json) · [Historical replay](evidence/historical.json) · [Per-step parity](evidence/parity.json) · [Compute ledger](evidence/provider.jsonl) · [Reproduce the result](REPLAY.md)

## Skills used

[codex] Native-space cycle for byte-identified learned weights, per-layer integer carry, immutable original-ID tape, and differential replay through the shared native step.

## Still open

[codex] The [actual checkpoint trainer](../antigravity-5/butterfly-v3/src/main.rs) split each dialogue at 257 input IDs for 256 next-token transitions. Its only epoch trained on all non-BOS targets inside each window. The cross-window transition was dropped and an answer-only phase never ran. A static census of all 71,876 TRAIN records recovered the exact epoch total of 100,045,687 supervised transitions. TRAIN had 4,745 two-response records and zero with three or more responses. In only 203 of those 4,745 records did one window contain both full user prompt blocks and the full second answer through EOS. This measures prompt-block co-location. The segment files do not locate the evidence for a particular answer. On the held side, 94 of 128 questions request the third, fourth or fifth response. [Training stream census](evidence/train-exposure.txt) The shortest held opening is 451 IDs. None of the 128 entire held openings fit in one such training window. This is an exposure mismatch; individual answer-bearing source mentions can be much closer to the final question. The scorer shows measurable passage dependence, but both output stages suppress the intended first token. The next learned-weight comparison should give the checkpoint both the complete source and the answer-bearing question during training, and separately measure whether training changes group admission and conditional token rank at the same weight size and update budget. Use separate training groups and an untouched comparison panel. Score both members independently and free-run through EOS. A clean reference to the current mechanism is the unchanged compact checkpoint. The [ordinary-client assessment](CLIENT.md) traces the missing UORBFY03 import and proposes a text-to-token parity test using the same tokenizer, dialogue encoding and recurrent transition. No open-ended human quality verdict, browser test or byte-matched long-context comparison ran in this unit.

## Plain summary

[codex] The model reads two different histories differently. Almost every first choice collapses to the same two tokens. For this checkpoint, fixing the group selector alone or the final token score alone leaves most of the errors intact. These results isolate the two learned choices that should be tested together before expanding model size.
