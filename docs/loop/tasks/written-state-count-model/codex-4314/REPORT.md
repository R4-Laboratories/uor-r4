# What the first-token weights learned from the corrected donor

**[codex] Neither fitted readout emitted the requested first token on any of the 800 controlled openings.** The full assistant marker and the causal question state did change where the model read. At 748 of those 800 selected addresses the fitted table contained no entry for the requested token. The first failure is visible before the frozen writer. This is a result about one trained opening readout on one complete D11 trunk. [Registration](SHEET.md) · [GPU results](evidence/results.json)

## The donor fix

The previous answer-first tape disagreed with the ordinary client at the first answer ID on each of its 4,345 actual assistant turns. We rebuilt the same 4,060 D87 TRAIN groups using the literal protocol-2 serving tokenizer for visible history, role marker, leading separator and answer. A separate verifier rebuilt the complete prefix from original source text through the serving dialogue methods. It matched the full visible prefix, the device's final 192-ID window, all 3,763,868 visible answer IDs, the masks, and all 4,345 detached reasoning documents. Of the genuine openings, 2,196 start with standalone space (ID 223) and 2,149 start directly with a lexical ID. The 285 earlier assistant EOS boundaries are visible masked history. The 4,060 terminal EOS targets are supervised. The original selection was reconstructed from 262 shards. [Donor implementation and falsifications](donor/README.md) · [Full readback](donor/RECEIPT.json)

Both arms received the same 118,915 supervised post-colon positions, comprising 4,345 genuine openings, 28,671 other visible positions and 85,899 detached reasoning positions. They also received the same 800 counterbalanced TRAIN openings. Each of those 119,715 events visited five addresses with a +2 target update. The contrast rows also applied −1 to the alternate and −1 to space at each address. Each arm received 598,575 positive updates, 8,000 negative updates and 598,575 exposure visits in the same ordered input batches. Integer additions within each GPU batch commute. The raw data reconstruct the colon-only space score of 37,604 and the marker arm's genuine-opening fallback space score of 3,592. [Fit receipt](evidence/training.jsonl) · [Input identities](evidence/approved.sha256)

The common parent is the existing complete #4312 answer-first trunk. The newly trained weights are sparse signed opening scores. Colon-only uses five constant addresses. The treatment uses the exact six-ID marker and five projections of its pre-marker geometric recurrence, with backoff when support is below four. The packed readout sizes are 54,196 versus 205,708 bytes, with five versus 5,512 learned addresses. The fits match in data and update budget. The treatment combines marker recognition, question-state routing and the extra capacity that routing creates. The original writer and trunk bytes are identical in the two deployments. [Native fit and rank contract](SHEET.md) · [Packed-state readback](product/FITTED.json)

## Fresh question and fact swaps

Each held group has two people, two distinct one-token answers, two assignments of those answers to people, and a question about each person. The four held wording families differ from the four contrast TRAIN families. Fact position is balanced. The 200 groups contain 400 question swaps and 800 first-token predictions. A preference requires a strictly higher signed score. Equal scores can have different ranks because the fixed unigram prior and then the lower original ID break ties. [Panel design and exclusions](panel/README.md)

| Controlled result | Colon only | Full marker and question state |
|---|---:|---:|
| Both questions favor their own answer within one assignment | 0/400 | 2/400 |
| Both questions favor their own answer under both assignments | 0/200 | 0/200 |
| All four requested answers beat both the rival and space | 0/200 | 0/200 |
| Requested answer scores above standalone space | 0/800 | 120/800 |
| Requested answer wins the gate / is emitted | 0 / 0 of 800 | 0 / 0 of 800 |
| Space is emitted | 800/800 | 545/800 |
| Median correct-answer rank among 4,096 IDs | 523 | 1,142 |

The preregistered matched group test has zero discordant groups, exact two-sided p = 1. The 95% exact interval for the marker arm's group rate is 0–1.83%. Its one-sided lower bound for the four-way answer-over-rival-and-space criterion is zero. The registered full-reply and EOS follow-up was therefore not run. [Group statistics](evidence/results.json)

**Where the scores disappear.** Swapping only the question changed the recurrent code on 388/400 pairs and changed the selected address on 376/400. Comparing each fixed answer token with itself under its own and the other question gives 51 score increases, 51 decreases and 698 ties. The marker arm gave the requested token a positive score on 29/800 openings and a negative score on 23. In all of the remaining 748, the selected row existed but had no word for that target. Of the 120 cases where the answer beat space, six had a positive target score, 107 had a missing target entry and seven had a negative target score. Relative to colon-only, the full-vocabulary target rank improved in 51 cases and worsened in 749. The best target reached rank 2. All 189 occasions on which the treatment writer changed the gate's selection began with a wrong gate ID. [Signed-score breakdown](evidence/score-membership.json) · [Held trace digests](evidence/held.sha256)

Six hundred held target occurrences were seen as positive targets in contrast TRAIN. Of these, 548 were absent at the selected address. All 200 held occurrences with answer IDs withheld from contrast TRAIN were absent there. None of the 800 occurred as a genuine **donor opening target**. Across other eligible donor colons, 758/800 occurred. Each candidate label occurs exactly once in the presented facts. These facts motivate learning a transferable score on candidate occurrences. They do not establish an architectural limit. [Exposure audit](panel/DONOR-EXPOSURE.json)

### Both answer assignments in one held group

The question is “Which call sign goes with Paul [or Zoe] on the trail crew?” The numbers are signed score and full-vocabulary rank in parentheses. This illustrative group, `33cd8a53cacadd96ccfde8ba`, is one of the two that recovered strict two-way preference in one assignment. The archive includes every group and the independently preselected first example in `analysis-v1/example.json`.

| Assignment and question | Requested answer | Room | Water | Space |
|---|---|---:|---:|---:|
| Colon-only, all four questions | varies | 16 (738) | 160 (141) | 37,604 (1) |
| Paul = room, Zoe = water. Ask Paul. | room | 0 (1,329) | 0 (733) | 27 (1) |
| Paul = room, Zoe = water. Ask Zoe. | water | 0 (1,329) | 0 (732) | 8 (1) |
| Paul = water, Zoe = room. Ask Paul. | water | 0 (1,331) | 2 (8) | −4 (4,096) |
| Paul = water, Zoe = room. Ask Zoe. | room | 2 (21) | 0 (743) | 24 (1) |

The swapped assignment supplies evidence for each requested word over its rival. Water still ranks eighth when Paul is asked about. Room still ranks twenty-first when Zoe is asked about, behind space. Reassigning the same two labels back to their original owners leaves both answer scores at zero.

## Transfer and verification

On five previously unused CoQA passages, the two original human questions per passage yielded 0/5 strict two-way preferences in either arm. Both models correctly emitted the two genuine immediate spaces and missed all eight immediate lexical targets. After each genuine space the ordinary trunk again emitted space. The intended lexical ID 19 ranked fourth in both cases, with target/space masses of 42,919/195,448 and 38,581/195,264. These ten openings are descriptive. The two questions in each passage differ in encoded length. [Delayed lexical steps](evidence/later-lexical.jsonl) · [Natural panel and limits](panel/README.md)

Real CUDA positive, wrong-state, near-marker, signed-negative, no-support and cold-reload controls passed. Both arms reproduced their held traces in reverse order. An independent reconstruction enumerated all 4,096 original candidates and all three probe scores for every held opening from the saved state. Cold ordinary Rust serving matched the device on all five opening addresses, selected address and exposure, selected sparse-row checksum, target/rival/space scores and ranks, selected gate ID and actual writer emission on **810/810 openings per arm**. The result identifies the first-token decision. Earlier D11 trunk/carry words were outside this capture. The measured producer is `85d2f4e9fee20f97ab555e8bde231821ee76e6e5`. Later carrier validation pins the canonical tokenizer, bounds the nested allocation and rejects unsupported base types before reading their payload. The release replay and product digests are in the [product receipt](product/FITTED.json). [Device finite checks](evidence/finite.json) · [Measurement source](evidence/device-source.sha256)

On the planted GPU fixture, the marker gate selected the designated lexical IDs 462 and 1,400 at the two different states. The frozen writer emitted IDs 31 and 11. Eight raw-prompt ordinary-client fixture runs reproduced all eight device emissions. This fixture demonstrates that a learned gate can make the required preference flips. It also exposes a separate writer alias that a future source-positive experiment must check. Among held prompts neither arm took the 4,096-ID nonpositive-score fallback. The longest selected sparse row contained 1,804 entries for colon and 101 for marker. Marker routing adds one four-block recurrence on 40–181 pre-marker IDs in this panel. The ordinary Rust CLI loaded both fitted artifacts. The current Studio worker's schema allowlist excludes this experimental carrier. Browser execution and speed remain unmeasured. [Product fixture and observed work](product/FITTED.json)

One rented Vast RTX 4090 completed the two fits plus weight serialization/readback in 1.13 seconds. All held scores, reverse replays and supplemental lexical probes completed 8.95 seconds after fit start. The provider reported 444 seconds of uptime at $1.309/hour, approximately **$0.16** of compute. The session copied the results to Wick ZFS and released the instance. [Compute receipt](evidence/compute.json) · [Pinned recovery and score tables](compute/README.md)

## Next decision

Keep the corrected ordinary encoder. For grounded openings, test a shared geometric score on each candidate source occurrence conditioned on the requested subject. Train the same score across answer identities and hold out whole wordings and answer IDs. Compare correct, swapped and absent candidate sources with the same writer and donor dose. Require the right occurrence on both questions under both assignments, then measure ordinary first-token emission against space. A pass there would justify full replies through EOS and a separate natural-language transfer check. The present experiment gives no reason to increase the dose of this sparse token-by-question readout or claim useful general conversation. [Handoff](PICKUP.md)
