# The refit reverses the signed noun ranking at the original governor

## What did not work

All four previously correct first/later switches failed on the **later** instruction after the ordered-interval median refit. The reader still found the correct signed governor. At that governor the original later `snuffer` went from three transport units cheaper than the earlier `snuffer` to nine units dearer. The best legal full tuples preserved both margins. These four pairs cross two facts and the two signed roles within the same lantern wording family. The original mixed TRAIN set contained zero matched first/later instructions. Its comparisons all favored the requested occurrence after refit. [Four source traces](evidence/four-switches.tsv) · [TRAIN comparison](evidence/train-original.json)

## What changed

The ordered-interval scorer, two frozen 13,336-byte state payloads, tuple bank, D11 and exact writer stayed fixed. A measurement kernel enumerated all sites at the original signed governor. For each of the two occurrences it forced that exact link and let the opposite governor, opposite noun and heading compete over the native tuple list. The unrestricted prediction used the existing CUDA operator and the original text and byte/BPE addresses. Lexical and transport costs are nonnegative integers. Local reads order those two costs lexicographically. Completed tuples order total lexical cost, total transport cost and bank cost lexicographically. Ties, missing completions, a different governor and third sites remain distinct outcomes. [Pre-registration](PRE-RUN.md) · [Device implementation](device/README.md) · [Non-author fitness review](audit/FIT.md)

The owner moved physical execution to a leased Vast RTX 4090 while Wick's card was busy. The binary was built on Wick for SM89. The same rented device ran the finite fixture, then the research reads. Wick received the raw results and supplied the immutable inputs and previous physical traces. The ordinary writer received no input in this unit. First wrong emitted token and exact replies through EOS remain unmeasured. [Venue amendment](VENUE.md) · [Hardware and finite controls](device/pod-finite-receipt.json)

## Measurements

The table covers both facts and both signs. Each entry gives requested / same-spelling rival transport cost, lower wins. Every lexical cost in the comparison was zero. Both forced completions existed and each native winner was unique. For positive `keep`, the bank costs of completions through the first and later sites were 572 and 536. For excluded `drop`, they were 600 and 548. Transport decided every comparison before the bank tiebreak.

| Selector | At correct governor, before | At correct governor, after | Best full tuple, before | Best full tuple, after | Returned occurrence |
| --- | ---: | ---: | ---: | ---: | --- |
| First | 54 / 55 | 64 / 65 | 80 / 81 | 84 / 85 | First both times |
| Later | 51 / 54 | 73 / 64 | 77 / 80 | 93 / 84 | Later before, first after |

For both signs the requested later attachment used transport center 3 before refit and center 0 after. The rival continued to use center 3. Its transport cost rose by 10 and the requested cost rose by 22. Replacing the unconstrained read with the best complete tuple at the fixed governor produced the same winner in every case. Other sites can have cheaper local attachments in isolation. Across full eligible completions, the two enrolled occurrences occupied first and second place in these examples. The opposite noun was always `room` in place of `taper`; the heading was the correct third occurrence of `snuffer`. None of these pairs had both full tuples correct. [Addresses, competing links, centers and tuple ranks](evidence/four-switches.json)

| Fact and signed governors | First occurrence, BPE | Later occurrence, BPE | Later request after refit |
| --- | --- | --- | --- |
| Cuff and orchard, `keep` and `drop` | [70,73) | [77,80) | [70,73) on both signs |
| Textile and marsh, `keep` and `drop` | [67,70) | [74,77) | [67,70) on both signs |

Across the full historical panel, 4/32 pairs selected both correct signed noun occurrences before and 0/32 after. These are spent diagnostic cases. On the 256 original TRAIN prompts we enrolled 288 comparisons, covering 256 neutral rivals and 32 additional heading rivals. Before refit, 264 comparisons preferred the target locally, 16 lost and 8 tied. After refit, all 288 favored the target at the governor, in the conditioned tuple and in the native read. Zero original comparisons repeated the failing later-side transition. [Historical counts and TRAIN rival census](evidence/summary.json)

On 32 separately generated TRAIN-only first/later pairs, neither state selected the requested occurrence on both sides of any pair. The requested link and complete tuple both changed from cheaper to dearer on **12/64 prompts**, including **8 later** and **4 first**. A third site won the unrestricted read before and after on all twelve. For example, on one later instruction the governor `forget` compared two occurrences of `protocol`. Requested / rival transport was 94 / 98 before and 96 / 94 after. The conditioned totals were 108 / 112 and 111 / 109. The native read selected `came` before and `what` after. The TRAIN-derived experiment reproduces the local reversal and its persistence in the conditioned tuple. It recorded zero complete paired regressions and zero exact failed-side native transitions. [TRAIN-derived transitions and source examples](evidence/train-derived.json)

### Reproduction at the historical source distances

Before the six primary reads, we registered a separate four-pair TRAIN-derived diagnostic, triggered by zero complete reproductions in the 32-pair panel. Each historical failure had 25 intervening BPE tokens from the first noun to its governor and 18 from the later noun. The primary derived pairs never had that combination. We removed the same three tokens from the existing TRAIN planning-review frame to match those distances. The two facts and both signed verbs came from TRAIN. Each pair still differs by just one selector token. Selection of the frame and deletions used the prompt geometry alone. The deletion changes surrounding text as well as distance, so this experiment establishes recurrence at that layout and leaves the independent causal effect of distance open. [Prospective rule](TRAIN-LAYOUT.md) · [Matched panel](matched-train/README.md)

**All four matched pairs reproduced the entire measured signed-source transition.** Both frozen and refitted states selected the correct governor. The original state followed both instructions on 4/4 pairs. The refit stayed on the first occurrence on 4/4. The noun was `proposal`, the governors were `accept` and `reject`, and the facts were school and road. On school the two occurrences were BPE [63,67) and [70,74). On road they were [65,69) and [72,76). A separate integer replay agreed on all 41,968 scalar fields. [Eight raw-address reductions](evidence/distance-matched-rows.jsonl) · [Four paired comparisons](evidence/distance-matched-train.json) · [Independent replay](evidence/independent-layout.json)

| TRAIN selector | At correct governor, before | At correct governor, after | Complete transport before, accept / reject | Complete transport after, both signs |
| --- | ---: | ---: | --- | --- |
| First | 66 / 75 | 68 / 85 | 105 / 114 for accept, 104 / 113 for reject | 97 / 114 |
| Later | 67 / 70 | 81 / 68 | 106 / 109 for accept, 105 / 108 for reject | 110 / 97 |

On all four matched later instructions the requested attachment continued to use transport center 3 in both states. The historical later attachments changed from center 3 to center 0. Relative transport cost reversed in both cases. In the raw packed vectors, changing the selector modified historical coordinates 22, 24, 25, 26, 27 and 29. On TRAIN it modified coordinates 20, 21, 23, 28, 30 and 31. Each set held for all eight compared links in its cohort, covering each distance four times. These are zero-based coordinate indices. The behavioral reproduction spans two different sets of selector-sensitive coordinates. [Original integer vectors and support audit](evidence/selector-axes.json)

## Checks run at the head

The standalone native reader and the instrumented reader agreed byte for byte on the synthetic fixture. Eight malformed request files were rejected. Across the six primary physical reads, the independent integer implementation reproduced 2,091,030 scalar fields on 768 native predictions and 832 signed comparisons across both states. It also checked 64,568 archived fields. The raw native predictions for both states matched the previous Wick captures on all 64 historical and 256 original TRAIN prompts. A fresh reduction and six independent replays from the primary archive reproduced 12 output files byte for byte. The rental was deleted after all 213 job files were secured on Wick. At the quoted rate of $0.869/h, 671 seconds of provider uptime implies an estimated $0.16 of compute. [Finite checks](evidence/finite-checks.json) · [Independent audit](audit/README.md) · [Rental receipt](evidence/compute.json) · [Restore the evidence](RESTORE.md)

## Skills used

The native-space cycle guided byte identity, shared integer operators and bit-exact replay. The UOR research architect guided enrollment, conditioning on the actual governor, and the separation of a local ranking change from an unrestricted original-address switch.

## Still open

Use the four reproducible TRAIN-derived pairs as regression cases for the next training test. On new TRAIN examples, require the desired original occurrence to beat its same-word rival at the correct governor on both selector variants. Score the unrestricted full tuple against all eligible third sites in parallel. Fund the comparison with the same update budget as the prior median, protect the 24 original near identities, and evaluate once on newly enrolled families. The present measurements identify a target margin for those updates. A repair and ordinary grounded generation remain open. [Previous median experiment](../codex-4299/REPORT.md) · [Grounded-reply milestone](https://github.com/UOR-Foundation/uor-r4/issues/2030)

## Plain summary

The refit makes the later noun more expensive to reach from the right instruction word. That accounts for the four observed first/later failures. On the TRAIN-derived prompts, unrelated words can also beat both intended occurrences. An experiment that measures only the distance to the gold noun would miss those competing reads.
