# Smaller integer updates preserve near source addresses but leave far nouns unresolved

## What did not work

On the Wick RTX 4090 the protected update kept all 24 previously correct near training pairs. It lowered common far signed hinge from 1,464 to 1,320 and recovered zero new complete far training pairs. On fresh wording it kept all 40 near wins and still won 0 of 64 far pairs. The governor was correct on every protected held prompt. The first failing read was an original noun address. The registered gate stopped ordinary generation, so the first wrong **emitted token** and exact replies through EOS are **NOT_RUN**. M2 remains open. [Source gate](evidence/source-gate.json) · [Ordinary client refusal](evidence/writer-release.json)

## What changed

We started from the same 13,336-byte native state, mixed training panel and signed first-sweep votes as #4291. The tokenizer, governing-verb reader, joint source reader, D11 and writer stayed fixed. Each arm considered 232 one-coordinate moves and 14 whole-center moves at each of two decisions. The moves shift i16 transport coordinates inside existing centers. Both arms first read the identical batch. They then took different first moves, so Wick scored three distinct batches. Selection used only training observations. A complete training pair requires both original governors and all three original noun addresses on both opposite instructions. [Order](ORDER.md) · [Registered comparison](PRE-RUN.md) · [State and scoring implementation](fit/README.md)

## Measurements

| Training state | Elected move | Common far hinge | Strict far edge wins / 256 | Original near pairs kept / 24 | Complete common far pairs / 64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parent | — | 1,464 | 136 | 24 | 40 |
| Protected, first | 234 | 1,392 | 136 | 24 | 40 |
| Protected, second | 240 | 1,320 | 136 | 24 | 40 |
| Unconstrained, first | 237 | 1,048 | 152 | 8 | 40 |
| Unconstrained, second | 245 | 632 | 168 | 8 | 40 |

At the parent, 241 of 246 proposals preserved all 24 near identities, 98 reduced far hinge, and 96 did both. Of those 96, 75 kept the far count and 21 lost eight complete far pairs. None won a previously missing far pair. After protected move 234, another 241 preserved the near identities, 94 improved hinge, and 92 did both. Of those 92, 74 kept the far count and 18 lost eight. No move in that set recovered a missing pair. From the unconstrained first state, none of the 246 second moves restored all original near pairs. These are counts for a fixed menu from three particular starting points. [Candidate counts, vectors, pair gains and independent audits](evidence/training.json)

The two selected protected moves apply the same nine one-unit displacements to center 2 in the two signed banks. Across all 256 common far edges, each of the original 136 wins, 16 ties and 104 losses kept its sign. The unconstrained final state converted 16 original losses to wins, eight losses to ties and all 16 ties to wins. Those conversions still closed no additional complete training pair. Single-edge loss and exact instruction-pair recovery gave different information. [Signed transitions](evidence/training.json)

The first candidate that lost a protected near source was move 0. On the DEFER member of `supplier-contract/park/0`, it replaced `estimate` at absolute BPE [97,101) with `may` at [102,104). The first loss among proposals that improved far hinge was move 237, the unconstrained winner. It kept both governors on `supplier-contract/park/16` but selected `purchase` at [97,101) for `estimate` at [104,108). The protected rule rejected this move. [First rejected addresses](evidence/training.json)

All 24 originally incomplete common far pairs failed in both instruction directions after each protected update. Each of their 48 prompts had both governors correct and exactly one wrong linked noun. The joint reader chose the strongest wrong signed rival in all 48. Its address lay between the governing verb and the target every time. The rivals were `board`, `space` and `piece`, each on 16 prompts. All 144 units of the reduction in hinge came from the 16 `piece` edges. Their signed margins rose from −25 to −16 and stayed on the losing side.

For example, on `enamel-display/road/18` the reader found `keep` on DEFER and `drop` on TAKE, both at [100,101). It attached each to `piece` at [102,103) instead of `medallion` at [112,115). Move 234 improved DEFER and move 240 improved TAKE. In each signed read the cost of the true `medallion` address fell from 31 to 22, while the intervening `piece` cost 6 throughout. The lower cost still selected `piece`. No joint noun or governor address moved on any of the 128 common far training prompts. [Original rival costs and source addresses](evidence/far-residuals.json)

### Fresh source transfer

Each pair contains TAKE and DEFER versions with opposite instructions and the same facts. The four held wording families contained eight new target nouns, genuine prior dialogue, a same-spelling rival at another address, and additional nouns in the long constructions. Each distance had 64 pairs. A source prompt is exact when all three original noun spans match. The learned-state pair counts below also matched the stricter count requiring both correct governors. Family and intervening wording covary with verb distance on this panel. [Panel construction and disjointness](panel/README.md) · [All nine arms by family and distance](evidence/transfer.json)

| Fresh source reader | Near pairs / 64 | Far pairs / 64 | All pairs / 128 | Exact source prompts / 256 |
| --- | ---: | ---: | ---: | ---: |
| Native D11 | 0 | 0 | 0 | 12 |
| Unfitted reader | 0 | 0 | 0 | 0 |
| Signed governor and gap alone | 0 | 0 | 0 | 0 |
| Parent | 40 | 0 | 40 | 88 |
| Protected, two updates | 40 | 0 | 40 | 88 |
| Unconstrained, two updates | 32 | 16 | 48 | 100 |
| Fixed far-only fit from #4291 | 28 | 16 | 44 | 116 |
| Supplied correct links | 64 | 64 | 128 | 256 |
| Supplied exchanged links | 0 | 0 | 0 | 0 |

The two states scoring 16 far wins recovered **disjoint pairs**. The unconstrained state won eight boatyard and eight leather-tool pairs. The fixed far-only state won eight observatory and eight orchard pairs. Relative to the parent, the unconstrained state lost eight near pairs and the far-only state lost twelve. The protected update missed all 16 comparator gains. A score of 16 alone would hide which wording each state actually reads. [Pair identities](evidence/movements.json)

Comparing original bytes and absolute BPE intervals directly, the protected state returned the identical three-address tuple as the parent on all 256 held prompts. The unconstrained state changed 32 far tuples, all gains, and 20 near tuples, all losses. [Per-prompt address transitions](evidence/held-changes.json)

On all 128 far prompts, the protected reader found both requested governing verbs. The first wrong address was the affirmed noun on 68 prompts and the excluded noun on 60. On the DEFER version of `observatory-inventory/snow/10`, the protected reader chose `our` at [104,105) for `sextant` at [105,108). On the TAKE version it chose `chalk` at [96,97) for `gnomon` at [63,67). The parent made the same choices. The fixed far-only reader recovered all three original addresses in both directions, including the second occurrence of `gnomon`. These offsets index the conversation's source tokens. [First failing spans and original token intervals](evidence/held-sources.json)

## Checks run at the head

Physical Wick training and grading inside its exclusive lease took 29.91 s. Held collection took 5.54 s. Independent integer replay matched three batches containing 9,921,984 state bytes, 379,392 signed edges and 189,696 joint read rows. The held replay confirmed the learned-state source sets on all four sets of 256 prompts. Supplied and exchanged links produced the expected opposite results. The finite fixture distinguished a safe move from an equally improving move that replaced a previously correct near pair while keeping the total count unchanged. [Training replay](evidence/train-replay.json) · [Held replay](evidence/held-replay.json) · [Device fixture](evidence/finite-device.json) · [Independent finite replay](evidence/finite-replay.json)

Two successive shared-device jobs delayed the primary run. Under the owner's short-rental authorization, a prospectively registered replica repeated the first training decision on another RTX 4090. A missing fixture path stopped its first finite run. A fresh package and independent replay passed before the provider training permit was issued. Wick and Vast then produced byte-identical first-round outputs across all 746 manifested files, including every candidate state, signed edge and source choice. The provider lease lasted 511 s, approximately $0.19 at the quoted compute rate. The controller confirmed destruction and zero active pods. Wick executed both training decisions and the entire fresh test. [Supplemental registration](SUPPLEMENT.md) · [Supplemental FIT](audit/VAST-FIT.md) · [Cross-device comparison](evidence/hardware-parity.json) · [Rental receipt](evidence/rental.json) · [Immutable private restore](RESTORE.md)

The primary training capture occupied 330,758,540 apparent bytes, the fresh source capture 9,479,925 and the grade 2,445,225. One post-restore snapshot of the full study tree recorded 4,005,973,622 apparent bytes and 591,831,552 allocated bytes on ZFS. The exact peak was not continuously sampled. The private measurement and supplemental archives together used 12,394,663 uploaded bytes. [Storage observation](evidence/storage.json) · [Restore instructions and immutable revisions](RESTORE.md)

The measured commands on Wick were `measure-wick.sh run v1` with its [pinned release](evidence/released-inputs.sha256), `authorize-held-wick.sh v1 v1 v1`, `held-wick.sh run v1 v1` with the resulting permit, and `grade-held-wick.sh v1 v1 v1`. The independent programs replayed every recorded batch before unblinding. Static shell syntax, Rust formatting, local document links and `uor-harness check --only harness.native_space_contract` passed. Both separate immutable downloads reproduced their local restore receipts. [Primary restore](evidence/restored-primary.json) · [Supplement restore](evidence/restored-supplement.json)

## Skills used

The native-space cycle guided the shared i16 state and byte-for-byte readback. The research-architect skill guided the equal-budget control, original-address criterion and conditional writer test.

## Still open

On the 24 unresolved training pairs, freeze the state, governor and writer. Rank every original candidate for the failed link. On device, compare the native read with omitting just the strongest intervening rival at its exact address, plus an equal-size control omitting a nonwinning intervening address. Record the new top rival, the gold noun's rank and whether both instruction directions recover. This distinguishes a single blocking occurrence from several intervening occurrences outranking the target. Use the observed rivals to propose the next integer update and keep the 24 near-pair identity checks. A supplied address omission is a diagnostic. A model advance still requires selecting the source from text alone on fresh wording. [M2 grounded replies](https://github.com/UOR-Foundation/uor-r4/issues/2030)

## Plain summary

The model can now adjust a small part of its stored state while keeping the short instructions it already reads correctly. In a long instruction it still attaches a verb to a nearer word. In the example above, it found “keep” but chose “piece” when the request named “medallion.” We can now test which other words, if any, also outrank the correct one before doing more training.
