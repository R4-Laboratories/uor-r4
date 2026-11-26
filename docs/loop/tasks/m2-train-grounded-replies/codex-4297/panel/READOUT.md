# Recorded ordered-bridge source decisions

The ordered representation kept 16 of the 24 original protected near TRAIN pairs at the frozen base and zero full far pairs. Across the 68 proposals in each of two rounds, zero proposals recovered any of the eight lost original near pairs, even when inspected without the protection constraint. Thirty-six proposals per round reduced all-outranker loss, down to 8,800 from 25,792. Both elections kept the base state bytes. [Candidate reducer](train-candidates.jq)

| Physical round | Proposals | Protected near identities retained | Total full near pairs | Full far pairs |
| --- | ---: | ---: | ---: | ---: |
| 1 | 68 | 0–16 out of 24 | 0–24 | 0 |
| 2 | 68 | 0–16 out of 24 | 0–24 | 0 |

Up to eight wins on previously unprotected near pairs explain the gap between the two near counts. The eight lost original identities are all in the supplier-contract family. In the first such pair, at physical rows 128 and 129, the same frozen state reads both original governors and both gold attachments span three BPE IDs. The endpoint reader selects `estimate` at bytes [208,216) and BPE [97,101) on both prompts, affirmed on DEFER and excluded on TAKE. The ordered packet changes the representation of a competing interval from the same governor and the joint read selects `now` at bytes [199,202) and BPE [95,96) on both prompts. The joint ordering compares summed lexical cost, summed transport cost and frozen bank cost in that order. With lexical cost zero in both cases, the new winner's transport sum of 59 beats the runner's 60. The selected tuple is unique on both prompts. This switch follows the geometry presented to the same centers before any update. [Original address reducer](train-source.jq) · [Reader score order](../../codex-4288/learner/README.md)

## Fresh selector contrast by changed role

Each role has 32 endpoints in 16 selector pairs. Both governors were correct on all endpoints for the arms below. The strongest and all-rival arms elected byte-identical final states by different sequences. The ordered arm retained its original state and its contrast readout matched the ordered base.

| Arm | Changed role | Exact controlled endpoints | Wrong endpoint at the other same-spelling site | Changed selection across peers | Correctly followed the selector on both peers | Full pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Supplied correct | Affirmed | 32 | 0 | 16 | 16 | 16 |
| Supplied correct | Excluded | 32 | 0 | 16 | 16 | 16 |
| Supplied exchanged | Affirmed | 0 | 0 | 0 | 0 | 0 |
| Supplied exchanged | Excluded | 0 | 0 | 0 | 0 | 0 |
| Strongest and all, each | Affirmed | 10 | 10 | 0 | 0 | 0 |
| Strongest and all, each | Excluded | 10 | 10 | 0 | 0 | 0 |
| Ordered all | Affirmed | 8 | 6 | 6 | 2 | 0 |
| Ordered all | Excluded | 8 | 6 | 6 | 2 | 0 |

The first physical pair is the gauge drawer DEFER pair in the [panel record](README.md). The supplied correct control selects site 5 then site 9, with one eligible tuple and bank costs 628 and 604. Strongest and all select `item` at bytes [161,165) and BPE [82,84) on both prompts. Their winning transport sum of 44 beats 50 on both prompts even though each runner has a smaller bank cost. Ordered all selects `the` at bytes [169,172) and BPE [85,86) on both prompts, with winning transport sum 61 against 62. Its winning bank cost changes from 620 to 612 across the selector edit. All three learned arms' choices are unique, with 386 eligible tuples and both signed governors at the correct original addresses. The first wrong occurrence is the affirmed noun at BPE [105,106) in both endpoint arms and [112,113) in the ordered arm.

In the first correctly followed ordered pair, at hall-light DEFER rows 32 and 33, the excluded `sconce` changes from bytes [97,103) and BPE [61,64) to bytes [118,124) and BPE [67,70). The unique joint keys are `[0,89,564]` and `[0,94,560]`. The unchanged affirmed role still selects BPE [107,108) in both prompts, with gold at [97,100). The ordered representation has four such correct role flips overall. The frozen state produced these flips, and every learned arm misses the opposite signed noun on every contrast endpoint. [Held contrast reducer](contrast-readout.jq)

## Replay from the physical records

On Wick, `panel/analysis-v1` carries the candidate census, both physical TRAIN traces, per-role and per-direction held counts and the exact first, first-correct and first-incorrect changed-source witnesses. The readout uses the sealed `measurement-v1`, `held-v1` and `panel/panel-v1` inputs. A second preparation reproduced the full analysis manifest byte for byte. The reduction uses only existing physical captures.

```bash
root=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-all-outranker-interval/panel
(cd "$root/analysis-v1" && sha256sum --strict --quiet -c SHA256SUMS inputs.sha256)
bash "$root/methods-v6/analyze-wick.sh" v1 v1 v3
cmp "$root/analysis-v1/SHA256SUMS" "$root/analysis-v3/SHA256SUMS"
```

| Record | SHA-256 |
| --- | --- |
| `analysis-v1/SHA256SUMS` | `8bb60153e001e8be126d9823241ad4ea1c95d05c7603ef60f56262bd6b870c74` |
| `analysis-v1/train-candidates.json` | `3c7525e310bfba2163d9b9c7d690f74e140c2001c934d0ac581db723a002fe93` |
| `analysis-v1/train-source.json` | `a1de71799d9112670d29f9fb6c8074800c65d2f15e48bf31ab4b3caceb057599` |
| `analysis-v1/by-role.json` | `de9cbc1bae15ab5a0166eb65f9a2506d359820e06941ebcc940a3f2ebe7d3f73` |
