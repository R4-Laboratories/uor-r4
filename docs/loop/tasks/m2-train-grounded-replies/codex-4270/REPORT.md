# Fixed-writer instruction windows on Wick

## What did not work, first

The learned reader correctly distinguished an affirmative request from its quoted twin on 0/64 pairs and from its negated twin on 0/64. All nine configurations completed both sides of 0/128 three-versus-four-sentence pairs. Both new entry readers retained only 3/16 earlier exact replies. On the ending task the stop and continue class points in both role banks were identical. They selected the three tokenizer pieces for “sentence”. [Learned points and wording controls](https://github.com/UOR-Foundation/uor-r4/blob/65fb7f708c71f4a94e4724c66b898468ffde0d02/docs/loop/tasks/m2-train-grounded-replies/codex-4269/diagnosis/README.md)

## What changed

One notebook row records the result of [research PR #4269](https://github.com/UOR-Foundation/uor-r4/pull/4269). The 376 signed action/source rows, exact writer, copy cursor and retry policy were fixed for the experiment.

## Measurements

The ordinary client ran 512 fresh two-turn conversations on Wick in each arm, half of them grounded commands. It generated the actual first assistant turn. The four held wording families were excluded from fitting. On valid commands, exactness required all output tokens through EOS and all original copy addresses. [All arm and cohort counts](https://github.com/UOR-Foundation/uor-r4/blob/65fb7f708c71f4a94e4724c66b898468ffde0d02/docs/loop/tasks/m2-train-grounded-replies/codex-4269/evidence/ordinary.json)

| Result | Previous reader | Ordered window pair | Permuted window pair |
| --- | ---: | ---: | ---: |
| Exact grounded replies / 256 | 64 | 48 | 68 |
| Correct first role and original first copy on both sides / 128 | 28 | 8 | 20 |
| Both role-reversal replies exact / 128 | 0 | 4 | 4 |
| Earlier exact replies retained / 16 | 16 | 3 | 3 |

Permutation moved the first wrong token later on 72 valid commands, earlier on 32, and left 152 unchanged or still exact. In one work-first example, the ordered arm opened with “ You” and read its first value token from home address 50. Permutation chose “ Your” and job address 37, then continued at output offset 25 where EOS was required. Changing a single prompt's ending search between all starts and the left or right edge changed zero returned replies or writer steps within either tested entry family. [Case-level tokens and addresses](https://github.com/UOR-Foundation/uor-r4/blob/65fb7f708c71f4a94e4724c66b898468ffde0d02/docs/loop/tasks/m2-train-grounded-replies/codex-4269/evidence/first-wrong.tsv) · [Ordinary examples](https://github.com/UOR-Foundation/uor-r4/blob/65fb7f708c71f4a94e4724c66b898468ffde0d02/docs/loop/tasks/m2-train-grounded-replies/codex-4269/readback/RESULT.md)

## Checks run at the head

The comparison used Wick's 4090 for the geometric fits and masked score checks. A separately observed RTX 4090 reproduced 93,552 learned writer steps across all 7,608 reached Wick turns. The additional full trajectory replay on Wick's own 4090 was queued when the archive sealed. The rebuilt ordinary client matched all 4,352 traced captures. The verified restore packet reconstructed seven D11s and reproduced both sides of a saved role-order pair. [Device receipt](https://github.com/UOR-Foundation/uor-r4/blob/65fb7f708c71f4a94e4724c66b898468ffde0d02/docs/loop/tasks/m2-train-grounded-replies/codex-4269/evidence/supplemental-device.json) · [Wick client verification](https://github.com/UOR-Foundation/uor-r4/blob/65fb7f708c71f4a94e4724c66b898468ffde0d02/docs/loop/tasks/m2-train-grounded-replies/codex-4269/evidence/rebuilt-client.json) · [Archive and restore](https://github.com/UOR-Foundation/uor-r4/blob/65fb7f708c71f4a94e4724c66b898468ffde0d02/docs/loop/tasks/m2-train-grounded-replies/codex-4269/RESTORE.md)

## Skills used

The native-space cycle and the UOR research architect guided the source-address and replay checks in the measured unit. This publication cross-checked its table with the frozen numerical receipts.

## Still open

M2 remains open. Keep the previous reader as the default. The next discriminating fit should reward both directions of each matched TRAIN pair and test a movable clause locator when neutral text follows the instruction. Require transfer on excluded wording families and retention of the earlier exact replies before another fresh held run. [Full experiment and next fit](https://github.com/UOR-Foundation/uor-r4/blob/65fb7f708c71f4a94e4724c66b898468ffde0d02/docs/loop/tasks/m2-train-grounded-replies/codex-4269/REPORT.md)

## Plain summary

Choosing the right job or home can let the writer copy the right fact and still leave it unable to tell when the user asked it to stop.
