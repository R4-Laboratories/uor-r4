# Joint reader fit on excluded wording families

## What did not work

Codex, 25 November 2026. Research PR #4272 tested whether jointly choosing the two native points for each decision helped a frozen 376-row writer follow a change in instructions. Fresh access required preserving all 16 retained ordinary conversations and making strict two-sided decisions on quoted admission, negated admission, first role and ending. The full joint reader kept 16/16. Quoted and negated admission each scored zero, so the fresh conversations stayed unopened. The signed reader remains the default and M2 remains open. [Research report](https://github.com/UOR-Foundation/uor-r4/blob/d44d208761f8f73eac776c06bb94c8b940471a35/docs/loop/tasks/m2-train-grounded-replies/codex-4272/REPORT.md)

## What changed

The M2 notebook records the joint fit and the matched independent comparison. All rows in the held wording family were excluded together, across both neutral placements, both labels and every fact group. The development folds reused the same fact vocabulary.

## Measurements

In 512 ordinary conversations on Wick with genuine generated assistant history, 256 asked for the task and 256 required deferral. The signed, same-pool independent and joint readers completed 60, 32 and 52 original-source-authenticated replies through EOS respectively. On the 256 prompts requiring deferral, 0, 72 and 56 replies matched the stock parent through EOS. Of those, 0, 48 and 24 explicitly selected DEFER. The joint reader scored 12/128 strict opening flips and 8/128 strict ending flips. The independent comparison scored 4/128 and 8/128. Moving the neutral clause from before to after the command shifted the joint opening count from 12 to 0 per 64 pairs and its ending count from 0 to 8. [Whole-family counts](https://github.com/UOR-Foundation/uor-r4/blob/d44d208761f8f73eac776c06bb94c8b940471a35/docs/loop/tasks/m2-train-grounded-replies/codex-4272/eval/OOF.md)

The joint fit recovered 38 grounded replies and lost 18 relative to the independent one. On one job/home pair the independent fit continued at token 34 instead of emitting EOS in both replies, while joint was exact on both. On one negated pair the independent reader was exact on both sides. Joint emitted “ You” where EOS was due at token 26 on the active instruction. Its negative twin failed on token 1. Across the 204 inexact grounded replies under joint, 118 first failed at token 1. [All first wrong tokens and original sources](https://github.com/UOR-Foundation/uor-r4/blob/d44d208761f8f73eac776c06bb94c8b940471a35/docs/loop/tasks/m2-train-grounded-replies/codex-4272/eval/oof-first-errors.tsv)

## Checks run at the head

The signed D11, training material, ordinary captures and verified archive live on Wick ZFS. The occupied physical Wick accelerator had its own fit and verification queued when the archive sealed. A leased RTX 4090 ran the numeric fit and a separate lease on that same GPU replayed 17,237 learned writer steps from 2,112 reached turns. All steps matched. The private 67,361-byte packet reconstructed all 18 D11s and both full arms reproduced every retained raw conversation after download. [Pinned artifacts and primary Wick receipt paths](https://github.com/UOR-Foundation/uor-r4/blob/d44d208761f8f73eac776c06bb94c8b940471a35/docs/loop/tasks/m2-train-grounded-replies/codex-4272/RESTORE.md)

## Skills used

The native-space cycle guided artifact identity and ordinary replay checks. The UOR research architect guided the distinction between geometric double wins, observed physical copy and complete replies.

## Still open

How many held admission pairs can a single pair of the allowed training points separate at *both* neutral-clause positions? An oracle diagnostic on development families can give the ceiling for this representation without entering serving. That measurement would distinguish a gap in point selection from a gap in the represented instruction-to-clause relation. [Registered study and fit objective](https://github.com/UOR-Foundation/uor-r4/blob/d44d208761f8f73eac776c06bb94c8b940471a35/docs/loop/tasks/m2-train-grounded-replies/codex-4272/ORDER.md)

## Plain summary

The writer can repeat the right facts once it gets the right instructions. A clause that quotes or rejects an instruction still sends it down the wrong path on unseen wording.
