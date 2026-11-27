# First-token copy and the self-fed successor

## What did not work

The learned reader chose the wrong position on 411/800 answerable questions and copied on 524/600 unsupported prompts. Of its 389 correct first words, 338 were followed by the wrong next token.

## What changed

The ordinary client has an opt-in source commit that verifies the original occurrence before top-eight truncation. It uses the existing reader weights and consumes its single copy action at the opening. [Ordinary product](../codex-copy-id/product/README.md)

## Measurements

Of 400 question pairs, Generate, learned Copy and supplied correct Copy respectively got both openings right on 0, 133 and 400 pairs. The corresponding two-token results were 0, 8 and 33 pairs. [Full report and negative controls](../codex-copy-id/REPORT.md)

## Checks run at the head

The two GPU traversals matched. Cold Rust reproduced all 21,976 observed transitions. The eight same-ID wrong addresses were refused. The [GPU receipt](../codex-copy-id/evidence/device-replay.json), [native client fixture](../codex-copy-id/product/CLI-qualified.json) and [private archive replay](../codex-copy-id/evidence/storage-replay.json) bind the checks to the measured inputs.

## Skills used

The native-space-cycle skill guided integer operator identity, original-ID preservation and cold replay.

## Still open

The learned reader reached 66/200 four-way openings against the 113/200 threshold. Instruction-conditioned continuation and unsupported reads have separate [follow-up tests](../codex-copy-id/PICKUP.md).

## Plain summary

The copy writes a selected word faithfully. On fresh swaps it exposed both the reader's wrong addresses and the ordinary decoder's difficulty choosing punctuation or EOS after a correct word.
