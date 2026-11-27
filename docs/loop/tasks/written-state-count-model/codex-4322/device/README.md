# First-period margin interventions

Build on the registered sm_89 GPU pod with nvcc, ptxas and the CUDA Driver API library. The build checks the source digests for the original choice implementation and geometric instruction enumerator. It preserves the original ABI, scoring and commit entries, and audits the generated PTX and SASS.

```sh
device/build-image.sh balanced NEW_BALANCED
device/build-image.sh weak NEW_WEAK
device/build-image.sh both NEW_BOTH
NEW_BALANCED/finite ORIGINAL_CHOICE_CUBIN baseline
NEW_BALANCED/finite NEW_BALANCED/choice.cubin balanced
NEW_WEAK/finite NEW_WEAK/choice.cubin weak
NEW_BOTH/finite NEW_BOTH/choice.cubin both
```

Each of the eight balanced first-period passes alternates STOP then COPY across exactly one slot per source row. It starts each class cursor at `epoch * (row_count / 2) mod class_count` and cycles through the original record order for that class. The 768-row scientific input therefore uses 384 slots per class per epoch. The weak-member pass keeps the original pair order. It updates the member with the lower signed margin when that margin is below eight, including its bias. For a tie it selects STOP when epoch plus the zero-based opposing-pair ordinal is even, and COPY when that sum is odd. Stages after the first period follow the original update sequence in each image.

The finite program loads the supplied cubin and compares every weight word, raw score and audit counter with the planted scalar result for that arm. It uses two opposing pairs at each of the first two stages, an imbalanced first-period stream, a deliberate feature alias and a shared feature that can make the pair gap and signed-margin conditions disagree.

Use the frozen #4321 driver to fit each arm from zero. Supply the frozen #4321 image to every scientific replay, together with the fitted arm's native words.
