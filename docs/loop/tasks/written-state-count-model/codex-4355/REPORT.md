# Correcting every rival fixes the current word and exposes shared-weight erasure

## What did not work, first

[codex] Both policies failed the shared-checkpoint gate at both seeds. The full geometric correction won all 3,888 structurally attainable admissions immediately in every epoch. At epoch eight only 22 and 36 of these wins survived. The sparse policy raised only the correct token's address words. It made zero geometric updates in either final epoch, so its 100% retention refers to already-correct rows. The registered group-disjoint refit remained unopened. This experiment measured the first lexical choice on supplied human TRAIN prefixes. It produced no new exact-through-EOS or open-ended chat score. [Gate](SHEET.md) · [Independent recount](evidence/recount.json)

## What changed

[codex] The old policy corrects one rival at a time. The new policies compute the minimum maximum individual i16-word displacement needed to beat every other original token. They use the same frozen P4, address operators, 584-byte attachment, writer, 3,891 training targets and two eight-epoch schedules. Every real mutation is followed by a new 4,095-ID device election. When the reachable box cannot win, both new policies leave all words fixed. The target is an input to the training update. The ordinary client receives the causal prefix and bound learned words. [Integer update](runner/src/full_election.cu) · [Device and host controls](runner/src/device.rs)

## Measurements

[codex] The table uses the same 3,886 rows from 3,703 conversation groups that are strictly attainable under both address operators. The last column counts wins at the end of training using a single shared state. An immediate win uses the state immediately after the corresponding row's visit. [Epoch evidence](evidence/epochs.json)

| Epoch-eight geometric arm | Seed | Immediate strict wins | Retained from those wins | Final strict wins |
|---|---:|---:|---:|---:|
| Previous single-rival | 435300 | 310 | 59 | 95 |
| Raise target words only | 435300 | 122 | 122 | 122 |
| Full correction | 435300 | 3,886 | 22 | 22 |
| Previous single-rival | 435301 | 319 | 38 | 52 |
| Raise target words only | 435301 | 116 | 116 | 116 |
| Full correction | 435301 | 3,886 | 36 | 36 |

[codex] The frozen prior won 80 common rows at both seeds. Matched ordered won 123 and 87 with target-only updates, and 74 and 28 with the full correction. Full geometry retained 0.57% and 0.93% of common immediate wins, against the 80% requirement. Relative to matched ordered its checkpoint difference was −1.34 and +0.21 percentage points; the paired conversation-group 95% intervals were [−1.75, −0.93] and [−0.03, +0.44]. The registered endpoint demands at least +5 points against all three controls. Across all 3,891 admissions, the old strongest-rival updates reproduced all 32 previous epoch records and the independent projection matched each recorded historical geometric checkpoint decision, ceiling and zero election. Full matched ordered achieved its own structural ceiling of 3,889 immediate wins every epoch. [Historical projection](evidence/history-projection.txt)

[codex] Under the full geometric policy in the final epoch, 3,837 and 3,855 row visits made updates. Across each epoch, those updates changed individual words 38,801 and 39,021 times, with total absolute i16 displacements of 364,382,388 and 365,701,550. The target-only policy could not force a win on 3,769 and 3,775 admissions from its current words. On the original TRAIN ordinal 0, the full update moved the correct ID 93 from rank 737 to 1 and from 311 to 1. It later fell to rank 166 behind ID 166 at seed 435300 and to rank 335 behind ID 686 at seed 435301. [First-token trace](evidence/sentinel.json)

[codex] A post-run inspection of the 104 reachable words found 93 and 91 at −32,768 in the two full geometric checkpoints. Neither checkpoint had a word at +32,767. Target-only instead put 62 and 60 at +32,767 and none at the lower limit. Each full geometric bank had between 33,194 and 55,789 integer steps of unused range beyond the spread between its smallest and largest active word. Those counts establish one-sided saturation. They leave open the respective contributions of saturation and conflicting prompts. [Bank limits](evidence/rails.json) · [Active ranges](evidence/bank-ranges.json)

## Checks run at the head

[codex] The six standalone arithmetic tests included an exhaustive two-bank/two-slot reference. Twelve host grading tests passed. The sm_89 physical finite suite covered full replacements, both policies, strict vs tied original IDs, clipping, the maximum 65,535-word move, a fractional Q48 predecessor and subsequent erasure. The earlier P4 and retention CUBINs reproduced byte for byte. The independently recounted 249,024 visits and all 32 new epoch summaries matched. Nine ordinary client invocations on a single pinned TRAIN prefix verify the committed winner, score, prior, ties, addresses and applicable carry for zero and eight fitted heads. Each fresh process took 17.21–17.32 s on Wick, with peak RSS between 3,887,456 and 3,888,244 KiB. [Prospective FIT](FIT.md) · [Cold receipt](evidence/cold.json) · [Reproduction](REPLAY.md)

[codex] The instrument took 40.95 s of process wall time, including 26.45 s loading the base. Fitting the old and two new policies across both seeds and both address schemes took 10.22 s in total. Peak one-second device memory was 4,314 MiB. The ledger recorded 294 s at $0.869/hour, approximately $0.071 at the listed compute rate. All 30 result and job members verified on Wick before the rental was deleted. The private archive restored all 114 checksummed members on Wick and its forced immutable download matched the compressed SHA-256. [Run timings](evidence/run.json) · [Immutable restore](evidence/restore.json)

## Skills used

[codex] The native-space cycle governed the bounded integer updates, shared prediction code, unchanged export layout and ordinary-client witness. The fresh harness selected one D85 contract rule and passed. [Native-space specification](../../../../harness/native-space.json)

## Still open

[codex] First test the numeric limit exposed here. Add the same integer offset to every reachable word of each bank so that the bank's current range is centered, preserving i16 values. Every candidate uses one address per bank. Its score therefore changes by the same sum of bank offsets, in score units, and the original-ID ranking is invariant before another update. Verify the common shift, every pairwise score difference, original-ID tie order and cold commit before comparing recentered versus ordinary updates. The change supplies headroom without adding learned words. Whether that headroom helps later shared predictions requires measurement. If erasure persists, test pairs of actual prompt constraints for incompatible required differences on the same words before expanding model capacity.

## Plain summary

[codex] The ordinary client reproduced the saved first-token elections. During training the update put the correct token first on every individually attainable opening. The shared weights lose nearly all those corrections as other prompts train. The next test can distinguish an integer-range problem from a more fundamental conflict over which words the prompts share.
