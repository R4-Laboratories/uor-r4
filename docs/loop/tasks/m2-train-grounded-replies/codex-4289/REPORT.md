# Far source attachment improved and lost the inherited near pairs

## What did not work

The updated reader lost all eight near pairs the inherited reader won. On the fresh held panel the updated reader went from 16 to 48 correct far pairs out of 64 and from 8 to 0 correct near pairs out of 64. Its net gain was 24 pairs. The registered release required near retention, a gain over native and empty readers at each distance, and a full signed-link pair at both distances. All three near criteria failed. **Fresh ordinary first wrong reply tokens and exact replies through EOS are NOT_RUN.** [Registered gate](PRE-RUN.md) · [Held results](evidence/source-gate.json)

## What changed

Wick trained only the two existing banks of eight 32-coordinate i16 governor-to-noun prototypes. The governor reader, original word and tuple proposals, tokenizer, D11, joint ranker and ordinary writer were fixed. Both fitted arms started at the same 13,336-byte parent state and used the same eight integer sweeps and TRAIN-only state selection. Each eligible edge attracted its nearest center toward the correct noun when its margin was below one. The contrastive arm also pushed the nearest center away from the currently cheapest wrong occurrence. The wrong set contained every admissible original occurrence in that sign’s native tuple slot. The cheapest wrong occurrence was recomputed after each sweep. A positive margin means the gold source beats **every** wrong occurrence. [Update](fit/README.md) · [Run and state identities](evidence/run.json)

The training set contained 128 fresh far pairs from eight wording families, with 9–16 intervening BPE IDs. The independent held set contained 64 near pairs at 1–3 IDs and 64 far pairs at 10–17 IDs across four new wording families. The two members exchanged only the signed governors. Each prompt contained another occurrence of the same noun. All eight held target nouns were fresh in the audited text comparisons. Four familiar verb pairs, the conversation protocol, and the general instruction scaffold were shared. Fact set and requested reply length co-vary. [Panel](panel/README.md) · [Independent FIT](FIT.md)

## Measurements

All 512 signed training edges were eligible. The frozen reader started with 256 strict losses, eight ties and 248 wins. Its strongest wrong occurrence beat the gold by 1–24 integer L1 units on the losing edges. The contrastive fit reversed every one, including all 124 affirmed and 132 excluded losses. Those 256 finished 1–21 units ahead of the current strongest rival. It also resolved all eight ties and kept all 248 previous wins. Of the 256 original losses, 152 had a different strongest rival after training. None had a remaining rival tied with or ahead of the gold. The total unit-margin loss fell from 3,120 to zero at sweep 6. [Contrastive transitions](evidence/contrastive-losses.json) · [Per-occurrence movements](evidence/movements.json)

| Status before the fit | Edges | Contrastive result (loss / tie / win) | Attraction-only result (loss / tie / win) |
| --- | ---: | ---: | ---: |
| Strict loss | 256 | 0 / 0 / 256 | 152 / 24 / 80 |
| Tie | 8 | 0 / 0 / 8 | 8 / 0 / 0 |
| Strict win | 248 | 0 / 0 / 248 | 12 / 0 / 236 |

The original strongest rival in each of the 256 strict losses was a different word from the target. The most frequent were `item` (64), `the` (48), and `measure`, `we` and `work` (32 each). The remaining 48 losses were spread across six other words. The same-spelling rival stayed in the candidate set. In one checkpoint example the required link was `clear` → `ticket` at original BPE [118,119). The reader preferred `the` at [106,107) by five units. After training `ticket` won by 13. Attraction alone elected sweep 8 with loss 1,524. It flipped 36 affirmed and 44 excluded losses. On the training set the unchanged joint ranker won 64/128 pairs with the parent state, 72/128 with attraction alone and 128/128 with repulsion. [Attraction-only transitions](evidence/attraction-losses.json) · [State trajectories](evidence/run.json)

A held pair requires both prompts to select the exact affirmed, excluded and later-reference occurrences in original bytes and absolute BPE. Every correct learned pair also carried both correct governor addresses on both prompts.

| Frozen writer, source arm | Near pairs / 64 | Far pairs / 64 | Total pairs / 128 | Exact prompts / 256 |
| --- | ---: | ---: | ---: | ---: |
| Native D11 | 4 | 0 | 4 | 12 |
| Unfitted reader | 0 | 0 | 0 | 0 |
| Learned governors with gap alone | 0 | 0 | 0 | 0 |
| Inherited transport | 8 | 16 | 24 | 56 |
| Attraction only | 8 | 16 | 24 | 72 |
| Strongest-rival update | **0** | **48** | **48** | **112** |
| Correct supplied links | 64 | 64 | 128 | 256 |
| Exchanged supplied links | 0 | 0 | 0 | 0 |

The strongest-rival update kept all 16 inherited far wins and added 16 harbor-records, eight civic-hearing and eight repair-school pairs. Each of these families finished 16/16 far. Reading-room stayed 0/16. The eight lost near pairs all came from repair-school. Attraction alone kept exactly the same winning pairs as the inherited reader at both distances. Across the new near and far prompts the updated reader still selected both correct governors 256/256 times, and every winner was unique. It selected the neutral same-spelling occurrence in 44 near prompts and zero far prompts. [Held gate by source arm](evidence/source-gate.json) · [Named pair movements](evidence/movements.json)

The failure is visible at an original address before generation. In all eight lost near repair-school pairs, the TAKE half retained all three sources and the DEFER half moved the affirmed link from `handbook` to the immediately preceding `this`. The excluded noun and later reference stayed correct. In pair `glass/16`, the source moved from BPE [98,101) to [97,98). At far distance in repair-school pair `glass/18`, both prompts recovered `handbook` from an earlier choice of `steps`. The remaining far errors were 16 wrong affirmed nouns and eight wrong excluded nouns, all in reading-room. Eight near prompts had a correct full triple, each paired with a failing mate. [Original address traces](evidence/movements.json) · [First source errors](evidence/source-errors.json)

## Checks run at the head

The lab verified the source-built native-space rule and independently checked the release before training. All 11 corrected physical fixture assertions passed. The grade fixture tested 512 raw signed edges and 15 paired release cases, including beating a prior rival while a new one still wins or ties. An earlier synthetic fixture failed its summary syntax and contained noncanonical inactive test bytes. Both failures were preserved and corrected before the measured collection. [Physical fixture](evidence/device-fixture.json) · [Grade fixture](evidence/grader-fixture.json) · [FIT](FIT.md)

On Wick, the independent Rust verifier reconstructed the complete elected and final states and the frozen lexical and unused bytes. For each of the two real fits it matched 4,608 signed per-sweep reads and 19,072 wrong original candidate records with zero disagreements. Both repeated fits and cold original-address decisions matched. The untouched #4288 ranker and separate blind checker agreed for the inherited, attraction-only and contrastive states on both splits. A second instrument reproduced all 768 native and supplied held address comparisons. The physical collection `measurement-v2` took 10.59 seconds under the RTX 4090 lock, at zero rental cost. A preceding occupied lease ran no measurement. The grade is `grade-v1`. The private archive restored 101 files, and the reconstructed margins, ranks, gate and error traces matched the originals byte for byte. [Run receipt](evidence/run.json) · [Control comparison](evidence/control-parity.json) · [Restore](RESTORE.md)

## Skills used

The native-space cycle guided the frozen-state, integer update and exact replay checks. The research-architect skill guided the M2 source dependency and the conditional release to the ordinary writer.

## Still open

Keep the governor reader, joint ranker and writer fixed. The next experiment should train a single transport state on fresh matched near and far examples. Include near training pairs that the inherited reader already wins and far pairs it loses. Compare the inherited and far-only states with a mixed update using the same budget. Track whether the near and far corrections ask the same center coordinates to move in opposite directions. Select the state on TRAIN with an explicit near-retention condition, then test unseen wording at both distances, including a new family with the structure that remained unresolved here. The current transfer panel has been observed. Reserve a new panel for the decision. [Grounded-reply milestone](https://github.com/UOR-Foundation/uor-r4/issues/2030)

## Plain summary

The same frozen reader can recover far original noun addresses after training against its own wrong choices. Far-only training also displaced the short-distance choices it previously got right. The next question is whether one set of native coordinates can preserve both behaviors well enough to test the writer on new conversations.
