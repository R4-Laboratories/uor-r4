# Learning which noun the governor acts on

## What did not work

The fitted reader won **8/128 held source pairs**. All eight came from one near-distance wording family. It found both signed governors in every held prompt but failed to attach both to their correct nouns on every far prompt. The preregistered gate required a paired improvement at both distances. **First wrong reply token and exact replies through EOS are NOT_RUN** on this panel. The ordinary writer was held fixed throughout. [Source gate](evidence/source-gate.json) · [Registration](PRE-RUN.md)

## What changed

We trained from the current user’s text, with the signed verb and noun intervals from the 256 exposed #4286 examples as training targets. Each sign learned four lexical and eight directed transport prototypes with an integer Manhattan-distance fit on Wick. The serialized reader state occupies 13,336 bytes. During held inference the learned arm received the user tape represented with the fixed D11's roots, the complete native proposal set, the saved banks and the fixed D11 scope banks. It considered every addressable word as a governor. It received no annotation, answer, list of candidate nouns from the evaluator, or paired prompt. Correct and exchanged controls received a separate signed-key and unsigned-edge packet. [Native learner](learner/README.md) · [Bank isolation](evidence/state.json)

The 128 fresh pairs vary governor direction, distance, the placement of the exact same-spelling rival and first requested role. Each pair exchanges only the two governing verbs. Training and evaluation share the protocol, generated first-turn history, general two-clause task scaffold and eight verb spellings. The independent held author introduced four wording families, 32 templates, eight target nouns and four fact values absent from training in their respective comparisons. Fact set and required reply length co-vary. This design measures attachment of familiar governors in fresh contexts. [Panel and split](panel/README.md) · [Independent FIT](FIT.md)

## Measurements

A source win requires the affirmed noun, excluded noun and later reference to have the exact original raw-byte and absolute BPE intervals. Both members must win for a pair to count. A full-link win also requires the exact two signed governor intervals and the two corresponding noun endpoints. Every trained source win on this panel also passed that stronger test. [Sources, ties and pair transitions](evidence/source-gate.json)

| Wick source arm | Near pairs / 64 | Far pairs / 64 | All pairs / 128 | Exact single prompts / 256 |
| --- | ---: | ---: | ---: | ---: |
| Native D11 | 0 | 0 | 0 | 4 |
| Unfitted reader | 0 | 0 | 0 | 0 |
| Learned governors with BPE gap alone | 0 | 0 | 0 | 0 |
| Learned governors and directed transport | **8** | **0** | **8** | **16** |
| Training with only noun targets exchanged | 0 | 0 | 0 | 0 |
| Training with both signs exchanged | 0 | 0 | 0 | 0 |
| Correct supplied links | 64 | 64 | 128 | 256 |
| Supplied links with noun endpoints exchanged | 0 | 0 | 0 | 0 |

Training yielded 128/128 full source pairs. On held text the fitted reader made 256 unique decisions, with both governors right in all 256. It selected the wrong affirmed noun on 156 prompts. On 84 more it got the affirmed noun right and missed the excluded noun. These two groups account for all 240 failures. In 24 failures the chosen triple included the neutral same-spelling rival. The other 216 failures included different distractors. At far distance the reader got each individual signed noun right on 40/128 prompts and never got both right on the same prompt. The supplied control reached the right triple on every one of those prompts. [Training scores](evidence/training.json) · [Held source errors](evidence/source-errors.json)

Two original-address traces illustrate the distinction. Intervals are half-open BPE positions on the original conversation.

| Held case | Verified governors | Required read | Actual read |
| --- | --- | --- | --- |
| Near editorial TAKE | `accept`, `reject` | Affirmed `brief` at [61,64), excluded `draft` at [91,93) | Another `brief` at [80,83), then `you` at [93,94) |
| Far auction TAKE | `clear`, `block` | Affirmed `bid` at [61,63), excluded `deal` at [100,102) | Correct `bid` and later heading, but `you` at [102,103) as the excluded noun |

The learned attachment gained eight complete pairs over the same governors attached by proximity and retained every native winning pair. Exchanging only the training noun targets preserved every signed governor decision and eliminated those eight wins. Both distance and wording-family coverage matter here. All eight gains belonged to the editorial family. [Per-family and per-distance results](evidence/source-gate.json)

## Checks run at the head

A synthetic test caught an earlier CUDA build reporting the affirmative endpoint in the excluded slot. That build ran no measured examples. On the repaired build all 14 finite assertions passed, three 13,336-byte fit variants matched the independent state reconstruction, and the same initial state produced byte-identical fit, cold-reload and repeat predictions. The trained lexical banks were byte-identical under noun-target corruption while both transport banks changed. [Fixture outcomes](evidence/device-fixture.json) · [FIT and retained negative](FIT.md)

On the actual prompts, the previous independent ranker and its Rust reduction agreed on 768 native and supplied original-address comparisons. A separate Rust checker, which had no labels, reproduced all fields of the physical trained and gap decisions across both splits, including the 1,024 unique winners, both signed links, ties and rival costs. The same checker found 126 differing fields in the superseded finite capture. The registered physical collection took 6.01 seconds including repeat fits and controls under Wick’s RTX 4090 lock. Preparation, synthetic checks and verification have separate receipts. Rental cost was zero. [Control replay](evidence/control-parity.json) · [Held replay](evidence/held-replay.json) · [Training replay](evidence/train-replay.json) · [Restore](RESTORE.md)

## Skills used

The native-space cycle defined the shared integer update, lossless bank serialization and exact replay checks. The research-architect skill guided the M2 decision, held wording split and conditional writer release.

## Still open

Keep the signed governor bank and the writer fixed. Train the directed edge against the best *wrong original occurrence on the same prompt*. Build new far training pairs in which the same spelling serves as the governed noun in one example and an incidental mention in another. Include intervening words and check that the present reader actually loses on this training material before spending on updates. The earlier within-prompt margin experiment updated whole-scope banks and found its training comparisons already satisfied. This comparison would update only the governor-to-noun transport, require a measured change to its wrong-versus-correct score, and reserve new wording families for the near/far test. Novel verb meanings and unassisted role order and completion remain separate open questions for M2. [Earlier margin experiment](../codex-4282/REPORT.md) · [Grounded-reply milestone](https://github.com/UOR-Foundation/uor-r4/issues/2030)

## Plain summary

The reader learned where to find the eight familiar verb spellings. It sometimes connected a nearby verb to the right original noun in new wording. Every long-distance pair still broke at the noun attachment stage, before a writer token was emitted.
