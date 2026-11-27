# Where the first wrong token originates in two frozen checkpoints

## What did not work

Neither learned checkpoint completed all four instructions on any of the 64 fresh factual contexts. The content-link checkpoint completed 104/256 scoped replies. The balanced baseline completed 128/256. Supplying the correct action after B recovered some linked replies, but a later decision cannot repair a reply that already emitted EOS or copied an unwanted word at token 3. Both artifacts received four valid factual addresses, the opening word A and its period.

## What changed

Only the diagnostic release selection changed. The first intervention supplied the correct stop/copy action immediately after the supplied period. The post-B intervention ran only if the request included B and the writer actually copied B from its original address after the correct prefix. The combined arm allowed both interventions. Every subsequent action, including after C, remained the checkpoint's own. The serving client received the protocol tokens and four addresses, with the reference confined to grading and the diagnostic oracle. [Pre-registration](SHEET.md) · [Device transitions](device/README.md) · [Independent integer replay](product/README.md)

## Measurements

Each fresh context has four versions over the same facts and question. Switching the operative clause or changing which clause asks for one word flips the requested answer length. Each axis contains 128 matched edges over the same 256 scoped replies. A quartet succeeds only when all four replies actually emit the reference through EOS.

| Checkpoint and release supplied | Exact scoped replies / 256 | First wrong at token 3 | First wrong at token 4 | Correct B reached / 128 required | Clause-order pairs / 128 | Selector pairs / 128 | Whole contexts / 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced, learned | 128 | 128 | 0 | 8 | 0 | 8 | 0 |
| Balanced, first | 184 | 0 | 72 | 128 | 56 | 56 | 24 |
| Balanced, post-B | 128 | 128 | 0 | 8 | 0 | 8 | 0 |
| Balanced, both | 256 | 0 | 0 | 128 | 128 | 128 | 64 |
| Linked, learned | 104 | 112 | 40 | 64 | 8 | 8 | 0 |
| Linked, first | 184 | 0 | 72 | 128 | 56 | 56 | 24 |
| Linked, post-B | 144 | 112 | 0 | 64 | 16 | 16 | 0 |
| Linked, both | 256 | 0 | 0 | 128 | 128 | 128 | 64 |

For the linked checkpoint, fixing only token 3 made 80 replies exact and exposed a token-4 error on another 32. Forty replies had already reached the correct B and then made the wrong choice. The balanced and linked states use identical learned words at the later releases, so correcting the early decision brought both to the same 72 remaining failures. The balanced model's apparent zero native token-4 errors hides that it reached B on only eight of the 128 replies that required it. [Linked counts and wording families](evidence/candidate-fresh.json) · [Balanced counts](evidence/balanced-fresh.json)

The learned linked wins on each axis occurred in the check-mark family. Post-B treatment added eight matched edges on each axis in the instruction-aside family, with no complete quartet gain. On 40 of 64 contexts, neither single intervention completed the quartet and both were required. The balanced model’s eight learned selector-edge wins came from a different family, rule elimination.

The fresh direct controls add 16 one/two-word pairs and 16 two/three-word pairs. The learned linked model completed zero of the former and all 16 of the latter. The learned balanced model completed zero in both groups. Supplying the first choice alone brought both models to 32/32 direct pairs, including the learned exit after C. The scope experiment uses eight authored wording families repeated across eight factual layouts. Token counts matched within all 128 clause-order reversals and 96/128 selector reversals. These paired edges overlap. [Panel design and seal](panel/README.md)

The same facts give a concrete trace. On folio RF0500, “Instruction: read lines one and two. Aside for my notes: read line one alone.” required `thunder. care` and EOS. The linked model scored +9 at the period and copied `care` (ID 690) from address 34. At B it scored +1 and copied `goat` (ID 2942) from address 50 instead of EOS (ID 1), making token 4 the first error. Reversing those two sentences left the requested reply unchanged, but the period score fell to −45 and the model emitted EOS at token 3. Forcing just the first decision restored `care`, then exposed the same +1 mistake at token 4. Supplying both decisions gave the correct four IDs `[3912, 16, 690, 1]`. [Fresh paired trajectories](RESTORE.md)

The historical scoped diagnostic reproduced both stage-cut predictions on the same IDs. Post-B alone took the linked checkpoint from 8 to 24/224 exact DEV pairs and from 40 to 56/224 previously evaluated HELD pairs. First-only reached 128 and 160 respectively. Both reached 224 on each. These old panels calibrate causal attribution; the 64 new contexts measure transfer. [Earlier DEV replay](evidence/candidate-dev.json) · [Earlier HELD replay](evidence/candidate-held.json)

On the protected panel, both models retained the original 50 pair identities with every treatment. The linked checkpoint lost all 75 previously passing balanced first-period DEV pair identities and all 32 direct-count DEV pair identities. The post-B treatment recovered zero of those identities, while first-only recovered every one in both cohorts. The two models also shared the same 50/100 earlier passing two/three-word DEV pairs; correcting post-B took both to 100/100. [Identity joins](evidence/retained.json)

## Checks run at the head

The leased RTX 4090 replayed 2,044 prompts for each of two frozen checkpoints, four arms per prompt, in forward and reverse orders. Scientific execution took 144.46 seconds. Total rental lifetime was 9 minutes 48 seconds from creation to confirmed deletion. The advertised $0.869 per hour implies about $0.14 in compute charges before any provider extras. The measured results were copied to Wick ZFS and checked against the device hashes before the rental was deleted. [Device manifest](evidence/device-output.sha256)

Both frozen GPU images passed 12 planted transitions, including a duplicate-token wrong-address case, and the host passed nine reachability gates. The independent packed-D11 path matched 62,474 arm transitions in each device direction. The fresh ordinary serving processes matched all 16,157 learned transitions across 4,088 model–prompt combinations. Every learned transition matched the earlier raw history on the 1,724 archived requests per checkpoint, including the old paired identities. Downloading the sealed research packet reproduced the fresh ordinary client and diagnostic records on both artifacts. [Signed FIT](FIT.md) · [Finite controls](evidence/finite.json) · [Client parity](evidence/parity.json) · [Restore instructions](RESTORE.md)

The focused integer boundary tests passed 2/2, as did the device host tests. Building the full integer library test target with the additional P1 serving feature hit existing unresolved imports in the unrelated `stack/tests/pair_rows.rs` tests. The release binaries with that feature built and ran through the normal client.

## Skills used

The native-space-cycle skill guided verification of the persistent integer state, the physical source identity, and the original native writer. All scientific scoring used one rented RTX 4090. The ordinary-client checks used integer serving on Wick.

## Still open

Train the release choices jointly and protect earlier direct requests by identity. This experiment supplied the opening and factual sources. The product still needs to find its own sources, generate its own opening, and sustain useful answers on questions outside these authored reply shapes. [Next fit and product test](PICKUP.md)

## Plain summary

The early reader can end a reply before the second word, and the later reader can continue after the right second word. Both errors occur on the same fresh instructions. The frozen writer can finish all 64 controlled contexts when the two choices are supplied. Learning when to take those actions from unfamiliar wording remains the task.
