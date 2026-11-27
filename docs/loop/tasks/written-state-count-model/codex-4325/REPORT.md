# Training the period and post-copy decisions together

## What did not work

The two trained checkpoints each completed **0/64 fresh four-way contexts**, including 0/32 whose four tokenized prompts had equal length. Neither trained checkpoint completed a fresh direct count pair. The joint fit's 128/256 exact scoped replies equal the number a constant one-word response would get on this balanced slice. It did produce 24 correct longer replies, but no selector-only switch made it choose the right first action on both members. The study's transfer condition failed. Keep balanced as the integration baseline. [Preregistered comparison](SHEET.md) · [Decision](evidence/decision.json)

## What changed

Two D11s started from the identical linked state and 4,096 zero signed i16 words. Both used the same ordered root links from the instruction, with the post-B stage permuting addresses into the same residual table. The period, original copy mechanism and after-C rule came from the existing writer. Both saw 768 TRAIN rows, of which 256 were verified older TRAIN rehearsal. The earlier development and held panels contributed zero training rows. The new held split supplied 512 fresh requests. One independent agent authored the four direct contrasts without seeing the training language. These are agent-authored prompts in a bounded copy-and-stop protocol. [Panel provenance](panel/README.md)

Four epochs gave each model 6,144 scheduled update slots and 98,304 coordinate opportunities. The first-only arm used every slot at the period. The joint arm spent 3,072 slots at the period and 3,072 after a verified copy of B. The training writer physically copied B on all 496 eligible original rows before they could receive late supervision. At inference the learned arm received only the prompt and four source addresses, with A and its period supplied as in the preceding study.

The first fit changed 1,645 residual words in 12,704 signed coordinate writes. The joint fit changed 2,081 words in 11,920 period writes and 4,848 post-B writes. Initial state, total opportunity budget and TRAIN example IDs were the same. Additional slots revisit only examples that require B. Consequently the first-only arm saw 1,088 STOP and 5,056 COPY targets at the period. The joint arm saw 1,088 STOP and 1,984 COPY there, then 1,536 of each at B. The experiment estimates the effect of reallocating supervision under this schedule. Distinguishing the effects of target mix and shared-coordinate interference requires another contrast. [Actual fit receipts](evidence/fit.json) · [GPU operators](device/README.md)

## Measurements

Each scoped context crosses the requested selector with clause order on identical facts. Each axis has 128 edges over the same 256 prompts. Token 1 is A, token 2 is the supplied period, and token 3 is the first learned release. The final column conditions only on runs in which the correct, physically addressed B is supplied through the first decision and the second decision remains learned.

| Measured checkpoint | Exact scoped rows / 256 | First wrong at 3 | First wrong at 4 | Native correct B reached / 128 | Native correct post-B / reached | Exact clause / selector pairs, each / 128 | Whole contexts / 64 | Correct post-B with first choice supplied / 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Balanced, frozen | 16 | 128 | 112 | 112 | 0/112 | 0 / 0 | 0 | 1 |
| Linked, frozen | 24 | 136 | 96 | 96 | 0/96 | 0 / 0 | 0 | 1 |
| First-only fit | 32 | 120 | 104 | 112 | 8/112 | 0 / 0 | 0 | 8 |
| Joint fit | 128 | 128 | 0 | 24 | 24/24 | 8 / 0 | 0 | 80 |

The joint fit produced 104/128 correct short replies and 24/128 correct long ones. It emitted EOS before B on the other 104 long requests. Its 128 token-3 failures comprised those premature stops and 24 unwanted copies on short requests. Supplying just the first choice exposes 48 remaining post-B failures; supplying both yielded 256/256 scoped and 512/512 total fresh replies in every model. The eight exact joint clause-order edges belong to the ink-marking family. [Fresh held results](evidence/held.json)

In the matched selector-only comparisons, the joint fit stopped on both sides of 104 pairs and copied on both sides of 24. An additional audit of the existing trajectories tested whether the change after B tracked requested length. On 64 matched direct pairs asking for two versus three words, supplying the first choice gave the joint model 32 pairs where both members stopped at B and 32 where both continued. Zero pairs made the correct stop/copy switch. It followed one common action for both bakery wordings and the opposite action for both trip wordings. The 80/128 improvement on the scoped post-B check is a gain on requests that all want to stop there. The mixed direct control establishes its limit. [Paired actions](evidence/paired-actions.json)

For example, the evening-edition request required `lemon. ash` and EOS. First-only copied `ash` (ID 2330) from address 30, then scored +42 at B and appended `bark` (ID 2222) at token 4. The joint fit scored −130 at B and stopped correctly. On an upper-drawer request for the same reply, first-only again copied the right B and added an unwanted word at token 4. Joint instead scored −196 at the period and emitted EOS at token 3. Supplying that first decision let it copy the same `ash`; its −27 post-B score then finished the reply. [Examples and complete raw trajectories](RESTORE.md)

On TRAIN, joint completed 582/768 replies across fresh examples and rehearsal, and 16/64 of the fully crossed new training contexts. Eight came from the pending-instruction family and eight from the valid-window family. The first-only arm completed 276/768 and 0/64. On the separate HELD set the joint fit completed 192/512 individual replies, including 128 scoped and 64 direct. The 16 training-context successes yielded zero held-context successes. [TRAIN results](evidence/train.json)

The original protected 50 pairs all failed under both new fits. Each fit also retained 0/32 previously passing balanced direct DEV pairs. Of the 75 passing balanced prior first-period pairs, first-only retained zero and acquired 25 different ones. Joint retained 37 and acquired 26 different ones. On the earlier scoped HELD panel the joint fit increased the total from 40 to 48 exact pairs, but kept only 16 of those original 40. It also lost all eight earlier linked clause-edge wins and all eight selector-edge wins on the previous crossed panel. These comparisons explain why a higher total does not license promotion. [Historical identities by cohort](evidence/retention.json)

A post-run analysis reconstructed all 4,096 joint residual words from the signed per-slot updates alone. Both stages wrote 402 of the same addresses. At 205 addresses their net changes opposed each other, including 20 where the final word cancelled to zero. The overlap is real. Its contribution to any particular prompt's decision needs a stage-specific replay. [Signed write accounting](evidence/write-accounting.json)

## Checks run at the head

The first AOT build used integer expressions whose variable division compiled into floating-point instructions. The instruction scan stopped that image before training. The accepted build uses an integer shift and bounded subtraction. Its sm_89 cubin has SHA-256 `43c00194967c4ad875b642274a9e9eede1251dd63c9be67ea293f1012409b0e7`. The finite controls check zero-residual equivalence, signed margins, shared and distinct stage addresses, source refusal and the frozen after-C behavior. Each fit reran from the original bytes and reproduced its slot audit and weights; a separate process repeated the training and matched the serialized results again. HELD entered the GPU only after both checkpoint files were sealed. [FIT](FIT.md) · [Finite receipt](evidence/finite.json)

Metering recorded 5 minutes 43 seconds between the scientific start and last score. The first fit call took 12.36 seconds including both fits, authentication, serialization checks and repeat controls. The timed first update pass took 0.974 seconds first-only and 0.723 seconds joint on this 768-row tape. Total rented RTX 4090 lifetime was 14 minutes 27 seconds, from creation at 22:17:57 UTC to deletion at 22:32:24 UTC. At the listed $0.869/hour that implies approximately $0.21 before any provider extras. The full ZFS return matched the device manifest before deallocation. [Fit timings](evidence/fit.json) · [Device return manifest](evidence/device-output.sha256)

Eighteen packed-client comparisons covered 9,208 model/prompt combinations, 37,075 ordinary client transitions and 143,292 diagnostic transitions per GPU execution order. The trusted ordinary client took only the prefix and supplied addresses and reproduced every learned token, source position and action. A fresh private-store download rebuilt the two fitted D11s and reproduced their full held rows in both the ordinary client and cold verifier. [Parity receipts](evidence/parity.json) · [Artifact restore](RESTORE.md)

On Wick's AMD Ryzen 9 9950X, the median of three warm, sequential 512-request runs measured 6,655 emitted steps/s for balanced and linked, 6,541 for first-only and 6,588 for joint. Median time per request was 0.686, 0.681, 0.717 and 0.470 ms respectively. Joint emitted just 1,584 steps on this tape, compared with 2,320 for linked. Its lower request time accompanies much shorter answers. The measurement includes client I/O, prefix work, trace serialization and the supplied opening and period. It establishes no standalone decode throughput or browser speed. Both fits add 8,192 signed-state bytes over linked. [Warm client timings](evidence/serving-cost.json)

## Skills used

The native-space-cycle skill guided the train/serve state contract. GPU fitting updated the exact signed words read by packed integer serving. No float master or conversion intervened between optimization and replay.

## Still open

The measured failure is the first learned release under a selector change. Even with a supplied first choice, the post-B reader did not distinguish fresh paired direct requests for two versus three words. Source discovery and opening generation remain open. [Next experiment](PICKUP.md)

## Plain summary

Joint training changed the learned weights and improved individual replies. Its reader still gave the same decision to opposing instructions in every fresh selector pair. The model needs to distinguish *which instruction governs* before it can rely on these gains in a conversation.
