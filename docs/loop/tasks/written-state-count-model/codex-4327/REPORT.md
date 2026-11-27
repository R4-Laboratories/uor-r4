# Paired release updates learned the training contrast and lost it on new wording

## What did not work

Both trained models made 0/128 correct selector switches, 0/128 clause-order switches and 0/64 direct two-versus-three switches after an authenticated common B on the new HELD panel. Across 64 four-cell contexts, neither completed one through EOS, including all 48 with equally long tokenized prompts. The paired state stopped after the supplied period on every one of the 512 inputs. It completed 192/512 requests against 224/512 with the matched row-wise control. The learned updates failed the registered transfer condition. Keep balanced as the integration baseline. [Test sheet](SHEET.md) · [Full grading method](analysis/grade.sh)

## What changed

Both checkpoints started with the same 40,966 frozen linked bytes and a 4,096-word zero i16 residual. Scoring, the memory reader, authenticated original-address copy and integer writer stayed fixed. The two policies used one identical reordered schedule of 6,144 exposures, preserving each original epoch, stage, row, target and multiplicity from #4325. The control applied ordinary margin updates. For opposed siblings, the challenger only wrote addresses visited by one sibling. Each direction tested its own signed margin of eight. Rehearsal and unpaired occurrences used the original update policy. Every B-supervised example went through the actual copy on the GPU. Labels and pair identities were confined to training and diagnostic controls. [Device implementation](device/README.md) · [Exposure identity audit](evidence/exposures.json)

The 768 TRAIN rows included 256 archived rehearsal rows. There were 512 opposed selector groups, 256 direct one/two groups and 256 direct two/three groups across four epochs; the two arms each had 98,304 coordinate-write opportunities. On matched groups, first-period row-wise writes hit 7,878 shared and 746 distinguishing coordinates; the paired policy hit zero shared and 5,760 distinguishing coordinates. At B the corresponding totals were 1,759/49 and 0/320. Different actual update counts follow from which signed margins failed and which coordinates were eligible. The design matched the maximum budget, exposures and opportunities. [Schedule and finite controls](evidence/device-finite.json)

## Measurements

Two authors separately supplied eight new scoped wording families and four direct families. These are structured agent-authored requests, with repeated facts within each family. The two fits sealed before the new HELD tape went to the training pod. The controlled writer receives the original factual positions and emits the supplied opening word and period. It then has to choose when to stop or copy another source token. Token positions below are one-based. All four controls scored 512/512 when both release actions were supplied. [Panel construction and overlap](panel/README.md)

| Fresh HELD, 512 prompts | Exact through EOS | First wrong at token 3 | First wrong at token 4 | Strict selector / 128 | Strict direct after B / 64 |
|---|---:|---:|---:|---:|---:|
| Frozen balanced | 208 | 192 | 112 | 0 | 32 |
| Frozen linked | 224 | 272 | 16 | 16 | 32 |
| Matched row-wise | 224 | 256 | 32 | 0 | 0 |
| Matched paired | 192 | 320 | 0 | 0 | 0 |

The row-wise control completed 160/256 shorter and 64/256 longer replies. Paired completed 192 shorter replies and zero longer replies. The two fits tied on one author's rows at 96/256. The other author's rows fell from 128/256 to 96/256. On the 320 prompts requiring a second source word, the learned trajectories reached that original B on 96 row-wise and zero paired prompts. Supplying only the correct first choice put both fits at the same authenticated B; 216/320 and 184/320 respectively then made the right next choice. On 32 direct crate pairs, paired reversed the two requested actions. On the other 32 direct pairs it stopped on both sides. The 128 selector pairs each stopped twice under both fits. In the paired state, 72/128 longer requests actually had a lower first-period score than their matched shorter request. A single retrospective integer threshold, keeping the scores fixed, could strictly separate at most 16/128 selector pairs.

**A first-error trace.** The same tray held “chalk, arm, mouth, lion”. Asking for one slot requires IDs `[3355,16,1]`; asking for two requires `[3355,16,1559,1]`. Row-wise copied “arm” from address 27 on both requests (scores +37 and +39). The first wrong token on the short request was token 3, observed 1559 instead of EOS 1. Paired changed both scores to negative (−139 and −113) and emitted EOS on both. It recovered the short reply and introduced a token-3 error on the previously exact long reply. Neither model learned the switch on this pair.

**Learning and interference on TRAIN.** At their final checkpoint, row-wise and paired made 40/128 and 64/128 fresh TRAIN selector switches and 32/64 and 64/64 direct two/three switches after an authenticated B. Immediately after each selector pair's own last matched update, 114/128 had both signed choices correct under paired. Subsequent updates lost 53 of those 114 pairs and gained three different ones, leaving 64/128. These are identities on one shared trajectory; intervening updates include other prompts and stages. The measure does not isolate the last-stage contribution. On direct two/three the paired arm kept all 63 switches it had immediately after those last updates and gained one more by the final checkpoint. [Attribution method](accounting/last-pair-update.jq)

**Retention.** Both fits completed 8/64 fresh TRAIN quartets. Across the 768 training prompts the row-wise state completed 540 replies and paired completed 474. On the 50 previously exact protected pairs neither fit retained a complete pair. Row-wise completed 25/100 individual protected replies and paired completed zero. Some older panels improved and some regressed. On the previously exposed #4325 HELD set the paired model completed more original pairs than row-wise (24 versus 16) but lost all eight previously exact joint pair IDs; row-wise kept all eight. [Grades across eight panels](evidence/grade.md) · [Update and threshold accounting](evidence/update-effects.json)

**Ordinary chat.** The actual prompt-only discrete client generated identical token IDs for the frozen base and both fitted artifacts on all 22 turns of 16 separate conversations. Six follow-ups included the assistant's actual generated history. All three models scored 0/22 useful replies, 0/22 exact replies and 0/22 natural EOS at a 16-token cap. Eighteen replies contained only whitespace. Asked how many of six pears remained after eating two, the client emitted `Let's%haps.gthon.gthon.g`. These are program outputs; usefulness was judged against the previously frozen rubric. The generic discrete decoder does not read the trained source-choice residual. The separate grounded helper that does read it rejects a request without four factual positions, and also supplies the opening and first period. This directly limits what the grounded scores can establish about a usable chat model. [Ordinary prompt-only records](evidence/product-ordinary.json) · [Grounded artifact parity](evidence/product-parity.json)

## Checks run at the head

The leased 4090 passed positive, inverted-label and identical-support finite controls at both stages. Swapping targets reversed both learned signed decisions and original-address commits. A wrong-address copy was refused. The host reconstructed every changed residual word from the signed device ledger. A second full fit from untouched initial bytes reproduced both fitted states and write traces. The scientific evaluator was the unchanged #4325 binary/cubin, with no JIT, in both input orders. The cold D11 verifier compared every diagnostic path and ordinary grounded-client emission on the 512 new rows. [Non-author FIT](FIT.md) · [Physical device authorization](evidence/go.json) · [Device manifests](evidence/device-inputs.sha256) · [Restore instructions](RESTORE.md)

The paired and row-wise fits changed 1,703 and 1,995 residual words respectively. Training time reported inside the first fit was 1.20 seconds for paired and 1.36 seconds for row-wise. Including device authentication, verification and the independent repeated fits, both complete checkpoints sealed in 28.6 seconds. Training plus all GPU forward/reverse scoring took 5 minutes 15 seconds. The rented RTX 4090 was allocated for 12 minutes 44 seconds at $0.869/h, an estimated $0.18 before provider extras. All 102 returned files verified on Wick before the controller released and deleted the pod. [Fitted identities and timings](evidence/fit.json)

## Skills used

The native-space-cycle protocol kept training updates and restored integer model words in the same space. Original source addresses, actual copy, EOS and self-fed tokens were part of the replay check.

## Still open

The ordinary chat entry point lacks an input-derived path to these release decisions. The limited controlled path still needs to learn which instruction governs the action across genuinely new wording. [Next experiment](PICKUP.md)

## Plain summary

The geometric trainer can write different scores for opposing training instructions. On new wording these weights made a universal early stop. A usable chat model needs its normal decoder to find a source on its own and carry it into a learned decision, and the decision itself must transfer.
