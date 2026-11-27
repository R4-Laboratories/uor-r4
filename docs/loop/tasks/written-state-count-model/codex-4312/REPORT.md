# Two full native checkpoints changed their outputs and both failed the chat test

Codex development result for [#4063](https://github.com/UOR-Foundation/uor-r4/issues/4063), following the [paired registration](SHEET.md), [non-author fit](FIT.md), and [pre-run decision](GO.md). The primary endpoint was useful completion of a grounded question through the model's own EOS on 200 fresh human dialogues. Both arms scored **0/200**. The observed paired difference was 0 percentage points, exact two-sided discordance p = 1. With no discordant outcomes in 200 pairs, the one-sided 95% upper bound on the probability of a discordant completion in this sampled population is 1.49%. The same one-sided bound holds for each arm's individual zero-success rate. The registered bar for increasing the training budget was missed.

## Where the replies failed

| Held result | Reasoning first | Answer first, reasoning separate |
|---|---:|---:|
| Correct chat content, allowing a cap | 0/200 | 0/200 |
| Chat ended by natural EOS | 0/200 | 0/200 |
| First disagreement with the human reference at token 1 | 183/200 | 182/200 |
| First disagreement at token 2 | 17/200 | 18/200 |
| Whitespace-only decoded replies | 179/200 | 161/200 |
| Correct recall reply through EOS | 0/64 | 0/64 |
| Both variants of a counterfactual recall pair correct through EOS | 0/32 | 0/32 |
| Prose top-1, teacher forced | 346/8,192 | 346/8,192 |
| Prose top-8, teacher forced | 500/8,192 | 569/8,192 |
| Prose top-8 on tokens novel to the causal prefix | 135/3,736 | 155/3,736 |
| Free prose continuations ending in EOS | 0/128 | 7/128 |

In three chats answer-first moved the first reference disagreement from token one to two. Two moved the other way. Both arms matched only the single-space token (ID 223) on every chat whose first disagreement was at position two. In 136/200 conversations the complete generated token stream was identical. Among the 64 changed streams, 50 differed at their first emission. Reasoning-first produced exactly 32 spaces on 165 questions and 32 newlines on 14. Answer-first produced 32 spaces on 161. All 400 replies used the 32-token allowance. One reasoning-first reply emitted a malformed special token and was retained. Neither arm opened a chat with the donor's stock “ We” token (799). [Anonymous paired statistics](evidence/blind-stats.json) and [decoded and replayed observations](evidence/readback.json).

The two chat trajectories chosen before training locate the failure inside the model. On their first step the human reference tokens were 376 and 1771. Both native gates selected 223, and the ordinary writer emitted 223 in all four model–prompt combinations. The authenticated successor addresses in the two initial windows were 161 and 165 in both arms. Across all eight preselected trajectories per arm, the writer preserved its gate's selected ID on 509/512 reasoning-first and 508/512 answer-first steps. These observations locate the wrong first choice upstream of the final writer on both traced chats. The all-chat score measures the frequency of failure, while the internal traces cover those two chat starts. [First-step device and client readback](evidence/readback.json).

Every recall response reached the cap. The closed code grader recognized no correct or competing five-digit claim. For prose, answer-first added 69 top-eight hits, a gain of **0.842 percentage points** with a paired whole-article 95% interval of **+0.415 to +1.270 points**. The gain for novel targets was 0.535 points (95% +0.053 to +1.035). Teacher-forced prose top-one prediction was exactly 4.22% in both arms. These are secondary measures of next-token ranking. For the separate open-ended human panel, 21 of 32 pairs generated identical token streams and both arms reached their 128-token cap on every prompt. Human usefulness scores await independent ratings.

## What was trained

Both deployments are complete, freshly initialized models from the released 19-section native D11 trunk family. The earlier 4 GB S0–S6 P4 is a historical reference. Both new arms use seed 4312, context 192 and the original 4,096 token IDs, with the same integer read and writer. Each arm received all 4,060 original TRAIN groups. The geometry optimizer visited 259,604 canonically matched target occurrences across all groups, in 1,015 batches and 3,960,128 coordinate-by-row trials. The following stage fitted the unigram and all 12 sparse output tables on 13,983,847 target occurrences per arm. The optimizer updated the base geometry first and the complete readout was fitted under the resulting state. This run tested one common initialization and this staged training recipe.

| Native result | Reasoning first | Answer first |
|---|---:|---:|
| Accepted geometry update batches | 533 | 539 |
| Committed coordinate moves | 1,121 | 996 |
| Changed token-vector words / gate words from initialization | 3,547 / 11 | 2,903 / 12 |
| Serialized checkpoint bytes | 1,184,813,288 | 1,201,932,060 |
| Base geometry fit | 194.912 s | 209.633 s |
| GPU count fit | 8.750 s | 10.831 s |

Fifteen of the 19 deployed section payloads differed. The unigram and the deterministic frame and route sections matched, as expected. Reasoning-first retained the original visible reasoning and answer sequence. Answer-first placed the answer after the assistant marker and moved the original reasoning and separator behind a separate state reset. Thus the hidden reasoning also lost its original user context. Every moving target was matched by original event and token ID. Both arms kept the historical one-visible-EOS-per-donor-document policy. The visible stock-token count fell from 3,897 of 4,345 donor openings to zero. Neither model learned to answer the 200 held questions. [Training-pair record](evidence/training-pair.json) and [source reconstruction](donor/RECEIPT.json).

## Grading, parity and recovery

The 200 main prompts use different human dialogues/passages with genuine earlier human answers. To fit context 192 the builder selected from predetermined source windows containing the current answer evidence. It selected 152 factual and 48 polar questions. The frozen deterministic rubric accepted all 200 retained reference answers and 545/600 independent alternate human answers. It rejects contradictory, echoed, malformed and unterminated completions. This is a controlled measure of assisted passage answering. The separate human packet presents authentic open-ended OASST2 histories, both complete model outputs, their termination reason, a common rubric, and a 16/16 balanced hidden position assignment. Distribute only its HTML to raters. Whitespace-only answers are shown with their character counts. The HTML runs offline and exports completed ratings. [Panel protocol and calibration](panel/README.md).

The independent custodian scored all 848 anonymous free-running replies and 784 teacher-forced records. The sealed manifest commitment was `ecd7c4e98ebfb8736028eab84e3e5a39f8277f4f5327bc0562be4e8ff6695d5d` before assignment was disclosed. Opaque left corresponds to answer-first and opaque right to reasoning-first. A cold client verified the native read and recurrence, successor address, selected original ID, cells, ordered top eight, and emitted token at all 1,024 registered trajectory steps. Sixteen independent ordinary `uor-chat` invocations reproduced complete raw trajectories, and two additional vocabulary-bound text invocations exercised the product dialogue entry. Raw-ID CLI lengths were set to the GPU's observed termination length. [Blind seal](evidence/blind-seal.sha256) and [cold replay](evidence/readback.json).

The RTX 4090 completed training and packing both models in 8 min 30 s and held generation plus teacher scoring in another 48 s. The provider ledger reports 1,814 seconds of total instance uptime at $0.869/h, or $0.438 at the compute rate. This includes AOT build, finite qualification and archive time. The instance was deleted after SHA-256 verification on Wick ZFS. The complete checkpoints, source inputs and sealed results were also downloaded again from [private store revision 36bb4722](https://huggingface.co/datasets/caseyallard/uor-r4-store/tree/36bb47220ba0d60e07f0a6cb177da16b10902b84/results/codex/visible-answer-20261127) and verified against the Wick copies. [Archive checksums](evidence/store.sha256) and [resource accounting](evidence/compute.json).

## Restore the checkpoints and human-rating packet

To serve either result, extract its checkpoint archive from the private store and extract `tokenizer.json` from `input/paired-inputs.tar.zst`. The integer CLI used in this study accepts `uor-chat --discrete answer-first/artifact.d11 --tokenizer tokenizer.json --protocol 2 --say "Explain how a seed grows into a tree." --max-tokens 32`. The two archive members named `artifact.d11` are primary model artifacts. Their verified SHA-256 values begin `1b8e1240` for reasoning-first and `47214eda` for answer-first. The full digests are in the training receipt. The human-rating file is `delivery/open-chat-ratings.html` inside `evidence/scoring-and-replay.tar.zst`.

## Research decision

Keep the verified integer writer for the next diagnostic. On fresh pairs with different questions and answers but the same literal assistant-role suffix, measure the first **content** token's rank in the native gate and in the final writer. Compare the intact prompt with a crossed-question control and the common-token prior. Record whether the read code and recurrence distinguish the two questions, and whether changing either state moves the intended token above space. This tests whether general language supervision reaches the deployed selection rule and localizes the next training change. The prose rank gain alone does not justify a scale-up. No milestone is promoted by this run.
