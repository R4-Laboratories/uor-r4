# A fitted memory gate changes token scores while the chat opening stays wrong

Codex development comparison for [#4063](https://github.com/UOR-Foundation/uor-r4/issues/4063). A leased RTX 4090 fitted and tested native integer memory weights inside the ordinary P4 prediction path. The primary natural cohort used 60 recorded conversations with genuine earlier assistant replies and no exact opening found in the screened D87 prompt bank. [Pre-registration](SHEET.md) · [Aggregate evidence](evidence/quality.json)

## What did not work

Keep frozen P4 as the default. The fitted gate reduced mean conversational Brier by 0.0000203 on 3,944 primary targets. Its paired 95% interval spans −0.0001410 to +0.0000978. Top-token accuracy went from 1,699 to 1,697 correct, with zero recoveries and two losses. Article Brier worsened by 0.0002369, outside the preregistered prose guard. Both comparisons against the frozen and pooled fits give the same primary interval because every pooled weight fitted to zero. [Paired estimation](panel/src/bin/s4-paired.rs) · [All six arms](evidence/quality.json)

All six arms missed the human reference at token one on every one of the 24 free-running conversations. Twenty-three learned replies began with the same “ We” token and the remaining reply with a newline. Only one chat changed its first token relative to frozen, from “ The” to “ We” where the recorded reply began with a standalone space. On the six chats that changed, the first model-to-model differences were at positions 1, 29, 29, 29, 29 and 37. Three learned chats emitted EOS and none reproduced the recorded reply. The score records agreement with one reference. The recurring generated opening across unrelated questions gives separate evidence about responsiveness. [Generation grading and first divergences](evidence/quality.json)

The retrospective TRAIN inventory locates a concrete mismatch in the supervision. In the exact 4,060 D87 records underlying R, 3,897 of 4,345 authenticated assistant starts (89.69%) begin with the same “ We” token. The same token starts 1/64 natural fitting replies and 0/24 human free-run references. One identical 16-token opening appeared in 22/24 frozen replies and 23/24 learned replies. Only one authenticated R opening contained that exact sequence. The original and serving tokenizers agree on all 4,096 token byte surfaces and on the six-token assistant marker. D87's ingest supervises the donor's reasoning immediately after that marker when reasoning is present, followed by the answer. This identifies a documented training incentive and a measured distribution mismatch. A controlled retrain is needed to establish its causal contribution. [Corpus inventory](evidence/openings.json) · [D87 ingest](../../../../../docs/labs/m2-student-corpus-ingest/instrument/src/lib.rs)

## What changed

S5 proposes the authenticated successor at a specific causal token address. S4's unique positive role vote selects one of 15 integer weights indexed by role and Pick class. During ordinary emission the selected weight transfers conserved Q48 mass between the proposed token and the strongest other token. The five additional role-blind weights share eligibility, source, rival, training loss and candidate grid. Three conditioned weights fitted away from zero, to +20, −4 and −16. The grid examined 13,151 reference targets, including 4,096 R targets and 9,055 natural fit targets. It found an eligible source on 11,918. The 124-byte sidecar binds the fitted state to the 3,987,940,843-byte P4 file and the exact fitting inputs. Training and ordinary decoding used the same original token identities and fixed-point arithmetic. [Operator](../../../../../crates/uor-r4-discrete/src/kn_ceiling/runtime/s4_source.rs) · [Device runner](runner/README.md)

The ordinary client now preserves an earlier assistant EOS inside an open multi-turn prompt. Its validation still catches a terminal EOS at entry. During delivery review, the client was changed to skip repeated token IDs before constructing the conjunction key, avoiding error allocations on repeated history. Both client builds replay the fixed measurements. This unit qualifies the opt-in path through the one-shot native client. [Client integration tests](../../../../../crates/uor-r4-integer/tests/discrete_p4_serving.rs)

## Measurements

Brier is a full-vocabulary loss where lower is better. The teacher-forced rows supply earlier reference tokens. Free runs use each arm's own emissions. The 68 conversations with a known D87 opening form a secondary cohort. Older corpus and verbatim answer overlap are unmeasured. [Panel construction and exclusion rules](panel/README.md)

| Teacher-forced cohort | Targets | Frozen / learned top-1 | Frozen / learned mean Brier | Learned minus frozen, paired 95% interval |
|---|---:|---:|---:|---:|
| 60 screened conversations | 3,944 | 1,699 / 1,697 | 0.708148 / 0.708128 | −0.000141 to +0.000098 |
| 68 known-opening conversations | 5,667 | 2,485 / 2,485 | 0.699774 / 0.699857 | −0.000064 to +0.000218 |
| 128 Wikipedia articles | 8,192 | 3,311 / 3,316 | 0.729960 / 0.730196 | +0.000094 to +0.000392 |
| 512 reused DEV-tune anchors | 512 | 296 / 297 | 0.503333 / 0.503679 | −0.000121 to +0.000886 |

On the 60 primary chats, 3,659 gold positions had an eligible source. The learned gate transferred mass on 689, changed 16 top tokens, and retained 442/1,435 correct targets absent from the causal prefix. The wrong-address control scored 1,702 correct and the shuffled-role control 1,699. This configuration shows no dependable advantage for the authenticated source. Source authentication and usefulness are separate measurements. [Paired counts, role admission and controls](evidence/quality.json)

| Free-running treatment | Chat first token correct / 24 | Chat EOS / 24 | Exact chat through EOS / 24 | Replies differing from frozen, chat / article | Exact article to 64 / 24 |
|---|---:|---:|---:|---:|---:|
| Frozen | 0 | 2 | 0 | 0 / 0 | 0 |
| Disabled, learned file loaded | 0 | 2 | 0 | 0 / 0 | 0 |
| Learned | 0 | 3 | 0 | 6 / 4 | 0 |
| Role-blind | 0 | 2 | 0 | 0 / 0 | 0 |
| Wrong address | 0 | 2 | 0 | 5 / 1 | 0 |
| Shuffled role | 0 | 1 | 0 | 11 / 3 | 0 |

Two of the three learned EOS emissions were earlier than the recorded EOS and 21 replies reached the 128-token cap. The S5 successor matched the reference opening in 1/24 learned conversations. Six reference openings appeared anywhere in the initial prompt. The learned gate acted on five opening decisions. On articles, 10/24 first tokens matched in both arms. One article moved its first wrong token from position two to three. All arms reached the 64-token article cap on all 24 articles. [Per-arm comparison](evidence/quality.json)

The RTX 4090 fitted in 1.102 seconds, scored the teacher panel in 1.549 seconds and generated 26,772 tokens across all six arms in 6.389 seconds. Instrument load and finite checks took 26.459 seconds, internal fixed-prefix replay 0.108 seconds, and the complete runner 35.609 seconds. Peak sampled device use was 4,248 MiB and peak host RSS was 4,071,356 KiB. These are development-harness stage clocks. The separately hosted P5 trainer on Wick recorded 68 seconds for the opening DEV-tune sweep and 249, 295 and 562 seconds for its three epochs including their reevaluations. The P5 authors report 1,107 seconds of training, 272 seconds for sealed evaluation and a frozen-S1 decision after a 40 ppm Brier worsening. That is an independent campaign on another GPU. [Unsealed clocks](panel/CHECKS.md) · [P5 authors' result](https://github.com/UOR-Foundation/uor-r4/pull/4311)

## Checks run at the head

The GPU finite test on the leased RTX 4090 recovered the signed planted optima +64 and −64, and +12 when pooled. Device score and Rust state replay compared 417,690 raw legal-token masses over 16 fixed DEV prompts and one real multi-turn prompt in six modes. A further 24,570 raw masses covered natural abstention, actual transfer and changed-winner witnesses. Every recorded address tied back to its causal token. The public executable independently checked 1,160 steps per build across 20 cases, covering 4,750,200 mass words by digest, plus route, source, weight, carry-related fields and emission. This includes a complete fixed conversation and article in all six arms. [Hardware, source and fitness pins](FIT.md) · [Replay verifier](cold-compare.jq)

A reporting expression initially failed to summarize the completed cold comparisons. The corrected reporter checks each case against the planned identifier, arm, domain, kind and step count. An independent replay of all 40 saved CLI comparisons reproduced the two identical 20-case summaries, SHA-256 `a6d978c0d9212ad17073c9181a071bd14f484382f4925033aa05b2b058b5b98a`. The registered independent grader and a hardened version returned identical bytes on all 288 real generated streams. The hardened grader additionally rejects an emitted BOS and a treatment trace whose S4 source object is missing. [Grader](panel/src/grade.rs) · [Cold recount](cold-summary.jq)

Both instances were deleted. The first exited before any science ran, with 184 seconds in the provider ledger. The replacement recorded 921 seconds. At $0.869/hour their combined estimated compute cost was $0.267 before provider fees. All 203 rental files verified on Wick ZFS, along with the separate pre-existing full-input archive. The 11,550,152-byte private restore package has SHA-256 `be37474b7970479da756693f83da4f9c31121770747add55efecaae4657f8ea8`. It includes both cold clients, the CUBIN and its disassembly, raw observations, fitted weights, tokenizer, natural prompts, independent grades, restore instructions and a manifest of 246 files. The 3.99 GB P4 remains at its original verified Wick restore location. [Pinned private evidence](https://huggingface.co/datasets/caseyallard/uor-r4-store/blob/46822861077840a6cd1d4cd32950b9cfb24b2984/results/codex/s4-output-20261126/s4-native-evidence.tar.gz)

## Skills used

The native-space cycle supplied the integer-state and cold-replay contracts. The UOR research architect skill aligned the comparisons with ordinary, trained, end-to-end prediction.

## Still open

The next development comparison should change what predicts the visible assistant turn. Produce two complete, matched native checkpoints from the same donor record groups. Preserve the current ingest in one. In the other, supervise the final answer at the visible assistant boundary and keep any donor reasoning in its own explicitly identified training segment. Rebuild the corpus-derived token counts and output weights in each arm. Keep the native vocabulary, role markers, turn-ending policy, initialization, state size and measured update budget matched. Use frozen P4 as an evaluation anchor. Test new multi-turn prompts with useful-answer-through-EOS judgments blind to the arm, independently varied final questions, a recall guard and prose positions requiring new tokens. Register sample size and scoring before training. Start with at least 200 independent conversations for the proposed capability panel. A fall in stock openings without better useful completed answers would leave answer segmentation insufficient. [D87 ingest](../../../../../docs/labs/m2-student-corpus-ingest/instrument/src/lib.rs) · [Program roadmap](../../end-to-end-integration/codex-4307/REPORT.md)

## Plain summary

The trained weights reached the public client's token choice and changed real outputs. They did not fix the first word on any held chat. Improving how the training corpus represents the assistant's answer is a more direct next experiment than enlarging this memory gate. Frozen P4 remains the baseline for that experiment.
