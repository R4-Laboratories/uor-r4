# The writer can follow three grounded instruction decisions

## What did not work

Codex, 25 November 2026. Across 512 conversations held out by wording family and 128 new colon-free conversations, the fitted integer reader recovered the correct three instruction fields with their source evidence on zero of 640 cases. On the new wording, it selected the annotated scope addresses on zero of 128. These are two-turn scripted conversations with real prior assistant output. M2 and [grounded reply training](https://github.com/UOR-Foundation/uor-r4/issues/2853) remain open.

## What changed

The experimental client in research PR #4278 admits a source-bound instruction, opens on the requested original fact and carries its completion action until the third verified source period. The same 376 signed writer rows and native copy mechanism serve every arm. Each field can be supplied for a causal control or selected from the integer artifact in the ordinary client without answer tokens. The eight development folds test unseen wording applied to facts present in other training families. Each of the four colon-free wording styles uses one associated fact pair reused from development.

Supplied state produced 256/256 grounded development replies exact through EOS and 256/256 explicit deferrals matching the stock payload. It recovered all 36 grounded cases where five previous readers had never attempted the copy. The stock parent itself capped at 48 tokens on 100 exclusions. That allowed 156/256 excluded replies through EOS and 68/128 strict complete groups. The previous native reader had 46/256 exact grounded replies, 88/256 explicit deferrals with stock EOS, and 2/128 complete groups. On the 128 new colon-free scripts, supplied state gave 64/64 exact grounded replies and 64/64 explicit stock deferrals through EOS.

## Measurements

| Source of the three fields | Held-family grounded /256 | Held-family explicit exclusion with EOS /256 | New wording grounded /64 | New wording explicit exclusion with EOS /64 |
|---|---:|---:|---:|---:|
| Previous native reader | 46 | 88 | 9 | 9 |
| All supplied | 256 | 156 | 64 | 64 |
| Only scope learned | 144 | 44 | 36 | 8 |
| Only first role learned | 56 | 156 | 24 | 64 |
| Only ending learned | 158 | 156 | 32 | 64 |
| All learned, ordinary text | 22 | 44 | 6 | 8 |

The ordinary artifact recovered six of the 36 previously unattempted positives and retained all 16 historical controls. It completed zero of 128 held-family groups and zero of 32 new-wording groups. On new-wording exclusions it returned the literal stock reply through EOS on 56/64, including 48 unresolved prompts that fell back. Native returned that stock reply on 17/64. On the 256 grounded positives the assembled reader moved the first error later on 38 and removed it on 20, moved it earlier on 44 and added an error to 44 native successes. It left 108 wrong at the same index and two exact. For the assembled reader the 256 grounded dev conversations split into 168 without grounded writer entry, 54 with a wrong first source, 12 whose first error was at third release, and 22 exact. All 98 held-family failures with only the ending learned first diverged at the third release.

Swapping the first role on the red-dyer/red-wharf prompt changed token 1 from ` You` to ` Your` and the first copy from home address 53 to job address 37. Both addresses contain the same “ red” token. Swapping only the requested ending preserved the first 33 tokens and replaced EOS at token 34 with ` Your`. This isolates the two decisions that can lose a complete reply after admission.

The correct scope tuple was in the candidate set on all 640 original cases. The learned score ranked a rival ahead of it on 112/128 new prompts and tied on the remaining 16. With the supplied correct scope, every one of those 128 prompts offered a candidate overlapping the annotated opening phrase and a candidate overlapping the annotated ending phrase. Some annotations included multiple clauses. The study did not establish which clause governed each action. An integer candidate feature identical to a distinct wrong source in the same prompt did not explain these misses. Transfer of the learned ranking across wording families remains the open mechanism.

## Checks run at the head

Changing the requested role on the original 128 colon-free cases also changed scope-mention order. A separately registered 128-case supplement swapped only the two governing sentences per original, producing 256 distinct scripts. On 64 affirmative pairs holding actual mention order fixed, supplied state recovered both requested first sources in 64/64 and both requested endings at the third release in 64/64. The assembled model scored 0/64 on both dimensions. Native achieved 12/64 and 0/64. This is a paired wording transfer measurement on a frozen fit. Same-prompt supplied swaps establish the individual fields' causal effects.

Two RTX 4090 fits produced identical integer coordinates. CUDA and portable Rust agreed on the original 74,240 scalar rank words and an independent GPU writer matched 43,168 bound steps in the supplied/control capture, including 1,280 third-period releases. The learned run verified another 34,526 writer steps and 1,024 third releases. The implementation uses shared host token framing and projection. Signed-parent fallback output comes from the ordinary client. [Complete scientific report](https://github.com/UOR-Foundation/uor-r4/blob/09aa5915c2ee5989f38faa99e85e87aae0ced40f/docs/loop/tasks/m2-train-grounded-replies/codex-4278/REPORT.md) · [Ordinary first-token movements](https://github.com/UOR-Foundation/uor-r4/blob/09aa5915c2ee5989f38faa99e85e87aae0ced40f/docs/loop/tasks/m2-train-grounded-replies/codex-4278/evidence/first-error-dev.json) · [Control and learned results](https://github.com/UOR-Foundation/uor-r4/blob/09aa5915c2ee5989f38faa99e85e87aae0ced40f/docs/loop/tasks/m2-train-grounded-replies/codex-4278/evidence/ordinary.json) · [Counterbalanced pairs](https://github.com/UOR-Foundation/uor-r4/blob/09aa5915c2ee5989f38faa99e85e87aae0ced40f/docs/loop/tasks/m2-train-grounded-replies/codex-4278/evidence/counterbalanced-pairs.json)

All nine fitted D11 artifacts were reconstructed from a pinned 32,806-byte private packet and a shared signed parent. The restored models reproduced all 25 checked complete ordinary conversation traces byte for byte. A Wick ZFS archive verifies 69,382 files. [Artifact, archive and replay receipts](https://github.com/UOR-Foundation/uor-r4/blob/09aa5915c2ee5989f38faa99e85e87aae0ced40f/docs/loop/tasks/m2-train-grounded-replies/codex-4278/RESTORE.md)

## Skills used

The native-space cycle guided train/serve replay. The UOR research architect guided the distinction between field selection, source-authenticated output and whole conversations.

## Still open

Freeze the proposer and writer and compare the current equal-byte reader with an integer margin trained on the true ordered source and the strongest wrong source within each prompt. Isolate scope first using supplied first role and pending action. Require paired correct direction, original source authentication, historical retention and increased exact replies through EOS before assembling the next artifact. The four-family reserved panel remains unopened.

## Plain summary

The writer can deliver the requested facts in the requested order when it identifies what the user meant. The model often considers the right words and selects a different occurrence. Ranking those sources is the next trainable decision.
