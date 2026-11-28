# The question changes the recurrent state, but equality discards the difference

## What did not work

[codex] Stop the four-block equality-mask configuration before GPU training. In all nine fresh pairs, changing the question changed its native 16-bit recurrent code. With every eligible original occurrence competing, **none of the nine** paired decisions admitted a strict one-point solution, even when each pair could choose its own 64-byte table. A single table shared across the panel also failed. Retain balanced. This closes the tested score modification with the existing 513 words frozen.

## What changed

The finite instrument uses the frozen D11 to capture the recurrent state immediately before the assistant marker and the state through each candidate's original position. For block b, bit b is one if the request and site roots agree. The resulting mask selects one signed i16 word to add to the existing site score. NoRead retains its frozen score. The actual, fixed-zero, swapped-question and question-removed arms share the same original candidate addresses, base scores and word bounds. The native marker reader independently attests the request code. The fixture compares one copy of the added table across both sides of each pair and separately across all pairs.

Three natural-source pairs use complete passages from previously unused articles, with newly authored questions. Six controlled pairs exercise requested person, explicit support, and repeated token IDs at different original occurrences. An independent reader graded the 18 questions from text before seeing the original byte keys or model traces. All 15 supported byte intervals and the three unsupported labels agreed. This is a finite representability test on chosen examples; the rates are not population estimates. [Panel preparation and custody](panel/README.md)

## Measurements

| Request state | Pairs feasible for just the two required inequalities | Pairs feasible against all candidates and NoRead | Pooled solution |
|---|---:|---:|---|
| Actual | 3/9 | 0/9 | None |
| Fixed | 3/9 | 0/9 | None |
| Other question in the pair | 5/9 | 0/9 | None |
| Question removed | 4/9 | 0/9 | None |

The counts were identical at margins one and eight. They are ceilings under separately chosen weights for each arm, not performance of a trained model. At margin one the three actual-query solutions covered one authored support pair, one natural support pair and one same-ID rank reversal. Only one of those three specific tables still met its inequalities under the fixed query; two did so under the swapped query. None of the nine pairs satisfied both assigned inequalities with all added words zero. Strict NoRead in this test means outranking the candidates, while the ordinary reader can additionally abstain on a lexical tie. [Paired constraints, exact witnesses and positive-cycle certificates](evidence/verdict.json)

At original position 21 on one support pair, the source retained state 56095, base score 847 and mask zero under request codes 30044 and 1458. The supported question requires word zero to be at least −851. Its unsupported twin requires that very same word to be at most −853. For all nine whole-turn failures the solver also found an individual candidate comparison whose score deficit cannot be repaired because the rival and gold index the same word. Across the 15 positive prompts, **13** have at least one rival tying or beating the correct frozen score with the same mask. In **zero** of those prompts does a tying or higher-scoring rival have the same full 16-bit source code as the correct source. This localizes a specific loss of information to the compressed comparison; distinct codes alone do not imply a useful learned decision. [Original positions and masks](evidence/sites.json) · [Full-code collision check](evidence/alias.json)

On the 96 prior authored TRAIN support reversals, the original 16 feature indices remained identical at the shared address. The new mask was also unchanged on 66/96. All 96 failed the bounded one-slot support constraints at both margins. At margin one, 91/96 negative sides needed a correction below −4096 just to get beneath the frozen NoRead score. The other five had incompatible requirements on the same word. Both the loss of distinctions and the size of the inherited scores constrain a fix. [TRAIN readback and exact inequality audit](probe/README.md)

## Checks run at the head

The 18 fresh ordinary prefixes contained 1,148 eligible occurrences. The product and an independent recurrence replay agreed on 1,202 integer codes and all 18 pre-marker state reads. Reversed input reproduced the same capture after restoring row order. The archived scorer reproduced address/score/feature observations for both frozen arms, including the #4330 readback. Four cold ordinary client invocations on two prompts each matched the corresponding arm-specific captured first writer event and committed ID. The balanced and expanded frozen clients themselves chose different first IDs on both prompts (315 versus 2774 and 21 versus 2446). Planted solver controls verified opposite-direction constraints, frozen offsets and exact ±4096 bounds. Each reported feasible integer assignment satisfies every edge; each impossibility certificate is a closed cycle of incompatible inequalities. This unit checked original-address and first-event parity. It generated no newly trained checkpoint and measured no changed reply or EOS rate. [Probe and replay](probe/README.md) · [Decision rule](GATE.md) · [Restoration](RESTORE.md)

## Skills used

The native-space-cycle protocol governed original state identity, causal prefix boundaries, frozen artifact loading and integer replay.

## Still open

The project still needs a question-conditioned interaction that can alter the correct site's rank over rivals, clear NoRead where appropriate, and survive normal self-fed serving. A signed native transport retains information that equality discards. Its paired feasibility and transfer to new language have not been established. [Next finite test](PICKUP.md)

## Plain summary

The model received a different question signal in every pair. Comparing the source and question with four equality bits collapsed too many distinct sources onto the same correction. GPU rental and training cost were zero.
