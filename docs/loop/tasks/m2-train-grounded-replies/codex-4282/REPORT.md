# Does a within-prompt training margin repair scope transfer?

## What did not work

The starting UORACT05 already chose the correct original-address tuple on both sides of all 2,048 training pairs, at the maximum registered margin on every side. Those pairs had only 41 different paired geometric inputs. Wick ran two equal-byte fits. Both reproduced the existing scope banks in all nine artifacts. Across 896 held conversations the gold tuple was present on every prompt, but it won on only 24/512 old development prompts and none of the 128 earlier colon-free or 256 new wording/fact prompts. The tested change leaves the signed default and M2 unchanged. [Research and physical evidence in #4281](https://github.com/UOR-Foundation/uor-r4/pull/4281)

## What changed

The instrument now compares the correct ordered source spans against every distinct rival in the same prompt. It retains wrong-address competitors even when the wrong source would make the same answer-or-defer decision. A finite positive case changed the winning original address and a deliberately indistinguishable pair preserved the tie. The measured training data supplied 85,760 same-admission wrong-address alternatives. The four first-role/ending banks, writer, cursor, codebook and D11 byte length stayed fixed.

## Measurements

The first role and ending were supplied. All results below are ordinary second-turn returns after genuine assistant history, scored against original fact addresses and an independent stock parent.

| Result in both isolated arms | Held wording family, 512 cases | Earlier colon-free, 128 cases | New wording and facts, 256 cases |
|---|---:|---:|---:|
| Exact grounded fact copy through EOS | 144/256 | 36/64 | 112/128 |
| Explicit DEFER and exact stock through EOS | 44/256 | 8/64 | 0/128 |
| Literal stock through EOS, any route | 84/256 | 44/64 | 16/128 |
| Both opposed replies user-visible exact through EOS | 72/256 | 16/64 | 16/128 |
| False admissions | 120/256 | 20/64 | 80/128 |

All 256 fresh gold triples were strictly outscored. The gold median rank was 32. Exact features of the fresh affirmed and excluded occurrences were absent in the pooled training comparison, while all 256 referenced gold occurrences had an exact feature counterpart. Neither the rank nor the first wrong returned token moved between the two fits. The 16 missed fresh grounded replies split into eight first errors at token 1 and four each at tokens 9 and 10. The 80 falsely admitted exclusions split 40, 20 and 20 at those same positions. Explicit DEFER, literal stock fallback and the 48-token ceiling have separate denominators.

Correct supplied original spans yielded 128/128 original-source grounded replies through EOS. All 128 exclusions returned exact stock, with 48 through EOS and 80 capped by the parent. Swapping the two governing spans on the identical prompt flipped all 256 choices. Both unassisted artifacts retained all 16 historical conversations through EOS. General grounding, conversation and reasoning continue to require separate evidence.

## Checks run at the head

Both physical fits returned the baseline fit bytes in all nine folds. The loaded action readers, tokenizer, nested signed writer and D11 byte length matched the originals. All 912 pairs of ordinary traces agreed after removing the outer artifact hash. The independent GPU readback matched 207,872 selection words and 41,912 supported writer steps. The coverage inventory used original byte and token addresses on the held prompts and checked its candidate streams against physical replay.

## Skills used

The native-space cycle and UOR research architect guided the physical replay and controlled grounding comparison.

## Still open

Construct new training pairs with opposing clause-to-occurrence bindings where the frozen reader actually makes source errors. Reserve independent wording families and compare improvements on **both** opposed endpoints, all original source addresses, through-EOS replies and historical controls. If a stronger training panel can move source margins on TRAIN yet fails to transfer, test a geometric link between the governing clause and the addressed occurrence with the same writer and byte budget. The official M2 bars for unassisted saved and fresh questions remain open. [Grounded-reply training issue](https://github.com/UOR-Foundation/uor-r4/issues/2853)

## Plain summary

The starting reader already won every training comparison. On fresh text it often copied the right fact for the wrong reason and also answered commands the user excluded. The next training data must make that distinction observable. The separately integrated ordinary-client stack depends on draft #4139. [Source, per-case results and restoration](https://github.com/UOR-Foundation/uor-r4/blob/5c67204bf00481e4cdb1357eb56c82a2f7258548/docs/loop/tasks/m2-train-grounded-replies/codex-4281/REPORT.md)
