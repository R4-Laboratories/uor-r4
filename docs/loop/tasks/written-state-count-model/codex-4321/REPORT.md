# A learned release action follows word-count wording on one fresh family

## What did not work

On the primary test the learned action made both replies exact on 50/200 fresh pairs. All 50 successes came from the two explicit word-count directives in the ceiling family. There were zero complete pairs in the transcript, editing and handoff families. At the earlier release, immediately after the opening period, the learned action chose to copy on both sides of all 200 pairs. The unchanged native writer first emitted an incorrect `g` (73) at token 3 on all 800 prompts. The trained arm emitted EOS on all 800, but only 325 emitted their exact references through EOS. [Independent score](evidence/held.json)

## What changed

Only the native release weights were trained. The #4319 D11, native writer, authenticated source copy and supplied first period stayed fixed. Each prompt offered the same four verified original word addresses A–D to both sides of its pair. The serving inputs were the prompt and these four positions in semantic order. A and the period were supplied. At each eligible release a negative learned score selected EOS, a positive score copied the next original word, and zero retained the ordinary proposal. The next decision operated on the actual resulting prefix.

The rented RTX 4090 trained 24,582 bytes of signed i16 state in 272 ms on 384 TRAIN pairs across four wording families, using eight pre-registered epochs. It updated distinct native root and ordered adjacent-root addresses and a bias at each of three release positions. The stage tables share the feature calculation with serving. At the common release the final weights correctly flipped 192/192 TRAIN primary pairs and 60/192 TRAIN first-period pairs. Of 768 reachable TRAIN actions at the first period, 576 requested copy and 192 requested EOS. That imbalance is a possible contributor to the observed first-period behavior. The available measurement does not isolate its effect. [GPU training receipt](evidence/train.json) · [Lossless packed artifact](evidence/artifact.json)

## Measurements

Each test had 200 independent factual pairs, four held wording families, two directive pairs per family before punctuation, and eight single-token factual words unique to that split. Facts, question, original positions and marker were identical within each pair. The shorter primary reference was A. B then EOS, the longer A. B C then EOS. The earlier test used A. then EOS versus A. B then EOS. Both sides always had another eligible factual word. Physical fact order was permuted. The two tests have different held wording families. Exactness was graded from the original committed IDs through emitted EOS with a five-token cap. [Frozen panel design](panel/README.md)

| Action source | After B: shorter exact / 200 | After B: longer exact / 200 | After B: both / 200 | After period: shorter exact / 200 | After period: longer exact / 200 | After period: both / 200 |
|---|---:|---:|---:|---:|---:|---:|
| Native | 0 | 0 | 0 | 0 | 0 | 0 |
| Learned, own instruction | 75 | 175 | 50 | 0 | 75 | 0 |
| Learned, actual opposite instruction | 25 | 125 | 0 | 0 | 100 | 0 |
| Correct supplied action | 200 | 200 | 200 | 200 | 200 | 200 |
| Opposite supplied action | 0 | 0 | 0 | 0 | 0 | 0 |

**Primary release after the same B.** The learned path actually reached B on both sides of all 200 pairs. Fifty had the correct EOS/copy flip, 125 copied on both sides, and 25 stopped on both. There were zero inverted flips and zero abstentions. All 50 correctly flipped pairs completed exactly through EOS, gaining 50 pairs over both native and swapped. Among the 150 inexact individual replies, all first errors moved from native token 3 to token 4: 125 shorter replies copied C instead of emitting EOS, and 25 longer replies emitted EOS instead of C. The native arm never reached B. At the shared decision 150 pairs assigned the longer reply the higher score, including many with the same sign. [Paired action and trajectory counts](evidence/pair-effects.json)

**Release immediately after the period.** All 200 shorter replies first failed at token 3 by copying B instead of emitting EOS. All 200 longer replies copied B correctly. Seventy-five then stopped correctly and 125 first failed at token 4 by copying C instead of emitting EOS. The longer wording raised the first-period score on 125 pairs, lowered it on 50 and tied on 25. Both scores were positive in every pair. The swapped state gave more individual exact replies on this test (100 versus 75) and completed zero pairs. Across both tests, no learned action abstained at the common decision. Native reached the cap on all 800 prompts. Every learned EOS came from the trained action. [Exactness and first mismatches](evidence/held.json)

| Primary held family | Correct EOS/copy flips / 50 | Both replies copied | Both replies stopped |
|---|---:|---:|---:|
| Explicit total word count (ceiling) | 50 | 0 | 0 |
| Transcript boundary | 0 | 50 | 0 |
| Keep or drop a word (editor) | 0 | 50 | 0 |
| Document handoff | 0 | 25 | 25 |

On a pair requesting “exactly two words” or “exactly three words,” the post-B scores were −2 and +13. The actual outputs were `black. orange` then EOS and `black. orange sugar` then EOS. For “Keep word two. Drop word three,” the score was +5 and the writer copied `blue` after `green. orange`. The “stays on the page” variant in the handoff family admits another plausible human reading. The other three families and the literal outputs are separable in the receipts. There are eight distinct directive pairs per test before punctuation, so success on 50 replicated facts covers only two successful primary directive pairs. [Every directive, signed score and replication count](evidence/wordings.json)

## Checks run at the head

A finite instrument first identified an unsigned CUDA margin that could skip incorrect-sign updates. Its action signs happened to pass the miniature fixture, but two of the 12 scores were wrong. The original image, binary and traces were retained. The corrected signed comparison matched all 12 independent numerical values before the registered fit. Three additional zero/copy/stop calibration states matched cold serving on all 32 finite prompts each, with both oracle directions and deliberate source or terminal errors checked. [Rejected calibration](evidence/rejected-finite-v1.json) · [Corrected calibration](evidence/accepted-finite-v2.json) · [Independent FIT](FIT.md)

The GPU reloaded and traversed all 800 held rows in reverse order. Forward plus reverse scoring took 41.816 s. Across all five arms the cold D11 reproduced the 800 GPU rows and all 17,850 recorded transitions. The separate ordinary client received only tokenized prompts and source positions and matched all 3,725 learned transitions. The checkpoint is 1,202,177,144 bytes, including the unchanged base and 24,582 bytes of learned state. It was reconstructed and replayed from the private archive. The raw results were verified on Wick before the rented 4090 was deallocated. [GPU](evidence/device.json) · [Cold verifier](evidence/receipt.json) · [Serving API](evidence/client.receipt.json) · [Restore](RESTORE.md)

## Skills used

The Codex lab used the native-space-cycle skill. The trained signed words, stage, source address, original ID and committed action were preserved through D11 serialization and verified against GPU traces. [Product code](../../../../../crates/uor-r4-integer/src/discrete_d11/native_source_choice.rs)

## Still open

The causal controls locate two gaps. The first-period head already misses many opposing pairs on TRAIN and misses all held pairs. After B the head fits TRAIN and transfers on explicit counts alone. The actual instruction can control the end of a sentence through trained weights in this supplied-address task. Source discovery and open-ended chat remain open. [Next experiment](PICKUP.md)

## Plain summary

The ordinary client can use trained weights to stop after one factual word or copy another. It did that reliably for two numerical word-count formulations on new facts. The next target is the first-period decision on training pairs and the different ways people express a boundary at the same word.
