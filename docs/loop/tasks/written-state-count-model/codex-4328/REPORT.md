# Ordinary chat can copy a requested word and still fail before, at, or after the period

## What did not work

The unmodified chat client completed 0/48 reference replies through EOS. Enabling its existing authenticated opening raised correct supported first tokens from 4/36 to 13/36, yet still completed 0/48. The occurrence reader accepted all 34 fenced prompts and rejected all 14 unfenced prompts. It made a false copy on all ten unsupported prompts it parsed. None of the twelve unsupported prompts produced the requested fallback. [Panel](panel/README.md) · [Supplied and disabled controls](evidence/control-summary.json)

## What changed

The ordinary Wick client received 48 raw text messages, with labels, source positions and expected replies available only to independent grading and diagnostic processes. The base, balanced and paired artifacts ran with the native default writer, the existing authenticated-copy option and an explicit occurrence-reader ablation. The new ablation retains the lower marker and trunk. The prior read-mode option still selected an occurrence on all 12 stock probes. The trace records each original source, copy claim, gate, proposed ID, commit, successor and termination. The reference runs reopen the artifacts without the trace. [Pre-run review](FIT.md) · [Raw client protocol](runtime/README.md)

Seven supported prompts naturally reached the correct source, committed its token and generated the requested period. An opt-in release path now consults the frozen stage-zero weights in the real decoder at that point. Its runtime trigger knows only that the actual first copy was authenticated and the actual second event generated a period. An independent address check identifies which triggered replies had the correct opening. All 48 raw prompts were submitted to each of the three release artifacts. [Conditional protocol and action semantics](release/README.md) · [Independent conditional FIT](release/FIT.md)

## Measurements

Positions below are zero-based in the protocol-framed prompt. Error positions are one-based reply tokens including EOS. Each count refers to the same fixed, authored development panel. The 432 primary traces repeat 48 inputs across nine combinations. Repeated inputs and matched wordings are dependent.

| Ordinary mode, per artifact | Correct first ID on 36 supported | First wrong at 1 / 2 / 3, all 48 | Any EOS | Exact through EOS |
|---|---:|---:|---:|---:|
| Native writer | 4 | 44 / 2 / 2 | 5 | 0 |
| Authenticated copy | 13 | 35 / 6 / 7 | 0 | 0 |
| Occurrence disabled | 0 | 48 / 0 / 0 | 0 | 0 |

The native writer's five terminal replies emitted EOS at the opening. With the authenticated writer, the 24 fenced supported prompts split into 13 correct original addresses and 11 other occurrences copied. All 12 unfenced supported prompts failed before candidate ranking. On the delivery quartet, replacing the blank line with a space reduced correct sources from 3/4 to 0/4. The words and their order were identical within each matched pair. Both questions selected their own correct original source on 5/18 question-change pairs. Both orders retained the requested person on 5/18 fact-order pairs. [Pair identities and whitespace-only comparisons](evidence/matched.json)

On the four fenced duplicate-label prompts, the reader copied the correct occurrence on all four. With both facts in the same order, changing Rosa to Oscar moved the source from address 19 to 36 even though both emitted ID 1074, ` green`. Supplying the correct original source gave 24/24 eligible copies and 10/24 correct following periods. The four distinct wrong-address claims carrying the identical ID were at the minimum writer distance and were all refused. Missing claims produced zero copies. A correct address check therefore protects the write, but selecting and claiming another person's fact can still pass it. [Control definitions](controls/README.md) · [Counts](evidence/control-summary.json)

A counterexample shows where the wrong word originated. Asked for Carter's crate mark, the reader ranked the correct ` blue` at address 17 sixth (score 3,793) and copied Gomez's ` red` from address 32 (score 7,161). Reversing the fact order recovered Carter's ` blue`. Asked for Daria's actual folder color after “Daria's folder is not blue. Ruben's folder is red,” the reader selected Ruben's ` red` from address 19 with score 6,346 against a NoRead threshold of −5. The writer copied exactly that wrong source. The other two unsupported prompts lacked the fence and failed framing. Parsed semantic abstentions were 0/10.

### What changed at the first real period

The native client emitted ID 73, `g`, immediately after all seven correct unassisted word/period openings. Six needed EOS and one needed ` red` (677). The explicit release path actually ran on 16 prompts per artifact. Nine of those triggers followed an incorrect or unsupported source. The grader excludes them from the seven verified boundaries while retaining them in the full reply counts. [Conditional outcomes](evidence/release-summary.json)

| Stage-zero weights in the ordinary client | Exact / 48 | Any EOS / 48 | First wrong at 1 / 2 / 3 | Token-three action on 16 triggered replies |
|---|---:|---:|---:|---|
| Linked reference with zero additional residual | 3 | 7 | 35 / 6 / 4 | 7 EOS, 9 native ties |
| Row-wise residual | 0 | 0 | 35 / 6 / 7 | 7 unresolved copy requests, 9 native ties |
| Paired residual | 3 | 7 | 35 / 6 / 4 | 7 EOS, 9 native ties |

The paired residual and zero-residual reference produced identical ID sequences across all 48 prompts. Compared with the native client after an authenticated opening, both recovered Carter’s ` blue.` on the reversed-fact prompt and the two same-ID ` green.` replies for Rosa and Oscar. For all three, the first error at token 3 became exact EOS. On Carter the signed scores were −4 (reference), +20 (row-wise), and −45 (paired). The row-wise copy request had no verified second source, so the client preserved its native `g`. Four of the seven EOS completions under paired and reference had already copied the wrong first token. Three of those four were unsupported requests.

The remaining four verified boundaries included a command to stop after the North code and a command to continue with the South code. They had different fact orders. Both began with the correct native ` blue.`; one needed EOS and the other needed ` red`. The native reply gave `g` on both. In both cases all three release weights received a zero score. In source, the trained feature extractor searches for a question-mark ID after the last blank line before the assistant marker. Without that ID it returns zero before it visits a learned feature. All nine triggered prompts missing this delimiter scored exactly zero under all three states. Seven other triggered prompts had an addressable instruction, and only three of those seven had the correct source. Accordingly the experiment identified three real exact recoveries, zero incremental wins from the paired residual, and no recovered continuation. [Delimiter audit](evidence/release-scope.json) · [Native instruction extraction](../../../../../crates/uor-r4-integer/src/discrete_d11/native_word_ending.rs)

## Checks run at the head

The native writer and model words stayed frozen. The served raw prompts were retokenized independently and graded against the actual commits. The ordinary control compared every traced and uninstrumented output. The supplied-address verifier replayed the full first two ordinary authenticated writer events. The focused occurrence tests passed 3/3 on Wick with a compatibility correction to an existing unrelated test import in the isolated test build. The untouched checkout's test-target build hit that pre-existing import error. Production binaries were built without that correction. [Runtime source and finite-test receipts](runtime/README.md) · [Reproduction and archive verification](RESTORE.md)

The revision-pinned archive was downloaded afresh. All 432 ordinary grades, all 144 conditional grades and all 48 native control rows reproduced. The cold sample reran 72 ordinary and 12 conditional raw-text requests and matched the archived outputs. Wick executed the primary 876 client invocations in 422 seconds; that includes repeated process starts and artifact loading. [Runtime receipt](evidence/runtime-receipt.json) · [Cold replay manifest](evidence/control-readback.sha256)

## Skills used

The native-space-cycle protocol bound the original token addresses, integer reader and writer events, and serialized weights to the self-fed client trajectory. This was frozen integer serving on Wick. No model fit or GPU rental ran.

## Still open

The first upstream gap is selecting an asserted source or declining to copy in an ordinary user turn. After a correct copy the ordinary writer still needs a reliable next-token choice, a reachable instruction state, and a second authenticated address for continuations. [Next experiment](PICKUP.md)

## Plain summary

The model can follow some changes in who the user asks about and can preserve the exact original word through its writer. It also confidently copies a different person's fact and overlooks an entire single-paragraph prompt. The next weight-training run should expose those choices to training and evaluate them in the ordinary client.
