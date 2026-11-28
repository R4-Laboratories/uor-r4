# Ordinary geometric weights for end-of-reply decisions

Lab: codex. [Order](ORDER.md)

## What did not work, first

The fitted EOS states completed **0/90 exact replies**. The base writer already diverged before the reference EOS in all 90 cases. Even an oracle that knew exactly when to stop could have recovered **0/90**, including **0/63** references short enough for the output budget. The geometric fits therefore failed the registered promotion rule. Keep the default off.

The donor subset had prior exposure. Every strict held opening is single-turn. This experiment cannot qualify transfer to new conversations, multi-turn recall, or the grounded stop/continue decision after a copy. Twenty-seven of the 90 reference replies need more than the registered 192 generated IDs including EOS. The release signal only promotes EOS. Once the prior proposes a wrong content ID, this operator can keep it or end the reply at that step.

## What changed

A versioned, opt-in D11 state adds sixteen signed integer lookups to the ordinary writer before commitment. All original IDs and the legacy default are preserved. The geometric arm uses relationships between the actual read roots, recurrent roots and committed writer vector; the ordered arm indexes pairs of original causal IDs. Both learn 4,096 signed i16 words. Zero or a nonpositive score keeps the ordinary ID, including an EOS the ordinary writer already proposed. A positive eligible score emits the bound EOS. Authenticated copies and supplied controls retain precedence. [Operator](../../../../../crates/uor-r4-integer/src/discrete_d11/native_occurrence/termination.rs) · [Registered trial](SHEET.md)

## Measurements

On the 4,060 historic D87 TRAIN records, 440 supervised final EOS and 129,641 real immediate content continuations met the causal-prefix and preceding-content requirements. Exact matching by preceding character class, completed reply length, opening-length bin and genuine history retained 237 TRAIN, 24 internal development and 47 internal held pairs. Distinct connected components stayed in one partition. The fit never selected a threshold, epoch or seed on the 24 development pairs. Previously exposed data, including long answer overlap between partitions, limit interpretation. [Supply and overlap](supply/README.md)

The Wick RTX 4090 trained two fixed seeds per arm. Each fit made 7,584 paired update opportunities in 32 epochs. Each of the four cold repeats reproduced its 4,096 signed words and update counters exactly. The 46.27-second fit pipeline includes ordinary feature capture, device updates and held scoring, two complete fits per setting and artifact packing. No rental charge was incurred. A geometric D11 occupies 1,202,185,796 bytes and an ordered D11 occupies 1,202,185,795 bytes, each with a separate shared 265,393-byte tokenizer. The payloads are exactly equal at 8,192 bytes and the serialized arm names account for the one-byte file difference. The writer base is 1,202,177,144 bytes.

| Teacher-forced action on 47 held pairs | STOP emits EOS | CONTINUE keeps a non-EOS ID | Both actions correct |
|---|---:|---:|---:|
| Frozen native writer | 0 | 46 | 0 |
| Fixed punctuation with native fallback | 4 | 42 | 0 |
| Geometric seed 1 | 38 | 36 | 28 |
| Geometric seed 2 | 39 | 37 | 30 |
| Ordered token seed 1 | 35 | 30 | 21 |
| Ordered token seed 2 | 35 | 32 | 25 |

These continuation counts credit keeping a content opportunity open, even if the original writer chose the wrong content ID. The geometric arms separately made 11 and 10 premature stops at continuing boundaries. The fixed punctuation rule is matched away by design within each pair. Its native fallback can still depend on the prefix. The disabled and fixed baselines both got zero complete pairs. The broader long-context comparison of count and persistent geometry remains open.

### Self-fed ordinary replies

Every held turn was an original plain-text user message sent through `uor-chat --say`. The frozen client emitted the wrong first ID on 58/90 replies. On all 32 remaining replies it emitted the wrong second ID, including 29 whose first ID was the correct standalone space 223. Space was the most common erroneous emission, on 56/90 replies. A held action win credits keeping the path open. At those 47 internal reference positions, geometry actually chose the correct next content ID on just 0/47 and 1/47, respectively.

| Arm | EOS emitted /90 | EOS earlier than reference /90 | Reached cap /90 | First mismatch at token 1 / 2 | Exact through EOS /90 |
|---|---:|---:|---:|---:|---:|
| Frozen, disabled, zero (each) | 2 | 2 | 88 | 58 / 32 | 0 |
| Fixed punctuation, reconstructed | 18 | 16 | 72 | 58 / 32 | 0 |
| Geometry seed 1 | 90 | 81 | 0 | 61 / 29 | 0 |
| Geometry seed 2 | 90 | 82 | 0 | 65 / 25 | 0 |
| Ordered seed 1 | 90 | 88 | 0 | 58 / 32 | 0 |
| Ordered seed 2 | 90 | 88 | 0 | 59 / 31 | 0 |

The geometric models emitted EOS as the very first ID on 14 and 19 replies. Each also emitted EOS after only whitespace on 51 and 47 replies. Relative to the base, the first mismatch moved earlier on three and seven replies and later on none. Exact-match grading compares with the recorded reference. Open-ended usefulness and human preference were not graded on this panel. [Sealed scores](evidence/held.json) · [Descriptive analysis and exact-reference ceiling](evidence/descriptive.json)

## Checks run at the head

[codex] The preregistered finite CUDA exercise compared signed scores, words and saturated updates to an independent Rust implementation, including target swap, ties, same-key cancellation and malformed inputs. The input stage independently checks real protocol prefixes, targets and truth masks. The CUDA scorer receives blind causal inputs for held scoring. The CPU and device use the same packed words. The study measures CUDA fitting and the decision over shared integer keys; full-device writer parity and browser serving remain unmeasured. [FIT](FIT.md) · [Device checks](device/README.md)

All 376 fitted CUDA boundary scores and committed IDs matched actual cold client readback, including the writer vectors, codes and both feature sets. Off and zero controls each preserved all 94 boundary proposals and all 90 complete trajectories. The panel ran 1,194 ordinary invocations on Wick. At most eight CPU serving clients ran concurrently, and the tool-seal to result-seal interval was 14 minutes 38 seconds. This timing includes cold reopening a 1.20 GB file per invocation; it is not a warm generation benchmark. The immutable primary summary digest is `53deea940160337b3c7c7955570c6689501e4be84e0ac1d4356a4338d0b710f3`. The result manifest is `82c499bf5fe5380c618f15c43e1502702cd918528a0cc43b773ea9deeb3f3eb3`. An early console tally silently compressed adjacent repeated device rows. The tables here come from the full binary record parser and sealed cold-client returns.

The post-fit trace audit found a span-step diagnostic that retained its tentative ID after EOS promotion. The corrected executable reports both the span proposal and the final commit. An invented supplied-source reply copied original token 1044, then the span proposed 316 while the learned gate committed EOS ID 1. The corrected client reproduced every frozen feature row and eight complete ordinary outputs across all four learned states byte for byte. All four D11s also rebuilt byte-identically from a single 7,568-byte compressed weights-and-bindings archive plus the shared base. [Composition evidence](evidence/composition.json) · [Restore and replay](RESTORE.md)

The exact-head review also detected a compatibility defect outside the registered text panel. On an original prefix where the base raw parity interface emitted `[1,73,1010]` through a three-ID cap, the first packaged Off/trace implementation stopped at `[1]`. The corrected interface preserves all three raw IDs with the gate disabled or trace-only. When EOS merely occupies the final slot of a one-ID raw request it correctly reports a length cap. Opted-in generic mode and ordinary text generation still close on EOS. The main study's raw boundary panels each requested one ID and graded that ID and state; its self-fed panels used text. [Cold regression](trial/raw-parity.sh)

## Skills used

[codex] The native-space cycle guided the device update, signed-state packaging, cold integer replay and ordinary committed-ID checks.

## Still open

[codex] Useful spontaneous language and original-source retrieval on unseen requests remain open. The next lexical study must demonstrate that an actual trained score can raise a previously absent content token through ordinary commit. [Next experiment](PICKUP.md)

## Plain summary

The learned geometric weights often recognized a real answer ending when given the human's answer so far. When the model had to supply the words itself, every reference had already diverged by token two. Changing when it stopped could not recover a single exact response. Train the first content decision against the strongest competing token before expanding stop-policy work.
