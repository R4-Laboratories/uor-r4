# Removing the strongest intervening noun exposed more wrong occurrences on every prompt

## What did not work

Removing the previously winning noun address recovered **0/48 prompts and 0/24 complete opposed pairs**. The correct noun remained strictly behind at least one original occurrence at its correct governor on every row. The matched nonwinner control also improved the gold rank by one on every row and preserved every original wrong joint choice. Neither omission supplied a working reader for fresh instructions. The writer did not run. First divergent emitted tokens and exact replies through EOS are **NOT_RUN**. [Arm and family counts](evidence/results.json) · [All 48 original-address traces](evidence/prompt-traces.tsv)

## What changed

This is a diagnostic of the frozen 13,336-byte protected state elected by #4293. We reused its 24 incomplete common far TRAIN pairs. Each pair consists of TAKE and DEFER versions with opposite instructions. Both governing words and the heading were already correct. Each prompt missed exactly one noun in the joint tuple. The matched control had to be a distinct eligible occurrence strictly between the same governor and target. The preregistered interval ordering selected an occurrence of `the` on all 48 prompts. No word spelling entered the selection. The 208 other training rows received no mask. [Registration](PRE-RUN.md) · [Original-address enrollment](panel/README.md)

On the Wick RTX 4090, the diagnostic copied the native attachment table and emptied one noun site only for the affected polarity. The same original word remained available as a governor, heading or noun of the other polarity. The original prompt, candidate indices, tuple bank, D11 and learned integer state stayed fixed through all three reads. The instrument also captured the complete signed cost list at the known correct governor. Cost is minimized. Gold competition rank is one plus the count of strictly cheaper candidates, and tied candidates are reported separately. The joint reader independently selects both signed links and the heading. [Device path and finite tests](device/README.md)

## Measurements

| Wick arm on the 48 residual prompts | Gold ranks and number of prompts | Both governors correct | Heading correct | Exact prompts | Both prompts exact |
| --- | --- | ---: | ---: | ---: | ---: |
| Original read | 3, 5 and 6 on 16 each | 48/48 | 48/48 | 0/48 | 0/24 |
| Omit the original winning occurrence | 2, 4 and 5 on 16 each | 48/48 | 48/48 | 0/48 | 0/24 |
| Omit the matched nonwinner | 2, 4 and 5 on 16 each | 48/48 | 48/48 | 0/48 | 0/24 |

| Instruction family | Original winning noun | Gold rank after omitting it | New winning noun in the joint read |
| --- | --- | --- | --- |
| Planning review, 8 pairs | `board` | 3 → 2 | `the` on all 16 prompts |
| Enamel display, 8 pairs | `piece` | 5 → 4 | `found` on 8, `the` on 8 |
| Building evacuation, 8 pairs | `space` | 6 → 5 | `back` on all 16 |

All joint choices were unique. The first wrong read remained the affirmed noun on 24 prompts and the excluded noun on 24 in every arm. The complete signed candidate counts were 1,920 native and 1,872 in each intervention. Every surviving cost matched its native value. The matched `the` beat the gold noun on every row despite losing to the original winner. This explains why removing either one improves rank by exactly one, while only removing the winner changes the output address. [Per-prompt choices, signed ties and token intervals](evidence/prompt-traces.tsv)

Across the 48 original reads there were **176 strictly cheaper occurrences**. All 176 were closer to the governor than the gold noun in both bytes and BPE. Of these, 160 lay between governor and gold. The other 16 were occurrences of `drill` immediately before the governor, with zero intervening BPE IDs. The planning family had two competing addresses per prompt, the enamel family four and the evacuation family five. In the enamel pair, the four competitors were `piece`, `the`, `found` and `door`. The matched control was cheaper than gold by 3–7 integer cost units across the panel. These counts describe occurrences across reused training prompts. They do not establish how often this happens in ordinary language or which coordinate caused it. [All 176 addresses, costs and centers](evidence/all-outrankers.json)

On `enamel-display/road/18`, `keep` on DEFER and `drop` on TAKE were both at absolute BPE [100,101). Both native reads chose `piece` at [102,103) at cost 6. The gold `medallion` at [112,115) cost 22 and ranked fifth. Removing `piece` raised it to fourth in both directions. On DEFER, `the` at [101,102) and `found` at [103,104) tied at signed cost 19. The full tuple uniquely chose `found`. On TAKE, `the` won at cost 15. Removing only the matched `the` kept `piece` as the joint winner for both instructions. These are original input-token addresses. [Both full witnesses](evidence/medallion.json)

## Checks run at the head

After Wick rebooted, strict verification of the preceding physical measurement and residual manifests succeeded and the protected state matched its recorded hash. The boot time from the kernel epoch was 18:02:12 UTC. An initial clock receipt treated local time as UTC. Its replacement uses Linux `/proc/stat` directly. The preceding PR had merged at 15:50:26 UTC. The new measurement used a separate ZFS run directory and a process-checked exclusive GPU lease. Its measured command was `run-wick.sh run v1` with the sealed release. The lease recorded 0.36 s for verification and capture, with 229,656 KiB peak host RSS. The new physical capture occupied 1,411,702 apparent bytes. The experiment used no rental. [Recovery receipt](evidence/restart.json) · [Released inputs](evidence/released-inputs.sha256) · [Measurement](evidence/measurement.json)

The finite GPU fixture admitted a true recovery, a second surviving rival, equal-surface distinct addresses, role isolation and ties. The first finite validator had a syntax error. A fresh build with the corrected validator passed before the non-author FIT. The neutral device output matched the previous learned output byte for byte on all 256 original rows. All 416 comparisons on the unmasked rows agreed. The verifier checked 77,316 fields from the joint and independent reads and 46,608 fields from the signed rank lists. The reference retained the complete opposite-side candidate list. The GPU captured the failed side's individual costs and the opposite side's independent projection. Independent and primary graders agreed on all 144 enrolled prompt-arm results. [FIT](FIT.md) · [Device fixture](evidence/device-fixture.json) · [Independent verification](evidence/verification.json) · [Unchanged rows](evidence/unchanged.json) · [Grade comparison](evidence/grade-replay.json)

The private archive contains 212 files. A separate Wick download at the immutable revision matched the original upload. Regrading the downloaded physical reads reproduced all 11 result files byte for byte. The local restore, publication readback and independent download had identical replay receipts. After the three checks, the complete study used 100,841,472 allocated bytes on Wick ZFS. [Restore procedure and dependency chain](RESTORE.md) · [Download and regrade receipt](evidence/restored.json)

## Skills used

The native-space cycle guided the state and original-address parity checks. The research-architect skill guided the intervention, matched nonwinner and decision criteria.

## Still open

The model needs to select the requested original noun in the full tuple on an unmodified prompt. Giving that noun the lowest signed link cost is a training objective for the next comparison. The tuple score can also settle which address wins. The current signed transport has 32 coordinates for direction, distance, punctuation, neighbors of the noun and projections of the two BPE IDs at each edge of the gap. Interior order beyond those edges is absent from the signed transport. The joint scorer separately uses the frozen tuple bank. The observed proximity pattern leaves open whether the training signal or the signed representation is limiting. [Transport encoding](../codex-4288/export/src/main.rs) · [Joint reader](../codex-4288/learner/learner.cu)

The next trainable comparison can distinguish these explanations in three matched arms. Use current transport with the strongest rival, current transport with every occurrence that beats gold, and a same-width ordered projection of the full bounded governor-to-noun interval with every outranker. The first two isolate the training signal. The second and third isolate the representation. Keep the original governor, tuple bank, writer, state size and update budget fixed. Preserve the 24 original near pair identities and count complete far pairs before testing unseen wording. Include pairs with the same endpoint words, gap and punctuation but different scope-bearing middle text, and counterbalance the nearer distractor. Every candidate remains available at inference. [M2 grounded replies](https://github.com/UOR-Foundation/uor-r4/issues/2030)

## Plain summary

The reader found “keep” and “drop.” When its first wrong choice, “piece,” was removed, it still preferred “found” or “the” to “medallion.” We now know which other words outrank the target, their exact addresses, and how much cheaper the reader makes them. Those comparisons can supply a training signal without deleting words from a user's instruction.
