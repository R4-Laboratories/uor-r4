# Which signed writes move the first wrong reply token?

## What did not work

None of the four reconstructed states answered all four versions of a single held scoped instruction correctly through EOS (0/64 contexts). On the 128 matched selector switches, the period-only component chose STOP and COPY correctly on eight. Adding the post-copy writes erased exactly those eight and introduced zero new correct switches. The other 120 already missed under the period writes. All eight erased switches belong to the forwarded-note family across repeated facts. The two stages interfere, but separating their weights alone cannot explain or close the first-stage gap. Balanced remains the integration baseline. [Registered test](SHEET.md)

## What changed

No optimizer ran. Starting from the same 40,966 linked bytes, we reconstructed each stage's net signed writes from the 6,144 slots of the actual #4325 joint fit. Four checkpoints append 4,096 signed 16-bit words apiece. The sum matches the archived 49,158-byte joint weights byte for byte. A component includes only the updates that occurred on this trajectory; fitting its stage alone would change which margins trigger updates. The same SM89 image, geometric scorer and integer writer replayed four rollouts per prompt. In learned mode, the opener and first period are supplied, the writer performs its own release decisions, and a copy must come from its original factual address. [Reconstruction receipt](evidence/reconstruction.json) · [Device inputs](evidence/device-inputs.sha256)

## Measurements

The 512 previously exposed held rows comprise 256 crossed instructions and 256 direct requests. A wrong token at position 3 is the decision immediately after the opening word and period. Position 4 is the decision after the next copied word B. The table shows actual emitted IDs and terminal EOS, including early stops. The final column tests the direct two-versus-three choice after manually supplying only the first choice and authenticating the same original B in both prompts.

| Retained writes | Exact through EOS / 512 | First wrong at 3 | First wrong at 4 | Selector switches / 128 | Direct switches after B / 64 |
|---|---:|---:|---:|---:|---:|
| None, frozen linked | 56 | 232 | 224 | 0 | 0 |
| First period | 72 | 184 | 256 | 8 | 0 |
| After B | 192 | 320 | 0 | 0 | 0 |
| Both, measured joint | 192 | 320 | 0 | 0 | 0 |

Across the 256 scoped rows the four states completed 24, 8, 128 and 128 respectively. The B-only state stopped at the first decision on every held prompt. The full state correctly answered 24 long scoped requests and 104 short ones. Its 128 exact scoped replies come from a different set than the 128 correct short replies under B-only. With the first choice supplied, the full state closed 80/128 required scoped B releases. Supplying both actions gave 512/512 exact replies in every arm. No unassisted source discovery or open-ended chat was tested.

**One loss by identity.** The two variants of forwarded-note pair `jf-held-Jh0500-o1` request “lemon.” and “lemon. shelf”. Their period scores (short, long) are (+37, +112) without residual writes, (−8, +146) with period writes, (−379, −256) with B writes and (−424, −222) with both. Native negative means STOP and positive means verified COPY. The last intervention increased the separation from 154 to 202 while pushing both scores below zero. On the longer request the first error moved from token 4 to token 3. Period-only copied “shelf” (2169) and then emitted 2222 instead of EOS (1). Full emitted EOS at token 3 before copying “shelf”. The short reply stayed exact in both states. Across clause-order pairs, the period component and the full state each made eight strict switches, but the two sets of eight were disjoint.

**After a common B.** Period writes alone pushed all 320 eligible held scores after a supplied first copy above zero. Post-B writes alone pushed all 320 below zero. On the 64 direct two/three contrasts, the full weights ranked the longer instruction above the shorter on all 64 but crossed zero on neither member pair (0/64 correct switches). In the bakery wording both full scores were positive (+2, +21). In the trip wording both were negative (−38, −29). Even choosing the best single threshold in hindsight can split only 32/64 direct pairs. At the first period it can split only 24/128 held selector pairs, with 64 pairs already strictly ordered backwards by the full model. These are upper bounds for changing a single global threshold with ranks held fixed. No threshold was fitted.

**Where the optimizer wrote.** Each one of the 128 held selector pairs visits different sets of geometric addresses. Each of the 64 four-cell contexts also has a nonzero selector-by-clause-order difference after subtracting the two STOP cells from the two COPY cells. Every held direct two/three pair visits different addresses after B. An identical residual representation therefore does not force these failures. The existence of a common assignment of weights that generalizes across authors is still open.

For the original TRAIN rows with opposed selector siblings, 7,769 of the 8,432 realized period writes (92.1%) landed on an address visited by both siblings. Such a write moves both period scores by the same amount and cannot change their separation. Only 663 writes changed their separation. The imbalance was greater in matched direct post-B training, where 1,077 of 1,104 writes (97.6%) landed on shared addresses. In 42 of the 69 direct post-B slots that wrote anything, every write was shared. Furthermore, on all 64 exposed held direct pairs, the post-B component *reduced* the longer-minus-shorter score difference. The first-period component *increased* it on all 64. These findings motivate spending matched updates on the coordinates that can move the two instructions in different directions. They do not establish transfer. [Address and write accounting](evidence/geometry.json) · [Paired margins and token traces](evidence/grade.md)

## Checks run at the head

An initial device manifest enumerated only the zero residual. The author and independent pre-run reviewer both caught it. No research rows ran under that manifest. The corrected gate binds all four expected weight hashes, code, seven tapes and the physical 4090. The finite device test checked the same-token wrong-address and missing-address controls and the fixed stop after C. [FIT](FIT.md) · [Device authorization](evidence/go.json) · [Finite controls](evidence/finite.json)

Across seven panels, all 3,324 aligned first-period and 2,004 authenticated-B scores satisfied full = period + post-B − zero exactly. The independently reconstructed address contributions also matched all four GPU states on 254 TRAIN and 192 HELD pair differences and on 64 four-cell mixed differences per split. The zero and full endpoints reproduced 28 historical files byte for byte across both input orders, totaling 13,296 records. All four D11 artifacts ran through the ordinary client on the 512 HELD inputs, and the full artifact matched the archived joint D11 byte for byte. [Score and threshold cross-checks](evidence/component-crosscheck.json) · [Product parity](evidence/product-parity.json) · [Archive and restore](RESTORE.md)

On TRAIN the full state recovered 48/128 fresh selector switches, 32/64 direct post-B switches and 16/64 exact four-way contexts. None transferred to these three HELD measures. All 50 original protected pairs passed through EOS in the zero state. Period alone, B alone and full each retained 0/50 complete pairs. These are exact identity comparisons. [Seven-panel grade](evidence/grade.md)

The scorer completed in 7 minutes 32 seconds. The controller recorded 11 minutes 4 seconds between allocation and destruction at $0.869/hour, about $0.16 of GPU rental before provider extras. The shared board post emitted a warning; the append-only controller ledger recorded allocation, lease, release and deletion. All 96 returned files verified on Wick before teardown.

## Skills used

The native-space-cycle protocol fixed the state layout, geometric scorer, original addresses, integer writer and serialization boundary. Ordinary D11 verification replayed the same four state images.

## Still open

An open-ended conversational model must find facts and produce its own opening. The present experiment supplies that entry to isolate a failure in trained instruction-dependent weights. No milestone or product artifact was promoted. [Next experiment](PICKUP.md)

## Plain summary

Later training overwrote eight useful instruction switches. Most were missing before the later training ran. More individual prompts stopped correctly while the model continued giving the same response to opposed instructions.
