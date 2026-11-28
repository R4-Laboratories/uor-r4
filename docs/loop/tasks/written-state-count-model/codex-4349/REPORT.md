# First lexical token through the ordinary integer client

Lab: codex. [Order](ORDER.md)

## What did not work, first

[codex] The 10,650,172-byte historical UORBFY01 cannot qualify as the frozen treatment under charter §3.3. Its coarse step computes a dense matrix-vector product over 64 independent learned rows of 256 coefficients. Its next step repeats the operation for all rows in the chosen fine cell. A shorter shortlist does not remove either dense operation. The exact row and coordinate loops are in [the trained implementation](../../../../../crates/uor-r4-native-vector/src/butterfly_rotor.rs). The investigation did not execute this checkpoint or alter #4339's historical measurements. V3 additionally uses a bounded squashing gate.

No new learned response or improvement in first wrong token on human prompts has been measured. The last relevant independent ordinary generation measurement was 0/90 reference replies exact on a different checkpoint in [#4345](../codex-4345/REPORT.md). The historical ordered count deployment uses different data exposure and far more bytes than the compact checkpoint.

## What changed

[codex] Added an opt-in, state-bound lexical rank to the ordinary P4 client. The input is the actual causal P4 session, including its last-BOS reset and sliding window. Four canonical relative Cl(0,6) lookups give each legal original ID a signed contribution on top of its frozen P4 mass. The winner is committed and goes back into both ordinary contexts. The head takes 520 bytes, plus 64 bytes for the P4 binding and checksum. A zero bank preserves the prior and its lowest-ID tie policy. Training has not determined any of the words. This implementation scans all 4,095 legal IDs and recomputes the recurrence over the current prefix. Its latency, GPU parity and generalization require measurement. [State contract](NATIVE.md) · [Ranking](../../../../../crates/uor-r4-integer/src/discrete_d11/native_occurrence/lexical.rs) · [Client election](../../../../../crates/uor-r4-integer/src/discrete_d11/native_occurrence/lexical_commit.rs)

## Measurements

[codex] Wick ordinary-client fixture with opening IDs [0,2,5], carry 0x2222 and two self-fed steps. The strongest prior was ID 7. The absent candidate was ID 23, in the same coarse cell. Their addresses differed at one of the four lookups. All arms evaluated the full legal vocabulary. Scores are signed ranking values, not probabilities.

| Signed words | First two emitted IDs | Winner's first-step score | Winner's four addresses |
|---|---|---:|---|
| Zero | 7, 7 | 85,717,396,606,069 | 10, 74, 138, 202 |
| Favor absent candidate | 23, 23 | 562,945,658,453,872 | 10, 70, 138, 202 |
| Favor prior winner | 7, 7 | 648,650,170,158,197 | 10, 74, 138, 202 |

The unmodified prior mass of ID 23 was 12,884,901,744. This finite intervention proves the scoring path can overcome that gap. Extreme hand-set weights and an artificial vocabulary give no estimate of what a fit could learn.

[codex] Wick streamed and hashed the complete real P4 deployment, then measured its native candidate codebook without a model forward pass. Across 4,095 legal IDs there are 4,007 distinct four-root fingerprints. Of 30,913 unordered ID pairs sharing first and last roots, 30,824 have different full fingerprints. The remaining 89 pairs get the same geometric adjustment at every carry. Both space 223 and EOS 1 have unique full fingerprints. No held-gold row-level limit can be inferred from the vocabulary census alone. [Measured codebook](P4-ROOTS.md)

[codex] The ordered control now has a finite exact-window reference that matches the historical compact fit's EOS/BOS and nonoverlapping 257-ID chunk boundaries. The compact model plus its tokenizer occupied 10,915,565 bytes. The P4 measurement base occupies 3,987,940,843 bytes before tokenizer or the 584-byte attachment. They are different budgets. The standalone count comparison still needs the GPU producer, budgeted trie, cold client and an eligible geometric treatment at the same total bytes. [Target census and serialized budget](ORDERED-CONTROL.md)

## Checks run at the head

[codex] Wick release-mode finite P4 suite passed five tests, including a real uor-chat subprocess against a synthetic serialized P4 deployment. The lexical trace covers disabled-by-omission versus zero, original-ID flips in both directions, equality after a second BOS, independent cold replay of the second step, the 2,048-ID boundary, wrong bound state and a corrupted word. The original codebook census replay was byte-identical twice on Wick. A private, version-pinned restore reproduced the codebook receipt. The scope review and exact-head correction are recorded in [the build decision](DECISION.md). The native operator's three synthetic tests cover 1,024 action checks and bit-exact serialization. The ordered reference's five synthetic tests verify the historical window and predecessor rules. The three operator tests also ran on the M1 host. No laptop training, real model evaluation or rented GPU. [Finite checks](FINITE.md)

## Skills used

[codex] Native-space cycle for the carry, original IDs, relative read addresses, portable integer words and exact replay boundary.

## Still open

[codex] Exact-head review found both historical butterfly modules compiled into the original serving dependency archive. The corrected P4 build gates them out before compilation. The selected Wick native-vector archive retains the Q31 operators and has zero butterfly symbols. The archived research package still exposes them under its historical default. Audit the separately selected GPU training build before dispatch. Then train the new words against the strongest actual rival on first lexical content, including genuine opening spaces. Register blocked full-root matches at row level and two new wording assignments. A gain must survive self-fed ordinary replies, genuine history and fresh grading. Hardware and training time for useful language remain unknown. The compact comparator needs a real byte-matched geometric artifact and the same supervised windows.

## Plain summary

The ordinary client can now emit a word missing from the prompt when four shared geometric lookup values favor it. This removes one obstacle to training the lexical writer. The experiment measured that change on a controlled fixture. The current real checkpoint still has to learn which word to favor on an unseen conversation.
