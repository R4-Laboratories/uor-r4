# Exact feasibility of the neighboring-root score

This gate uses the checksum-verified #4334 Z3 and HiGHS build and its independent integer checker. The retained solver outputs cover the earlier, newer and pooled full source-and-NoRead systems. The supported-question split checks the 1,056 original-address comparisons, as well as gold versus NoRead, across the 13 supported newer queries and on each of those queries alone. The split repeats with the four prior candidate masks. Every design uses actual, fixed, swapped, removed and zero-weight arms, frozen and zero offsets, and margins 1 and 8. Where an actual-arm table exists, the checker replays it unchanged under the controls. The two-supplied-site mask grants the reader the annotated sites and measures only the reduced ranking problem. [Prior exact gate](../../codex-4334/gate/README.md) · [Candidate masks](../../codex-4334/rivals/README.md)

The original tools assumed that the fixed and actual query coordinates in a single row had equal feasibility. That assertion stopped the initial edge solve. The edge-only build now independently certifies both arms and reports disagreements. It preserves the requirement that every actual row has a fixed control. The clamped gate still checks the inherited equality using both original tools. The verifier records exact edge counterexamples and requires the older verifier to reject the same earlier result stream at its equality assertion.

With fresh result directories on Wick, use the three authenticated restores and the new feature tapes.

```sh
bash gate/solve.sh "$RELATIVE" "$SIGNED" "$REQUEST" "$FEATURES" "$RESULTS"
bash gate/verify.sh "$RELATIVE" "$SIGNED" "$REQUEST" "$FEATURES" "$RESULTS" "$VERIFIED"
```

The independent checker reconstructs every constraint from its original position, token ID, captured offset and signed coordinates. The verification also checks that the clamped inputs equal the archived Tail32 inputs after the bank permutation. It applies the same permutation to every archived weight vector and certificate coefficient vector. The weight dot products, integer box bounds and counterfactual replays then agree under the permutation. It also compares every independently solved clamped status and requirement count with the corresponding archived status and count. For the joint features, replay recomputes both predecessor certificates on their original comparison sets and solves the full comparison set.

Each proof receipt names the relevant design. The short certificate search exhausts single zero coefficient vectors and oppositely directed pairs on pooled native source ranking at margin 1 with zero offsets. A zero vector gives an immediate contradiction. Otherwise the search divides each coefficient vector by the greatest common divisor of its entries and matches primitive opposite directions. The independent checker replays any short contradiction against the full set of relevant comparisons. A miss excludes proofs with one or two terms that cancel to zero. The proof-support receipt gives a second minimality check. If a verified zero-sum certificate has (n) positive terms and rank (n-1) modulo 257, the rational rank is (n-1) as well. Its only linear relation up to scale uses all (n) terms, so no proper subset of those terms can cancel.

Preserve the two verdict directories and their SHA256SUMS manifest, and the independent receipts and their separate SHA256SUMS manifest. The scratch directories contain reconstructed input tapes and can be regenerated. The verification deliberately submits the permuted clamped evidence to the edge checker and requires a coefficient mismatch.

## Native edge results

The pooled actual edge table fails to rank the gold occurrence above its competing original addresses, despite each of the 13 newer supported questions admitting its own table. A pooled table can put each gold occurrence above NoRead in all four margin and offset conditions. The earlier, newer and combined full all-source and NoRead systems fail in every arm and condition.

| Newer supported candidate set | Comparisons | Pooled actual edge | Margin 1, zero offsets |
|---|---:|---|---|
| All eligible sites | 1,056 | Impossible | 33 terms give (0 geq 128{,}267) |
| Word starts | 556 | Impossible | 32 terms give (0 geq 557{,}599) |
| Passage sites | 843 | Impossible | 33 terms give (0 geq 262{,}863) |
| Passage word starts | 452 | Impossible | 31 terms give (0 geq 142{,}089) |
| Two supplied sites per question | 13 | Feasible | A bounded integer witness exists |

The all-site and passage word-start proofs use coefficient sets of ranks 32 and 30 modulo 257, respectively. Each certificate's own support is irreducible. The search across all 1,056 comparisons found no one- or two-term cancellation. The old absolute and Tail32 certificate vectors each change in all 33 comparisons and cannot establish the edge failure. The new evidence comes from solving the complete comparison set.
