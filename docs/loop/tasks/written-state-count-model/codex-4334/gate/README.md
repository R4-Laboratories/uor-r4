# Finite signed score gate

The gate reuses the verified [#4333 solver and box-certificate checker](../../codex-4333/RESTORE.md). The patch adds an independently checked split between gold versus all wrong original source positions and gold versus NoRead, on the 13 supported cases from the newer exposed panel. Each split runs with one table across the panel and with one table per case. The unmodified full comparison design restores all unsupported cases, includes the NoRead threshold, and checks each of the two historical panels and their union, with results for individual cases and pairs. Both designs compare margins 1 and 8 with frozen and zero offsets. They test actual, fixed, swapped, removed and zero-weight features. When the actual problem is feasible, its exact table is replayed under the other features.

Repeat on Wick with the two [checksum-verified #4333 restores](../../codex-4333/RESTORE.md) and the [matched #4334 feature matrix](../transport/). Set `PRIOR` to the #4333 restore, `OLDER` to its prerequisite #4332 restore, `FEATURES` to the new extractor output, `TOOLS` to a new directory and `RESULTS` to an existing private parent directory.

```sh
bash gate/setup.sh "$PRIOR" "$TOOLS"
for cut in zero separator tail32; do
    bash gate/run.sh "$TOOLS" "$PRIOR" "$OLDER" "$FEATURES" "$cut" "$RESULTS/$cut"
done
```

The setup checks the pinned Z3 and HiGHS toolchain and builds the solver and checker offline. The run compares the zero-cut files byte for byte to the archived signed captures. It then compares all fields outside the four signed feature vectors with the corresponding zero-cut file before solving. The independent checker reconstructs the comparisons from source positions, token IDs and frozen scores, and replays all published solver words and certificates using integer arithmetic. It also checks that the inherited 33-term proof was valid under the zero cut and recomputes its coefficients and box bound at the new cut using its original addresses and multipliers.

`rank.jsonl` holds the 560 split verdicts per cut. The `old` and `fresh` files each hold 1,120 full-design verdicts and `joint` holds 2,200. The `*-check.json` files contain the replay counts, including the share of actual tables that pass directly under each control. `rank-proofs.json` summarizes pooled verdicts and counts their proof terms by the position of the rival relative to the supplied question separator. `prior-certificate.json` gives the aggregate coefficient and bound comparison for the previous proof. Preserve input and output digests with the private packet.

The same proof and witness checker can replay an existing result stream without Z3 or HiGHS. For example, use the independently regenerated fresh capture for one cut.

```sh
"$TOOLS/target/release/verify-signed-verdicts" \
    "$FEATURES/fresh-$cut.jsonl" "$PRIOR/panel/truth/key.jsonl" "$RESULTS/$cut/rank.jsonl" --rank
```

The packed tables and examples are private. These are finite feasibility results on exposed cases with gold addresses available to the solver. They supply a capacity bound and cannot establish out-of-sample behavior, fitting quality, CUDA qualification or ordinary inference quality.

## Supported-case results

On each cut, all 13 gold addresses could beat their 1,056 wrong-source comparisons with one actual table per question, at both margins with either set of offsets. A single shared table was impossible on each cut for every arm, margin and offset setting. The independent checker replayed 560 split records per cut, with 444 feasible and 116 impossible in each set.

The pooled NoRead-only comparison was feasible for actual, fixed, swapped and removed geometry at both margins with both offset settings. The same actual solution failed when directly replayed under any of the controls. The zero-weight baseline failed this pooled comparison in all four conditions.

| Frame | Offsets | Margin | Terms | Passage / question rivals | Weighted minimum | Box maximum |
|---|---|---:|---:|---:|---:|---:|
| Root zero | Frozen | 1 | 24 | 18 / 6 | 15,405,791 | 11,919,360 |
| Root zero | Zero | 1 | 33 | 27 / 6 | 150,844 | 0 |
| Separator | Frozen | 1 | 33 | 26 / 7 | 99,469,614 | 0 |
| Separator | Zero | 1 | 32 | 25 / 7 | 1,954,257 | 0 |
| Tail32 | Frozen | 1 | 26 | 20 / 6 | 3,065,781 | 2,367,488 |
| Tail32 | Zero | 1 | 33 | 31 / 2 | 8,365,541 | 0 |

Each row shows a rechecked contradiction for actual-query gold versus wrong sites. When the combined coefficient vector is zero, the contradiction holds for every weight vector of this form. A nonzero vector with a positive box maximum proves impossibility within the stated bounds.

The prior 33-term proof keeps its weighted minimum 150,844, but all 33 coefficient vectors change in both relative frames. The same multipliers now give a box maximum of 2,883,715,072 for the separator cut and 2,661,187,584 for Tail32. The relative-frame proofs in the table are new certificates.

The checker rejected a separator certificate stream supplied with Tail32 features. It also rejected flipping the claimed result of replaying the actual NoRead table under the fixed control.

## Full all-site and NoRead results

All 180 shared full-panel systems across the three cuts, three panel selections, five arms, two margins and two offsets were impossible. The absolute-anchor full result streams reproduced the prior three streams byte for byte. Across split and full designs the checker verified 5,000 records each for root zero (3,516 feasible), separator (3,524 feasible) and Tail32 (3,672 feasible). Every remaining record carried a verified impossibility certificate.

At margin 1 the actual-query results were:

| Panel | Feasible rows per cut | Feasible pairs with frozen offsets per cut | Feasible pairs with zero offsets, root zero / separator / Tail32 |
|---|---:|---:|---:|
| Earlier nine pairs | 18 / 18 | 8 / 9 | 8 / 8 / 8 of 9 |
| Newer nine pairs | 17 / 18 | 8 / 9 | 6 / 7 / 7 of 9 |
| Combined 18 pairs | 35 / 36 | 16 / 18 | 14 / 15 / 15 of 18 |

In the combined panel at margin 1 the separately solved swapped arm matched the corresponding actual pair counts. With frozen offsets the fixed arm passed 9, 9 and 15 of 18 pairs across the three cuts. With zero offsets it passed 0, 0 and 11. The removed arm passed 9 pairs with frozen offsets and zero with zero offsets at each cut. Direct replay of each feasible actual pair table under fixed, swapped or removed features passed zero pairs across both offset settings and all cuts.
