# Study of the D21 preceding-delimiter requirement

The historical D26 recurrence carrier bound no original assertions across the token-verified passage separator on either exposed panel. Removing only D21's preceding-delimiter requirement admits 256 genuine same-word matches across all 36 prompts. This raises candidate coverage to 13 of 28 supported prompts and yields the correct original address under at least one trace on 3 of 28. It also yields a selected address on 7 of 8 unsupported prompts.

The emitter enumerates the original ordered-role query projection and original equal-token slices in front of the cut. Before admitting a slice it checks the D21 START class, original whole-word bounds and the requirement that the fact start precede the query start by at least two positions. It distinguishes those checks from the preceding delimiter and confirms that the original assertion iterator agrees with the resulting native membership. The unmodified contextual candidate path provides the assertion fence, lexical candidates, completed-copy filter, recurrence coordinates, scoring and selection. Each native pair must exactly reproduce the public selection and full trace. The historical positive replays source position 51. Four additional native pairs across Tail32 pass the same trace check, while the public method rejects every delimiter-only pair.

The true cut uses the fixture's token-verified separator. Tail32 instead splits at the ordinary trailing 32 tokens. In both views only whole projected query spans within the current-user range and facts wholly before the selected cut are enumerated. The emitter reads no truth keys. The grader joins authenticated exact source positions after emission. Counts of eligible or chosen addresses deduplicate positions within each prompt across all bindings. A supported prompt can contribute both a correct and a wrong choice because different bindings can choose different sources.

| Cut and arm | Earlier fenced gold / eligible gold / any correct selection | Newer fenced gold / eligible gold / any correct selection | Unsupported prompts with a selection |
|---|---:|---:|---:|
| True separator, native | 0 / 0 / 0 of 15 | 0 / 0 / 0 of 13 | 0 of 8 |
| True separator, delimiter relaxed | 14 / 8 / 0 of 15 | 13 / 5 / 3 of 13 | 7 of 8 |
| Tail32, native | 1 / 1 / 0 of 15 | 1 / 1 / 1 of 13 | 1 of 8 |
| Tail32, delimiter relaxed | 11 / 6 / 1 of 15 | 13 / 5 / 3 of 13 | 7 of 8 |

In each arm, every gold address reachable inside a fence is a projected lexical start in at least one of the retrieved intervals. The completed-copy requirement removes 14 of 27 of those addresses in the true-separator relaxation before scoring. On that view every supported prompt also has a wrong eligible address, and 21 of 28 have a wrong selected address. All eight unsupported prompts have an eligible site. Their 67 deduplicated wrong candidate sites and 19 wrong selections are counted per prompt. The relaxed Tail32 view has 64 candidate sites and 17 selections across the eight unsupported prompts.

The exact-ID slice census finds 263 matches across the true separator and 278 across Tail32. Seven in each view fail the original whole-word right bound. None fail START or the positional gap. All 256 surviving true-boundary pairs fail the native preceding delimiter. At Tail32, 267 of 271 surviving pairs fail it. These counts concern slices whose identical query token sequence has already passed the original query projection.

This is an exhaustive diagnostic under the supplied split, original carriers and these exposed prompts. It measures source-address coverage over all enumerated bindings and establishes neither a query reader nor an end-to-end reply.

```sh
bash relax/run.sh "$RELATIVE" "$SIGNED" "$REQUEST" "$D30_RAW" "$RESULTS"
```

The runner builds a study-only diagnostic in a fresh staged #4332 checkout, authenticates the D26 artifact, pinned historical positive and #4334 input tapes, verifies reversed-input replay, and grades against the restored sealed keys. Preserve the two forward observation streams, two summaries, control receipt and `RELAX.sha256`. The running model source in the shared checkout has no study entrypoint.
