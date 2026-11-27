# Assistant opening boundary audit

This post-run audit counts fixed training events from the registered [#4312 fitted plan](../../codex-4312/FIT.md). Run it on Wick with the original donor and tokenizer files. The full source-event map and geometry subset must each agree with their mask. A source assistant event requires an exact six-ID marker in a masked frame after a prompt, followed by an eligible reasoning or answer token within the lineage group's visible document. The audit verifies the visible EOS and auxiliary BOS framing separately for each group. Counts after ID 28 measure the input to the query-only count family. That family uses the preceding token as its key.

For genuine openings, the audit counts the immediate target and first non-whitespace token, preserves zero-length and multistep whitespace runs, and compares to retokenizing the same source piece with the protocol-2 `" " + text.trim()` rule. The output contains aggregate counts, vocab IDs and token surfaces.

The separate `held-label-boundary` binary verifies all 810 original targets using the frozen panel and tokenizer. It encodes each trimmed reference answer without the protocol-2 leading space and emits only pair ID, side, cohort, the canonical and alternate IDs, and status. Use the sidecar for post-run comparisons of sealed ranks. Pair comparisons require distinct alternate IDs. Keep the sidecar and all source data on Wick.

```sh
prior=/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-visible-answer
current=/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-question-swap
tokenizer="$prior/panel/inputs/tokenizer.json"
build_target=$(mktemp -d "$current/boundary-v1/replay-target.XXXXXX")

cargo build --release --locked --offline --target-dir "$build_target"

"$build_target/release/question-swap-boundary-audit" \
  "$prior/donor/pair-01" "$prior/qualification/pass03/plan-aot/plan.json" "$tokenizer"
"$build_target/release/held-label-boundary" "$current/panel/frozen-v2" "$tokenizer"
```

The sealed `boundary-v1/donor-summary.json` has SHA-256 `924a799f29741880e3e3ffacca262f57a8a4cd90a441c11c75677f02ee376087`. The private `boundary-v1/held-label-boundary.jsonl` has SHA-256 `b2ecd23c514de40cd723f92b078255795c09249d18bf3ede0ab949c2efc6a9a3`.
