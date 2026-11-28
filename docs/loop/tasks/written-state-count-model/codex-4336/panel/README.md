# Authenticated span-release serving panel

This finite panel uses three fresh authored fact families, each with 7, 8 or 9
actual bound tokenizer IDs in the first value. Within each family two fact orders,
two wording families and the two reply requests are fully crossed. Each of the 12
matched pairs shares all facts and source tokens. Its question alone requests
termination after the first fact or continuation into a second original fact.
A different original fact in every passage has the same first token ID. Three
additional rows request unspecified, denied or expressly excluded information.
The author file, raw prompts, independent grade key, exact authority, perturbed
claims and scheduled arms were sealed on Wick before decoding.

The grader rebuilds protocol-2 IDs and the legal user range from the raw authored
prompt with the artifact-bound tokenizer. It aligns the authored value surfaces
to original byte boundaries and assigns absolute prompt addresses. It tokenizes
the separately assembled human response and appends the actual protocol EOS to
check exact replies and the first different committed token. The grader checks
the unchanged historical and current-base ordinary client records, and compares
trace-on and trace-off emission on bounded anchors. It reads the audit trace to
check claimed and copied addresses, self-fed input length and previous tokens,
the requested next action, whether EOS occurred, and whether a cap, delimiter,
rejection or authenticated endpoint preceded it.

Historical, current-base and new ordinary authenticated runs cover all rows.
Stock defaults and quiet authenticated runs cover one paired anchor per length
and all three unsupported rows. The explicit diagnostic runs supplied correct,
native and swapped next actions throughout the supported grid. Anchor runs
exercise a same-first-ID address from the competing fact, a claim ending one
token early or late, missing claim, missing proof, and authority tied to the wrong
prompt. Two anchors exercise a second claim with the wrong address. The
delimiter control deliberately binds matching claim and authority through the
period after a true 7-token value, exposing delimiter release independently of a
claim/authority mismatch. Three unsupported examples receive an original
lexical claim without fact authority. The quiet correct-action control covers
the six supported anchors. All runs use balanced D11, protocol 2, greedy
temperature zero, seed zero and a 16-token decode limit.

Use a private fresh output directory on Wick with the pinned tokenizer, clients
and balanced D11 from the study restore instructions. The program builds against
the repository tokenizer and has no dependencies on the diagnostic transition.

```sh
cargo build --offline --locked --release --manifest-path "$PANEL_SOURCE/Cargo.toml" \
    --target-dir "$TARGET"
"$TARGET/release/span-panel" freeze "$TOKENIZER" "$AUTHOR" "$PANEL"
"$TARGET/release/span-panel" run "$TOKENIZER" "$PANEL" "$HISTORICAL" "$BASE" \
    "$DIAGNOSTIC" "$BALANCED" "$CAPTURE"
```

The capture wrapper can collect the `historical`, `new` and `diagnostic` job
groups independently. The `grade` command accepts those tapes and rewrites the
full record in the original frozen job order.

```sh
"$TARGET/release/span-panel" capture "$TOKENIZER" "$PANEL" "$HISTORICAL" "$BASE" \
    "$DIAGNOSTIC" "$BALANCED" "$OUT_HISTORICAL" historical
"$TARGET/release/span-panel" capture "$TOKENIZER" "$PANEL" "$HISTORICAL" "$BASE" \
    "$DIAGNOSTIC" "$BALANCED" "$OUT_NEW" new
"$TARGET/release/span-panel" capture "$TOKENIZER" "$PANEL" "$HISTORICAL" "$BASE" \
    "$DIAGNOSTIC" "$BALANCED" "$OUT_DIAGNOSTIC" diagnostic
"$TARGET/release/span-panel" grade "$TOKENIZER" "$PANEL" "$CAPTURE" \
    "$OUT_HISTORICAL/records.jsonl" "$OUT_NEW/records.jsonl" "$OUT_DIAGNOSTIC/records.jsonl"
```

The deterministic outputs are `records.jsonl`, `scores.jsonl` and `summary.json`.
The raw decoder records retain every JSON field except the relocated absolute
artifact path, together with the subprocess stderr. Full prompts, keys and
per-case data remain in private custody. A word sequence through EOS and a
correct original-address sequence have separate scores. Supplied actions
establish a serving ceiling for the frozen panel and do not grade learned action
selection or source admission.

The 7- and 8-token spans yielded 16/16 exact replies through EOS with the
original addresses when both source and action were supplied. All eight
corresponding stop/continue pairs passed. With the same authenticated source
and native action, all 16 first errors occurred immediately at release. Token
offsets are zero-based. The stop target at release was EOS (ID 1), and the
continuation target was ID 269. The native writer emitted ID 210 on all eight
7-token rows, and IDs 16 or 465 on the eight 8-token rows. All 9-token rows
reached the eight-token cap in the traced oracle, native and swapped arms.
Swapped actions yielded the opposite reply through EOS and from the respective
original addresses on all 16 rows that reached the authenticated end.

On the ordinary authenticated baseline, 22 of 24 supported prompts caused an
original-source claim and copy. Every first token and selected original copy
address missed the labeled target. Three supported rows ended at EOS with
incorrect replies. The unspecified and denied cases also caused false original
claims and copies. The excluded case did not. Supplying a lexical claim without
authority rejected the diagnostic at offset zero on all three unsupported rows.
The ordinary fallback preserved its original behavior, including the two false
source claims. The historical generated-period scorer triggered on 12 supported
and one unsupported ordinary row. It triggered zero times in the correct,
native and swapped diagnostic arms with an authenticated first span. The
diagnostic supplies a different release condition after its own source commits.

See [aggregate evidence](../evidence/panel.json) for the full action and length
breakdown, wrong-token counts and replay checks.

## Replay the guarded client

The library guard checks the tokenizer's content identifier before using the
fixed delimiter and EOS IDs. The captured CLI checks the tokenizer file digest
against the artifact. Use the recorded guarded executable on Wick after the
cold restore and choose fresh private directories for `GUARD_PANEL` and
`GUARD_TRANSFER`. The panel and transfer manifests cover deterministic
outputs. The new panel's `inputs.json` records the guarded executable digest
separately. [Replay receipt](../evidence/guard-replay.json)

```sh
GUARDED="$RUNS/20261127-codex-semantic-span/guard-target/release/uor-chat"
guard_sha=$(jq -er .guarded_client_sha256 "$UNIT/evidence/guard-replay.json")
printf '%s  %s\n' "$guard_sha" "$GUARDED" | sha256sum --strict -c -
"$RESTORE/target-panel/release/span-panel" run "$CORE/tools/tokenizer.json" \
    "$RESTORE/panel" "$CORE/tools/uor-chat" "$RESTORE/target-base/release/uor-chat" \
    "$GUARDED" "$BALANCED" "$GUARD_PANEL"
(cd "$GUARD_PANEL" && sha256sum --strict -c "$RESTORE/payload/REPLAY.sha256")
bash "$UNIT/transfer/run.sh" "$RUNS" "$GUARDED" "$GUARD_TRANSFER"
cmp "$RESTORE/payload/transfer/SHA256SUMS" "$GUARD_TRANSFER/SHA256SUMS"
(cd "$GUARD_TRANSFER" && sha256sum --strict -c "$RESTORE/payload/transfer/SHA256SUMS")
```
