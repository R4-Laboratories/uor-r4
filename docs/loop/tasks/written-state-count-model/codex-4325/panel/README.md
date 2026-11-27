# Matched two-release fit and held transfer

| Tape | Fresh scoped quartets | Fresh direct one/two pairs | Fresh direct two/three pairs | Rehearsal rows | Total rows | Post-B stop/copy pairs |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 64 | 64 | 64 | 256 | 768 | 112 |
| HELD | 64 | 64 | 64 | 0 | 512 | 64 |

Each quartet crosses which of two incompatible clauses appears first and which clause governs. The two axes hold the four original factual occurrences, their absolute token positions, the question, and the format instruction fixed. The governing clause requests one or two words. Its reference is `A . EOS` or `A . B EOS`. Direct two/three pairs use `A . B EOS` and `A . B C EOS`. All fresh rows carry four original source values. The licensed count is encoded only in user wording and the offline reference. Scoped pairs share one selector and exchange clause order. The keys also index the other comparison, which shares clause order and changes the selector.

There are eight scoped and four direct wording families per split. Each scoped family contributes eight contexts with rotated source values and physical field order. Each direct family contributes 32 deterministically instantiated lists. All 128 clause-order edges per split have equal encoded lengths. Selector edges have equal lengths in 80 TRAIN comparisons and 64 HELD comparisons. The stricter four-way equal-length subgroup contains 40 TRAIN and 32 HELD quartets. Of the direct pairs, 64 per split have equal encoded lengths. Fresh prefixes span 90–164 TRAIN tokens and 89–159 HELD tokens. The same five-token reply cap applies to all rows.

The panel agent authored the scoped templates and TRAIN direct templates. The separate Codex product replay agent authored all four HELD direct count contrasts without the TRAIN wording. The panel agent adapted that agent's suggested topics into four-entry lists and supplied the format instruction so both members keep identical facts. The Rust builder instantiates synthetic names, values and layouts. Common words and release semantics recur across the splits.

## Fit data and identities

The sole fit tape is `train/rows.jsonl` with joined `train/keys.jsonl`. It contains 512 fresh rows followed by 192 rows selected from the #4321 TRAIN tape used by #4322 and 64 rows selected from #4323 TRAIN. The selection includes both sides of each pair and preserves each selected scientific row's original fields. The added `row_id` is the original pair ID followed by its side. The keys carry `split`, `cohort`, `source_issue`, `source_panel_issue` on rehearsal, and `source_row_id`. The [TRAIN receipt](train-receipt.json) binds every selected historical row ID and both original split manifests. Both fits read this same combined tape.

The fresh [HELD receipt](held-receipt.json) pins its separate family census. On each fresh row, `context_id` joins the full context. The one-based `one_word_clause_ordinal` and `operative_clause_ordinal` are null on direct controls. `equal_length_context` selects equal-length quartets or direct pairs. `equal_length_pair` and `equal_length_selector_pair` report the two individual matched edge lengths. `selector_swap_id` is populated only on scoped rows.

The [independent jq audit](independent-audit.json) authenticated all 640 output pairs and screened 5,188 archived rows, including #4324 fresh, plus 3,908 archived rows excluded from training. Each fresh split has zero exact overlaps in normalized prompts, normalized contexts, normalized directive wording, encoded prefixes, source token IDs and pair IDs against the prior archive and the other split. The combined fit tape has zero exact prompt or source overlap with the excluded rows. Rehearsal fields match the original TRAIN data. The builder also checks normalized authoring templates against the archived signatures.

## Reproduce

The read-only panel is on Wick at `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-joint-release/panel-v1`. An independent repeat produced identical rows, keys, receipts and manifests at `panel-replay-v1`. [SEALED.sha256](SEALED.sha256) lists identities relative to the run directory. [PREPARATION.sha256](PREPARATION.sha256) lists the exact inputs and executable relative to `panel-prep-v1`. The first attempted prep rejected a 187-token prefix at the fixed writer bound. The approved wording uses one fact header per context. Preparation and validation execute no model.

From this directory on Wick, with `RUNS` pointing to the directory containing the prior runs and `NEW_PANEL` an unused path:

```sh
cargo build --locked --offline --release
target/release/joint-release-panel prepare "$RUNS" "$NEW_PANEL"
(cd "$NEW_PANEL" && sha256sum --strict -c SHA256SUMS)
for split in train held; do
  (cd "$NEW_PANEL/$split" && sha256sum --strict -c SHA256SUMS)
done
```

Run the independent structural and overlap audit with Bash process substitution and `jq`:

```bash
source_choice="$RUNS/20261127-codex-source-choice/panel-v1/train"
source_scope="$RUNS/20261127-codex-scope-trace/panel-v1/train"
old_splits=(
  postperiod/{finite,held}
  source-choice/{train,finite,held-primary,held-secondary}
  period-margin/{dev,held,protect}
  scope-trace/{train,dev,held}
  release-factorial/fresh
)
prior_tapes() {
  local kind=$1 selection=$2 split
  for split in "${old_splits[@]}"; do
    if [[ $selection == excluded && $split == */train ]]; then
      continue
    fi
    cat "$RUNS/20261127-codex-${split%/*}/panel-v1/${split#*/}/$kind.jsonl"
  done
}
jq -en \
  --slurpfile train_rows "$NEW_PANEL/train/rows.jsonl" \
  --slurpfile train_keys "$NEW_PANEL/train/keys.jsonl" \
  --slurpfile held_rows "$NEW_PANEL/held/rows.jsonl" \
  --slurpfile held_keys "$NEW_PANEL/held/keys.jsonl" \
  --slurpfile source_4322_rows "$source_choice/rows.jsonl" \
  --slurpfile source_4322_keys "$source_choice/keys.jsonl" \
  --slurpfile source_4323_rows "$source_scope/rows.jsonl" \
  --slurpfile source_4323_keys "$source_scope/keys.jsonl" \
  --slurpfile old_rows <(prior_tapes rows all) \
  --slurpfile old_keys <(prior_tapes keys all) \
  --slurpfile blocked_rows <(prior_tapes rows excluded) \
  --slurpfile blocked_keys <(prior_tapes keys excluded) \
  -f audit.jq
```
