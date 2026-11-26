# Signed-slot guard recovery and state lineage

The optional recovery menu uses the two archived signed transport states. For each
recipient, the other state supplies one complete affirmative slot and one complete
excluded slot at their same ordinal. The independent ordinal loops visit all eight
choices per sign, giving 128 registered positions across the two recipients. This
includes positions that reproduce an existing native state. The manifest preserves
enumeration order and the lineage records the recipient, donor, two slot ordinals
and actual tail movement. Rust copies the native little-endian coordinate words.

Stage `native.rs`, `crossover.rs` and `prepare-wick.sh` in a fresh versioned
methods directory under the study's `fit/recovery` Wick ZFS root. Run
`bash prepare-wick.sh v1`. The command verifies both archived state hashes and
the inherited ordered bundle tool. Each generated 13,336-byte state packs as a
13,352-byte ordered bundle and unpacks to the exact input bytes. The resulting
`crossovers-v1/states/candidates.tsv` uses the same name and state basename format
as the primary fitter. The runner writes a UTC receipt, a complete member seal
and a portable archive.

For the post-fit byte audit, stage `native.rs`, `lineage.rs`, `schedule.jq` and
`audit-wick.sh` in the same methods directory. Run
`bash audit-wick.sh v3 STUDY/rented-train-v1/fit-out`, replacing `STUDY` with the
absolute study root on Wick and using a fresh receipt version. The audit checks
every candidate's byte shape, immutable words, manifest ordering and native
receipt presence. It reconstructs all 57 scheduled initial states and all
logged accepted updates word-for-word from the physical transport coordinates
and the submitted GPU assignments. It also checks the fixed track schedules,
budget, objective acceptance ordering and rank-selected initial starts.
The fitter's GPU assignments and objective values are inputs to this byte replay.
The independent source-ranking audit evaluates the underlying model predictions.

The 17-proposal shadow stores counts, the selected proposal ordinal and training
objective before and after. Its checkpoint supplies no independently exported
state, guard result or fresh result.
