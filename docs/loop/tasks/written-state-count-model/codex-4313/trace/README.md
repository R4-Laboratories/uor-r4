# D11 question-swap trace

Run `question-swap-trace gpu ARTIFACT INPUT.jsonl OUTPUT.jsonl` with
`CUDA_DISABLE_PTX_JIT=1`. Build the `cuda` feature for the device capability
using the scorer's existing AOT integer PTX admission. The command opens a
new output file after both GPU passes agree. The `info` command records the
embedded image target and exact PTX and cubin digests. The `finite` command
uses a synthetic D11 artifact with planted count rows.

Each input row carries a string `pair_id`, integer `side` (0 or 1), string
`cohort`, original ID array `prefix`, and original first-content `target`
ID. The scorer cross-probes the opposite side's target as `rival` and the
space ID 223 as `space`. The prefix length and SHA-256 bind the exact sequence
as contiguous little-endian u32 words. Each output retains both identifiers,
the full original prefix, the three IDs, and the artifact SHA-256.

## State and emission

- `read_vector` and `read_code` are the four geometric read blocks and their
  packed orbit code at the input of the count gate. `routes` carries each
  enabled level's gate, selected root, ordered binder nodes, transport frames,
  and transported roots. A gate with no contributing node has an empty list.
- `recurrent_frames` are the four terminal compose indices. `recurrent_sum`
  is the 32-word signed sum entering the four scale-matched snaps.
  `recurrent_vector` is the normalized sum after alignment. Its root IDs
  encode `recurrence_code`. The read and recurrence fields are captured at
  their distinct sites in the scoring function.
- `source_position` is a zero-based original prefix position. `source_token`
  is the original successor ID at that position. Both are null when no
  matching causal predecessor exists. `query_token` is the final original
  prefix ID. The gate's choice and weighted count are `gate_selected` and
  `gate_mass`.
- `writer_vector` is the read after the native count-gate rotation.
  `admitted_cells`, `top8` and `emitted` come from the final geometric
  writer. `active_candidates` counts unique sparse-pool IDs.
  `positive_candidates` counts that pool's positive-mass IDs.
  `writer_candidates` counts IDs in the admitted cells.

## Ranks and ties

The three probe records are `probes.gold`, `probes.rival`, and `probes.space`.
All ranks start at one and tie counts include the probe itself where eligible.

- `mass` is the unsigned integer gate score. With raw counts in
  `gate_score_terms` its value is
  `C_q + 64 C_qr + 2 C_r + 32 C_qc + 8 C_rr + pointer_bonus`. The pointer
  bonus is 256 for the matching successor ID and zero otherwise.
  `pool_member` is membership in the union of the query, read and recurrence
  marginal rows and the optional pointer. Some inactive IDs have positive
  raw mass from a joint table. `eligible_positive` requires both pool
  membership and positive mass.
- `gate_rank` orders eligible positive IDs by decreasing mass, decreasing
  unigram count, then increasing original ID. If no positive eligible ID
  exists, the stored `prior_selected` is the fallback and alone has deployed
  rank one. Every other ID receives null. `gate_full_rank` gives the matching
  ranks for selectable IDs and appends remaining vocabulary IDs by unigram
  count and original ID. `mass_ties` counts all original IDs with this raw
  mass, and `eligible_mass_ties` counts only positive eligible IDs with it.
- `prior_rank` orders every original ID by decreasing fitted `prior_count`
  then increasing ID. `prior_ties` counts every original ID with the same
  count. `prior_selected` is the artifact's stored fallback ID.
- `writer_rank` orders IDs in admitted cells by increasing geometric L1
  `writer_distance` then increasing original ID. An ID whose `writer_cell`
  is outside those cells receives null. `writer_full_rank` appends excluded
  IDs by increasing distance and ID. `writer_distance_ties` counts all IDs
  with this distance. `writer_admitted_ties` counts the admitted IDs with it.

A second launch reverses input and probe order and compares every returned
word by original row. `duplicate_replay` is true only after this comparison.
Run `question-swap-trace client-parity ARTIFACT GPU.jsonl` as a separate
process for independent replay through the integer artifact reader and
native client writer. It checks the recorded state including the recurrent
frames and sum, routes, source, writer ranks, prior ranks, and emitted IDs.
The finite fixture separately asserts the GPU sparse-gate mass and ordering
against planted answers, inactive mass, zero mass, and ties.
