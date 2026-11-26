# Strongest third-site transport fit

The fit begins at the protected median-bilateral-3 UORSTT01. The production
entry point checks SHA-256
`e1adbc1c8edbc0e30fa81e610d5fd47baf7ea44e178bb78ae11a44d4fe4f7977`.
It reads a frozen word bank, an ordered UORTXT01 TRAIN packet, and the headerless
five-column TRAIN TSV already used by #4302. Columns are row, sign (+1 written
as `1`, or `-1`), governor word ordinal, requested site ordinal, and the
same-spelling rival site ordinal. TSV pairs are first then later, exchange the
requested and rival physical addresses, use the same governor and sign, and
cover each physical row once. The upstream packet seal establishes original
spellings. No protection or fresh capture enters the fitter.

## Native objective

For each row, replay the archived strict lexical sign and original span
nonoverlap tests. Pin the input governor and sign at the requested site.
Minimize over every legal tuple containing that link, independently minimizing
the opposite signed attachment at its site. The heading remains free. Take
the cheapest complete tuple across every remaining eligible original signed
site after excluding the requested site and its same-spelling rival. Pin the
same governor and sign at each third site. Both costs use the inherited exact
order of summed lexical L1, summed transport L1, and frozen heading bank cost.
Every proposal reranks all legal complete third sites. The fitter records
minimal sites, the number of eligible physical tuples, the number of minimizing
link/tuple completions, and a representative. Earliest physical tuple and
lowest tied opposite governor ordinal determine the representative used to
form a proposal.

When lexical tiers agree, use the integral margin
`third_complete_transport - target_complete_transport` and hinge debt
`max(0, 8 - margin)`. A strict full-tuple win uses all three cost tiers.
Distinct lexical tiers can still supply full-tuple evidence and contribute no
transport debt. A requested sign that fails the strict comparison, an
overlapping governor/site, an absent target role, an uncompletable target, and
an absent complete third site have distinct status counts. Skipped rows
contribute no wins or debt. Pairs require both eligible first/later rows in
order to count as strict pair wins. Pair debt is the larger of the two
row debts, with additional total row debt recorded by sign.

## Frozen search envelope

All four tracks start at the protected native state. Buffered GPU proposals
derive their signed discrete L1 subgradient votes from the current winning
target and third completions, toward the target and away from the third.
The votes include their exact selected opposite attachments and nearest
transport centers. Six directions use all active hinges, worst-side hinges
per pair, the pinned role only, the opposite role only, first variants
only, or later variants only. Test each at 1, 2, 4, 8 and 16 units. Also test
each of the 16 signed slots in isolation at 1, 4 and 16 units using the first
direction. In each signed bank, test the axis with the largest absolute vote
in both directions at 1 and 4 units. Ties choose the lowest axis. Missing or
inactive votes produce a duplicate position. Updates saturate i16 and affect
only coordinates 20 through 31 of active transport slots.

The balanced track prioritizes the smaller sign's strict pair win count, then
total strict pair wins, positive margin pairs, strict row wins, worst-side
debt, and total debt. The debt track puts worst-side debt and total debt first,
then the smaller sign's strict pair count, total strict pair wins, positive
margin pairs, and strict row wins. The two signed tracks prioritize the stated
sign's strict pair wins, worst-side debt, total debt and positive margin pairs,
then the smaller sign's pair wins, total pair wins, overall worst-side debt,
and overall total debt. Lower total L1 movement from start and proposal order
break ties between proposals. Accept only a strict improvement in the track's
objective, for at most four rounds per track.

One start plus four tracks times four rounds times 86 proposals gives at most
1,377 objective evaluations. The separate replay budget is 16 track-base
replays and 65 snapshot replays. Export the start, each accepted state and up to
48 first-round menu positions. This menu covers the six global one-unit
moves, one- and four-unit moves in each signed slot, both single-axis
directions and sizes per sign, and the balanced and debt picks among global
proposals. States deduplicate byte-for-byte. The first-round menu permits
external protection and combined-arm scoring to check small geometric edits
as well as the trained winners. The fit consumes only TRAIN inputs.

## Build and run on Ada SM89

Build the staged source archive with CUDA 12.8 and a compatible Linux host
compiler, Rust and jq. Set `CUDA_BIN` to the directory containing nvcc and
cuobjdump if needed. The build verifies the source digest manifest, builds
native SM89 code ahead of time, checks the device instruction scan and
compares unrestricted attachment/prediction machine code to the inherited
competition executable. The fit uses the inherited UORSTT01 writer and
round-trips every candidate without changing bytes outside its active tail.
There is no CPU objective or update path.

```sh
tar -xzf third-fit-sources.tar.gz
bash sources/build.sh build
bash sources/run.sh finite build finite
bash sources/run.sh fit build BANK TRAIN_TEXT TRAIN_REQUESTS PROTECTED_START train-fit
```

The finite check runs both signs and both prompts. A better local distractor
is a worse legal complete tuple, the cheaper same-spelling rival is excluded,
and a known bilateral tail update changes the strongest third from site 2 to
site 3 while improving complete margins from -2 to +2. Test the fit's actual
accepted update, a tied set of third sites with multiple minimizing tuples,
a missing third, an ineligible signed target, rejection of a repeated row,
byte identity outside active tail coordinates, and exact repetition. The
inherited #4302 competition executable supplies an independent reduction of
fixed-governor sites for candidate costs, site ties, tuple ties and eligible
counts. Its unrestricted GPU capture must be byte-identical to the fitter's.
The parent run independently replays raw TRAIN and fresh venue captures.

`candidates.tsv` gives stable names and state basenames. The baseline is
`start`. Each `NAME.native.jsonl` is the ordinary unrestricted TRAIN replay
through archived attachments and prediction with no governor or site supplied.
`objectives.jsonl` contains fixed-governor training terms. `schedule.jsonl`
records the seed census, menu/archive mapping, and track choices.
`complete.json` records the search size and snapshot count once all files
have closed. `run.sh inspect` emits one state's TRAIN objective and replay.
`run.sh validate` checks file shape and pairing without touching the GPU.
