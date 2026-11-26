# Signed paired occurrence fit

The executable takes one ordered TRAIN packet with one contested sign per first/later
pair. Both signs must occur across the file. Each headerless five-column request is
`row, sign, governor, requested site, distinct rival site`, tab separated. Consecutive
rows form one pair, first then later. The packet checker verifies coverage, sign, paired
governor identity and the exchange of distinct original byte/BPE addresses. The panel
seal binds same spelling, edit order and all source identities. Regression, near and held
requests are outside this executable's fit input.

The device uses the archived typed `Geometry::transport`, `closest`, `usable` and
joint-score ordering. It caches the invariant lexical minima and tied governors for the
opposite signed role at each site and frozen bank costs for all original triples.
It recomputes the nearest signed transport slot and exact L1 for every proposed state.
For each arm it fixes the authored governor and one of the two sites, considers all original
legal triples and minimizes each opposite signed attachment over the tied best lexical
governors. The tuple objective orders lexical costs, transport costs and frozen bank costs.
Unrestricted snapshot replay uses the archived `attachments` and `predict` kernels.

Let each selector's local margin be rival minus requested nearest-center L1. Its local
debt is `max(0, 1 - margin)`. A fixed-governor joint win is a strict lexicographic tuple
win. With equal lexical costs its transport margin likewise subtracts the requested from
the rival cost. Its joint debt is `max(0, threshold - margin)`, where threshold is zero
if the requested frozen bank score already breaks the tie, otherwise one. A differing
lexical tier decides the joint win independently of transport and contributes no movable
joint debt. Each pair must win both selector rows to count. The device records the larger
and sum of the two debts separately and aggregates scores by sign. For a tied minimizing
joint completion, the vote uses the earliest physical tuple and the lowest tied governor
ordinal in its opposite-role site. Real candidate scoring retains the entire tie count.

There are 57 fixed starting positions. Two archived seeds, two entire opposite-seed sign
substitutions, and three componentwise quarter mixes precede 32 individual signed-slot
substitutions and 18 bilateral substitutions across seeds. Bilateral substitutions
cover each slot and the pair of slots 0 and 3, which the inherited #4300 local winner trace
implicates. Empty/inactive or duplicate mixes still use a schedule position. Candidate 0
is the original and 1 is the median.

Eight tracks start from original and median with balanced ranking, entire-sign swaps
with respective signed priorities, midpoint with local priority, and the best initial
centers under joint, balanced and local ranking. Each round uses the current nearest
centers, exact realized losses and frozen joint completions to take signed integer votes
toward the requested geometry and away from the rival. Six directions cover all active
local hinges, paired worst-side local hinge, paired worst-side joint hinge including
opposite attachments, the sum of the two worst-side directions, and that sum limited to
each sign. Steps are 1, 2, 4, 8, 16 and 32 for each direction. The most strongly voted
slot for each sign also gets slot-only hybrid steps 2, 8 and 32. The strongest hybrid
axis for each sign is tested in each direction with steps 1, 4 and 16. Every update is
saturated to signed i16 and affects only axes 20 through 31 of active slots. Each round
tests all 54 branches and accepts one strict improvement or stops, for at most six
rounds per track.

All track priorities first require valid completions. The balanced track maximizes the
smaller sign's both-objective pair win count, then their sum, minimizes the two worst-side
sums combined, maximizes joint then local pair wins and minimizes total debts. The local
track maximizes local pair wins, minimizes local worst-side then total local debt,
maximizes the smaller sign's both-win count and joint pair wins and minimizes joint
worst-side debt. The joint track swaps the local and joint terms. The signed tracks
prioritize their sign's both wins, combined worst-side debt, and individual local then
joint wins before panel-wide both wins and combined worst-side debt. Lower distance
from original, then proposal order, break ties between proposals. These tie-breakers
cannot by themselves earn an accepted step.

There are at most 2,649 scored attempts and 105 exported candidate positions, including
the initial positions. Base and snapshot replays are counted separately. The first
17 proposals from original are recorded as a shadow objective checkpoint. TRAIN
selection can score every exported state independently against the protected packet
and strongest alternative original occurrence. The fitter never sees those targets.

## Run on Linux SM89

The source archive includes the exact inherited CUDA sources and their checksums.
Use an ahead-of-time CUDA toolkit with Ada SM89 support, a compatible host compiler,
Rust for the finite fixture, and jq. Wick builds against CUDA 13.1. Use the renter's
toolkit and driver-supported CUDA version to build the same source on the rental.
Set `CUDA_BIN` if its nvcc and cuobjdump are apart from `/usr/local/cuda/bin`.
If the venue supplies only a runtime, the Wick build needs a compatible Linux/glibc
and CUDA driver before use. The build checks SM89 SASS, rejects PTX and scans device
instructions. The sealed case uses original CUDA device parser and ABI.

```sh
tar -xzf paired-fit-sources.tar.gz
bash sources/build.sh build
bash sources/run.sh finite build finite
bash sources/run.sh validate build BANK ORDERED_TEXT REQUESTS ORIGINAL MEDIAN unused
bash sources/run.sh fit build BANK ORDERED_TEXT REQUESTS ORIGINAL MEDIAN new-output
```

The finite fixture checks strict improvements on both selectors in both signed banks,
byte identity outside active tail fields, deterministic repetition, an exchanged-site
negative, rejection of a repeated pair row and wrong sign, and rejection of immutable
state drift. Both a seed and a learned artifact are compared against the inherited
exact fixed-governor scorer for cost, tie counts, eligible completions and complete
native replay bytes. Production model fitting belongs to the parent study release.

`candidates.tsv` maps stable candidate names to basenames of raw `UORSTT01` snapshots.
Each snapshot has an inherited `.native.jsonl` full packet replay. `objectives.jsonl`
gives both signed costs, transport margins, physical request ordinals, conditioned
winners, eligible counts, tie counts and paired margins. `schedule.jsonl` gives each
initial position and each fitted path/round with before and after loss, selection,
state deduplication and cumulative score counts. `budget-17.json` is the shadow objective
checkpoint and `complete.json` records counts and state size after all records close.
