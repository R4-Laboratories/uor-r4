# SM89 governor link learner

This experiment fits original-address links on labeled text, then rereads the same type of text and searches the frozen proposer’s original tuples on one physical SM89 backend. The state is an integer bank of lexical and directed transport centers for each sign. The signed tokenizer, feature exporter and native proposer are inputs to this program.

## Packets

All numbers are little-endian. Each magic is eight ASCII bytes and is followed by a `u32` row count. Counts must agree across packets and every row preserves its original zero-based ordinal. Each address is four `u32` words in byte-start, byte-end, absolute-token-start, absolute-token-end order. All intervals are half-open. Bytes are relative to the current raw user turn, with any framing blank excluded.

| Magic | Fields per row |
|---|---|
| `UORTXT01` | `u32` word, site and tuple counts, followed by every original word as address plus 96 signed `i16` lexical coordinates, every native site as address plus 96 signed `i16` D11 coordinates, every directed transport as 32 signed `i16` coordinates in word-major site order, and every frozen native tuple as three `u32` site ordinals in affirmed, excluded and heading order. |
| `UORLNK01` | Two supervised links, affirmed followed by excluded, each with a `u32` governor word ordinal and a `u32` noun site ordinal. |
| `UORAID01` | The affirmed and excluded signed governor word keys as two `u32` ordinals, then two unordered unsigned edges, each with a `u32` governor word ordinal and a `u32` noun site ordinal. |

The aid reader resolves both signed governor keys and both edge keys to complete byte and token addresses before matching. A valid unmatched key has a reported governor, an absent noun, and no admitted tuple. The exchange arm swaps the two unsigned noun endpoints. The text and trained-state readers used for learned prediction never open the labels or aid packets.

Words and sites have separate address domains, limited to 128 each per row, with up to 512 rows and 4,194,304 tuples per file. Site coordinates follow the frozen point bound of −24 through 24. Duplicate original byte occurrences within one domain and duplicate tuple index triples are rejected. The same spelling or vector at distinct addresses is allowed. An edge requires disjoint governor and noun byte intervals. Signed intervention keys must be distinct. Its edge endpoints must be valid and distinct in each domain.

## Transition and state

Each sign fits two banks on the GPU using its supervised original word and edge. The lexical bank has 32 slots of 96 signed 16-bit coordinates. The directed bank has eight slots of 32 signed 16-bit coordinates. Farthest-first starts at the lexicographically least sample, then adds the sample with maximum nearest-center Manhattan distance. Equal nonzero distances choose the lexicographically least sample. Identical vectors cease to create slots when the maximum distance reaches zero. Two batch L1 steps follow. Each sample joins its nearest center, where assignment ties select the earliest center created by the deterministic initialization. Per coordinate, the center moves one integer unit in the direction of the sum of signed sample deviations. Exact coordinate matches cast zero votes. No coordinates can move outside their observed ranges. The optimizer scratch and temporary candidates do not persist.

The state file is magic `UORSTT01` followed by exactly 13,328 payload bytes. The affirmed and excluded banks follow in sign order. Each sign has a `u32` lexical count, the full lexical center array, a `u32` transport count, and the full transport center array. All inactive slots are zero and the full file is 13,336 bytes. The input fit banks used for heading resolution remain a separate frozen dependency, drawn from the historical primary fitted words. The same state layout is uploaded, fitted, saved, restored and read through the same native integer operations. The lexical/transport nearest-center operator is shared by fit assignments and prediction. The fit-time replay uses the same predict kernel as detached restore. There is no carried conversational state or emitted answer token in this experiment. Replay compares the original absolute token intervals and byte intervals, decision indices, ties, and complete costs. The build and released-run manifests bind tool, state, packet and frozen dependency identities outside the state bytes.

Prediction considers every original word. Its lexical distances to the two sign banks admit it only to the strictly nearer sign. For each site in that sign’s early slot in at least one frozen native tuple, every nonoverlapping word of the sign competes by lexicographic `(lexical L1, transport L1)`. The fitted transport uses the word-to-site coordinates. The governor-gap arm keeps the lexical bank and substitutes original token interval gap for transport. Overlap or adjacency in token space has zero gap. A tuple’s score orders summed lexical distances, summed transport distances and the frozen native L1 cost, in that order. Ties across distinct triples or distinct original governors abstain. A link endpoint common to all joint minimizers stays visible for diagnostics. If there is no legal joint tuple, the per-sign projections remain visible and the tuple stays absent. The underlying frozen native arm minimizes the three frozen-bank distances over all native tuples by the same exact L1 and abstains on tied triples. Unfitted provides the same joint operation with empty centers.

The joint search evaluates each tuple from its two per-site sign minima. This is exact for the joint ranking since the score’s first two coordinates are additive, each governor can be admitted to at most one sign, and every native tuple has distinct site ordinals. Order affects reported ordinals only. Receipt output includes ordinal-free byte and token endpoints for identity.

## Invocation

Use an already verified Wick build, with the common GPU lease held, `CUDA_DISABLE_PTX_JIT=1` and `CUDA_FORCE_PTX_JIT=0`. Invoke the production runs only after that run’s release.

```sh
textlinks-sm89 train frozen.words train.bin train-labels.bin learned.state train-live.jsonl
textlinks-sm89 learned frozen.words train.bin learned.state train-cold.jsonl
cmp train-live.jsonl train-cold.jsonl
textlinks-sm89 learned frozen.words text.bin learned.state learned.jsonl
textlinks-sm89 governor_gap frozen.words text.bin learned.state governor_gap.jsonl
textlinks-sm89 native frozen.words text.bin native.jsonl
textlinks-sm89 unfitted frozen.words text.bin unfitted.jsonl
textlinks-sm89 correct frozen.words text.bin aid.bin correct.jsonl
textlinks-sm89 exchanged frozen.words text.bin aid.bin exchanged.jsonl
```

Substitute `train-swapped` to exchange both heads’ complete signed supervision per training row. Substitute `train-endpoints` to exchange only the nouns and preserve the training governors’ signs. Both variants restore through `learned`. Output paths must be fresh.

Each JSONL row has a `row`, `arm`, `links`, `independent_links`, `collision` and `tuple`. The two link lists have affirmed then excluded results. Each entry has nullable `word` and `site` ordinals and nullable `governor` and `noun` addresses, represented as objects with two-element `bytes` and `tokens` arrays. Link diagnostics include `word_ties`, `site_ties`, `link_ties`, `eligible`, and `best` and `rival` costs of `[lexical, transport]`. The collision flag records incompatible independent projections even when joint search finds a legal assignment. The tuple has a nullable `ordinal`, `sites` and `addresses`, the exact tied joint decision count in `ties`, the candidate tuple count in `eligible`, and `best` and `rival` costs of `[lexical, transport, bank]`. `bank` and `rival_bank` also expose the bank coordinate directly. Missing costs are null.

## Wick finite verification

From a source stage on Wick, run `bash run-wick.sh build v1` and `bash run-wick.sh finite v1 v1`. The synthetic fixture does not consume measured training or held prompts. The checked build contains SM89 cubins and no embedded PTX. The finite phase leases the device, executes all three fits, compares every saved state byte to an independent Rust reference with verified nonzero coordinate updates, and compares live and detached replay byte-for-byte. It also checks transparent noun, governor and heading ties, two distinct same-projection noun addresses, transport reversal, joint assignment against incompatible independent minima, sign and endpoint corruption, complete order-invariant supplied-edge replay, and rejection of dangling, colliding and trailing input.
