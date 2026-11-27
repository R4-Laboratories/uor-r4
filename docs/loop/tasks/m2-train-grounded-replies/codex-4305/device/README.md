# Original-address intervention on the signed argument reader

The instrument reads UORTXT01, UORSTT01 and the original tuple bank on SM89. The archived CUDA closure supplies native attachment, tuple prediction, cost ordering, tie handling and serialization. The added kernel copies native attachment ranks and changes candidate admission for one signed role. Both paths call the same native predictor. The integer state is downloaded and serialized after scoring so its original bytes can be verified.

- Baseline copies every native rank.
- G admits only the designated original governor for the contested sign at each native role site it can legally attach to. It bypasses the strict comparison with the opposite lexical bank for that exact word and computes the actual own lexical and learned transport distances. Word/site overlap and original role participation still apply.
- N clears the contested sign's attachment ranks outside the declared site set. Inside the set it copies native ranks, including the strict lexical sign gate.
- GN applies both restrictions. Wrong-governor and displaced-site arms substitute their input ordinals independently and together. The opposite sign's native ranks, original tuple ordinals, heading sites and frozen bank are used in every arm.

G can admit a word to both signs when that word qualifies natively for the other sign. The native tuple operator retains those routes. The sidecars report them so any claim requiring two distinct governor addresses can check that condition explicitly.

## Build and execute

Stage the source closure from the prior cold packet `methods/scorer` with `source.sh PRIOR_SCORER NEW_SOURCE`. The stage script checks both inherited translation units against their sealed hashes. The CUDA build runs on the leased SM89 device host with no PTX or host scoring fallback.

```bash
bash SOURCE/build.sh SOURCE NEW_BUILD
NEW_BUILD/argument-score BANK UORTXT01 UORSTT01 MASKS.tsv NEW_OUTPUT
cmp NEW_OUTPUT/baseline.jsonl NEW_OUTPUT/native-reference.jsonl
cmp NEW_OUTPUT/state.readback UORSTT01
```

The manifest binds the executable and compilation record, source scan, ELF and PTX listings, and disassembly. The baseline uses the unmasked ranks through the intervention kernel. The native reference bypasses that kernel entirely. The archived scorer and protected captures are additional baseline checks.

Each mask line declares one original row, signed role, two distinct governor ordinals and two equal-cardinality disjoint sets of site ordinals.

```text
row sign governor wrong_governor good_count good_sites... wrong_count wrong_sites...
```

Ordinals are zero-based; sign is +1 for affirmation or -1 for exclusion. Every row is written in its original packet order. A sparse mask leaves unlisted rows native in every arm. The site mask has no answer ordinal or spelling. Gold and same-spelling-swap complete-link controls can be passed as two additional UORAID01 paths. Their explicit key and edge ordinals are applied as recorded and produce `gold.jsonl` (`arm:correct`) and `swapped.jsonl` (`arm:exchanged`).

## Read the output

The nine arm files are `baseline.jsonl`, `G.jsonl`, `N.jsonl`, `GN.jsonl`, `wrong-G.jsonl`, `wrong-N.jsonl`, `GN-wrong-G.jsonl`, `GN-wrong-N.jsonl` and `GN-wrong-both.jsonl`. The filename identifies the intervention. Their JSON is the native `learned` replay format, with independent and joint link ordinals, exact byte and BPE endpoints, tie counts, collision, and original tuple ordinals and costs. The native predictor sets the tuple ordinal to null when there is no joint winner or more than one minimum-cost assignment. Signed endpoint fields can still project a common word or site from tied routes, or reflect the independent reads when no tuple is compatible.

Every arm also writes `routes.ARM.jsonl`. Each row records:

- `mask` gives the supplied sign, forced word and sorted admitted site set, with nulls where there is no intervention.
- `forced_lexical` holds the forced word's own and opposite lexical distances. It is `[null,null]` without a supplied governor.
- `original_tuples` counts the intact tuple table. `compatible_tuples` counts tuple ordinals with eligible links at both signs. `compatible_assignments` sums the products of admitted governor counts for each compatible tuple.
- `best_tuples` counts distinct original tuple ordinals at the joint lexicographic minimum. `best_assignments` includes every tied governor assignment within those tuples, including same-governor assignments where admitted. Both counts are zero when there is no compatible tuple.
- `compatible_same_word_assignments` and `best_same_word_assignments` are the corresponding counts for routes whose two signs use the same governor. These count one occurrence per original tuple admitting that reused word.
- `ranks` is indexed by native sign and site. Every entry gives its eligible word count, exact best and rival costs, and every governor achieving the best local cost.

## Finite witness

`finite.rs` writes a small native-format fixture from integer constants. It runs no model. Row 0 separately exercises forcing a governor excluded by the native gate, retaining three genuine sites, displacing each factor and changing the compatible opposite site and heading. Row 1 has two tied original tuples and two tied governors at each sign, so the joint read abstains with eight best assignments. Row 2 mirrors the contested sign, excludes a word/site overlap under G and exposes a route using the same governor at both signs. The gold and swap controls choose different exact site ordinals by construction.

```bash
rustc --edition 2021 SOURCE/finite.rs -o generate-finite
./generate-finite FIXTURE
(cd FIXTURE && sha256sum fixture.bank fixture.state fixture.text masks.tsv gold.aid swapped.aid > SHA256SUMS)
bash SOURCE/finite.sh SOURCE FIXTURE BUILD NEW_FINITE_OUTPUT
```
