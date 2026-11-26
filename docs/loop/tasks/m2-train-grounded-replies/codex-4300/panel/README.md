# Source-occurrence panel

Lab: codex. These inputs identify the original addresses to compare before any learned scoring. The historical and mixed TRAIN packets, references and coordinate maps are byte-for-byte copies of the qualified #4299 panel. The new prompts use the predecessor's canonical selector generator and both canonical packet exporters.

| Cohort | Physical rows | Pairs | Requested-versus-rival comparisons | Exact selector edits |
| --- | ---: | ---: | ---: | ---: |
| Historical selector | 64 | 32 | 64 | 32 |
| Original mixed TRAIN | 256 | 128 | 288 | 0 |
| TRAIN-derived selector | 64 | 32 | 64 | 32 |

The historical input includes every original selector pair and flags the four changes the frozen reader got right and the full refit lost. The pair IDs use the `lantern-cupboard-sort` family and the `cuff-and-orchard` and `textile-and-marsh` fact IDs. Suffix `/10` is affirmative and `/2` is excluded. The eight endpoint ordinals are 32–35 and 40–43. Each pair has a `first` and `later` endpoint that exchanges the exact target and rival addresses.

For the original mixed TRAIN census, a rival is a distinct native site whose exact surface bytes equal the requested noun's, whose bytes are disjoint from the signed governor, and which appears in the same signed slot of at least one native tuple. This eligibility comes from packet structure and original labels. The scorer checks the learned sign test and legal completions for each state. There are 128 far requests and 160 near requests, 144 per sign. The registered neutral occurrence accounts for 256 requests. On 32 near rows the heading is also eligible as a signed noun under another heading, so the census includes that second request. Across the 128 registered TRAIN pairs, none of the text changes consists of a single equal-width `first` to `later` replacement.

The derived panel crosses the four far mixed TRAIN wording families, two TRAIN facts, two signs and both governor orders. It reuses their original nouns, governors, fact values, reply commands and phrase material. The checked frame vocabulary comes from the literal mixed TRAIN prompts. The selector intervention uses the frozen tokenizer's whole `first` and `later` lexemes. The generator changes one selector per pair exactly, with one changed BPE ID, equal prompt lengths and no shift in either physical noun occurrence, governor, opposite noun or heading. The canonical comparison verifies equal native tuple eligibility and identical endpoint features on all 64 candidate attachments, plus a change in their ordered interval features. The 64 derived attachments span 21–33 intervening BPE IDs. The historical selector attachments span 13–26. The output records zero shared historical selector target nouns, fact IDs, scope clauses or prompt digests, and zero shared original mixed TRAIN prompt digests. The edited prompts form a separate TRAIN-derived diagnostic cohort.

## Files and replay

Each cohort has the unchanged canonical `reference.json` schema and `ordered/coordinates.jsonl`. The headerless `requests.tsv` stores row index, sign, governor word index, requested site index and rival site index. Indices are zero-based within the packet. Sign is `1` for affirmative and `-1` for excluded.

`requests.jsonl` carries each request with the case, pair, stratum, historical flag and original matched addresses of the governor, both same-spelling sites, the opposite governor and noun, and the heading. Address objects include the original half-open user-text byte span, absolute prompt BPE span, BPE IDs and exact surface text. `rows.jsonl` preserves every row with its physical coordinate map and all of its requests. `census.json` reports denominators and pair identities.

The [receipt](receipt.sha256) pins two complete, byte-identical preparations. The partial attempts were retained on Wick. They stopped on a missing generator argument, a colon rejected by the user-turn wire format, a selector within the native candidate context, and a shell file-descriptor lifetime error. The final preparation checked the TRAIN construction and operator equivalence before comparing its provenance with historical inputs.

```sh
root=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-switch-source-competition/panel
for version in 1 2; do
    (cd "$root/panel-v$version" &&
        sha256sum --strict --quiet -c SHA256SUMS authority.sha256 train-derived.sha256)
done
cmp "$root/panel-v1/SHA256SUMS" "$root/panel-v2/SHA256SUMS"
# Repeat into a fresh panel version.
bash "$root/methods-v5/prepare-wick.sh" 3
```
