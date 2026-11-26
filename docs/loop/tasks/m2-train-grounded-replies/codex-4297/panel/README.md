# Complete governor-to-noun intervals on the original input tape

The endpoint and ordered packets use the same 32 signed i16 coordinates per attachment. The ordered arm replaces only zero-based coordinates 20–31. It preserves the existing projections through gaps of four BPE IDs, where the two IDs from each end cover the whole gap. For longer gaps the arm projects every intervening ID in both directions through the frozen D11 `Turn::instruction` operator and places coordinates 8–13 of each view in the existing six-coordinate slots. Each view uses the actual governor and noun offsets from the blind exporter.

Each call to the frozen operator admits 128 IDs. The bridge projects right-aligned chunks of 126 IDs, two periods of the native 63-phase rotation. Adding their coordinate sums preserves each original phase. The bridge admits up to 256 IDs and explicitly rejects longer spans. Each rotated root coordinate has magnitude at most two, so the absolute sum for one coordinate is at most 512. The finite oracle sums the native signed-axis table directly token by token and agrees in both directions at every length from 1 through 256. The 7- and 256-ID fixtures exchange two middle IDs with fixed punctuation, lengths and endpoint IDs. Both directions distinguish the orders in the new coordinates. The prior coordinates alias in both cases. The finite check also reproduces the native period-63 alias and rejects a 257-ID gap. [Bridge operator](bridge.rs) · [Exporter patch](exporter.patch)

## Original mixed TRAIN and new blinded panels

The original #4291 mixed geometry, coordinate map, reference, links, aid, distances and pair identities are copied byte for byte. The matched ordered export checks every word and site point, physical row and tuple, byte and absolute BPE coordinate, all attachment coordinates through 19, and all 32 coordinates on gaps up to four IDs. The held and contrast exports receive the same checks. In each cohort, the original coordinate map and the ordered map match byte for byte. The panels cover original gaps up to 136 IDs on mixed TRAIN and 145 IDs on primary held. Only plain user strings and the genuine first-turn history enter either blind exporter.

| Panel | Physical attachments | Identical at gap 0–4 | Changed at gap 5 or more | Gaps longer than one native call | Largest gap | Observed ordered coordinate range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed TRAIN | 906,336 | 96,016 | 810,320 | 264 | 136 | −38 to 38 |
| Held | 917,952 | 95,056 | 822,896 | 2,016 | 145 | −42 to 48 |
| Selector contrast | 256,424 | 26,840 | 229,584 | 0 | 122 | −46 to 42 |

The primary panel uses the frozen #4286 generator and exact recorded reply through EOS to `Hello.` The four new grammars contribute 256 untouched prompts in 128 opposed TAKE/DEFER pairs. Each pair flips its two signed governors at the same byte and absolute BPE positions. Each family contributes 16 near pairs and 16 far pairs, crossing both governor directions, both orderings of the same-noun rival and requested occurrence, and both requested fact orders. The two authenticated copy cases request three and four sentences. The generator verifies 512 native actions and 512 exact fact bodies through their periods. The reference replies span 27–37 IDs through EOS. [Frozen generator](../../codex-4286/panel/src/main.rs) · [Historical mixed panel](../../codex-4291/panel/README.md)

| Held family | Near gold BPE gaps | Far gold BPE gaps | Near rival nearer / equal / farther | Far rival nearer / equal / farther |
| --- | --- | --- | --- | --- |
| Mill spindle service | 1–3 | 13–16 | 0 / 0 / 32 | 16 / 0 / 16 |
| Mint die examination | 1–5 | 14–16 | 0 / 8 / 24 | 8 / 8 / 16 |
| Titration bench clearance | 1–3 | 9–13 | 0 / 0 / 32 | 8 / 0 / 24 |
| Piston bay parts | 1–2 | 17–18 | 0 / 0 / 32 | 16 / 0 / 16 |

The novelty check compares exact authored user text, encoded second turns, complete prompt arrays, recorded prompt digests, case and family IDs, fact IDs and values, templates, and whole target noun words. The primary check covers 3,856 earlier reference records, nine earlier grammar files and 512 teacher rows. The contrast check also includes the 256 new primary references and their grammar. Both need zero overlaps in each checked category. The same check rebinds the exact native addresses and verifies the gold, exchanged, same-noun rival and exchanged rival tuples against the blind tape on every new prompt. [Novelty and occurrence guard](guard.jq)

## Matched change of intended source

The additional 64 prompts in 32 independent far pairs compare two original occurrences with the same noun spelling. Each selector pair changes exactly one five-byte word and one BPE ID, from `first` to `later`. That change switches the original same-noun target while preserving its signed governor, the other signed noun, heading, action addresses and expected reply. The selector lies beyond the two endpoint IDs in each of the two possible attachments. Each attachment retains its exact original endpoint words, byte and BPE coordinates, length, punctuation, 32 endpoint coordinates and eligible tuple ordinals across the edited prompts. Its ordered coordinates distinguish the two versions. The verifier requires both candidate attachments on every pair, for 64 matched attachments. It checks the original signed governor's lexical point and both candidates' tuple-bank points. The prompts use both governor directions and both fixed TAKE/DEFER signs, 16 pairs each. Within each direction the two selectors exchange the nearer and farther candidate.

| Selector family | Far pairs | Gold BPE gaps | Eligible original tuples per prompt |
| --- | ---: | ---: | ---: |
| Gauge drawer selection | 8 | 10–27 | 194–386 |
| Vestry cabinet selection | 8 | 12–23 | 244–328 |
| Hall light selection | 8 | 11–24 | 102–174 |
| Casting furnace selection | 8 | 11–23 | 102–246 |

In one gauge drawer DEFER pair, the selector occupies raw bytes [155,160) and absolute BPE [81,82) in both prompts. The excluded governor stays at bytes [184,190) and BPE [89,91) as word 37. The two candidate sites remain at bytes [98,105) and [120,127), covering BPE [62,66) and [70,74), respectively. The intended excluded site is 5 for `first` and 9 for `later`. All 386 tuples and both endpoint vectors repeat exactly across the pair. Ordered coordinate 20 changes from −4 to −6 on site 5 and from 4 to 6 on site 9. These measurements describe the input geometry and intended addresses.

The supplemental rows use the same physical contract as the primary held panel. They use different wording families and pair identities. `reference.json` contains the ordinary `admission` field with a fixed sign between peers, and the exact selector under `contrast`. `pairs.json` labels the first and later peers explicitly. Compare these pairs separately from the 128 primary opposite-sign pairs. [Selector generator changes](contrast-generator.patch) · [Matched verifier](contrast.rs)

## Wick custody and replay

The sealed root is `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-all-outranker-interval/panel/panel-v1`. The private primary and contrast grammars are `../private/held-v5.json` and `../private/contrast-v2.json`. Earlier drafts are retained privately. The first two factual stems failed the frozen shared-first/divergent-second BPE requirement. A long earlier titration phrasing exceeded the frozen proposer extent, and shorter phrasing retained the complete 64-row family. The novelty guard also caught earlier reused fact IDs and three repeated templates. The final pack keeps the verified raw prompts, native coordinates and packet bytes.

| Path relative to the root | Contents |
| --- | --- |
| `{mixed,held,contrast}/blind/{text.bin,coordinates.jsonl}` | Original endpoint packets and blind addresses |
| `{mixed,held,contrast}/ordered/{text.bin,coordinates.jsonl}` | Complete ordered bridge packets and the matching address maps |
| `{mixed,held,contrast}/{reference.json,links.bin,aid.bin,pairs.json,distance.txt}` | Original source supervision, source controls and row-pair indices |
| `{held,contrast}/structural-links/` | Frozen source-control geometry and supplied native, correct and exchanged link packets |
| `{held,contrast}/fresh/` | Original user-only client scripts, ordered jobs and exact address sidecars |
| `{mixed,held,contrast}/bridge.json` | Full packet comparison and exact original gap census |
| `{held,contrast}/guard.json` | Novelty, strata, distance, role, polarity, rivalry and eligibility census |
| `contrast/matched.json` | Both original candidate addresses and endpoint and ordered views for every selector pair |
| `bridge-finite.json` | Order and phase controls with per-token native replay |
| `blind.sha256`, `labels.sha256`, `operator.sha256` | Independently verifiable blind, label and operator manifests |

The preparation verifies the preceding input manifests, extracts the stock first-turn record and verifies its exact bytes, seals blind geometry before source supervision, and seals the completed tree. The following commands recheck the immutable preparation and reproduce it in a new directory on Wick.

```bash
root=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-all-outranker-interval/panel
(cd "$root/panel-v1" && sha256sum --strict --quiet -c SHA256SUMS blind.sha256 labels.sha256 operator.sha256 authority.sha256)
bash "$root/methods-v5/prepare-wick.sh" 2 3
cmp "$root/panel-v1/SHA256SUMS" "$root/panel-v2/SHA256SUMS"
```

Wick `methods-v3/build-wick.sh` rebuilds the unchanged native dependencies, patched exporter and supplemental generator into a new build version using the original locked Rust dependency sets. The panel compares packets and source addresses using host-only data checks. It executes neither a learned source ranking nor a model reply.

The complete second preparation reproduced all 1,031 member hashes byte for byte. The complete first tree occupies 52,766,720 allocated bytes on Wick ZFS. The full seal, blind and label receipts and primary packet members are pinned in [receipt.sha256](receipt.sha256).

The [recorded source readout](READOUT.md) traces original protected TRAIN identities and the exact selector decisions on the separately captured held rows.
