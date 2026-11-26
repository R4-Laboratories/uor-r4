# Held original-address panel

The Wick seal contains 256 second-turn prompts in 128 opposed TAKE/DEFER pairs. Each family crosses near and far governor distance, governor-before and governor-after noun order, neutral-first and governed-first same-spelling occurrences, and job-first and home-first requests. The two fact sets call for three and four sentences, respectively. The two prompts in each pair vary only the two registered governor words, at the same byte and absolute BPE spans. The signed governor vocabulary is inherited from the TRAIN curriculum.

| Family | Near pairs | Far pairs | Near BPE gaps, either link | Far BPE gaps, either link |
| --- | ---: | ---: | --- | --- |
| Leather tool inspection | 16 | 16 | 1–3 | 14–17 |
| Observatory inventory | 16 | 16 | 1–3 | 15–17 |
| Orchard access | 16 | 16 | 1–2 | 9–13 |
| Boatyard rigging | 16 | 16 | 2–5 | 14–15 |

The boatyard governor-after far branch contains eight opposed pairs. Both governed nouns precede an intervening finite temporal clause with two additional nouns. A finite main clause carries each governor 15 IDs after its noun. The governors contribute zero IDs to each noun's six-ID window on these endpoints. The frozen generator enrolled both orders and both distances for every family. Reply references span 27–33 IDs through EOS.

The authenticated first assistant reply is the same 13-ID reply through EOS to `Hello.` carried by [the #4286 panel](../../codex-4286/panel/README.md). The preparation program re-extracts it from the original stock capture and compares the exact bytes to the frozen history. The unchanged Rust generator validates both source-copy bodies through their periods and both action spans for each prompt, with 1,024 uniquely registered correct, exchanged and incidental tuples across the panel. The heading matches the same spelling at two earlier addresses. The differently spelled noun occurs after both occurrences so the native proposer can enumerate all four tuples.

The text exporter takes only the current user strings and fixed history. The guard compares the blind and reference prompt digests, the four exact tuple identities and both governor addresses for every endpoint. Reference byte coordinates include the tokenizer's leading protocol blank, while blind byte coordinates index the authored user string. The guard converts the byte origins and checks absolute BPE ranges directly. It checks the near and far pair counts, then independently rejects a duplicate #4291 TRAIN digest and a perturbation of the first blind site's BPE addresses. Labels join only after blind export and these checks.

The novelty guard scans 2,560 reference records across the two #4283 reserve copies, #4286 and #4288 fresh panels, all generated #4289 and #4291 TRAIN grammar cells, and both earlier held panels. It verifies that family labels, fact IDs and values, exact authored prompts, any recorded prompt digests, and actual encoded prompt arrays are disjoint. It tests all 32 authored templates against eight prior grammars and all eight target nouns as whole words against the reference records and the 512 #4283 TRAIN user strings. The first two fact drafts failed the original shared-first/divergent-second BPE copy check. The sealed fact sets start with different stems and both passed structural enrollment. Preparation, guard checks and reproduction did not rank any candidate or generate a model reply.

A separate antecedent check covers 784 records from the original #4283 primary held panel, its pre-capture design, the 16 ordinary regressions and an earlier reserve copy. It compared exact authored text, encoded second turns, complete prompt arrays, row and family IDs, fact values, and the eight target nouns. All overlaps were zero.

## Wick custody

The sealed root is `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-protected-transport/panel/panel-v1`. The private authoring input is its sibling `private/held-v4.json` and the staged preparation scripts are under `methods-v4/`. The seal contains `held/reference.json`, `held/blind/{text.bin,coordinates.jsonl}`, `held/{links.bin,aid.bin,distance.txt,pairs.json}`, and the inherited native and supplied control packets under `held/structural-links/`. The ordinary client scripts, ordered jobs and authenticated structural sidecars are under `held/fresh/`.

| Record | SHA-256 |
| --- | --- |
| `private/held-v4.json` | `ff71ba32ec6f99a3d98de5d2ad4dd1bed60e0e71350d89f8404819e209bf2eb1` |
| `SHA256SUMS` | `22213dcba2081598d51e08cb573ca5a721ad26dcb04c20d4fd906a5cbae511cc` |
| `authority.sha256` | `bd42c8e91a83e5e310f87a8b40ecb518449e1edbf01b274f54734521592ff3b8` |
| `guard.json` | `e486f3cfab03ce3ad3804802d3ba64e2ac73dfd9354452d97f4bcc59f153d8b6` |
| `held/reference.json` | `033c2f7040c6e07dd4c57499c57ed0987b7481a1d48f318a8f9adb2ff7e47ea9` |
| `held/blind/text.bin` | `94953fca2bf60ae5ac72237b83370bddbf7885f32349ac4717ec8e1ff6513cb3` |
| `held/blind/coordinates.jsonl` | `171e46448cf206bf43dde8ccb8d5b75f7d167e1c9eb45229a8995dd3f0ef70b3` |
| `held/aid.bin` | `3e8ca396d7db8d392a798e14ca4e3dc9438882aa0c592430af80b2aad82c41a4` |
| Sibling `antecedents-v1/SHA256SUMS` | `494525299ba1d74f5f1022190105bab09b83516e7ed5a574fc90a23c758d1cfc` |

Two full preparations matched all saved members byte for byte. Use an unused positive output version for another reproduction on Wick.

```bash
root=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-protected-transport/panel
(cd "$root/panel-v1" && sha256sum --strict --quiet -c SHA256SUMS authority.sha256)
bash "$root/methods-v4/prepare-wick.sh" 3
cmp "$root/panel-v1/SHA256SUMS" "$root/panel-v3/SHA256SUMS"
bash "$root/methods-v5/verify-antecedents-wick.sh" 2
cmp "$root/antecedents-v1/SHA256SUMS" "$root/antecedents-v2/SHA256SUMS"
```
