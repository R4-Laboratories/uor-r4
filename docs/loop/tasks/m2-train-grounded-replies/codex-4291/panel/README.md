# Matched transport curriculum and fresh transfer panel

Both training arms have 256 endpoints in 128 TAKE/DEFER pairs. The first 128 endpoints are the same far block from grammar A, in the same order with byte-identical blind coordinates and a matching 39,449,740-byte geometry prefix including the packet header. The second 128 endpoints use grammar B. The mixed arm receives its near cells and the far-only arm receives the corresponding far cells. Each B near endpoint has exactly one far partner with the same family, fact set, governor direction, neutral placement, admission, first reply role, reply length and all user text outside the two governed clauses. The intervention changes the words and punctuation between the noun and governor as well as their distance in BPE IDs.

| Split | Near pairs | Far pairs | Families | Near gaps in IDs | Far gaps in IDs | Candidate tuples per endpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed TRAIN | 64 | 64 | 8 | 1–3 | 11–13 | 86–358 |
| Far-only TRAIN | — | 128 | 8 | — | 11–13 | 114–390 |
| Held | 64 | 64 | 4 | 1–4 | 11–17 | 52–320 |

The common A block uses planning-review, drill-rotation, building-evacuation and enamel-display. B uses supplier-contract, practice-class, storm-drainage and market-redemption. Its practice-class governor-first near branch reuses the short demonstrative construction from the previously successful repair-school branch with fresh full wording, nouns and facts. It contributes eight opposed pairs at one-ID gaps on both links. The pre-update TRAIN run determines whether the parent wins any of the available near pairs.

Held uses edition-approval, courtroom-playback, ferry-access and pottery-sorting. All four families cross both distances, two fact sets, governor direction, neutral placement, first requested role and admission. The pottery-sorting far governor-after branch places a finite second-person clause and an aside containing additional nouns after each target and before its governor. The branch contributes eight opposed pairs at 17-ID gaps on both links. This independently worded construction exercises the post-target competition relevant to #4289's unresolved reading-room family. Held replies span 28–37 nominal IDs through EOS. Fact set and requested reply length co-vary.

The signed vocabulary and authentic first assistant turn through EOS are fixed to the inherited protocol. The unchanged generator requires one exact instance of each registered correct, exchanged and two incidental tuples per endpoint, using original bytes and absolute BPE addresses. The heading refers to one original governed noun spelling that also has an incidental occurrence at another address. The differently spelled noun follows both earlier occurrences so the four structural tuples remain possible. Generator checks cover both action addresses and both fact copy bodies through their periods. The blind exporter receives only second-user text and the fixed history before the separate supervision join.

The guard verifies disjoint families, full authored templates, fact values and IDs, case IDs, full prompts and encoded prompt digests against the released #4286, #4288 and #4289 panels, including #4289's full generated A/B cells. Held's eight target nouns are absent as whole words from both active training arms, those historical panels and the 512 historical #4283 teacher user turns. The guard rejected injected duplicate family and digest records and a B partner with altered neutral placement. Structural preparation ran without ranking. One retained A draft failed to enroll a heading in the frozen proposer and one retained held draft shared an ending template with #4288. The sealed versions passed both checks.

## Private Wick custody

The read-only seal is `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-mixed-transport/panel/panel-v1`. The private sibling `private/` contains the three grammars bound by `private/GRAMMARS-v2.sha256`. Each of `mixed`, `far` and `held` contains `reference.json`, `blind/text.bin`, `blind/coordinates.jsonl`, `links.bin`, `aid.bin`, `distance.txt` and `pairs.json`. The mask has one line per reference row, with 0 for far and 1 for near. `layout/matched-b-endpoints.json` binds each alternative by its distance-invariant endpoint ID and zero-based ordinal. It records both corresponding opposed pair IDs and prompt IDs. `held/fresh/` contains authentic dialogue scripts, jobs and original-address structural sidecars. `held/structural-links/` also contains the inherited exact correct, exchanged and other supplied control packets.

| Sealed path | SHA-256 |
| --- | --- |
| `SHA256SUMS` | `72d23b3c1ef1c0fd37c816f8cc9140647f34f65683b7b2005a5d68f72367cb5f` |
| `authority.sha256` | `1f373ac5bdb3b31b34970a57c261633a96cab78603fc25cb8ef77f2af0ded979` |
| `guard.json` | `e55bad3caf914c3e5c37a5c4214a9df89871b3eed50adbc53b3596edebbe885d` |
| `layout/matched-b-endpoints.json` | `473d45861c848ad13b852af02cb4b0293de826185c8928c31b2a072578d03ea4` |
| `mixed/reference.json` | `b474449396acdcf616a5da89ed457cb3cbf79426734d61e7a86885942417af17` |
| `far/reference.json` | `f82612e536bfec5f8b1aaa8cec3163e55c00a85184196baf6d4d9337a4190220` |
| `held/reference.json` | `e20f6c544c5620f4c826636b6a9471303db7b91e438d5eac327a038424139350` |
| `mixed/blind/text.bin` | `fd0e73c63927ceaa18a613c9cae63e6cea1682f404fedd559a8fe95f0c1dcc65` |
| `far/blind/text.bin` | `1b8f34d3a4b35ce83cb37009e68347fe5f459a3aa8f3eb7894d7d24737b23aa7` |
| `held/blind/text.bin` | `8ee8ab72b26718268f689018ecdb81ec36b987e9c02e17f77f31dba2b878d24c` |

Reproduction on Wick matched every sealed file in `panel-v2` apart from two negative diagnostics that embed their staging paths. Each version verified against its own manifest. Use an unused positive number for another reproduction.

```bash
root=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-mixed-transport/panel
(cd "$root/panel-v1" && sha256sum --strict --quiet -c SHA256SUMS authority.sha256)
bash "$root/methods-v3/prepare-wick.sh" 3
cmp <(grep -vE '  ./layout/(duplicate-family|mismatched-b)\.stderr$' "$root/panel-v1/SHA256SUMS") \
    <(grep -vE '  ./layout/(duplicate-family|mismatched-b)\.stderr$' "$root/panel-v3/SHA256SUMS")
```
