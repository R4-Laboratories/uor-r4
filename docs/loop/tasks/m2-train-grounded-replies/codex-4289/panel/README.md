# Fresh protocol-2 occurrence panels

The Wick preparation reuses the sealed #4286 generator and #4288 plain-text exporter. Both splits retain the authentic historical first assistant turn through EOS and use the originally trained four signed verb pairs.

| Split | Endpoints | Opposed pairs | Wording families | Near gaps | Far gaps | Native tuples per endpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 256 | 128 | 8 | — | 9–16 | 100–372 |
| Held | 256 | 128 | 4 | 1–3 | 10–17 | 90–380 |

TRAIN includes every far cell from two independently worded four-family designs. Within each training family, governor position before or after the noun, same-spelling incidental position before or after the governed reference, requested fact role, and fact set are fully crossed with the two admission signs. Across the two designs, both spellings in each training noun pair take both the heading's reference role and the differently spelled governed role with the same verb pair. Within each endpoint, the referenced spelling occurs at distinct governed and incidental original addresses. The inherited tuple proposer requires the differently spelled noun to follow both earlier mentions of the referenced spelling for the correct, exchanged, and two incidental tuples to coexist. The differently spelled noun therefore retains that relative position.

Held has 64 near and 64 far pairs. Each family crosses both distances and the same structural factors, using two additional job/home fact sets. In both splits the two endpoints of an opposed pair differ only in the two governor words and their covering BPE windows. The scope and incidental occurrences retain separate, unchanged byte and absolute BPE addresses. Both far governor spans are outside the corresponding noun's six-ID context. The frozen generator verifies each of the four registered scope tuples exactly once, both action addresses, and both authentic fact copy bodies through their periods. Held nominal replies span 32–43 IDs through EOS.

`guard.jq` checks disjoint authored templates, wording families, fact values and IDs across the five designs from TRAIN, held, #4286 and #4288. It checks authored text, case IDs, prompt IDs and prompt digests across the new splits and the two preceding panels. All eight held target nouns are absent as whole words from the new TRAIN user text, the #4286 and #4288 panel text, and all 512 user turns in the sealed #4283 teacher pack. The duplicate family and duplicate training digest probes both fail closed. Structural preparation did not rank an endpoint.

The private Wick root is `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-far-transport-margin/panel`. The full authored grammars, unused generated near TRAIN cells, and structural probe versions are retained there. Initial structural drafts exposed the inherited 128-ID post-declaration tail limit, a fact pair without the required shared initial BPE ID, and one near branch with zero governor IDs in context. The final grammar revisions were enrolled using the same structural checks.

The sealed `panel-v1/{train,held}` folders each contain `reference.json`, `blind/text.bin`, `blind/coordinates.jsonl`, `links.bin` and `aid.bin`. The exporter receives only second-user strings and the fixed historical first turn when generating blind geometry. The separate label and supplied-control files are produced by joining coordinates back to the structural reference. Held additionally includes authentic dialogue scripts and `jobs.tsv` under `held/fresh/`, supplied-scope sidecars under `held/fresh/structural-controls/`, and precomputed #4286 structural packets under `held/structural-links/`.

| File under `panel-v1` | SHA-256 |
| --- | --- |
| `SHA256SUMS` | `0241228fa9f93f7362852d6be551dee465cb94299135d9c277b24991d85d329a` |
| `authority.sha256` | `65360b8d8376aacca9d51ee35f38c0d964ba33821449c8ec2028c1d5ea8e6d4d` |
| `guard.json` | `2c6e84f50b741ed90477c1c93aeb79c240f876a6fc8a9c9189f8aa24c5143bfc` |
| `train/reference.json` | `a227b2b7f4930bee43c611d706d90ebe3b88e2c568b50064e217f8b1f611df86` |
| `held/reference.json` | `bcc617eb26d2a32ed8332bce70146583517cd9d52e523e9b4bb9acbdf1c5d7cf` |
| `train/blind/text.bin` | `82dc970f304abcdaf584ab84e85694ba9042152af5049879b8bf337979427c66` |
| `held/blind/text.bin` | `cab8410798a376c9bde74139f0839f0ecafbb6d37c537c1a243047a2b16e7379` |

Wick preparation reproduced all content in `panel-v2`, with one manifest difference. The original negative family-control diagnostic includes the run's temporary staging path. Both manifests verify independently. On Wick, use an unused positive version for another reproduction.

```bash
root=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-far-transport-margin/panel
bash "$root/methods-v1/prepare-wick.sh" 3
(cd "$root/panel-v3" && sha256sum --strict --quiet -c SHA256SUMS authority.sha256)
cmp <(rg -v '  \./duplicate-family\.stderr$' "$root/panel-v1/SHA256SUMS") \
    <(rg -v '  \./duplicate-family\.stderr$' "$root/panel-v3/SHA256SUMS")
```
