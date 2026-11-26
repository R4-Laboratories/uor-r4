# Held text-to-governor panel

The training input is the 256 already exposed second turns in #4286's sealed `panel/panel-v1/fresh/reference.json`. The new held grammar was authored independently and enrolled once through the same pinned protocol-2 generator. The custody root on Wick is `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-text-governor-links/panel`. Keep its private grammar, prompts and grading labels out of training. The four signed governor pairs recur intentionally so the transfer test asks whether training identifies their links in fresh wording and novel noun contexts.

The held panel crosses four new wording families and two fresh job/home fact sets with both signs, near and far governor distances, governor placement before and after the noun, same-spelling competitor placement before and after the governed noun, and job-first and home-first requests. All 256 factorial cells are occupied. Each wording family contributes 64 endpoints and 32 opposed pairs. Each family and distance contributes 32 endpoints, with 16 per level of each crossed dimension. The fact sets carry three- and four-sentence replies respectively. Consequently, reply length and fact set move together.

The complete second-turn prompt includes the original generated first assistant response through EOS. Within all 128 opposed pairs the two governor words and their covering BPE windows change while the noun byte and absolute BPE intervals, every other prompt ID, and the expected reply stay fixed. Both governor signs have near gaps of 1–3 IDs and far gaps of 12–20 IDs. Both far governor spans are wholly outside the relevant six-ID context. The literal answers span 24–31 IDs through EOS. The original generator also confirms the job/home copy signatures and both action candidates.

The sealed `panel-v1/fresh/reference.json` retains the #4286 reference ABI. Its `native_scope` records the affirmed, excluded and referenced occurrences, and `neutral_rival` provides the other occurrence of the reference spelling at a distinct byte and BPE address. The unchanged proposer offers each of the four registered tuples exactly once per endpoint. Their cases are the correct tuple, the exchanged tuple, and each of those polarity orientations with the same-spelling neutral competitor substituted for the real source. The proposer enumerates 78–344 tuples per endpoint in total. The predecessor's preparation-only packet reader verified each candidate's exact address and preserved the whole candidate universe in `structural-links/geometry.bin`. The supplied correct and exchanged link packets keep both original governor addresses and exchange only their noun endpoints. Their exact bytes and separately serialized labels remain under the seal.

`split.jq` verifies that the held set reuses all four signed governor pairs, with each row's two signed governor strings recovered from the original bytes. Eight held target nouns are absent as whole words from the historical training user text. The splits share zero family names, authored templates, fact values, case IDs, full authored second turns, encoded prompts, or encoded prompt digests. The shared first-turn generation and protocol framing are controlled across splits. The guard accepted this split and rejected both a duplicated historical panel and a held row carrying a training prompt digest. Structural checks enrolled every cell before any ranking or writer execution.

| Wick artifact | SHA-256 |
| --- | --- |
| Historical training `panel-v1/fresh/reference.json` | `b95b52f26debef4dae7197baa2557b228fe60d583c466c8d3d78353271fe88fb` |
| Private held `grammar-v1.json` | `7421efc8dd4a3fcb34e8f22c44d2f2ed8161229928ba59fe0aef855a79c1181f` |
| Held `panel-v1/fresh/reference.json` | `fcc8031daf05d66a47fda878cbe3015ead90b1b9990dd67a268f316e716edce3` |
| Held `panel-v1/SHA256SUMS` | `85f267274746f65f79f60a3c7e4f8c830c779de5258e5dc63ceb1c063042a6bb` |
| Held `panel-v1/authority.sha256` | `c790e7c407cb9115db2da913de417f610206ab9a51f7090eeb2dbc3545655f81` |
| Held `panel-v1/enrollment.json` | `59bb572d06ff9972d417296e69f18ebb374148183a95f5e74043d3255a87c9c9` |
| Held `panel-v1/split.json` | `732f7055cf7db82dc32285a03d59e390da3b92b8d49f721bf7f2166195534b27` |

The sealed panel was reproduced byte for byte as `panel-v2` on Wick. To reproduce again, run the staged methods with an unused positive version. The wrapper checks the inputs, predecessor source and binary receipts, and the resulting held reference before sealing.

```bash
root=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-text-governor-links/panel
bash "$root/methods-v1/prepare-wick.sh" 3
cmp "$root/panel-v1/SHA256SUMS" "$root/panel-v3/SHA256SUMS"
```
