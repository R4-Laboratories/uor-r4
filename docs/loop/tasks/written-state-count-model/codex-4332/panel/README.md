# Fresh paired source fixtures

The panel has 18 single-user turns in nine pairs. Each pair keeps its full passage, exact prequestion token IDs, and both candidate source positions and IDs fixed while replacing the question. Four pairs reverse the requested source across distinct first lexical IDs. Two reverse the original position across identical first lexical IDs. Three pair one supported question with a question whose answer is absent.

| Pair type | Pairs | Supported turns | NoRead turns |
|---|---:|---:|---:|
| Authored distinct-ID reversal | 2 | 4 | 0 |
| Authored equal-ID reversal | 2 | 4 | 0 |
| Authored support flip | 2 | 2 | 2 |
| Natural distinct-ID reversal | 2 | 4 | 0 |
| Natural support flip | 1 | 1 | 1 |

The natural passages are intact from the pinned author-published SQuAD 2.0 TRAIN file. Three distinct articles were selected outside the article, exact passage, and normalized question identities used by the three sealed #4329 SQuAD tapes. Every question was newly written for this panel, including the unsupported natural question. The builder also screens normalized full prompts against the sealed #4330 panel. Historical exposure outside those checked inputs is unmeasured. [Earlier input receipt and attribution](../../codex-4329/supply/README.md)

The builder uses the product's protocol-2 tokenizer and token-class table. It requires exact byte round trip, the ordinary user bounds, an unchanged token seam after the two newlines, and 12 tokens of context reserve in the 192-token window. Both indexed sources open at the first canonical answer ID. The entire declared source span has the same token IDs as the canonical space-prefixed answer. Sites are in passage order and their half-open UTF-8 byte spans refer to the original passage. Case IDs are text-derived hashes. Input and key join only on case ID.

## Blind review and readback

The independent reader received the blind directory before inspecting any model output or the private key. The reviewer supplied support or NoRead and the half-open UTF-8 span for each supported case. The author-side join agreed on all 18 labels and spans, including which original occurrence to use when the answer IDs are identical. These fixtures establish independently checked source targets for the finite test.

On Wick, the private base is `/home/callard/uor-codex/20261127-whole-turn-recurrence`. The selected packet is `panel-v2`. The preliminary v1 files remain on Wick. V2 corrected two authored question spans and made one natural question's reference explicit before blind review. The private spec and original prose stay on Wick.

| File under the private base | SHA-256 |
|---|---|
| `spec-v2.json` | `014f414f347fc25602039f222f4a92c7bd97e48e3d161f57696e79f83ad7cef8` |
| `panel-v2/blind/SEALED.sha256` | `49e4da492eec5a7a7529ef242862a2371e8f917286d6401b37196386a744374e` |
| `panel-v2/blind/input.jsonl` | `a5fc2936ebc374595b5cb5d8916380c2a54addc89dd6bce0ff51705e0cf35d5c` |
| `panel-v2/blind/prompts.jsonl` | `3b081ef23e1444a74822c79cd50ea8e9b22c1ee9e3d8ab2125ac01e9a5119ad8` |
| `panel-v2/truth/SEALED.sha256` | `cbebd8c7a12768102abe2d83c3405f2c73761d1c2b52ca532455592ad0897736` |
| `panel-v2/truth/key.jsonl` | `db24c1a2945317970a0289e608f7f9236ca0fa15c63967880e6bd31dbd11ac6d` |
| `panel-v2/independent-adjudication/grades.jsonl` | `242407405f034121e46cd5fa6e709ee9a77fc7197cd9ba577ff836ec240574a7` |

Build the locked package on Wick from this directory. Use a fresh output location and compare both rebuilt tapes and their manifests. The original second preparation matched the two sealed directories byte for byte.

```sh
CARGO_TARGET_DIR="$WICK_BASE/panel-target" cargo build --release --offline --locked
umask 077
"$WICK_BASE/panel-target/release/whole-turn-recurrence-panel" \
  /mnt/plexpool/uor-r4-corpus/runs "$WICK_BASE/spec-v2.json" "$FRESH"
for tape in blind truth; do
  (cd "$FRESH/$tape" && sha256sum --strict -c SEALED.sha256)
  diff -qr "$WICK_BASE/panel-v2/$tape" "$FRESH/$tape"
done
```
