# Fresh signed-source panel

The panel contains 18 independent ordinary user turns grouped into nine same-passage pairs. Both source candidates have exact token IDs, original addresses and half-open UTF-8 byte spans. Each pair preserves the complete passage and its serialized prequestion tokens. A reversed request switches the supported address, and a support flip changes a supported request to NoRead while preserving both candidates.

| Pair | Pairs | Supported turns | NoRead turns |
|---|---:|---:|---:|
| Authored role reversal with explicit exclusion | 1 | 2 | 0 |
| Authored role reversal with one token ID at two addresses | 1 | 2 | 0 |
| Authored omitted role, explicit denial, unconfirmed assertion and excluded destination | 4 | 4 | 4 |
| Natural requested-source reversal | 2 | 4 | 0 |
| Natural requested-role support flip | 1 | 1 | 1 |

The three natural passages are intact paragraphs from separate articles of the pinned author-published SQuAD 2.0 TRAIN file. All questions were written for this panel. The builder screens natural article identity and passage hash against the sealed #4329 and #4332 inputs, normalized question against #4329 and #4332, and normalized full prompt against #4330 and #4332. The author-selected passages and questions are pairwise unique within the panel. Exposure outside those named records was unmeasured. [SQuAD input receipt and attribution](../../codex-4329/supply/README.md)

The independent reader received only the case IDs and prompts before supplying 18 support decisions and original byte ranges. The reader agreed with the author key on all 18 labels and ranges. The natural case beginning `979a8592` uses the explicitly named teaching institution as the target. The wording asks “where,” so a reader could interpret it at the granularity of city or country. The finite grade uses the agreed institution address. The other supported cases also use the agreed exact source spans.

The read-only label-free tape under `panel-v1/blind` contains `input.jsonl` (ordinary prefix IDs, user bounds and prequestion boundary), `prompts.jsonl` (raw text) and `opposing.json` (the symmetric case-to-other-case map). The private `panel-v1/truth/key.jsonl` binds both candidate sites and the supported target, which is null for NoRead. Case and pair identifiers preserve the #4332 builder's text-derived hash scheme. The specification, raw panels and independent grades are retained on Wick, outside Git.

| File under `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-signed-site` on Wick | SHA-256 |
|---|---|
| `panel-spec-v5.json` | `cf17b3204f487c3bfe872f0180a0e83a942f965e28009830630cc33cd4c35a34` |
| `panel-v1/blind/SEALED.sha256` | `490230556a8983369f77eccbebeab0fe873de7ecd1ef5174c0c1eabadfa6cbaf` |
| `panel-v1/blind/input.jsonl` | `268c54324a687792acfc9722809a060ba7cb18435c57be8b3af5cf57f8373929` |
| `panel-v1/blind/prompts.jsonl` | `a7c84c26baae3212e27bfad9f7c65ff7a33ba0925829a69769f759b2e7ff8ec4` |
| `panel-v1/blind/opposing.json` | `5a99530f2c4f4fe0ba15752480af20b2c9ae7e6119db9da5bd9215a139bf55c5` |
| `panel-v1/truth/SEALED.sha256` | `6321035ae5257ee57938bf78419b74433ae8261ad9b00ee8eee8a1e8411c8158` |
| `panel-v1/truth/key.jsonl` | `c056d8e4adc8998ad46164cf0b04978556e7c0cc4680b3bbcf52ded638aae68e` |
| `blind-reader-v1/grades.jsonl` | `ee07fb9b2b7782d72f7887f3133d2878cde922c271b9af9652b7cc9d31a05f69` |

Build on Wick in this directory and use a new output location. The first locked build and second generation matched both final tapes and their manifests byte for byte. A separate preflight of the unchanged #4332 builder produced the same prompt file byte for byte.

```sh
runs=/mnt/plexpool/uor-r4-corpus/runs
base="$runs/20261127-codex-signed-site"
prior="$HOME/uor-codex/20261127-whole-turn-recurrence/panel-v2"
export CARGO_TARGET_DIR="$base/panel-target"
cargo build --release --offline --locked
umask 077
"$CARGO_TARGET_DIR/release/signed-site-panel" "$runs" "$prior" "$base/panel-spec-v5.json" "$FRESH"
for tape in blind truth; do
  (cd "$FRESH/$tape" && sha256sum --strict -c SEALED.sha256)
  diff -qr "$base/panel-v1/$tape" "$FRESH/$tape"
done
jq -en --slurpfile author "$FRESH/truth/key.jsonl" \
  --slurpfile reader "$base/blind-reader-v1/grades.jsonl" \
  '($author | length) == 18 and ($reader | length) == 18 and
   ($author | map({case_id,target:.target.bytes}) | sort_by(.case_id)) ==
   ($reader | sort_by(.case_id))'
```
