# Original occurrence enrollment

Enrollment reads the sealed #4291 mixed TRAIN tape and the #4293 protected observations at updates 234 and 240. The [Wick preparation script](prepare-wick.sh) verifies the published manifests, checks the selected state bytes against the raw batch, and builds a host-only input verifier against the historical packet decoder and ByteBPE tokenizer. Restore the mixed input from the [#4291 custody record](../../codex-4291/RESTORE.md), the two protected batches and states from the [#4293 custody record](../../codex-4292/RESTORE.md), and the separately sealed residual summary from `far-residuals-v1`. The [residual recipe](../../codex-4292/grade/far-residuals-wick.sh) derives the list from already recorded batches.

The verifier decodes the binary packet and checks all 256 physical row IDs, prompt digests, word and site byte/BPE addresses, and listed tuples. It decodes the reference prompt's framed second user turn and checks the BPE interval of each original word and site against that tokenizer. It joins the 24 residual pair IDs to 48 common far rows, requiring both opposed instructions, both correct joint governors, the correct heading, and exactly one wrong joint noun per row at both protected updates. The wrong site must match the recorded strongest fixed-governor rival and the independent native winner on that side. The other side can select its gold noun jointly even when its isolated fixed-governor scorer prefers another site.

A control must occur in that sign's original tuple slot, have an eligible attachment under the fixed governor, and lie between the governor and target in original bytes and absolute BPE intervals. The verifier excludes all three gold sites, the selected wrong site, and all recorded tied strongest rivals at both protected updates. It orders the remaining occurrences by the BPE interval gap from the wrong site, absolute difference between interval starts, byte start, byte end, and site ID. It uses the original eligibility and the recorded winners without measuring counterfactual rankings.

On Wick the selection yielded eight eligible controls on 32 prompts and nine on 16. The first eligible occurrence under the fixed order has surface `the` on all 48 prompts, as permitted by the original candidate tape. The selected address is independent of its spelling. The 234 and 240 strongest-rival sets agree on every enrolled row.

## Sealed inputs

The seal is at `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-intervening-source/panel/panel-v1`. Its second preparation at `panel-v2` has the same manifest and member bytes.

| File | Contents | SHA-256 |
| --- | --- | --- |
| `SHA256SUMS` | Enrollment, antecedents and three rejected address fixtures | `6c8b8d3a98267078474b9b7167547cbf50bf54792f9e628afa158ca9753351c2` |
| `masks.tsv` | 256 physical rows, each containing failed sign, wrong winner site and control site | `63fa5dacbb2875d17c11a2706fc365463e49b1f8d54efc0ceddb13bcaaee7eeb` |
| `rows.jsonl` | 48 enrolled rows with the case and pair IDs, admission, gold and selected links and tuples, role, fact sources and ordered controls | `d6170b8625a5b295a2da1b29af5532efa962ffd54d53c03ec91bb3ba422af3b9` |
| `receipt.json` | Census, missing controls and prior rival agreement | `1260966f7de1478d88c3a4c4f46be60b7819b34124a41e30a4e98001c9c662a4` |

Sign 0 is affirmed and sign 1 is excluded. The 208 untouched rows use `-1 -1 -1`. The third field is `-1` when an enrolled row has no eligible control. Every enrolled row in this panel has a control. Byte and BPE intervals are half-open. Each occurrence retains its original row-local site ID. The guard rejected a misaddressed winner, an out-of-range control, and a control at the original gold address.

Verify the existing seal or reproduce it in an unused version on Wick.

```bash
root=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-intervening-source/panel
(cd "$root/panel-v1" && sha256sum --strict --quiet -c SHA256SUMS inputs.sha256)
bash "$root/methods-v2/prepare-wick.sh" prepare v3 v2
cmp "$root/panel-v1/SHA256SUMS" "$root/panel-v3/SHA256SUMS"
```
