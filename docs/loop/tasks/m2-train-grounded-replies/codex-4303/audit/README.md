# Independent original-address audit

The audit uses the sealed [prior-study Rust decoder and ranking verifier](../../codex-4301/audit/README.md). Its small extensions verify the two factor edits and compare the physical objective and lexical probes. The inherited decoder checks the original prompt, absolute BPE and byte spans, candidate site and governor ordinals, and the exact one-token first/later edit. The inherited ranker independently consumes the native integer state, text packet and frozen banks and checks each field of the returned scorer traces. No model execution was repeated to generate these independent checks.

The M objective audit enumerates completed tuples for both the requested site and every third site available at the same authored governor. Thirds exclude both of the original same-spelling occurrences. It recomputes the fixed role, opposite attachment and heading and records every minimizing third site, the first original tuple, ties, the transport margin, and separate statuses for missing sign, role or completion. The inherited scorer audit then computes target and strongest-third witnesses with both governors free. An unrestricted third may use the authored governor or an alternate one. The E audit traces the 64 original TRAIN labels into a geometric lower median, interpolates from native nearest centers and compares every serialized byte of all five proposals. It compares each physical word distance and center slot and counts membership changes using strict own cost less than opposing cost.

The complete TRAIN audit reproduced 37,837,362 native and competition fields across 295 matrix states, checked their original regression and near identities, and reproduced all 232 grid ranks. The chosen four states preserve 4/4 signed switches and 24/24 full near pairs each. For the selected factorial, E changes 60 bytes only in the affirmative lexical bank and M changes 14 bytes only in the active excluded transport tail. The combined state uses the same changed bytes as the single edits. The separate M lineage and E probes agree on 1,044,124 and 228,480 scalar fields respectively.

The held audit unpacked all five ordered bridges to the frozen original state bytes and reproduced 1,585,749 native, competition and lexical fields. Every arm yields 0/128 exact signed source and full pairs. All five share the same seven fixed-third paired wins, all in `salvage-tool-invoice`. The source-eligible `retain` word turns rise from 0/128 to 128/128 with E, with native own/opposing costs changing from 84/56 to 42/56. Sixty-four contested affirmative retain turns gain a fixed completion. The other 64 contested targets with no fixed completion are excluded `bypass` and `ignore`, each 32 turns. The free third beats the pinned target on all 192 rows that have a pinned target after E. On 96 of those rows every best free third governor has an address different from the authored governor. Other labeled governors retain their prior memberships. E changes membership for nine additional word surfaces in the complete packet. The correct supplied links recover all 256 signed and full rows, and the swapped same-spelling links recover zero.

The Wick evidence root is `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-third-margin-retain`. In a restored copy, the runnable verifier is `audit/build-v5/slot3-audit`. Check manifest hashes before running the commands. Use a fresh output directory on Wick for each replay.

```bash
root=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-third-margin-retain
runs=${root%/*}
audit=$root/audit
build=$audit/build-v5
scripts=$audit/source-v5
panel=$root/panel
train=$root/physical-train-v1
codec=$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json

bash "$scripts/archived-wick.sh" "$runs" "$build" "$root/physical-e-finite-v1" "$audit/replayed-archived"
bash "$scripts/panel-wick.sh" "$build" "$panel" "$codec" "$audit/replayed-census"
bash "$scripts/finite-wick.sh" "$build" "$root/physical-e-finite-v1" "$audit/replayed-e-finite"
bash "$scripts/margin-finite-wick.sh" "$build" "$root/physical-m-finite-v1" "$audit/replayed-m-finite"
bash "$scripts/train-wick.sh" all "$root" "$build" "$train" "$train/matrix" "$train/score" "$root/election-v1" "$audit/replayed-train"
bash "$scripts/held-wick-v2.sh" "$root" "$build" "$root/physical-held-v1" "$root/shipment/fresh-input" "$audit/replayed-held"
```

The held runner above is [held-wick.sh](held-wick.sh) in this source tree. The archived executable rebuild script checks the original sealed source before compiling tests and the verifier. Synthetic E and M fixture modes validate the historical inactive marker words and accept a separate parser view; real state mode checks the original bytes directly.

| Wick audit artifact | Manifest SHA-256 |
| --- | --- |
| `build-v5/SHA256SUMS` | `e0d8b534043b1f63e758eb3e2e18ad9201d7eaf8546ea6c6447ce10eed41ae8c` |
| `predecessor-v1/SHA256SUMS` | `5ca592a2161f7aa1c755878c2f3d9cb2dee2c3c46d7afffd3ad31b2b36d549a7` |
| `panel-v2/SHA256SUMS` | `977fda09edc5ae43e17f40d98d28112d2acedad698285b9ce4789c727ce849c7` |
| `e-finite-v2/SHA256SUMS` | `8ee3f9df10d6efb5660dc28d3ecb955fe2ccb93b8106e8693a9d4fee91f4790c` |
| `m-finite-v3/SHA256SUMS` | `551ae277dd991cfe58890f37ab7e2f473885025a7452ab9cb89a267e3cd06456` |
| `train-fit-v1/SHA256SUMS` | `a5e3a830b4d8c1136c70b67cd6175869829044ecea06b58f413e81f0993da286` |
| `physical-train-factors-v1/SHA256SUMS` | `0dc3ac76ed21f5e77a37666d4451e09099675b37a1d3948463e456a7dfa39dd2` |
| `train-finalists-v1/SHA256SUMS` | `7fef6b379a68d641ba41f31880c403786cd74d7922f76c314fe6d85843ea4cee` |
| `held-fit-v1/SHA256SUMS` | `76b5d1c59ad279c502a86e68972b49f99a493608daf20fe7f995fafb279b6135` |
| `train-all-v1/SHA256SUMS` | `35de55dcdaa3c0311c9d3c72fa87e3073e800750931356c814e41318d4de5e3a` |
| `physical-held-v1/SHA256SUMS` | `eddcd4ee3d1c62d4be7ff6997b3dabae9f1b75caf15b1b305e485e7984a4e1ef` |
