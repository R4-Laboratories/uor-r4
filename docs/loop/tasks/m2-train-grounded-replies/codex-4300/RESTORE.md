# Restore the source competition evidence

Wick ZFS holds the build, original packets, return receipts and self-checking snapshots under `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-switch-source-competition`. The archived raw outputs were produced on the single leased Vast RTX 4090 recorded in the [compute receipt](evidence/compute.json). The parent Wick capture and integer state are included in the primary snapshot. The smaller distance-matched snapshot uses the primary snapshot's authenticated immutable state and signed tokenizer. [Primary archive receipt](storage/receipt.json) · [Distance-matched archive receipt](matched-train/archive-receipt.json)

| Packet under the study root | Bytes | SHA-256 of measurement.tar.zst |
| --- | ---: | --- |
| `storage/archive-v1/packet` | 41,181,058 | `a13331a29bff362ade409def5c36dbef13e74261977b1c7b0f8bbc47027d5086` |
| `matched-train/storage/archive-v2/packet` | 1,296,025 | `73e8ba0338d892833077e7d9f5c14aa29466322d38dc3acdfdaddeaafd5ac200` |

The first ORDER commit and the sealed release contain the exact pre-run ORDER, SHA-256 `a350561435f8973f450a0ff6fc4254a0cb7275cba91518f3dc51f00605d3e23c`. The checkout adds the harness field name to the existing roadmap link after the run. To reconstruct the original methods manifest, use the [registered ORDER at the first commit](https://github.com/UOR-Foundation/uor-r4/blob/5ca1116b5bb0a5aaebea9af06f70acbd1acce8ed/docs/loop/tasks/m2-train-grounded-replies/codex-4300/ORDER.md).

Read-only checks for the existing Wick artifacts run from the study root.

```sh
set -eu
cd /mnt/plexpool/uor-r4-corpus/runs/20261126-codex-switch-source-competition
for packet in storage/archive-v1/packet matched-train/storage/archive-v2/packet; do
    (cd "$packet" && sha256sum --strict --quiet -c SHA256SUMS)
done
for result in release-v1 device/finite-v1 device/finite-v1-lease capture-v1 capture-v1-lease \
    grade-v1 train-comparison-v1 layout-grade-v1 audit/selector-axes-v1 \
    matched-device/capture-layout-v1 matched-device/capture-layout-v1-lease; do
    (cd "$result" && sha256sum --strict --quiet -c SHA256SUMS)
done
```

The [primary replay instructions](storage/README.md) and [distance-matched replay instructions](matched-train/README.md) rerun the integer verifier and reproduce the tabulated rows in fresh directories. For the primary and secondary packets, two independent extractions reproduced the same replay receipts. The separate unabridged pod job archive lives at `custody/pod-retirement-v1/switch-source-final.tar.zst`. Its digest is `8e3d6dc1a1fce0f3fc32484a89fff1211221e3e603dc59b6352e3068007fab9f`. It holds all 213 job files. The controller verified its three physical capture manifests against the individual Wick returns before terminating the rental.

The original-address comparison and numeric example extraction can be rerun with [compare-train.jq](compare-train.jq). Use `grade-v1/historical.rows.jsonl` for the measured historical reference and the individual cohort's graded rows for the second argument.

```sh
jq -ne --slurpfile historical grade-v1/historical.rows.jsonl \
    --slurpfile reads matched-device/replay-v1/rows.jsonl \
    -f train-comparison-v1/compare-train.jq | cmp - layout-grade-v1/summary.json
```
