# Restore or replay this comparison

The original raw captures and device inputs live on Wick ZFS at `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-governor-noun-links`. The unchanged pooled D11 lives in the preceding study at `20261125-codex-governing-source-pairs/models/fold-8/ordered-all.d11`, relative to the same runs directory. Its SHA-256 is `5c843187d34c3a6ca94560a161c8d82c3655d16dd65e283c36753d1e9f55e7b8`. The [predecessor's reconstruction record](https://github.com/UOR-Foundation/uor-r4/blob/76d65cef10497ef03e1038be15e02b0ce595258b/docs/loop/tasks/m2-train-grounded-replies/codex-4283/RESTORE.md) also identifies the private package for rebuilding that artifact from the signed parent.

The Rust study crates use the pinned reader sources from integration commit `76d65cef10497ef03e1038be15e02b0ce595258b`. On Wick, [the panel wrapper](panel/prepare-wick.sh) verifies its source authority before construction. The [device wrapper](device/run-wick.sh) builds against the retained source and CUDA inputs and checks their manifests. The [client wrapper](client/run-wick.sh) verifies the actual ordinary binaries and both D11s. The finite v1 source ranker is preserved alongside the v2 ranker that executed the held study. The exact row joins and raw writer reads are reconstructible from the saved device addresses and ordinary JSON traces.

To check the physical and ordinary captures in place on Wick, use the stored manifests.

```bash
set -euo pipefail
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-governor-noun-links
for dir in panel/panel-v1 device/build-v2 device/prepared-v2 device/score-v2 gate-v1 client/capture-v1; do
    (cd "$study/$dir" && sha256sum --strict --quiet -c SHA256SUMS)
done
```

The [private restore receipt](evidence/private-restore.json) names the exact dataset revision and download prefix. On Wick, the following restores the compact original-address panel, output records and measured summaries to a new directory and checks every member.

Wick downloaded the three objects at revision `8c14f5c4c3b78cf9417c802670999b16fc87dd49`, totaling 85,168 bytes, and verified all 30 archive members. Regrading the restored device endpoints reproduced the [source gate](evidence/source-gate.json). Regrading the 1,280 compact per-arm reply records reproduced the [independent token audit](evidence/token-audit.json). Both comparisons were byte-identical. The packet also contains a byte-verified copy of the [full ordinary summary](evidence/ordinary.json). Rebuilding that full summary from the original traces uses the raw records on ZFS. [Independent regrade hashes](evidence/restored-grades.sha256)

```bash
receipt="$study/restore-v1/receipt.json"
target=$(mktemp -d "$study/independent-restore.XXXXXXXX")
repo=$(jq -er .repo "$receipt")
prefix=$(jq -er .path "$receipt")
revision=$(jq -er .revision "$receipt")
"$HOME/.local/bin/hf" download "$repo" --repo-type dataset --revision "$revision" \
    --include "$prefix/*" --local-dir "$target/download" --quiet
packet="$target/download/$prefix"
(cd "$packet" && sha256sum --strict --quiet -c SHA256SUMS)
mkdir "$target/restored"
tar --zstd -xf "$packet/measurement.tar.zst" -C "$target/restored"
(cd "$target/restored" && sha256sum --strict --quiet -c "$packet/MEMBERS.sha256")
```
