# Reproduce the source comparison on Wick

The Wick ZFS root is `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-text-governor-links`. Source preparation uses the fixed D11 reader and proposer at integration commit `76d65cef10497ef03e1038be15e02b0ce595258b`. The writing artifact SHA-256 is `5c843187d34c3a6ca94560a161c8d82c3655d16dd65e283c36753d1e9f55e7b8`. The same parent's original pooled words and the exact signed tokenizer and generated history are pinned by the export wrapper. The GPU fit is its own 13,336-byte native state. The held gate did not release fresh ordinary-client captures. [Input manifest](evidence/released-inputs.sha256) · [Parent D11 restoration](../codex-4286/RESTORE.md)

The [panel method](panel/README.md) reproduces both author seals, including all same-spelling rivals. [Exporter](export/prepare-wick.sh) stages the string-only inputs and the maps from all original words and native proposals to prompt addresses. Use v2 for both the exporter and GPU. [Learner method](learner/README.md) identifies build, finite, training and detached scoring commands. The [released measurement](measure-wick.sh) verifies the FIT and input manifest and runs through the common Wick lease. Reproduce into new output directories. The independent [replay checker](parity/README.md) compares all joint decisions, including tie and abstention fields, without opening the labels.

The [private receipt](evidence/private-restore.json) pins the dataset revision, archive name, byte count and hash. It contains the fresh reference and private grammar, original-address maps, raw ranks and bank states, per-prompt decisions, summaries, and replay records. The original blind coordinate packets and uncompressed traces also remain on Wick ZFS. To restore the compact data to a new directory on Wick, set `study` to the root above.

```bash
set -euo pipefail
receipt="$study/restore-v1/receipt.json"
target=$(mktemp -d "$study/source-recovery.XXXXXXXX")
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

The Wick restore verified all 70 original files from the pinned remote revision. Regrading the restored 256 training records and 256 held records reproduced both full source scoreboards, the ordinary release decision and the source-error trace byte for byte. [Restored grade hashes](evidence/restored-grades.sha256)
