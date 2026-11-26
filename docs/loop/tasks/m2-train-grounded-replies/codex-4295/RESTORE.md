# Restore the original-address omission experiment

The private dataset `caseyallard/uor-r4-store` holds three objects at `results/codex/intervening-source-20261126-3d0d0f3cb490`, revision `0f8cc5180377935eeb26b64cf4f679cbfa80ce9c`. The three objects total 4,059,063 bytes. The archive has 212 manifested files and SHA-256 `3d0d0f3cb49018ea29d13c310155912ecbb33469e3ea4f855e44d44c371bcb45`. [Publication receipt](evidence/receipt.json) · [Independent download and regrade](evidence/restored.json)

The packet contains the three physical readers' outputs, ordered candidate costs, original addresses, intervention masks, build and finite-test receipts, registration, release, independent readback, grades and the supplemental analysis of every cheaper occurrence. The frozen state and original reference text, tokenizer coordinates and shared geometry come from the preceding Wick experiments. Regrading uses their pinned paths on Wick ZFS. Recover those dependencies first if restoring the study on a replacement drive. [Frozen state and primary measurement](../codex-4292/RESTORE.md) · [Original mixed panel and methods](../codex-4291/RESTORE.md)

## Verify an independent download

Run on Wick with the private dataset credential already configured. The commands create a new directory inside the study every time. They verify the archive against the published digest, then verify every extracted file and recompute the 11 recorded grade outputs. The archived `inputs.sha256` records physical paths, which change when restoring. All comparisons below use the actual result files.

```bash
set -euo pipefail
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-intervening-source
mkdir -p "$study"
restore_dir=$(mktemp -d "$study/restore.XXXXXXXX")
prefix=results/codex/intervening-source-20261126-3d0d0f3cb490
revision=0f8cc5180377935eeb26b64cf4f679cbfa80ce9c
"$HOME/.local/bin/hf" download caseyallard/uor-r4-store --repo-type dataset \
    --revision "$revision" --include "$prefix/*" --local-dir "$restore_dir/download" --quiet
packet="$restore_dir/download/$prefix"
extracted="$restore_dir/extracted"
mkdir "$extracted"
(
    cd "$packet"
    echo '3d0d0f3cb49018ea29d13c310155912ecbb33469e3ea4f855e44d44c371bcb45  measurement.tar.zst' | sha256sum --strict --quiet -c -
    sha256sum --strict --quiet -c SHA256SUMS
)
tar --zstd -xf "$packet/measurement.tar.zst" -C "$extracted"
(cd "$extracted" && sha256sum --strict --quiet -c "$packet/MEMBERS.sha256")
bash "$extracted/methods-v1/grade-wick.sh" "$extracted/measurement-v1" \
    "$extracted/panel/panel-v1" "$restore_dir/regraded"
compared=0
while read -r digest member; do
    [[ $member == ./inputs.sha256 ]] && continue
    cmp "$extracted/grade-v1/$member" "$restore_dir/regraded/$member"
    compared=$((compared+1))
done < "$extracted/grade-v1/SHA256SUMS"
test "$compared" -eq 11
```

On the existing Wick study, the sealed wrapper also compares the three downloaded Hub objects to the original upload and writes a compact replay receipt. Use a new restore version for each call.

```bash
root=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-intervening-source
bash "$root/storage-methods-v3/preserve-wick.sh" download v1 v2 \
    "$root/packet-v1/remote-v1/receipt.json"
```

The local archive replay, publication readback and separate download produced the same replay receipt, SHA-256 `93d0c6de72f03b97f8789003040bf4c53dd9d8f7714548978fe32988daa5ff40`. The recorded physical measurement manifest is `ae9471af842a490d39a9b114fd096e194cb61ba9158c495adbbcddae35278fb8` and the independently replayed address audit is `2df4107410187d6e4583a1b0dcb5218a5f58b292018213c07df8bc5cb46c6d22`. Regrading uses the existing GPU readings.
