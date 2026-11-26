# Restore the Wick source comparison

The immutable private dataset revision is `f988e81d12a8fe0cf0711e358c6ce3d218e3b352`. Its three objects total 50,218,483 bytes. The 49,601,300-byte compressed archive contains 4,862 files. It includes the enrolled 24 near-pair identities, all six training elections, the native state in each arm, the original fresh text and addresses, all 24 held source captures and the writer decision. It also preserves the first incomplete grade and the complete regrade of the same reads. [Private dataset at the measured revision](https://huggingface.co/datasets/caseyallard/uor-r4-store/tree/f988e81d12a8fe0cf0711e358c6ce3d218e3b352/results/codex/all-outranker-interval-20261126-ce1e6257e0a1) · [Publication receipt](evidence/archive.json)

The [registered order](registered-order.md) is byte-identical to the order in the Wick release. The repository's [ORDER.md](ORDER.md) adds the machine-readable item label to the same roadmap reference after measurement. When restaging the measured version from this checkout, copy `registered-order.md` to the incoming directory as `ORDER.md`. The published archive already contains the exact release.

The archive pins the client used for the measured no-pass decision. For that release, recover the runtime and dispatch scripts from the archive. The client in this checkout also covers partial admission, which the physical panel never reached. Its separate Wick fixture uses recorded client traces to test which arms would be dispatched. [Client admission](client/README.md)

## Reproduce the result

On Wick, use the established private-store login and the study's ZFS directory. The following downloads the three objects at the pinned revision into a new directory and replays the six elections and the two fresh panels without acquiring the GPU. Use the next unused `download` version for subsequent readbacks. [Storage tools](storage/README.md)

```sh
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-all-outranker-interval
method="$study/storage/methods-v4/preserve-wick.sh"
receipt="$study/storage/archive-v2/published-v1/receipt.json"
bash "$method" download v2 "$receipt"
sha256sum "$study/storage/download-v2/replayed/receipt.json"
```

For recovery on Wick using only this Git checkout and the private dataset, run the following from the repository root. The replay uses the archived Rust binaries on Wick and checks every archived member against its manifest before execution.

```sh
set -eu
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-all-outranker-interval
receipt="$PWD/docs/loop/tasks/m2-train-grounded-replies/codex-4297/evidence/archive.json"
repo=$(jq -er .repo "$receipt")
prefix=$(jq -er .path "$receipt")
revision=$(jq -er .revision "$receipt")
mkdir -p "$study"
cold=$(mktemp -d "$study/accepted-restore.XXXXXXXX")
"$HOME/.local/bin/hf" download "$repo" --repo-type dataset --revision "$revision" \
    --include "$prefix/*" --local-dir "$cold/download" --force-download --quiet
packet="$cold/download/$prefix"
(
    cd "$packet"
    printf '%s  SHA256SUMS\n' "$(jq -er .packet_sha256 "$receipt")" | sha256sum --strict -c -
    sha256sum --strict --quiet -c SHA256SUMS
)
mkdir "$cold/extracted"
tar --zstd -xf "$packet/measurement.tar.zst" -C "$cold/extracted"
(cd "$cold/extracted" && sha256sum --strict --quiet -c "$packet/MEMBERS.sha256")
bash "$cold/extracted/storage/methods-v4/restore-grade-wick.sh" \
    "$cold/extracted" "$cold/extracted/storage/archive-v2/layout.json" "$cold/replayed"
printf '%s  %s\n' "$(jq -er .cold_replay_sha256 "$receipt")" \
    "$cold/replayed/receipt.json" | sha256sum --strict -c -
```

The local replay and two independent downloads produced the same receipt, SHA-256 `d39c895314fb594072a5330b99587006f6408a29dad857262428d01adecb9a30`. Each replay matched all 133 reference files and reproduced the decision to leave the writer unrun on this panel. The completed archive and its readbacks occupied 623,859,200 allocated bytes on Wick ZFS. [Cold replay receipt](evidence/restored.json) · [Original protected identities](evidence/protected-near.tsv) · [Writer decision](evidence/writer.json)
