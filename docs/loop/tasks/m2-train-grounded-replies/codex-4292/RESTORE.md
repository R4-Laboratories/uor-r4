# Preserve and restore the protected integer transport comparison

The private archive preserves the exact candidate states, original-address rank traces, signed edge observations, frozen menu, both elections and rejection witnesses, independent audit receipts, and the held input and control evidence. It also preserves the released methods and decision, the build and lease receipts, the private grammar drafts, original-address labels and ordinary client evidence when the gate releases a capture. The complete production archive uses the [sealed held panel](panel/README.md) and the [pre-registered comparison](PRE-RUN.md).

The compact archive omits the held blind geometry and the earlier instrument's structural geometry. Its member list retains the original hashes, user strings, coordinate map, history, labels, and side information. The verifier regenerates both geometry files from the pinned Wick dependencies, compares the supplied packets, and checks every member of the complete held seal. The inherited TRAIN panel and states stay pinned to the shared [#4291 private archive and release](../codex-4291/RESTORE.md). The signed D11 and tokenizer use their original Wick paths and the [earlier addressed-source custody record](../codex-4286/RESTORE.md).

The independent restore replays every recorded proposal's native state, signed edges, and original byte and BPE read on the archived physical batches. It compares the independent receipt bytes, reruns each election from the saved batch, and checks the selected state files. It then re-joins the held raw traces, compares the per-pair source results, checks the independent structural-control addresses, and recomputes the held grade. These readbacks use already recorded physical outputs. They require no GPU dispatch or serving replay.

## Published archives

Both archives were published to the private `caseyallard/uor-r4-store` dataset. Six objects total 12,394,663 bytes. Wick downloaded every object at its immutable revision and compared the downloaded bytes with the sealed local packet.

| Record | Primary Wick comparison | Vast supplement |
| --- | --- | --- |
| Dataset path | `results/codex/protected-transport-20261126-da4a4ca4f04b` | `results/codex/protected-transport-vast-20261126-08530e0315a8` |
| Immutable revision | `f1a5584277f7975ce208f42f0df5f22648650b5e` | `0bc4bbfb25f6707d7c36ce3c326cb9184e80d382` |
| Uploaded bytes | 5,229,600 | 7,165,063 |
| Compressed archive bytes | 4,698,241 | 7,134,399 |
| Archived members | 4,211 | 266 |
| Archive SHA-256 | `da4a4ca4f04b884bb2b005a98a08b398b7115cc990b122398a0d54334e03317a` | `08530e0315a8a20db84922ab9d4df132e5b73e3f8cd38e9a3a857cefd3fa1686` |
| Replay receipt SHA-256 | `f79b4da089b5c2e6ba27a7c566d7ceabbd5a681d8deed0dca2f33e86f6e573e3` | `d24ba1c8a38c26ba775e068a3d47863a6fb57aaf351422315d0172b4234fd44a` |

The primary archive includes `measurement-v1`, `transfer-v1`, `grade-v1`, and `report-v1`. From the downloaded copy, Wick reconstructed two geometry files, independently replayed all three unique physical TRAIN batches and rejoined all nine held arms, and reproduced the held grade with zero disagreements. The held gate refused writer capture. The Wick publication receipt is `storage/archive-v1/restore-v1/receipt.json`. The local restore, publication readback and separate immutable download at `storage/download-v1/replayed/receipt.json` produced byte-identical replay receipts.

The supplement retains the provider payload, launch and lease receipts, the first attempt, signed finite and TRAIN checks, and the pod lifecycle and cost record. All 746 manifested Vast root device files and 249 grade files were byte-identical to their matching Wick files. The supplement records each digest and the matching path in the primary archive. Its cold restore rebuilt those 995 members from the downloaded Wick archive and checked the complete Vast capture and build manifests. The original raw files are retained on Wick. The Wick publication receipt is `storage/vast-archive-v1/download-v1/receipt.json`. The local restore, publication readback and separate immutable download at `storage/vast-archive-v1/restore-v3/receipt.json` produced byte-identical replay receipts.

## Local pack and restore

Use the staged scripts in the Wick storage directory. Substitute the actual sealed measurement, transfer and grade versions. With a failed TRAIN premise, pass `-` for both transfer and grade. With a passed held gate, give the captured ordinary client version as the final argument to `pack`. The packer reads the sealed release and checks that the measurement, audit authorization, transfer, grade, and any client receipts bind each other. It creates the compressed archive directly from the manifest members and reports the actual compressed byte count in `pack.json`.

```bash
root=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-protected-transport
method="$root/storage/methods-v3/preserve-wick.sh"
bash "$method" pack v1 v1 v1 v1
bash "$method" restore-local v1 v1
cat "$root/storage/archive-v1/local-v1/replayed/receipt.json"
```

The panel-only restore fixture is at `storage/fixture-v1`. Its compact archive is 154,360 bytes with SHA-256 `b1aa8093d430f8d0c9e619694c1b02dfda1b2a5aff84bbf51807c40f6b64d01f`. It reconstructed both geometry members and compared the complete held seal with zero disagreements. The fixture exercises compact custody before any candidate result is available.

## Private store and readback

Once publication is approved and the local replay passes, the uploader verifies the store's private status and takes the immutable revision from the commit response. It downloads all three objects at that revision, compares the archived bytes and member hashes, and reruns the restore. The publication receipt records the revision, dataset path, archive SHA-256, file count and uploaded bytes.

```bash
bash "$method" publish v1 v1 v1
cat "$root/storage/archive-v1/restore-v1/receipt.json"
```

For a later independent readback, use the immutable receipt and an unused Wick output directory.

```bash
bash "$root/storage/methods-v3/restore-grade-wick.sh" download \
    "$root/storage/archive-v1/restore-v1/receipt.json" "$root/storage/download-v2"
```

The supplement needs the primary archive's restored first batch. From the supplement's immutable revision, the following commands download the three objects and reconstruct its shared raw and grade members.

```bash
bundle="$root/storage/vast-archive-v1"
receipt="$bundle/download-v1/receipt.json"
download="$bundle/download-v3"
mkdir "$download"
repo=$(jq -er .repo "$receipt")
revision=$(jq -er .revision "$receipt")
prefix=$(jq -er .path "$receipt")
"$HOME/.local/bin/hf" download "$repo" --repo-type dataset \
    --revision "$revision" --include "$prefix/*" --local-dir "$download/download"
bash "$root/storage/vast-methods-v2/preserve-vast-wick.sh" restore v1 v4 \
    "$root/storage/archive-v1/restore-v1/extracted" "$download/download/$prefix"
```
