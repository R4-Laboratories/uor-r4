# Preserve and restore the fixed-assignment median study

The archive preserves the selected release and permit, both native TRAIN elections and all 36 candidate reads, fitted payloads, original mixed TRAIN and fresh held input packets, both held cohorts and their 26 physical lanes, source gates, versioned fixtures, calibration and independent audits, and the ordinary writer receipt. It also carries the report and post-hoc source analyses. The [layout](layout-v1.json) selects the accepted Wick versions. The archive script follows their seals to collect each study member once. It copies the historical identity records and any inputs used by the cold replay, and keeps references to other inherited material in the originating manifests.

The cold restore verifies the archive, extracts regular members into a fresh directory and reruns the two TRAIN elections from the physical candidate reads. It reconstructs all held rows and pairs from raw reads and the original coordinate maps, repeats eight typed address replays and regrades both cohorts. It compares the reconstructed artifacts byte for byte and reruns the independent Rust TRAIN and held audits. For a refused source gate it regenerates the writer refusal. For an admitted writer it regrades recorded token traces and checks the writer differential. The client fixture recheck uses captured historical traces.

The compressed object uses fixed tar metadata and sorted paths. A member manifest binds every regular file. The transport fixture requires byte-identical recompression and extraction and rejects an altered compressed object. Study originals remain on Wick.

```bash
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-fixed-assignment-median
methods="$study/storage/methods-v2"
bash "$methods/preserve-wick.sh" fixture v2
bash "$methods/preserve-wick.sh" pack v1 "$methods/layout-v1.json"
bash "$methods/preserve-wick.sh" local v1 v1
```

After a successful local cold replay, the authorized publisher verifies the existing dataset is private. It uploads to a content-addressed path, then forces a new download at the immutable revision and repeats the cold replay. A later restore uses that exact receipt to pin the packet, manifest and expected replay.

```bash
bash "$methods/preserve-wick.sh" publish v1 v1 v1
receipt="$study/storage/archive-v1/published-v1/receipt.json"
bash "$methods/preserve-wick.sh" download v1 "$receipt"
```

Use unused archive, fixture, publication and replay versions for additional attempts.

Archive v1 contains 1,985 regular members and is 18,274,913 compressed bytes. Both TRAIN elections, 26 held lanes and the writer refusal reproduced with zero disagreements across 184 compared files. The private revision and exact SHA-256 digests are in the [immutable receipt](receipt.json).
