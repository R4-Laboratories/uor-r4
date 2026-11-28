# Restore the whole-turn source supervision study

The sealed packet is 285,530 bytes with SHA-256 `8bfe82fcd3839afb78be3e3636c7ae03d1336bd756b3e061a6c7c0b062a0fddf`. Wick retains it at `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-whole-turn-census/archive-v1/whole-turn-supervision.tar.zst`. The packet contains the selected SQuAD 2.0 prompts and private labels with dataset attribution, the independent authored comparisons, the production tokenizer, locked Rust packages, and the numeric census, semantic review and ordinary replay receipts. The private Wick manifest binds the raw donor input and census rows, semantic review texts and traces, and original replay files in their local custody. [Revision-pinned private packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/8e3997c80db470262b34973065392e5970744d51/results/codex/whole-turn-source-supervision-20261127/whole-turn-supervision.tar.zst) · [SQuAD attribution](archive/ATTRIBUTION.md) · [Packet membership](evidence/archive-members.sha256) · [Private Wick fingerprints](evidence/archive-private-wick.sha256)

On Wick, set `UNIT` to this study's absolute source directory. Choose fresh absolute paths for `DOWNLOAD` and `RESTORE` inside the Wick run root. Cargo needs the locked crates in its cache.

```sh
RUNS=/mnt/plexpool/uor-r4-corpus/runs
revision=8e3997c80db470262b34973065392e5970744d51
remote_packet=results/codex/whole-turn-source-supervision-20261127/whole-turn-supervision.tar.zst
sha=8bfe82fcd3839afb78be3e3636c7ae03d1336bd756b3e061a6c7c0b062a0fddf
hf download caseyallard/uor-r4-store "$remote_packet" --repo-type dataset \
  --revision "$revision" --force-download --local-dir "$DOWNLOAD"
bash "$UNIT/archive/restore.sh" "$RUNS" "$RESTORE" "$DOWNLOAD/$remote_packet" "$sha"
(cd "$RESTORE" && sha256sum --strict -c READBACK.sha256)
```

The original data paths are pinned in the private Wick manifest. The rebuild uses the intact author-published SQuAD files from the study's Wick input directory when present, or downloads the same pinned TRAIN and DEV versions from the [publisher](https://rajpurkar.github.io/SQuAD-explorer/). The census rebuild also opens the exact corrected #4314 tape, raw #4312 files and #4315 historical receipt. The restored ordinary replay receipt documents the earlier serving experiment. Source reconstruction uses the tokenizer and corpus instruments only. [Census recipe](census/README.md) · [Natural candidate recipe](supply/README.md) · [Authored comparison recipe](contrasts/README.md)

The restore verifies 65 archived members and 33 separately retained Wick files. It rebuilds and byte-compares the 710 natural rows across all three splits, the 360 authored comparison rows, their two independent check summaries, and the 4,345 original private census rows and summary. The revision-pinned forced download matched the Wick packet byte for byte, and a fresh extraction and build matched all 13 outputs from the local cold readback. [Restored file hashes](evidence/archive-readback.sha256) · [Archive receipt](evidence/archive-receipt.json)

For interpretation of the original natural tapes, the subsequent [semantic label review](evidence/label-review.md) supersedes any descriptions in the sealed packet that imply all published impossible questions are genuinely unsupported.
