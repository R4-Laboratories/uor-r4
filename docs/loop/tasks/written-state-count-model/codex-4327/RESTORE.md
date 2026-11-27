# Restore the paired-update study and both D11 files

The revision-pinned packet occupies 9,155,685 bytes and has SHA-256 `f479544f286763b4f580f447cd8da56c5628e12c2ff98e355da24ce0197e9e57`. Wick stores a copy at `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-paired-release/product-archive-v1/paired-release.tar.zst`. The packet contains both original and repeated signed integer fits, schedule and realized write ledgers, authenticated B traces, the matched device binaries and AOT images, both GPU input orders for eight panels, panel keys and grading ledgers, the packed artifact identities, cold device-to-client comparisons, all raw generic CLI turns, and source to replay the checks. [Revision-pinned private packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/f1c5f09d966ae675b428128cca662970c924ebc9/results/codex/paired-release-20261127/paired-release.tar.zst)

Restore the [frozen #4319 D11 writer](../codex-4319/RESTORE.md) and set `BASE` to the file with SHA-256 `93147372d495b36fda03fa0115a983e75bf1fd1b132d826212b4e1e0e6a1e89a`. Build the integer serving tools and ordinary probe on Wick with the [product instructions](product/README.md). Set `UNIT` to the absolute path to this study in the checkout and `PROBE` to the compiled probe. Pick new paths for `DOWNLOAD` and `RESTORED`.

```sh
revision=f1c5f09d966ae675b428128cca662970c924ebc9
packet_sha=f479544f286763b4f580f447cd8da56c5628e12c2ff98e355da24ce0197e9e57
packet_path=results/codex/paired-release-20261127/paired-release.tar.zst
hf download caseyallard/uor-r4-store "$packet_path" --repo-type dataset \
  --revision "$revision" --force-download --local-dir "$DOWNLOAD"
mkdir "$RESTORED"
bash "$UNIT/product/readback.sh" "$RESTORED" "$DOWNLOAD/$packet_path" "$packet_sha" "$BASE" "$PROBE"
(cd "$RESTORED" && sha256sum --strict -c readback.sha256)
```

The readback checks the payload and measurement manifests, identical checkpoint repeats, matched exposure accounting, and original GPU-score threshold diagnostics. It regenerates and compares the row, pair and summary grades for all eight splits. It repacks both D11 files losslessly against the frozen writer, reruns grounded serving across all 512 fresh held prompts with the four-mode comparisons against both GPU orders, and compares the new prompt-only probe records to the frozen base and both fitted run transcripts. The Wick D11 files each occupy 1,202,201,720 bytes and live under the study's `product-v1/rowwise` and `product-v1/paired` directories. [Grounded parity and artifact identities](evidence/product-parity.json) · [Plain dialogue outputs and refusal checks](evidence/product-ordinary.json)

The forced private download matched the Wick packet byte for byte. Its clean extraction reproduced the full grading and accounting records, both packed artifacts, every grounded trace and every plain dialogue transcript. [Verified readback manifest](evidence/product-readback.sha256)

The original and frozen base dependencies are retained by their [joint-training recipe](../codex-4325/RESTORE.md), [component replay recipe](../codex-4326/RESTORE.md), and the per-file static device manifest in this packet. The saved scoring binaries emit `first-period` and `joint` filenames for this study's `rowwise` and `paired` arms, respectively.
