# Restore the four component states and replay the HELD client

Wick stores the four complete D11 files at `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-release-components/product-v1`. Each file is 1,202,201,720 bytes. The packed full artifact was compared byte for byte with the original #4325 artifact. [Artifact identities and HELD parity](evidence/product-parity.json)

The private packet occupies 9,364,823 bytes and has SHA-256 `986685ba566ee1cc6248599c2a2f4a6578f337d409921cd51245c4625c931b25`. It contains all four 49,158-byte integer states, original joint-fit and signed-write receipts, all seven panels and prior comparison records, both GPU execution orders, the HELD client responses, source-position traces and source to repeat the checks. [Download the revision-pinned private packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/cf7d06f85fa244ca7f2b0b7222cde0c46860c7bd/results/codex/release-components-20261127/release-components.tar.zst)

Restore the frozen D11 writer by its [base writer recipe](../codex-4319/RESTORE.md) and set `BASE` to the resulting absolute path. The expected base SHA-256 is `93147372d495b36fda03fa0115a983e75bf1fd1b132d826212b4e1e0e6a1e89a`.

Build the integer tools using the [product instructions](product/README.md). From a new empty directory on Wick, run the restore driver with the downloaded packet and its pinned digest. Set `UNIT` to the absolute path of this study's source directory.

```sh
PACKET_SHA256=986685ba566ee1cc6248599c2a2f4a6578f337d409921cd51245c4625c931b25
bash "$UNIT/product/readback.sh" "$PWD" "$PACKET" "$PACKET_SHA256" "$BASE"
```

The driver verifies every payload file, compares full historical GPU records in both directions, packs all four artifacts from the restored integer states, and checks the ordinary output and independent diagnostic replay on all 512 HELD rows. It compares restored requests, outputs and receipts to the archived bytes and writes `readback.sha256`. Process timings include model load and remain host measurements. [Fresh private readback hashes](evidence/product-readback.sha256) · [Historical GPU comparison](evidence/product-historical.tsv)

The fresh private extraction reproduced the signed-write reconstruction receipt and all seven grading ledgers and summaries. From the extracted `payload` directory, run the same checks.

```sh
cargo run --locked --manifest-path source/components/Cargo.toml -- \
  product-inputs-v1/trajectory "$PWD/new-components"
cmp components-v1/SHA256SUMS new-components/SHA256SUMS
(cd new-components && sha256sum --strict --quiet -c SHA256SUMS)
bash source/analysis/grade.sh "$PWD" "$PWD/new-grade" "$PWD/product-inputs-v1"
```

The fresh extraction also reproduced the sealed geometry receipts and completed the quartet threshold and feature audits. The geometry instrument reads the original #4325 directory layout. Recreate its small input tree with links to the verified packet files, then compare the geometry receipts to the sealed copies.

```sh
archive="$PWD/product-inputs-v1"
prior="$PWD/new-prior"
mkdir "$prior"
mkdir -p "$prior/measured-v1/raw/fit"
for file in schedule.jsonl joint-slots.jsonl; do
  ln -s "$archive/trajectory/$file" "$prior/measured-v1/raw/fit/$file"
done
for split in held train; do
  raw="$prior/measured-v1/raw/$split"
  mkdir -p "$prior/panel-v1/$split" "$raw/frozen" "$raw/trained"
  ln -s "$archive/panels/$split.jsonl" "$prior/panel-v1/$split/rows.jsonl"
  ln -s "$archive/panels/$split.keys.jsonl" "$prior/panel-v1/$split/keys.jsonl"
  ln -s "$archive/historical/$split/zero.jsonl" "$raw/frozen/candidate.jsonl"
  ln -s "$archive/historical/$split/full.jsonl" "$raw/trained/joint.jsonl"
done
cargo run --locked --manifest-path source/components/Cargo.toml -- \
  geometry "$prior" frozen-roots.u32le components-v1 "$PWD/new-geometry"
(cd new-geometry && sha256sum --strict --quiet -c ../geometry-v3/SHA256SUMS)
bash source/quartets/run.sh measured-v1 "$prior/panel-v1" "$PWD/new-quartets"
bash source/quartets/verify.sh new-quartets new-geometry new-grade "$PWD/new-features"
```

The GPU scorer and frozen runtime are inherited from the [joint-training packet](../codex-4325/RESTORE.md), and the linked comparison for the five earlier tapes is preserved in the [release factorial packet](../codex-4324/RESTORE.md). The component study dispatch and GPU receipt are in `measured-v1`. The archived manifests pin the physical 4090, frozen reader, writer, cubin, model states, source, commands and panel hashes.
