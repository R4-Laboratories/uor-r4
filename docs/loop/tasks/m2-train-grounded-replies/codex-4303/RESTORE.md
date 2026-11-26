# Reproduce the protected slot-3 source read

The ZFS study root on Wick is `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-third-margin-retain`. The portable file `storage/archive-v1/third-margin-retain-v1.tar.zst` is 105,202,275 bytes with SHA-256 `e3457681969e64ac68c2d1cee37ce8c65aae71fa63e82b77fcafe17452265114`. It contains 5,130 manifest entries covering 1,644,744,910 logical bytes. The included `MEMBERS.sha256` has SHA-256 `048d01e22cc4719d0d3eaeaafe19eb4d216761f5ac887bce804ef125066db47d`. It carries the tokenizer, frozen inputs, device source, fitted states, native captures, supplied controls, independent verifier, reductions and device receipts. [Packing and verification](storage/README.md)

The canonical private Hugging Face store also holds the portable archive, both receipts and their checksums at immutable revision `f15d6215df2024314e22bc114641161fab42c2e2`. All four objects were downloaded from that revision and matched the originals byte for byte, 105,203,244 bytes total. The full rental job capture and intermediate preparations stay on Wick. [Private result at the verified revision](https://huggingface.co/datasets/caseyallard/uor-r4-store/tree/f15d6215df2024314e22bc114641161fab42c2e2/results/codex/third-margin-retain-20261126-e3457681969e) · [Object hashes and readback receipt](storage/hf-receipt.json)

Run the following in a new empty directory on Wick. On another compatible Linux x86-64 host, download `third-margin-retain-v1.tar.zst` from the pinned private revision and set `archive` to that file. The cold replay consumes the extracted packet and writes a new verification directory.

```sh
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-third-margin-retain
archive="$study/storage/archive-v1/third-margin-retain-v1.tar.zst"
printf '%s  %s\n' e3457681969e64ac68c2d1cee37ce8c65aae71fa63e82b77fcafe17452265114 "$archive" | sha256sum --strict -c -
tar -I zstd -xf "$archive"
printf '%s  %s\n' 048d01e22cc4719d0d3eaeaafe19eb4d216761f5ac887bce804ef125066db47d packet/MEMBERS.sha256 | sha256sum --strict -c -
(cd packet && sha256sum --strict --quiet -c MEMBERS.sha256)
bash packet/methods/storage/replay.sh packet verified
```

| Relative Wick directory | Measurement | SHA-256 of directory’s SHA256SUMS |
| --- | --- | --- |
| `physical-train-v1` | Original fits, guards and 295 TRAIN candidate states | `6029544796f50947c9f405841beec5305423e7e3c80708d279f30e37b5994f5e` |
| `election-v1` | Frozen TRAIN election and five artifact hashes | `5067baf1db5744684701f6e19391763c75cff33f97cc892da99854d112f52bca` |
| `physical-held-v1` | Complete held GPU readings and address controls | `cb584f033bfe625cf26800a993acc081cdc5c90da30ef85e13c0626a023f9284` |
| `audit/train-all-v1` | Independently replayed all 232 comparison ranks and all 295 TRAIN candidates | `35de55dcdaa3c0311c9d3c72fa87e3073e800750931356c814e41318d4de5e3a` |
| `audit/physical-held-v1` | Independent replay of all held reads | `eddcd4ee3d1c62d4be7ff6997b3dabae9f1b75caf15b1b305e485e7984a4e1ef` |
| `panel/held-reduction-v1` | Per-prompt spans, original native routes and third-site competitors | `a2028738397039c25f4ac2cbc903f2badd846dc181fae282351d833de948d30f` |
| `storage/archive-v1` | Sealed portable tarball and receipt | `977dec3e4e8910b0dc6b8a245d6dc61ee3aa9ee231456c9b872e33ab8effcc0e` |
| `storage/cold-v1` | Fresh extraction, replay output and elapsed time | `1d7ac4796839ec3b860177a3c7c21705c3ca5c27ba9d180d8b05ab5c3c8e9bba` |

The 29.89-second cold replay regraded the full election, checked native artifact readback and parameter ownership, and recovered all five held grades, both controls and the exact guard identities. The first portable extraction exposed a missing panel-history member. The earlier pack and its failed cold attempt are preserved in the sibling `*-missing-panel-history` directories. The final pack includes the sealed member and passed in a fresh directory.

The complete raw rental job also survives independently at `physical-v1/job.tar.zst`, 131,506,765 bytes, SHA-256 `73fe1fb053f81f6bca6f8db3f24944ddea18a8b5a2e9679cca0e41361ad5e9ff`. All 2,196 files were extracted and checked against the original rental member manifest before deletion of the pod. [Device receipt](evidence/compute.json)
