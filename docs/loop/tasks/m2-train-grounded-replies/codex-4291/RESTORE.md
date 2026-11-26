# Preserve and restore the mixed transport comparison

The Wick study root is `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-mixed-transport`. The [panel recipe](panel/README.md) identifies the private grammars, matched row order, authenticated history and original address controls. The [registration](PRE-RUN.md) sets the curriculum premise, TRAIN-only elections, untouched transfer gate and separate #4289 comparison.

The private packet keeps the sealed measurement tree and grade. For a passed premise that includes both nine-state trajectories, all 18 physical checkpoint rank traces, signed diagnostics, coordinate votes, checkpoint scores, both elections, elected state bytes and the fresh and historical raw source runs. It also preserves the three private grammars, the near/far endpoint map, source labels, active methods, the physical build and review receipts, and client captures when supplied. The historical source material includes the original #4289 baseline and far rank traces and their authenticated panel identifiers. A failed premise packet records the two frozen parent reads and their eligibility and joint-pair grades.

The compressed packet keeps original prompt text and coordinates, supplying the sealed hashes for the large blind geometry files. The restored verifier regenerates these from the saved histories and user text with the tokenizer, D11 and exporter identified by `provenance/geometry-dependencies.sha256`. Those three files are shared, previously pinned Wick dependencies. The packet contains the independent signed-edge, coordinate-vote and joint-rank audit executables. Wick can replay all grading and checking steps without dispatching a fit or renting GPU time. The target for the compact upload is 16 MiB. `pack.json` reports the actual compressed upload bytes, archive SHA-256 and member count before any upload. [Predecessor tool restore](../codex-4288/RESTORE.md) · [Parent D11 restore](../codex-4286/RESTORE.md)

## Verified archive v1

The archive carries `measurement-v1` and `grade-v1` from `release-v1` and the signed `methods-v2` tree. The curriculum premise passed and the held source gate failed. The archive has no fresh ordinary writer capture.

| Record | Value |
| --- | --- |
| Private dataset | `caseyallard/uor-r4-store` |
| Dataset path | `results/codex/mixed-transport-20261126-9c25a6ec30a2` |
| Immutable revision | `c140f64ef263d1ba0dd8cea482b6d7c2fd972738` |
| Uploaded bytes | 2,898,193 across three objects |
| Compressed archive bytes | 2,715,069 |
| Original members verified | 1,158 |
| Archive SHA-256 | `9c25a6ec30a2cd21760848f8b5f18d6e2eebdcd2097555044a1fb1118b6bb1a0` |
| Methods manifest SHA-256 | `1480c9f725aef1a6f8de7e27fee4190f0dc582bc90dcffcecfc06be6d819c987` |
| Measurement manifest SHA-256 | `c1eb88fcbecd2a6658dee53b05d74862db0fbe42f45a2687345aa917f8a49dfb` |
| Grade manifest SHA-256 | `6e9a69121c72fe7e6e894fcc7a14eedffd8f94960aaf0b6fd66a175d09f9a74c` |
| Reproduced gate SHA-256 | `60279f721cbab14c3fd5de1fcdc0432f90e16fcd870fe68e3d99ed2a8fec5b53` |
| Reproduced bridge retention SHA-256 | `0037f204d30dd16a38e4efe26237aef27857d96e39ce41bd2554f7bf0a692831` |

The Wick receipts are under `storage/archive-v1`. `pack.json` records the compact bytes and member count. `restore-v1/receipt.json` pins the private revision. `local-v1/replayed/receipt.json`, `restore-v1/replayed/receipt.json` and the independent `storage/download-v1/replayed/receipt.json` are byte-identical. Each restore reconstructed all 18 checkpoint rank reads, the elections and coordinate vote observations, and reproduced every substantive fresh and historical grade artifact with zero disagreements.

## Pack and verify locally

Stage the two scripts under the private storage directory on Wick as `storage/methods-v2/`. The commands below record the completed v1 run. Use unused archive and restore versions for a new run. The packer selects the production methods sealed by the signed measurement release. A complete grade is required when the training premise passed, regardless of the held gate result. Supply the actual capture version as the final argument if the ordinary client ran.

```bash
root=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-mixed-transport
method="$root/storage/methods-v2/preserve-wick.sh"
bash "$method" pack v1 v1 v1
bash "$method" restore-local v1 v1
```

For a failed curriculum premise, use `bash "$method" pack v1 v1 -` followed by the same local restore. A failed-premise archive requires a sealed measurement and contains no held or bridge rank outputs.

The verifier checks the release-to-methods hash chain from the archive and records the exact production methods digest in its receipt. It compares the rebuilt coordinate maps and geometry hashes before reconstructing the parent premise from the authenticated physical reads. For a completed fit it compares every checkpoint's joint source rows and opposed pairs, both elections, elected bytes, the independent signed edge reconstruction and the full split vote readback. It then runs the authenticated grader's scoring commands with the extracted inputs. The fresh and historical scoreboards and pair identities, both source witnesses, the signed release gate, the writer endpoints and the historical 32/8 retention report must match byte for byte. The comparison includes the client token trace when one is present. The regenerated manifest and grade input receipt use the replay paths.

## Publish and restore the immutable revision

After the official grade and any authorized client capture are complete and the lab has confirmed publication, the SHA-pinned [compact uploader](../codex-4286/preserve-wick.sh) checks the existing dataset's private status and uploads the verified packet. The adapted uploader takes the immutable revision from the returned commit URL. It verifies the uploaded bytes and the archive members on readback, and the same replay runs from the downloaded files.

```bash
bash "$method" publish v1 v1
cat "$root/storage/archive-v1/restore-v1/receipt.json"
```

For an independent later download, use the published receipt and a new Wick output directory.

```bash
bash "$root/storage/methods-v2/restore-grade-wick.sh" download \
    "$root/storage/archive-v1/restore-v1/receipt.json" "$root/storage/download-v2"
```

The publication receipt records the private dataset path, fixed commit, archive SHA-256, byte budget and verified member count. A successful local or downloaded replay also has a separate `replayed/receipt.json` and output manifest.
