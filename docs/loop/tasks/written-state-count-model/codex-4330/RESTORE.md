# Restore the whole-turn occurrence fit

The private core packet is **7,814,112 bytes** with SHA-256 `86d1141253a1774773aec305d702d56de746ecdc8df3453d9db01778d9d4d34f`. It contains both GPU fit repetitions and epochs, the finite fixtures, sealed TRAIN and evaluation inputs, complete GPU candidate decisions, source grades, the exact run source and Linux product tools. The packet includes stable paths and checksums for the larger Wick artifacts. [Revision-pinned private core packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/c2c5daa20edd2bf7f95d181969c7d553adf8a81a/results/codex/whole-turn-source-fit-20261127/whole-turn-source-fit.tar.zst)

The two new D11 envelopes each embed the 1,202,177,144-byte balanced checkpoint. Use the copy already sealed on Wick or rebuild that checkpoint by following its prior private archive. Both routes give SHA-256 `5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa`. This study's packet supplies the 1,026-byte occurrence words and the exact fit or frozen provenance digest for each envelope. [Balanced reference and upstream archive](../codex-4322/RESTORE.md) · [Product commands](product/README.md)

A separate forced download of the #4322 archive at its original revision reconstructed the pinned balanced checkpoint from the verified #4319 base and recovered balanced fit. Wick retains that reconstruction under `archive-parent-readback-v1`. [Parent reconstruction checksums](evidence/archive-parent.sha256)

On Wick, choose fresh absolute paths for `DOWNLOAD` and `REPLAY`. Set `UNIT` to this study's absolute source directory.

```sh
RUNS=/mnt/plexpool/uor-r4-corpus/runs
BALANCED="$RUNS/20261127-codex-period-margin/cold-v1/balanced/artifact.d11"
revision=c2c5daa20edd2bf7f95d181969c7d553adf8a81a
packet_path=results/codex/whole-turn-source-fit-20261127/whole-turn-source-fit.tar.zst
packet_sha=86d1141253a1774773aec305d702d56de746ecdc8df3453d9db01778d9d4d34f
hf download caseyallard/uor-r4-store "$packet_path" --repo-type dataset \
    --revision "$revision" --force-download --local-dir "$DOWNLOAD"
bash "$UNIT/restore/readback.sh" "$DOWNLOAD/$packet_path" "$REPLAY" "$BALANCED" "$packet_sha"
(cd "$REPLAY/reconstructed" && sha256sum --strict -c SHA256SUMS)
```

The preserved original packet is in `$RUNS/20261127-codex-whole-turn-fit/archive-v1`. The recovered larger artifacts and candidate streams are listed in the packet's `WICK-BULK.sha256`.

The forced download matched the original packet byte for byte. The fresh replay reconstructed both D11 envelopes at their recorded digests, matched all 214,776 candidate records and 3,436,416 integer feature words across the 1,086 cases, and reproduced the full source grades byte for byte. Wick retains the replay in `$RUNS/20261127-codex-whole-turn-fit/archive-readback-v1`. [Cold replay checksums](evidence/archive-cold.sha256)

## Ordinary reply evidence

The **124,349-byte** private ordinary packet has SHA-256 `9988b44733293ff650c1f918d4e9f0f574d9b613034121d7d94d1b44109037c9`. It contains the per-case grades, arm summaries, free-chat counts, comparison, control manifest and scoring source. Wick keeps the full ordinary client traces under `ordinary-v1` and `ordinary-controls-v1`. Their paths and digests are in the packet's `WICK-BULK.sha256`. [Revision-pinned private ordinary packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/24ab9fa6bd6ea1fd60dd2b8b785c4c20b6d5ba5b/results/codex/whole-turn-source-fit-20261127/whole-turn-ordinary.tar.zst)

Use the core replay directory as `REPLAY`. Choose fresh absolute paths for `ORDINARY_DOWNLOAD` and `ORDINARY_REPLAY`.

```sh
revision=24ab9fa6bd6ea1fd60dd2b8b785c4c20b6d5ba5b
packet_path=results/codex/whole-turn-source-fit-20261127/whole-turn-ordinary.tar.zst
packet_sha=9988b44733293ff650c1f918d4e9f0f574d9b613034121d7d94d1b44109037c9
hf download caseyallard/uor-r4-store "$packet_path" --repo-type dataset \
    --revision "$revision" --force-download --local-dir "$ORDINARY_DOWNLOAD"
bash "$UNIT/restore/readback-ordinary.sh" "$ORDINARY_DOWNLOAD/$packet_path" \
    "$ORDINARY_REPLAY" "$REPLAY" "$BALANCED" "$packet_sha"
```

The forced ordinary download matched the sealed packet byte for byte. The original ordinary binary replayed a reserve support pair and one protected duplicate-address prompt across the three rebuilt artifacts. All nine full records matched their measured digests after excluding the relocated artifact pathname. Wick retains those cold replays under `ordinary-archive-readback-v1`. [Ordinary archive and replay receipt](evidence/archive-ordinary.json)
