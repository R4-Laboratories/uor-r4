# Restore the far transport comparison

The Wick ZFS study root is `/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-far-transport-margin`. The parent transport state comes from [#4288](../codex-4288/RESTORE.md). Its two lexical governor banks and the fixed D11 have separate entries in the [released manifest](evidence/released-inputs.sha256). The [panel method](panel/README.md) pins the original tokenizer, genuine history, source exports and private grammars. Both training and blind serving use the native i16 state and the predecessor’s integer distance operator.

The [SM89 fitter](fit/README.md) documents the eight updates, TRAIN-only state selection and synthetic controls. The released executable is `fit/build-v3/transport-margin-sm89`. The clean physical fixture is `fit/finite-v4`. The [independent Rust oracle](audit/src/main.rs) reconstructs every update and checks the entire serialized bank. The [measurement](measure-wick.sh) freezes selected states before blind ranking, checks joint decisions against a second implementation and compares 768 original-address control decisions. Use new version directories for reproductions. The [FIT](FIT.md) identifies the earlier failed instruments and the [preregistration](PRE-RUN.md) gives the release rule for the ordinary client.

The [private receipt](evidence/private-restore.json) identifies an immutable revision of the existing private dataset. The compact archive contains the grammars, both references and coordinate maps, source-supervision and intervention packets, parent and fitted states, per-prompt original and updated signed margins, every blind arm’s rankings, the full held gate and its inputs. The 76 MiB training and 60 MiB held blind CUDA input files are also on Wick ZFS. The sealed panel generator reconstructs them from the archived grammars and the pinned source instruments.

Restore to a new directory on Wick. Set the two version variables below to the collection and grade recorded in [the report](REPORT.md).

```bash
set -euo pipefail
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-far-transport-margin
receipt="$study/restore-v1/receipt.json"
recovery=$(mktemp -d "$study/restore-v1-replay.XXXXXXXX")
repo=$(jq -er .repo "$receipt")
prefix=$(jq -er .path "$receipt")
revision=$(jq -er .revision "$receipt")
"$HOME/.local/bin/hf" download "$repo" --repo-type dataset --revision "$revision" \
    --include "$prefix/*" --local-dir "$recovery/download" --quiet
packet="$recovery/download/$prefix"
(cd "$packet" && sha256sum --strict --quiet -c SHA256SUMS)
mkdir "$recovery/extracted"
tar --zstd -xf "$packet/measurement.tar.zst" -C "$recovery/extracted"
(cd "$recovery/extracted" && sha256sum --strict --quiet -c "$packet/MEMBERS.sha256")
measurement_version=v2
grade_version=v1
bash "$recovery/extracted/methods-v1/restore-grade-wick.sh" "$recovery/extracted" \
    "$measurement_version" "$grade_version" "$study/restored-grades-v$(date +%s)"
```

The [restore comparison](evidence/restored-grades.sha256) covers the signed edge tables and summaries for both fits, the source tables and scoreboards on both splits, the source endpoints supplied to the gate, the signed gate result and the source-error trace.
