# Restore the neighboring-root source evidence

The private packet contains the new edge feature tapes, the extraction and collision receipts, the edge and clamp solver records, the admission summaries and delimiter diagnostic, the ordinary-client grades and receipts, and the locked study sources. The feature manifest also covers the clamp, reverse and inverse-clamp tapes, which are regenerated during restore. The private #4332–#4334 packets supply the prompts, keys, tokenizer, preceding captures, locked runtime source, and solver packages. Keep these and the per-row outputs in private storage. [Revision-pinned packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/ff007bdd09ee55902fcae09735c81b0007a6c9bb/results/codex/neighboring-root-source-20261127/neighboring-root-source.tar.zst) · [Archive receipt](evidence/archive.json)

Restore on Wick with the offline Cargo cache, Rust toolchain, GNU core utilities, `jq`, `zstd`, `dpkg-deb` and authenticated `hf`. The two frozen D11 artifacts must be available at the canonical Wick locations listed in #4332. The ordinary replay uses the checksummed client from the #4330 core readback. On a machine without those files, recover them from the [balanced D11 packet](../codex-4322/RESTORE.md) and the [whole-turn reader packet](../codex-4330/RESTORE.md). The [#4332 instructions](../codex-4332/RESTORE.md) also list the canonical inputs for rebuilding its sealed panel.

Set `UNIT` to this study directory in the repository, and `PARENTS`, `DOWNLOAD` and `RESTORE` to fresh absolute directories with existing parent directories. Use the inherited #4332, #4333 and #4334 receipts from the same checkout. Point `D30_RAW` to the historical custody tree containing the sealed D26 carrier and its positive ordinary-client control. [D30 custody record](../codex-30/evidence/STORAGE.md) · [D26 carrier digest](../codex-26/evidence/storage.json)

```sh
RUNS=/mnt/plexpool/uor-r4-corpus/runs
D30_RAW=/home/callard/uor-codex/20261120-context-swap-30/custody-v1/raw
bash "$UNIT/archive/restore-parents.sh" "$RUNS" "$PARENTS"

receipt="$UNIT/evidence/archive.json"
repository=$(jq -er .repository "$receipt")
revision=$(jq -er .revision "$receipt")
packet=$(jq -er .path "$receipt")
hf datasets info "$repository" --revision "$revision" --expand private,sha --json |
    jq -e --arg revision "$revision" '.private == true and .sha == $revision'
hf download "$repository" "$packet" --repo-type dataset --revision "$revision" \
    --force-download --local-dir "$DOWNLOAD"
bash "$UNIT/archive/restore.sh" "$RUNS" "$RESTORE" "$DOWNLOAD/$packet" "$receipt" "$PARENTS" "$D30_RAW"
```

The restore authenticates the compressed packet before extraction and checks its complete file list and individual hashes. It rebuilds the label-free extractor against the pinned native source, compares the feature and replay receipts, reruns the keyed collision analysis, and independently verifies the sealed solver streams. The gate checks cyclic rotation against the #4334 rank and restricted-solver records and rejects a clamped certificate stream on the edge feature tape.

The admission replay verifies the historical carrier and control digests, rebuilds the original probe and the delimiter diagnostic, and compares the reconstructed tapes and summaries with their sealed hashes. The ordinary replay reruns the three artifact arms and checks the transcripts after excluding the absolute artifact pathname. It also checks the grades and summaries.

To rerun the solver search, set `RERUN` and `RECHECK` to fresh directories and use the recovered solver environment. The independently verified archived results remain under `$RESTORE/payload/gate`.

```sh
gate="$RESTORE/payload/source/gate"
bash "$gate/solve.sh" "$PARENTS/4334" "$PARENTS/4333" "$PARENTS/4332" \
    "$RESTORE/features" "$RERUN"
bash "$gate/verify.sh" "$PARENTS/4334" "$PARENTS/4333" "$PARENTS/4332" \
    "$RESTORE/features" "$RERUN" "$RECHECK"
```
