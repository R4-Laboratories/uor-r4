# Restore the passage-relative source-ranking witness

The private packet holds the relative source-feature and cut tapes, the solver’s witnesses and certificates, the independent verification receipts, and the locked study sources. The checksum manifest also covers the absolute and reverse captures and features, which are regenerated on restore. The earlier packets supply the original passages, prompt tapes, labels, captures, D11 artifacts, tokenizer, and external solver packages. [Revision-pinned packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/0c61787181f4fa666d683a7f50bbb9d5f54afbf3/results/codex/relative-source-finite-20261127/relative-source-finite.tar.zst) · [Archive receipt](evidence/archive.json) · [#4333 restore](../codex-4333/RESTORE.md) · [#4332 restore](../codex-4332/RESTORE.md)

On Wick, set `UNIT` to this study’s source directory, `SIGNED` to a completed #4333 readback, and `REQUEST` to the #4332 readback used to build it. Use absolute paths for these and for fresh `DOWNLOAD` and `RESTORE` directories. The build uses the existing offline Cargo cache.

```sh
RUNS=/mnt/plexpool/uor-r4-corpus/runs
repository=$(jq -er .repository "$UNIT/evidence/archive.json")
revision=$(jq -er .revision "$UNIT/evidence/archive.json")
packet=$(jq -er .path "$UNIT/evidence/archive.json")
sha=$(jq -er .sha256 "$UNIT/evidence/archive.json")
hf download "$repository" "$packet" --repo-type dataset \
    --revision "$revision" --force-download --local-dir "$DOWNLOAD"
bash "$UNIT/archive/restore.sh" "$RUNS" "$RESTORE" "$DOWNLOAD/$packet" "$sha" \
    "$SIGNED" "$REQUEST"
```

The restore authenticates both prior member manifests, both frozen D11 digests, the recovered prompt and key tapes, and the original captures. It rebuilds the feature instrument, reconstructs the cut state, and compares the forward and reverse tapes against the sealed digests. It then rebuilds the independent checker, verifies the archived verdicts on the reconstructed tapes, and compares the check receipts byte for byte. It also reconstructs the four candidate restrictions and verifies their archived verdicts on all three cuts. As a substitution check, the verifier must reject the absolute-frame verdicts on the separator-relative tape for a mismatch in the certificate coefficients or offsets.

To rerun the search itself, use the locked sources and checksummed solver packages recovered by the restore. Set `RERUN` to a fresh directory.

```sh
mkdir "$RERUN"
gate="$RESTORE/payload/source/codex-4334/gate"
for cut in zero separator tail32; do
    bash "$gate/run.sh" "$RESTORE/tools" "$SIGNED" "$REQUEST" \
        "$RESTORE/transport" "$cut" "$RERUN/$cut"
    bash "$RESTORE/payload/source/codex-4334/rivals/run.sh" "$RESTORE/tools" \
        "$RESTORE/transport/fresh-$cut.jsonl" "$SIGNED/panel/truth/key.jsonl" "$RERUN/rivals-$cut"
done
```
