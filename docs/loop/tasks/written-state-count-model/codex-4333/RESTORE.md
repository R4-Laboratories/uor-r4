# Restore the finite signed-source witness

The private packet contains the sealed fresh prompt and key tapes, blind source grades, the diagnostic and fresh signed captures, the ordinary fresh capture, the five solver result streams with their integer witnesses or certificates, the post-hoc source-ranking attribution, the independent checker, and the locked study sources. Keep the specification, prose, labels and individual rows in private storage. [Revision-pinned packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/d7429b0c18eb1ad518176021b46b25f7f105cbab/results/codex/signed-site-finite-20261127/signed-site-finite.tar.zst) · [Digest and readback receipt](evidence/archive.json)

The study reuses the two original D11 artifacts, the tokenizer and flat trunk, and the earlier labeled and captured pairs through the pinned #4332 cold restore. The fresh panel builder also checks the pinned #4329 dataset inputs and #4330 prompts through the canonical Wick paths. [#4332 restore](../codex-4332/RESTORE.md) · [Panel recipe](panel/README.md)

On Wick, set `UNIT` to this study's source directory, `PRIOR` to the completed #4332 cold restore, and `DOWNLOAD` and `RESTORE` to fresh absolute directories. Use the existing offline Cargo cache.

```sh
RUNS=/mnt/plexpool/uor-r4-corpus/runs
revision=$(jq -er .revision "$UNIT/evidence/archive.json")
packet=$(jq -er .path "$UNIT/evidence/archive.json")
sha=$(jq -er .sha256 "$UNIT/evidence/archive.json")
hf download caseyallard/uor-r4-store "$packet" --repo-type dataset \
    --revision "$revision" --force-download --local-dir "$DOWNLOAD"
bash "$UNIT/archive/restore.sh" "$RUNS" "$RESTORE" "$DOWNLOAD/$packet" "$sha" "$PRIOR"
```

The restore checks every archive member, both original D11 digests and the #4332 capture. It rebuilds the fresh panel and confirms the 18 independent byte-span grades. It then re-emits both signed captures and the reversed fresh capture from the ordinary artifact, and checks the original hashes and the row-order invariance. Finally it rebuilds the independent verifier and checks the archived old, fresh, joint, positive-only, omitted-row and pooled attribution records against the recovered signed features and labels.

To regenerate the primary and supplemental solver streams, set `SOLVE` to a fresh directory and use the packaged x86-64 solver tools on Wick. The script checks the recovered binaries, verifies each emitted witness independently, and compares the verdict status for every group. The Wick rerun reproduced all 4,460 solver records byte for byte. A separate rebuild using the same extracted solver environment also reproduced the 24 attribution records byte for byte. [Attribution replay](transport/README.md#post-hoc-source-ranking-attribution)

```sh
bash "$UNIT/archive/rerun-solver.sh" "$RESTORE" "$PRIOR" "$SOLVE"
```
