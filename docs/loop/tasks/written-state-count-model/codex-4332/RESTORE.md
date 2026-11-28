# Restore the finite request-to-occurrence witness

The private packet contains the sealed blind and key tapes, independent passage grades, raw integer captures, solver verdict, aligned-site and alias receipts, reverse-order and historical capture comparisons, four ordinary client records, the TRAIN support audit, and the locked Rust source and build dependencies. Keep passage prose, keys and per-row traces in private storage. The final v2 packet includes the source tree for the shared count runtime, which the preliminary v1 packaging omitted. [Revision-pinned v2 packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/a13ee14c1877b197cb0b2a9bfc1e203ed490e388/results/codex/request-site-finite-20261127/request-site-finite.tar.zst) · [Digest receipt](evidence/archive.json)

The two D11 artifacts are referenced by path and SHA-256. On a machine without the Wick copies, recover the balanced D11 from the [#4322 packet](../codex-4322/RESTORE.md) and the matching frozen whole-turn reader from the [#4330 packet](../codex-4330/RESTORE.md). The panel rebuild also checks the original SQuAD TRAIN digest, the three private #4329 tapes, the #4330 panel, and the #4330 tokenizer and class table at their canonical Wick paths. [#4329 input recipe](../codex-4329/RESTORE.md) · [Panel recipe](panel/README.md)

On Wick, set `UNIT` to this study's source directory and use fresh absolute download and restore directories. The restore uses the locked packages already available in Cargo's offline cache.

```sh
RUNS=/mnt/plexpool/uor-r4-corpus/runs
revision=a13ee14c1877b197cb0b2a9bfc1e203ed490e388
packet_path=results/codex/request-site-finite-20261127/request-site-finite.tar.zst
packet_sha=587e809244d91945da273f4c6c63817a522b60ec331e5ada5dc2980ceafd2ceb
hf download caseyallard/uor-r4-store "$packet_path" --repo-type dataset \
    --revision "$revision" --force-download --local-dir "$DOWNLOAD"
bash "$UNIT/archive/restore.sh" "$RUNS" "$RESTORE" "$DOWNLOAD/$packet_path" "$packet_sha"
```

The restore checks all packet members and both immutable D11 digests. It rebuilds the blind and key tapes, reruns the native artifact capture and the integer gate, compares both diagnostics, and verifies the independent reader's source-byte offsets. The archived historical and reverse-order capture comparisons and TRAIN analysis have separate reproduction commands. [Probe recipe](probe/README.md)
