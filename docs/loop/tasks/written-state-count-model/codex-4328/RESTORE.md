# Restore the ordinary source and release study

The private packet occupies 7,900,057 bytes with SHA-256 `b963b6f632ab5b5f5b36dd2b04c6f4ce12e14432f01be86c96fc2bdfa7991541`. Wick keeps the sealed original at `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-ordinary-source/archive-v1/ordinary-source.tar.zst`. Its manifest binds the exact sealed prompts and independent truth, all raw ordinary and release traces, supplied-address controls, graders and outputs, command sources, compiled Linux tools, and the tokenizer. [Revision-pinned private packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/66da603a5010860abf9bb29acc637755872ea737/results/codex/ordinary-source-20261127/ordinary-source.tar.zst)

The large D11 dependencies remain in their frozen Wick locations. To reconstruct them, use the #4319 base writer, #4322 balanced reference, #4326 zero-residual reference, and #4327 row-wise and paired artifact recipes. The original and conditional [input pins](FIT.md) bind the complete artifacts, tokenizer and executables. [Base writer](../codex-4319/RESTORE.md) · [Balanced reference](../codex-4322/RESTORE.md) · [Zero residual](../codex-4326/RESTORE.md) · [Row-wise and paired residuals](../codex-4327/RESTORE.md) · [Conditional pins](release/FIT.md)

On Wick, set `UNIT` to this study's absolute source directory. Choose fresh absolute paths for `DOWNLOAD` and `REPLAY`, with `REPLAY` inside `$RUNS/20261127-codex-ordinary-source`.

```sh
RUNS=/mnt/plexpool/uor-r4-corpus/runs
revision=66da603a5010860abf9bb29acc637755872ea737
packet_path=results/codex/ordinary-source-20261127/ordinary-source.tar.zst
packet_sha=b963b6f632ab5b5f5b36dd2b04c6f4ce12e14432f01be86c96fc2bdfa7991541
hf download caseyallard/uor-r4-store "$packet_path" --repo-type dataset \
  --revision "$revision" --force-download --local-dir "$DOWNLOAD"
bash "$UNIT/controls/readback.sh" "$RUNS" "$REPLAY" "$DOWNLOAD/$packet_path" "$packet_sha"
(cd "$REPLAY" && sha256sum --strict -c readback.sha256)
```

The forced private download matched the sealed Wick file byte for byte. Fresh extraction verified every member, regenerated both panel tapes, and reproduced all 432 ordinary grades and 144 conditional release grades, including the instruction-scope accounting. A fresh native verifier reopened the artifact and reproduced all 48 supplied-address and occurrence-control records, along with the matched contrasts.

Cold text replay matched 72 ordinary records across eight cases and nine arms. It also matched all 12 conditional records across three arms and four cases. The release cases covered the initial verified stop boundary (`c002`), a changed EOS (`c007`), an inactive natural prompt (`c013`), and the verified continuation boundary (`c034`). Comparison retains every recorded model field apart from the artifact pathname. The Wick readback is at `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-ordinary-source/readback-v1/`. [Verified readback manifest](evidence/control-readback.sha256) · [Full ordinary runner](runtime/README.md) · [Full conditional runner](release/README.md)
