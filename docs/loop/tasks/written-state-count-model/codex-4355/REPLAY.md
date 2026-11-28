# Reproduce the registered full-election comparison

The frozen source identities and group partition are those of [#4354](../codex-4354/REPLAY.md). Use the trained runner and ordinary release client in the [private results packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/blob/d1532be9a08110a30fb94d13b63fd5d57972e682/results/codex/20261128-lexical-full-election/packet.tar.zst), whose repository, immutable revision, path and SHA-256 appear in [the restore receipt](evidence/restore.json). Verify the compressed object, then extract it on Linux and run `sha256sum --strict -c PACKET.sha256` in the private extraction directory. The input tapes contain human conversations. Keep raw input IDs and per-visit traces private.

The shared P4 is stored separately on Wick at `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p4-qual-r2/artifacts/p4.d11`, SHA-256 `e0ffbba19deb34df085d002c1062dd43e88adf98a2aac6aa81d469ab98507776`. Wick keeps the new experiment under `/mnt/plexpool/uor-r4-corpus/runs/20261128-codex-full-election-training`. To rebuild use the source at `0aecb96e4766930fac8f6aee0faa2047aef6e9e8`, the locked runner manifest, and the sm_89 CUDA target. The ordinary client is the #4354 release binary. Set `unit` to this unit, `packet` to the private extraction and `p4` to the verified P4.

```sh
NVCC=/usr/local/cuda/bin/nvcc LEXICAL_CUDA_ARCH=sm_89 cargo build --release --locked --offline --features cuda --manifest-path "$unit/runner/Cargo.toml"
"$packet/bin/uor-lexical-full-election" verify-input "$packet/input" "$packet/internal-split.json"
bash "$unit/cold-check.sh" "$packet/bin/uor-chat" "$p4" "$packet/input" "$packet/attempt01/trial" "$new_cold_output"
```

For a separately registered GPU reproduction, first run `bin/uor-lexical-full-election finite` under the leased physical device lock. Stage the verified base as `p4.d11` beside the runner, input, split, approved manifest, frozen historical trajectory and dispatch script. Execute `dispatch.sh STAGE NEW_ATTEMPT GPU_UUID` under that lock. The historical control compares every stored epoch word and update counter for both address operators and seeds. The eight fitted ordinary client attachments preserve the same writer and word layout as #4354. During serving the native state elects the next original ID from the ordinary causal prefix.
