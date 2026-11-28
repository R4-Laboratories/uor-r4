# Reproduce the TRAIN retention measurement

[codex] The private [result packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/blob/18528d77bc109fc8c6a38e6059365497175b6195/results/codex/20261128-lexical-margin-retention/packet.tar.zst) contains the exact inputs, group split, #4353 replay references, runner, ordinary client, CUDA objects, private per-visit records, bound words, five cold outputs and resource logs. Verify the compressed [archive hash](evidence/restore.json), unpack to a private directory, and run `sha256sum --strict -c PACKET.sha256` there. The separately stored base P4 has SHA-256 `e0ffbba19deb34df085d002c1062dd43e88adf98a2aac6aa81d469ab98507776`. Wick's preserved copy is `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p4-qual-r2/artifacts/p4.d11`. The new run is under `/mnt/plexpool/uor-r4-corpus/runs/20261128-codex-margin-retention`.

Extract the [release-client packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/blob/4718b226107d17f0e03603112d00c4bdfde34954/results/codex/20261128-lexical-margin-retention/release-client.tar.zst) separately, verify [its hash](evidence/restore-release.json) and all `RELEASE.sha256` members, and set `release` to that directory. Linux tar preserves every checksum member.

Use the #4354 training source at `4ac43bdfd8cb2fce0be98233c752a42672511394`. Use the release source at `9a0272950` for the ordinary client, final cold checker and serving integration. Set `unit` to this task directory, `packet` to the extracted private directory, and `p4` to the frozen base. Rebuild with the registered Linux toolchain and CUDA sm_89 if binaries need replacement. Use private output paths that do not exist.

```sh
cargo build --release --locked --offline -p uor-r4-integer --no-default-features --features p4-only-client --bin uor-chat
LEXICAL_CUDA_ARCH=sm_89 NVCC=/usr/local/cuda/bin/nvcc cargo build --release --locked --offline --features cuda --manifest-path "$unit/runner/Cargo.toml"
"$packet/bin/uor-lexical-retention" verify-input "$packet/input" "$packet/internal-split.json"
jq -s --slurpfile reported "$packet/attempt01/trial/epochs.json" -f "$unit/recount.jq" "$packet/attempt01/trial/visits.jsonl" > independent.json
cmp independent.json "$packet/audit/recount.output.json"
bash "$unit/cold-check.sh" "$release/release-client/uor-chat" "$p4" "$packet/input" "$packet/attempt01/trial" "$new_cold_output"
```

The independent jq recount uses jq 1.8.1 on Wick. Checkpoints preserve the two choices of address operator as a versioned 584-byte bound native attachment. A separately registered device reproduction runs the packaged `finite` first, then `dispatch.sh STAGE NEW_ATTEMPT GPU_UUID` under the exclusive project lease. Put the base in STAGE as `p4.d11`, beside the approved manifest, staged runner, dispatch script, input and split. The ordinary client replays from the authentic already-consumed lexical boundary, which includes the standalone space when present in the record.
