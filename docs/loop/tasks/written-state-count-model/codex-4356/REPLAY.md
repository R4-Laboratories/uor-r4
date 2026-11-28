# Reproduce the native bank-recentering comparison

Use [the private packet receipt](evidence/restore-training.json) to select the repository, path and immutable revision. Verify the compressed SHA-256, extract to a private Linux directory and run `sha256sum --strict -c PACKET.sha256`. The packet contains authentic conversation data and every measured visit. The separate P4 base is identified by its complete SHA-256 in the approved manifest. Its Wick path is `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p4-qual-r2/artifacts/p4.d11`. The original #4356 run is under `/mnt/plexpool/uor-r4-corpus/runs/20261128-codex-bank-recenter/training`. Reconstruct `previous/` from the 16 files listed in `previous-members.sha256`, using the trial directory in [the #4355 immutable packet](../codex-4355/evidence/restore.json). Verify that manifest from inside the reconstructed directory.

For a new research run, build the engineering revision in [the pre-run FIT](FIT.md) on the sm_89 CUDA toolchain with the locked runner manifest. The original release binaries are in the packet. The independent recount in the final review revision includes the raw certificate reread verifier. Set `unit` to this unit directory and `packet`, `previous` and `p4` to the verified extraction, #4355 trial and P4 paths.

```sh
NVCC=/usr/local/cuda/bin/nvcc LEXICAL_CUDA_ARCH=sm_89 cargo build --release --locked --offline --features cuda --manifest-path "$unit/runner/Cargo.toml"
"$packet/bin/uor-lexical-centered-election" verify-input "$packet/input" "$packet/internal-split.json"
"$packet/bin/recount" "$previous" "$packet/attempt01/trial"
bash "$unit/cold-check.sh" "$packet/bin/uor-chat" "$p4" "$packet/input" "$packet/attempt01/trial" "$new_cold_output"
```

For a separately registered GPU collection, stage the files named in `approved.sha256`, including the verified P4 as `p4.d11`, and the two input trees. Run `bin/uor-lexical-centered-election finite` on the exclusive device before starting collection. Under that device lock execute `dispatch.sh STAGE NEW_ATTEMPT GPU_UUID`. It requires an unused result directory. The historical replay enforces both prior schedules, both address methods and update policies at each uncentered visit. The conditional proof tape contains exact Q48 priors and packed bank addresses for a second GPU read of each selected certificate side. The ordinary check starts 19 independent processes from the fixed first TRAIN prefix.
