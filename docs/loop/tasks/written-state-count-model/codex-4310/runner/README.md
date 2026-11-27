# S4 source-gate runner

The CUDA build uses the previously qualified P4 row scorer and a separate signed integer gate. The P4 scorer emits the full frozen legal-token score vector and raw route. The gate uses those frozen words for GPU grid fitting and evaluation. Each generation step supplies the treatment's own causal history. The reference continuation supplies targets for fitting and teacher-forced measurements.

Build with nvcc available on the leased RTX 4090 host or an authorized build host.

```sh
cargo build --release --locked --offline --features cuda
s4-output-runner finite
s4-output-runner run P4_D11 R_UORT DEV_TUNE_UORT FIT_JSONL HELD_JSONL OUT_DIR
```

The build emits sm_89 CUBIN and PTX by default. Audit all five entries in the exact build, including `p4_parity_batch_kernel`, `p4_bar_a_windows_kernel`, `s4_fit_accumulate`, `s4_fit_commit` and `s4_forward`. The runner requires pinned P4 and panel identities and creates a new results directory.

- `learned.uors4g01` stores the checked 124-byte sidecar.
- `measurement.json` records device artifact identity, input identities, per-cell losses and supports, replay counts and elapsed stage times.
- `teacher-records.jsonl` stores paired per-record GPU measures. The `zero` arm loads the learned gate in disabled mode.
- `parity-fixtures.jsonl` records 16 fixed open DEV prefixes and the fixed historical-EOS prompt with all six treatments for cold replay.
- `teacher-witnesses.jsonl` preserves the first available held nonzero transfer, changed winner and abstention for exact CLI replay.
- `generation.jsonl` stores token streams and raw trace digests for the fixed free-running prompts.

The natural panel and its generated examples remain under the data custody recorded in the study. Use the declared actual CUDA and ordinary client receipts to interpret the emitted metrics.
