# Regrade the two executions

Build the [independent Rust grader](../../codex-4364/grade/README.md) from the project with Cargo in release mode. It reads saved native state, the private token tapes and captures. It verifies actual token IDs, scores, stop reasons and bytes. The two shell programs here use the original Wick layout and the operator’s local compute ledger to assemble separate private input views and compare them. The private replication receipt provides a durable copy of the inputs and run evidence. Keep the views in a private working directory.

```sh
./stage-view.sh restored "$PWD/pilot"
./stage-view.sh replay "$PWD/repeat"
./compare.sh "$PWD/pilot" "$PWD/repeat" "$GRADER" "$PWD/verified"
```

Each view authenticates its own receipts first. The comparator checks direct equality of the native checkpoints, binaries, all raw GPU rows and all 12 ordinary client traces. For the short training and complete authentic receipts it permits a different physical device UUID. References to newly generated finite and training receipts must match their own input files before their hashes are excluded from the cross-run comparison. All other fields of the paired JSON reports are compared. The grader includes planted changes to one output ID, a checkpoint arm, a cold-client arm and a clipping flag. The job-custody check verifies the new lease and the ordered calibration, training and readout jobs. A different GPU UUID is recorded as a property of the particular replay.
