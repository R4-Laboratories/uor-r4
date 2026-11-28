# Replay the opening trace

Use the private `caseyallard/uor-r4-store` dataset with the path, revision and SHA-256 in the [parent packet receipt](../codex-4361/evidence/restore.json) and the [derived packet receipt](evidence/restore.json). The latter contains the physical score rows, binary, device and job records, raw pair ledgers and grading evidence. It deduplicates the six intake files against the parent. Verify each downloaded archive before unpacking into a fresh private directory and check every member against its `PACKET.sha256`.

From this unit directory on a leased RTX 4090, run the following with two verified, unpacked packets. The first command checks the exact binary, writer kernel, source artifacts, tape and fixture against this revision. The second runs the physical controls followed by all three checkpoints. Both output directories must be fresh.

```sh
./restore-intake.sh /private/4361 /private/4363 /private/new-intake
./capture.sh /private/new-intake /private/new-attempt
```

To build the instrument, use `cargo build --release --locked --offline --bin compact-v3-train-opening-margin` in this unit of a complete pinned UOR-R4 checkout on Linux. The [source](src/main.rs) reuses the [calibrated physical operator](../codex-4360/runner/src/bin/device.rs) and loads a precompiled sm_89 cubin. Archive intake hashes bind the executable used for this measurement. A different compiler build may produce different ELF bytes.

The [recount](recount/README.md) reads the stored hidden words, checkpoint rows, physical scores and complete transition signatures. Run the following for the supplementary grading and the model-free artifact inventory. The inventory covers the actual serialized parameters. The score identity reconstructs changes between two observed endpoints, including the signed hidden and weight terms for a fixed rival. Outputs are in the same order as the checkpoints supplied to the commands.

```sh
i=/private/new-intake/inputs
t=/private/4363/received/attempt01
cargo build --release --locked --offline --bins
target/release/weight_inventory "$i/base.uorbfy03" "$i/reset.uorbfy03" "$i/full.uorbfy03"
target/release/score_change "$t/rows.jsonl" "$i/base.uorbfy03" "$i/reset.uorbfy03" "$i/full.uorbfy03"
cargo run --release --locked --offline --manifest-path recount/Cargo.toml --bin assignment -- "$t/rows.jsonl"
```

The [paired-corpus instructions](pairs/README.md) give the original corpus and historical token-stream dependencies for reconstructing both rosters. Recounting the preserved private rosters alone only checks the downstream pairing.
