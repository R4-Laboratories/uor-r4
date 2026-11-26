# Cold recovery of third-site transport and signed eligibility

The packet includes the captured native GPU reads, the exact artifacts and
operator source, original byte and BPE panel manifests, controls, sealed
election and device receipts. It uses regular files and directories throughout.
Original authority manifests preserve provenance paths, and cold verification
consumes only extracted members. The source packet includes the signed tokenizer,
the frozen D11 projection and word bank, the base fitted state, both fits and
the independent Rust verifier and its tokenizer source.

On Wick, copy this source directory and the exact `codex-4297/bundle.rs` to the
run's `storage/methods-v1/`, then run:

```sh
bash wick.sh stage v1
bash wick.sh train v1
bash wick.sh seal v1 build-v5
bash wick.sh cold v1
```

Staging verifies the input and finite-run seals. The training stage adds the
measured proposal artifacts, paired captures and matched guard/reference
bytes. Sealing adds the election, held reads, selected state readbacks,
independent results and physical device receipts. If the measured election
includes projections, their closed source and raw capture travel with it.
Packing records compressed size, uncompressed member bytes, file count and
wall time. Extraction uses the archive digest and member digest in the archive
receipt. It does not need the staging tree.

The cold verification reruns the complete primary and optional projected
elections from raw GPU captures. It compares all election records, all 28
guard identities on both original spans for each selected edit, all five
fresh grades, the ordered readback bytes and both supplied controls. It
independently checks the held native ranking and word membership for five
states, recalculates exact signed parameter ownership, and replays the M and
E fits against their captured physical results. The winning M is traced
through its native byte snapshot and strongest-third score groups. The
optional projection check reconstructs the chosen bounded same-state
displacement and requires exact byte equality.

For a copy of the compressed archive on a compatible Linux x86-64 machine,
verify the archive SHA-256, extract it using `tar -I zstd`, then invoke
`bash packet/methods/storage/replay.sh packet NEW_OUTPUT`. GNU coreutils, Bash,
jq and a Rust compiler for optional projection validation are required. Cold
verification uses captured GPU evidence and runs the archived integer verifier
on the host.
