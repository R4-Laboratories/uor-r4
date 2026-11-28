# Restore the opposed opening pilot

Use the private `caseyallard/uor-r4-store` dataset. Download the three archives at the exact revisions in [this pilot's receipt](evidence/restore.json), the [#4361 receipt](../codex-4361/evidence/restore.json), and the [#4363 receipt](../codex-4363/evidence/restore.json). Use `hf download --repo-type dataset --revision REVISION --force-download REPO PATH --local-dir FRESH_DOWNLOAD` for each. Compare the archive bytes and SHA-256 with its receipt before unpacking. Keep the downloads, extracted packets, and replay outputs private.

Extract each archive in its own fresh directory and verify its member manifest:

```sh
zstd -dc ARCHIVE | tar -C EXTRACTED -xf -
(cd EXTRACTED && sha256sum --strict -c PACKET.sha256)
```

From the verified pilot packet, reconstruct the intake and two checkpoints in a fresh directory:

```sh
bash PILOT/tools/verify-restored.sh PARENT_4361 PAIRS_4363 PILOT FRESH_REPLAY
```

The verifier authenticates all three packet manifests, the graph-first TRAIN reservation, and the original intake. The intake references the existing step-3333 checkpoint, historical fixture, tokenizer, and serving cubin in #4361. It uses this pilot's training tape, role messages, and original comparison training cubin. The verifier applies each zstd patch to the frozen checkpoint and checks each reconstructed file against both the training report and the original device member manifest. It writes reconstructed checkpoints under `FRESH_REPLAY/attempt02/checkpoints/`.

The packet keeps both evaluator ELFs and their hashes in `evaluators.sha256`, the original training and finite logs, the refused attempt03, and the corrected attempt04 output. It also keeps the private cold CLI outputs and the source kit, schemas, roster fields, and build provenance. The original cold CLI is under `PARENT_4361/run/ordinary/uor-chat` with SHA-256 `15d5856ebf19200021700f77166a4caa641896c5f54ef0dbec279e0cc70a77e2`. For GPU and cold replay commands see the [readout instructions](serve/README.md). Use the corrected evaluator hash and attempt04 rows for cold replay.

The retrospective four-row calibration is in a separate 45-member [supplemental packet](evidence/capacity-restore.json). Force-download it at the recorded revision, compare the outer SHA-256, unpack it and verify its member manifest. Run `verify-capacity.sh` against its device result directory with trusted receipt SHA-256 `7fceaf35f9ef396e0fba91c79a31a3af8dba6383abcbaa1b9475a822bb0efdfe`. Preserve the earlier pilot and its original release record when verifying the supplement.
