# Restore the prospective repeat

Force-download the original pilot and its parents using the [pilot instructions](../codex-4364/RESTORE.md). Keep the resulting step-3334 checkpoint files and the original receipt. Force-download the private replication archive at its recorded revision from the [new receipt](evidence/restore.json). Compare the outer size and SHA-256. Extract to a private, fresh directory and verify its `PACKET.sha256`.

```sh
zstd -dc replication.tar.zst | tar -C "$EXTRACTED" -xf -
(cd "$EXTRACTED" && sha256sum --strict -c PACKET.sha256)
bash "$EXTRACTED/ops/verify-restored-wick.sh" "$EXTRACTED" "$PILOT_CHECKPOINTS"
```

The last command reconstructs the two byte-identical checkpoints and validates every original device member manifest for calibration, authentic training and physical readout. The packet retains the new run's verification evidence and references its original binary inputs by pinned hashes. The ordinary cold output and independent aggregate are included. The [comparison instructions](grade/README.md) describe the separate input views, exact-file comparison, provenance checks and planted errors.
