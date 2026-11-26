# Four-pair TRAIN layout capture

This runner serves [the secondary registration](../TRAIN-LAYOUT.md). It reuses the existing
primary rental's exact SM89 executable, frozen bank, before and after states, GPU UUID and
provider instance. It produces two captures of eight prompts under a separate result root.

The Wick packaging step checks the enrolled panel seal, registration seal, four pair identities,
both selectors for each pair and eight physical requests. The package contains the small ordered
geometry and TSV plus the registration, exact panel manifest and portable entry point. Tokens,
source coordinates and prompts remain available through the sealed Wick panel.

```sh
bash "$RECIPE/stage-layout.sh" package v1 "$STUDY/matched-train/panel-v1" \
  "$LAYOUT_PANEL_SHA" "$STUDY/layout-registration-v1"
```

The authorization step checks the main capture and grade ledgers, verifies a sealed grade of
zero full-signature recurrences across the 32 primary TRAIN-derived pairs, and verifies the
separate non-author FIT. It binds the exact portable payload, the already transferred main
research manifest, both state files, the binary and the original bank. The receipt also pins
the provider instance and GPU UUID used in the primary finite and research runs.

```sh
bash "$RECIPE/stage-layout.sh" authorize v1 "$STUDY/matched-device/package-v1" \
  "$STUDY/grade-v1" "$STUDY/capture-v1" "$STUDY/release-v1" "$STUDY/audit/layout-FIT-v1"
```

Transfer `package-v1/layout.tar.gz` and `authorization-v1/layout-authorization.tar.gz`
from `$STUDY/matched-device`. Confirm their Wick-provided SHA-256 digests. In the original
pod working directory, leave the authenticated `research/` tree in place.

```sh
tar -xzf layout.tar.gz
tar -xzf layout-authorization.tar.gz
bash layout/run-layout.sh "$LAYOUT_PAYLOAD_SHA" research capture-layout-v1 \
  "$GPU_UUID" layout-auth "$AUTH_SHA"
tar -czf layout-result-v1.tar.gz capture-layout-v1
```

The pod checks the registered device and existing input bytes before scoring. It records
hardware, GPU occupancy, both commands, timings, stderr, and a raw manifest for the two arms.
The launcher must hold the normal provider lease for this separate job.

Return the archive and `capture-layout-v1/SHA256SUMS` digest. Place the controller's lease,
probe, job log, original raw `SHA256SUMS` and digest in `$STUDY/matched-device/control-v1`
as `lease.json`, `pod-probe.txt`, `job.log`, `raw.SHA256SUMS` and
`raw-payload-sha256.txt`.

```sh
bash "$RECIPE/stage-layout.sh" import v1 "$LAYOUT_RETURN_TAR" "$LAYOUT_OUTPUT_SHA" \
  "$STUDY/matched-device/authorization-v1/layout-auth" "$STUDY/matched-device/control-v1"
bash "$RECIPE/stage-layout.sh" replay v1 "$STUDY/matched-train/panel-v1" \
  "$STUDY/matched-device/capture-layout-v1" "$AUDIT_BUILD" "$CODEC" "$METHODS"
```

Import preserves the raw manifest, checks absolute ancestry and seals the adjacent lease
receipt. The two arms use the original `before/{native,requests}.jsonl` and
`after/{native,requests}.jsonl` shapes. The replay command runs the existing independent
auditor and row reduction on that cohort and returns a separate sealed result.
