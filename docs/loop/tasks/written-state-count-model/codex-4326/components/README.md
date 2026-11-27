# Reconstruct the joint write components

The four states preserve the original 20,483-word linked prefix and its 4,096-word signed little-endian residual table. The components partition the recorded **joint** update trajectory. `period.i16le` holds the writes from its stage 0 slots, and `post-b.i16le` holds the writes from its stage 1 slots. The [prior fit](../../codex-4325/evidence/fit.json) also produced a separately optimized first-period arm.

On Wick, the predecessor is `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-joint-release` and the component run is `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-release-components`. The latter holds the measured files in `components-v1` and the audited ledgers in `geometry-v3`. For a fresh restoration, set `UNIT` to the #4326 task directory, `PRIOR` to the restored [#4325 run root](../../codex-4325/RESTORE.md), and `NEW` to an existing scratch parent.

Set `ROOTS` to the `roots` member included in the sealed component packet as `frozen-roots.u32le`. On Wick this came from `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-shared-source/staging/base-full-v1.tar.zst`, archive SHA-256 `472dd6345d367ff0455e8ca528037cc94ca625ac49f9d09a3118e1f8b8032a1a`. This member consists of 4,096 packed 16-bit roots in little-endian 32-bit words and has SHA-256 `875a0857dfef840d88a64c90a5777a6154a1674cde08a71429e2cca93fb7bb5a`.

```sh
CARGO_TARGET_DIR="$NEW/components-target" cargo build --offline --locked \
  --manifest-path "$UNIT/components/Cargo.toml"
ACCOUNT="$NEW/components-target/debug/release-component-accounting"
"$ACCOUNT" "$PRIOR/measured-v1/raw/fit" "$NEW/components"
(cd "$NEW/components" && sha256sum --strict -c SHA256SUMS)
cmp "$NEW/components/full.i16le" "$PRIOR/measured-v1/raw/fit/joint.i16le"

"$ACCOUNT" geometry "$PRIOR" "$ROOTS" "$NEW/components" "$NEW/geometry"
(cd "$NEW/geometry" && sha256sum geometry.json *.jsonl > SHA256SUMS)
```

The reconstruction checks the pinned schedule, GPU audit and serialized full state, including signed update deltas, per-slot unique addresses, the cap and margin decision, the frozen-prefix digest and every replayed word. It accumulates (i16) sums with checked arithmetic in chronological order and separately by stage. The greatest absolute running word was 21 in the period component, 32 in the post-B component and 33 in the joint sequence. The recombined state matches the archived 49,158 bytes and SHA-256 `2216c08d143fa15de31a7c31e009c791b5e56ad845f35bd89327afdbdcc29525`.

The geometry command independently replicates the address enumeration in the original [link operator](../../codex-4323/device/src/scope.cu) and its [stage permutation](../../codex-4325/device/src/joint.cu). It constructs each support in first-visit order, deduplicated after hashing. In each recorded writing slot it checks the epoch-dependent starting phase, the original stride across that ordered support and the stage XOR address permutation. From the original roots and sealed token rows it constructs signed feature differences and checks their residual contributions against arithmetic differences in **archived CUDA** scores. Post-B scores require a verified original B token and position in both traces. The command performs offline bookkeeping. The current CUDA replay supplies observed component actions and ordinary D11 checks. The [geometry receipt](../evidence/geometry.json) records the address and contribution census.

For a matched STOP/COPY pair, each address contributes +1 if only COPY visits it and −1 if only STOP visits it. Common addresses cancel in the pair margin. For the four-cell scoped design, the audit adds both COPY feature vectors and subtracts the two STOP feature vectors, giving address coefficients in `[-2, 2]`. A zero interaction vector and a nonpositive frozen COPY-diagonal minus STOP-diagonal imply that fixed-support residual reweighting cannot give all four cells their strict target signs. The observed four-cell vectors were nonzero in both splits. This establishes available feature interactions at this scale. Attaining all four signs across many contexts remains a separate question.

The ledgers preserve each pair identity and its contribution from the frozen scorer, period writes and post-B writes. Address counts across pairs or contexts count incidences. The audit distinguishes addresses never written by a stage from addresses with zero final weight after cancellation. Matched training-slot counts compare each capped write to the exact opposed sibling's support. These counts describe the realized schedule, including its conditional margin violations.
