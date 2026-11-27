# Matched release fitting

The two arms start with the same 40,966 linked-checkpoint bytes and 4,096 zeroed signed little-endian i16 residual words. Only the residual can change. Both stages use the existing first-occurrence distant-root addresses. Stage zero reads those addresses directly. Stage one XORs each address with 2,048, so both stages can update the same table. The frozen stage-two score and commit path come from the prior study. Scoring for the comparison uses the original #4325 scorer binary and cubin.

## Training schedule

Four epochs preserve the exact multiset of #4325 joint exposures by epoch, stage, row identity and action. Each epoch starts with every TRAIN row at the first period, then the #4325 alternating stop/copy post-B exposure pools with their cycling cursors. The driver validates and joins the sealed TRAIN keys to the typed rows. It groups opposed first-period selector siblings with the same one-word-clause ordinal and fixed facts, direct one/two rows at the first period, and direct two/three rows after B. Within each epoch and stage, it walks the original exposure order and places each available sibling immediately after the first unconsumed occurrence of the pair. Both arms receive this same group order. The schedule records the source offsets, identities, target actions and fallback reasons.

Rehearsal exposures, same-action stage-zero rows, stage-one rows lacking an eligible sibling, and surplus exposures without a remaining sibling all use the existing row-wise policy in both arms. Before fitting, the frozen writer authenticates A, supplies the period and commits B from its immutable original position wherever the post-B stage is scheduled.

## Updates and audit

The row-wise arm checks each member against its signed margin of eight in tape order, including the effect of the first member's writes on the second member's score. The paired arm checks both members' margins independently against their common pre-group state and applies signed unit updates only to addresses absent from the other member's support. Identical supports give no eligible update in the paired arm. Both arms cap each row at 16 distinct writes and use the #4325 evenly spaced first-visit rank selection with phase 17 times the epoch. Paired rank selection uses the filtered support. The cap budget and full read supports are matched, while the number of eligible distinguishing coordinates can be smaller.

The device audit reports raw scores before the group, before and after each member's own update, and after the group. It reports the corresponding signed margins and actual address writes classified by membership in both supports or just the writing member's support. A write on an unpaired row has no sibling for classification. The Rust driver reconstructs every state word from the audit, repeats each fit from the untouched starting state and requires matching state and audit bytes. It reloads the serialized state and requires identical GPU score triples.

## Build and commands

Build the host binary on Wick with the pinned Rust dependencies. Build the SM89 AOT image on the leased GPU worker. Its assembly checks hashes against the frozen scorer sources and scans both PTX and SASS for banned instruction classes. Run the finite calibration with PTX JIT disabled.

```sh
CARGO_TARGET_DIR=BUILD_DIR cargo build --locked --release --manifest-path device/Cargo.toml
CARGO_TARGET_DIR=BUILD_DIR cargo test --locked --release --manifest-path device/Cargo.toml
bash device/build-image.sh NEW_IMAGE_DIR
paired-release-device schedule TRAIN TRAIN_KEYS NEW_SCHEDULE_JSONL
CUDA_DISABLE_PTX_JIT=1 paired-release-device finite PAIRED_CUBIN PAIRED_SHA SCOPE_CUBIN SCOPE_SHA
CUDA_DISABLE_PTX_JIT=1 paired-release-device fit PAIRED_CUBIN PAIRED_SHA COPY OCCURRENCE BASELINE \
  BASE_DIR CLASSES READER_WEIGHTS MARKER TRAIN TRAIN_KEYS LINK_WEIGHTS NEW_FIT_DIR
```

The calibration compares the zeroed residual path against the historical linked cubin, checks addresses against hand-derived integer roots, and verifies stage phasing and authenticated commits. At both stages the assigned labels produce scores of −8 and +8 and commit EOS and the original source token. Swapping the labels reverses both signs and both commits. Other fixtures check identical support, reversing member order, a satisfied sibling margin, row-wise writes to shared support and equality of the fallback policy.

The fit produces `rowwise.i16le`, `paired.i16le`, `fit.json`, `schedule.jsonl`, `authenticated.jsonl`, per-arm slot audits and per-row fitted score triples. The file manifest in [the finite evidence](../evidence/device-finite.sha256) is relative to the Wick `device-build-v1` directory.
